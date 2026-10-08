"""LLM pseudo-labelling of pool messages via OpenRouter.

Runs OFFLINE, ahead of time. Nothing here is ever imported by the classify
pipeline -- hard rule 3 says inference makes zero network calls, so this file
and the trained model must stay strictly separate.

Design notes, i.e. why this does not look like the lecture's call_model loop:

1. BATCHED CALLS. The prompt carries the whole intent list plus 5 few-shot
   examples per class -- ~6.6k tokens for A, ~10.2k for B. Sending one message
   per call re-pays that for every message: 231M input tokens, about $460.
   Batching 50 messages per call amortises it down to ~4.9M tokens, about $13.

2. ID-KEYED OUTPUT, NEVER POSITIONAL. The risk batching introduces is that a
   model returns 49 results for 50 messages. Matching by position would then
   silently shift every label after the gap, producing wrong-but-plausible
   labels that QA would only see as unexplained mediocrity. So we ask for a
   JSON object keyed by message id, and verify every id came back.

3. RESUMABLE. ~515 calls is long enough that a crash mid-run matters. Results
   append to a JSONL file as each batch lands; re-running skips ids already
   present.

Usage:
    python src/labeller.py benchmark_a --dry-run     # cost estimate, no calls
    python src/labeller.py benchmark_a --limit 100   # small live test
    python src/labeller.py benchmark_a               # the real run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time

import pandas as pd
import requests
from dotenv import load_dotenv

from config import REPO_ROOT, artifact_dir, labels_dir
from data import load_intents, load_pool_unlabeled

load_dotenv(REPO_ROOT / ".env")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("LABELLER_MODEL", "anthropic/claude-sonnet-5")
EXTRA_HEADERS = {"X-Title": "Ostr-AI 2026 pseudo-labelling"}


def model_slug(model):
    """Filesystem-safe form of a model id, for namespacing output files.

    Output paths MUST include this. Without it the resume logic ("skip ids
    already in the file") happily serves one model's labels for another
    model's run: the run reports 0 batches and $0.00, and any comparison
    between models silently becomes a comparison of a model with itself.
    """
    return model.replace("/", "_").replace(":", "_")


def get_api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key or key.startswith("sk-or-v1-..."):
        raise SystemExit(
            "No OPENROUTER_API_KEY. Copy .env.example to .env and add your key."
        )
    return key


def load_fewshot(benchmark):
    """The 5-per-class examples the labeller is allowed to see.

    Deliberately reads fewshot.tsv and never labeled_examples.tsv: the other
    half (holdout.tsv) has to stay unseen for the QA gate to mean anything.
    """
    path = artifact_dir(benchmark) / "fewshot.tsv"
    if not path.exists():
        raise SystemExit(f"{path} missing -- run: python src/make_splits.py")
    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)


# Written from the 11 errors a human found in the 120-message uniform audit,
# plus the top confusion pairs where our two independent labellers disagreed.
#
# The reason this exists rather than a bigger encoder: RoBERTa-base, at twice
# DistilBERT's capacity, reproduced all 11 errors exactly. Two models trained
# on the same labels failing in the same places means the failures are in the
# labels. And 7 of the 11 sit on rows where BOTH sonnet and gemini agreed --
# so cross-checking labellers cannot find them either. Both frontier models
# share the same blind spot, and only an instruction that names the
# distinction can close it.
#
# Every rule below cites the audited message that motivated it, so the block
# stays auditable and nobody has to guess later why a rule is here.
RULES_V2 = {
    "benchmark_a": """DISAMBIGUATION RULES -- apply these BEFORE falling back to example similarity.

1. ACTUAL EVENT vs GENERAL QUESTION.
   Is the customer reporting something that happened to them, or asking how
   the service works in general? A message with no first-person event marker
   ("my", "I was", a date, an amount) is usually the general/policy reading.
   - "Do you have charges now when you pay with a card?" is a policy question,
     NOT a report that a fee was charged to them.

2. NOT YET vs FAILED.
   "Hasn't gone through yet", "still pending", "how long does it take" describe
   something IN PROGRESS. Never pick a category meaning the action FAILED, was
   REJECTED, or that the customer was UNABLE to do it.
   - "My ID verification hasn't gone through yet" is pending, NOT
     unable_to_verify_identity.

3. SUCCEEDED vs FAILED.
   Read whether the outcome actually happened. If the customer says they
   received it, were refunded, or it went through, do NOT pick a
   not-received / not-showing-up / failed category.
   - "I just got refunded for my purchase" means the refund HAPPENED.
   - If they received something but the AMOUNT was wrong, that is a
     wrong-amount category, not a not-received category.

4. WHETHER or HOW vs WHY.
   "Is X required?", "Can I avoid X?", "How do I check X?" ask WHETHER or HOW.
   "Why is X required?" asks for a REASON. Never use a why_* category for a
   whether/how question.
   - "Is identity verification required to use my account?" asks WHETHER.
   - "Is there any way to bypass identity verification?" is about avoiding it,
     not about having failed it.

5. DIRECTION OF MONEY.
   Decide whether the customer is SENDING or RECEIVING before choosing.
   "I haven't got my payment yet" is usually money INCOMING to them;
   "my payment hasn't gone through" is usually money OUTGOING from them.

6. CHANNEL: card purchase vs bank transfer vs cash withdrawal vs top-up.
   Match the mechanism the customer actually names.
   - "a purchase I made" is a card payment, not a transfer.
   - "while I was taking my money out" is a cash withdrawal, not a charge
     noticed on a statement.
   - Use transfer categories only when they describe sending money to a
     person or account.

When two categories still fit equally well, prefer the one whose examples match
the customer's SITUATION rather than merely their vocabulary."""
}


def load_all_labeled(benchmark):
    """All 10 labelled examples per class, for the 10-shot prompt.

    fewshot.tsv holds 5 per class; the other 5 went to holdout so the labeller
    could be QA'd against examples it had never seen. That gate is gone -- it
    read 98.4% while hand review of the same labeller's pool output read 85%,
    because holdout is curated and capped at 60 characters while the pool is
    not. Since we no longer believe the number it produced, reserving half the
    ground truth to compute it is a poor trade: those 5 extra examples per class
    are better spent showing the labeller what the categories actually look
    like, which is the thing it is demonstrably getting wrong.

    Reading labeled_examples.tsv directly rather than concatenating the split
    files, so this does not depend on make_splits having been run.
    """
    from data import load_labeled

    return load_labeled(benchmark)[["text", "intent"]]


def build_system_prompt(benchmark, rules="v1", shots=5):
    """The static half of the prompt: task, label space, and examples.

    Identical for every batch, which is exactly why batching pays off -- this
    block is the expensive part and we want to send it as few times as we can.

    Examples are grouped under their intent rather than listed flat, so the
    model sees the boundary between neighbouring categories rather than having
    to infer it from scattered pairs.

    rules="v2" appends the disambiguation block above. v1 is kept as the
    default and unchanged, so the labels already paid for stay reproducible
    and the two prompts can be scored against each other.
    """
    intents = load_intents(benchmark)
    if shots >= 10:
        fewshot = load_all_labeled(benchmark)
    else:
        fewshot = load_fewshot(benchmark)

    blocks = []
    for intent in intents:
        rows = fewshot[fewshot["intent"] == intent]
        examples = "\n".join(f"  - {t}" for t in rows["text"])
        blocks.append(f"### {intent}\n{examples}")
    examples_block = "\n".join(blocks)

    extra = ""
    if rules != "v1":
        block = RULES_V2.get(benchmark) if rules == "v2" else None
        if block is None:
            raise SystemExit(f"no '{rules}' rules defined for {benchmark}")
        extra = "\n\n" + block

    return f"""You label short customer messages with exactly one intent category.

There are {len(intents)} categories. You must use one of these names verbatim:
{", ".join(intents)}

Here are labelled examples of each category:

{examples_block}

RULES:
- Every message gets exactly one category, chosen from the list above.
- Use the exact category name, character for character.
- Never invent a category. If a message is ambiguous, pick the single closest
  category rather than refusing.
- Some categories are deliberately similar. Read the examples above carefully
  to tell neighbouring categories apart.{extra}

OUTPUT FORMAT:
Return ONLY a JSON object mapping each message id to its category name.
No prose, no markdown fences, no explanation. Example shape:
{{"101": "some_category", "102": "another_category"}}
Include an entry for every id you were given."""


def build_user_prompt(batch):
    """The per-batch half: just the messages, tagged with their real ids."""
    lines = [f"[id={r.id}] {r.text}" for r in batch.itertuples()]
    return (
        f"Classify these {len(batch)} messages. "
        f"Return a JSON object with exactly {len(batch)} entries, "
        "one per id.\n\n" + "\n".join(lines)
    )


def _extract_json(content):
    """Pull a JSON object out of a model response.

    Tolerates markdown fences and stray prose around the object. The greedy
    {.*} takes the outermost braces, which is what we want when the model
    wraps its answer in commentary.
    """
    if not content:
        return None
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def call_openrouter(messages, model, api_key, retries=4, timeout=180):
    """One chat completion, with backoff. Returns dict or None.

    Retries on anything: rate limits, timeouts, transient 5xx. A single failed
    batch would otherwise lose 50 labels.
    """
    for attempt in range(retries):
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers={"Authorization": f"Bearer {api_key}", **EXTRA_HEADERS},
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0,  # deterministic: same input, same labels
                    "max_tokens": 8000,
                },
                timeout=timeout,
            )
            if resp.status_code == 429:
                raise RuntimeError("rate limited")
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage", {}) or {}
            return {
                "content": data["choices"][0]["message"]["content"],
                "cost": usage.get("cost", 0) or 0,
                "in_tokens": usage.get("prompt_tokens", 0),
                "out_tokens": usage.get("completion_tokens", 0),
            }
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, then retried
            if attempt == retries - 1:
                print(f"    batch failed permanently: {exc}")
                return None
            wait = 2**attempt
            print(f"    call failed ({exc}); retry in {wait}s")
            time.sleep(wait)
    return None


def label_one_batch(batch, system_prompt, valid_intents, model, api_key, max_passes=3):
    """Label one batch, re-requesting any id that comes back missing or invalid.

    Returns (labels dict, cost, n_failed). Anything still unresolved after
    max_passes is left out entirely rather than guessed -- an unlabeled row we
    can see beats a wrong row we cannot.
    """
    labels = {}
    cost = 0.0
    pending = batch

    for _ in range(max_passes):
        if pending.empty:
            break
        result = call_openrouter(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_prompt(pending)},
            ],
            model,
            api_key,
        )
        if result is None:
            break
        cost += result["cost"]
        parsed = _extract_json(result["content"]) or {}

        # Keep only ids we actually asked for, with an intent that really exists.
        # A hallucinated category name is rejected here rather than poisoning
        # the training set downstream.
        wanted = set(pending["id"])
        for key, value in parsed.items():
            key = str(key).strip()
            if key in wanted and isinstance(value, str) and value.strip() in valid_intents:
                labels[key] = value.strip()

        missing = wanted - set(labels)
        if not missing:
            break
        pending = pending[pending["id"].isin(missing)]

    return labels, cost, len(set(batch["id"]) - set(labels))


def label_messages(
    benchmark,
    df,
    out_path,
    model=DEFAULT_MODEL,
    batch_size=50,
    workers=8,
    dry_run=False,
    rules="v1",
    shots=5,
):
    """Label every row of df (columns id, text), appending to out_path as JSONL.

    Resumes: ids already in out_path are skipped, so re-running after a crash
    or a Ctrl-C picks up where it stopped.
    """
    intents = load_intents(benchmark)
    valid_intents = set(intents)
    system_prompt = build_system_prompt(benchmark, rules, shots)

    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(str(json.loads(line)["id"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    todo = df[~df["id"].isin(done)].reset_index(drop=True)

    batches = [todo.iloc[i : i + batch_size] for i in range(0, len(todo), batch_size)]
    sys_tokens = len(system_prompt) // 4
    print(
        f"{benchmark}: {len(df)} messages, {len(done)} already done, "
        f"{len(todo)} to label in {len(batches)} batches of {batch_size}"
    )
    print(f"  model: {model}   prompt rules: {rules}   few-shot: {shots}/class")
    print(f"  system prompt: ~{sys_tokens:,} tokens, sent once per batch")

    if dry_run:
        msg_tokens = sum(len(t) // 4 for t in todo["text"])
        est_in = len(batches) * sys_tokens + msg_tokens
        est_out = len(todo) * 12
        print(
            f"  DRY RUN -- would send ~{est_in/1e6:.2f}M input + "
            f"~{est_out/1e6:.2f}M output tokens"
        )
        print("  no API calls made")
        return None

    api_key = get_api_key()
    lock = threading.Lock()
    totals = {"cost": 0.0, "failed": 0, "done": 0}

    def run(batch):
        labels, cost, failed = label_one_batch(
            batch, system_prompt, valid_intents, model, api_key
        )
        with lock:
            # Append as we go so a crash costs one batch, not the whole run.
            with open(out_path, "a", encoding="utf-8") as fh:
                for msg_id, intent in labels.items():
                    # model and prompt version stamped per row so the file says
                    # what produced it, independently of its filename
                    fh.write(
                        json.dumps({"id": msg_id, "intent": intent, "model": model,
                                    "rules": rules, "shots": shots})
                        + "\n"
                    )
            totals["cost"] += cost
            totals["failed"] += failed
            totals["done"] += 1
            print(
                f"  [{totals['done']}/{len(batches)}] +{len(labels)} labels "
                f"({failed} failed)  running cost ${totals['cost']:.4f}"
            )

    from concurrent.futures import ThreadPoolExecutor

    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, batches))

    elapsed = time.time() - started
    print(
        f"done in {elapsed/60:.1f} min -- cost ${totals['cost']:.4f}, "
        f"{totals['failed']} messages left unlabeled"
    )
    return read_labels(out_path)


def read_labels(out_path):
    """Read a labelling JSONL back into a DataFrame (id, intent)."""
    rows = []
    with open(out_path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(rows, columns=["id", "intent"]).astype(str)
    return df.drop_duplicates(subset="id", keep="last").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Pseudo-label pool messages via OpenRouter")
    ap.add_argument("benchmark", choices=["benchmark_a", "benchmark_b"])
    ap.add_argument("--limit", type=int, help="only label the first N messages")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true", help="estimate cost, make no calls")
    ap.add_argument("--rules", default="v1", choices=["v1", "v2"],
                    help="v2 adds the disambiguation block (benchmark A only)")
    ap.add_argument("--shots", type=int, default=5, choices=[5, 10],
                    help="few-shot examples per class in the prompt; 10 uses "
                         "labeled_examples.tsv, spending the holdout half")
    ap.add_argument("--show-prompt", action="store_true",
                    help="print the system prompt and exit, making no calls")
    args = ap.parse_args()

    if args.show_prompt:
        print(build_system_prompt(args.benchmark, args.rules, args.shots))
        return

    pool = load_pool_unlabeled(args.benchmark)
    suffix = f"_limit{args.limit}" if args.limit else ""
    if args.limit:
        pool = pool.head(args.limit)

    # The prompt version is part of the filename for the same reason the model
    # is: the resume logic skips ids already present, so two runs sharing a file
    # would silently serve the first run's labels for the second and report
    # "0 batches, $0.00". That bug already happened once, between two models.
    rules_tag = "" if args.rules == "v1" else f".rules-{args.rules}"
    shots_tag = "" if args.shots == 5 else f".shot{args.shots}"
    # Batch size is part of the prompt as surely as the rules are: it decides how
    # many 77-way decisions share one context window, and so how much attention
    # each message gets. Without it in the name a smaller-batch rerun would find
    # every id already present and label nothing.
    batch_tag = "" if args.batch_size == 50 else f".b{args.batch_size}"

    # labels/, not artifacts/: these cost money and cannot be regenerated free
    out = (labels_dir(args.benchmark)
           / f"pseudo_labels.{model_slug(args.model)}{rules_tag}{shots_tag}{batch_tag}{suffix}.jsonl")
    label_messages(
        args.benchmark,
        pool,
        out,
        model=args.model,
        batch_size=args.batch_size,
        workers=args.workers,
        dry_run=args.dry_run,
        rules=args.rules,
        shots=args.shots,
    )
    if not args.dry_run:
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
