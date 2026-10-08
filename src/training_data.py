"""Assemble the classifier's training set and the label mapping.

Kept separate from train.py because train.py needs torch (Colab only) while
this needs nothing but pandas -- so the data assembly stays inspectable
locally on Python 3.14.

Training set = fewshot.tsv (gold, 5/class) + the pseudo-labelled pool.
holdout.tsv is deliberately absent: it is the validation set, and putting it
here would make every reported number meaningless (hard rule 5).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from config import LABELS_ROOT, PRIMARY_LABELLER, artifact_dir, labels_dir
from data import load_intents, load_pool_unlabeled
from labeller import read_labels


def find_pseudo_label_file(benchmark, model=None):
    """Locate the pseudo-label JSONL, which is namespaced by labelling model."""
    d = labels_dir(benchmark)
    # Fall back to the recorded canonical labeller rather than guessing when
    # several exist -- benchmark A has a second labelling purely for the
    # agreement signal, and silently training on it would be a real bug.
    model = model or PRIMARY_LABELLER.get(benchmark)
    pattern = f"pseudo_labels.{model.replace('/', '_')}.jsonl" if model else "pseudo_labels.*.jsonl"
    matches = sorted(p for p in d.glob(pattern) if "_limit" not in p.name)
    if not matches:
        raise SystemExit(
            f"no pseudo-labels matching {pattern!r} in {d}. "
            f"Run: python src/labeller.py {benchmark}"
        )
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise SystemExit(
            f"several pseudo-label files in {d} ({names}). "
            "Pass --labeller-model to say which one to train on."
        )
    return matches[0]


def load_fewshot(benchmark):
    return pd.read_csv(
        artifact_dir(benchmark) / "fewshot.tsv",
        sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False,
    )


def load_holdout(benchmark):
    return pd.read_csv(
        artifact_dir(benchmark) / "holdout.tsv",
        sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False,
    )


def label_maps(benchmark):
    """intent name <-> integer id. Order comes from intents.txt so the mapping
    is identical on every machine and every run -- the saved model's head is
    only meaningful alongside exactly this ordering."""
    intents = load_intents(benchmark)
    return {name: i for i, name in enumerate(intents)}, intents


def class_prior(benchmark, model=None):
    """Estimated class frequencies in the wild, from the pseudo-labelled pool.

    Deliberately excludes fewshot: that is exactly 5 per class by construction,
    so folding it in would drag the estimate toward uniform and defeat the
    purpose. Used for the prior-weighted accuracy metric, which brackets the
    unknown test-set prior alongside plain balanced accuracy.
    """
    labels = read_labels(find_pseudo_label_file(benchmark, model))
    counts = labels["intent"].value_counts()
    intents = load_intents(benchmark)
    # every class gets a floor of 1 so an absent class cannot get zero weight
    counts = counts.reindex(intents).fillna(1.0)
    return (counts / counts.sum()).to_dict()


def agreement_flags(benchmark, primary=None):
    """Mark pool ids where two independently-run labellers gave the same label.

    Measured on the 100 human-judged messages: where the labellers agree, the
    human approved 91% of labels; where they disagree, 50%. So agreement is a
    usable confidence signal, and it costs one cheap extra pass rather than the
    re-labelling that turned out not to help (Opus scored the same as Sonnet on
    the gate -- it fixed four labels and broke five).

    Prompt variants are excluded, not just limit files. The signal is meant to
    be two labellers run INDEPENDENTLY disagreeing; the same model under a
    different prompt measures prompt sensitivity instead, which is a different
    thing and a much weaker one. Adding pseudo_labels.<sonnet>.rules-v2.jsonl
    silently rerouted this: sorted() put the two sonnet files first, so it
    started reporting 527 disagreements (5.7%, sonnet against itself) where the
    real two-labeller figure is 921 (10.0%), and --drop-disagreed would have
    dropped that wrong set of rows.

    Returns a DataFrame of id, agree. Empty if only one labelling exists.
    """
    files = sorted(p for p in labels_dir(benchmark).glob("pseudo_labels.*.jsonl")
                   if "_limit" not in p.name and ".rules-" not in p.name)
    if len(files) < 2:
        return pd.DataFrame(columns=["id", "agree"])
    if primary:
        want = f"pseudo_labels.{primary.replace('/', '_')}.jsonl"
        files = [p for p in files if p.name == want] + [p for p in files if p.name != want]
    a = read_labels(files[0]).rename(columns={"intent": "l0"})
    b = read_labels(files[1]).rename(columns={"intent": "l1"})
    m = a.merge(b, on="id", how="inner")
    m["agree"] = m["l0"] == m["l1"]
    return m[["id", "agree"]]


def human_judged_ids(benchmark):
    """Pool ids a human has already judged.

    These must be EXCLUDED from training: they are the only representative
    validation data in the project, and holdout cannot replace them -- holdout
    is curated and capped at 60 characters while test.tsv is 24.4% longer than
    that. Training on them would destroy the one honest estimate we have.

    BOTH judged sets count, and missing the second one cost us a real
    measurement. This returned only human_verdicts.tsv for a while, so all 120
    audit messages stayed in training -- and the audit was then used to measure
    the model those same rows had trained on. 10 of its 11 errors were rows the
    model had seen with that exact wrong label, so what looked like a label bias
    the model had generalised was in part simple memorisation, and the audit's
    85.8% was inflated against the leaderboard's 81.25%.
    """
    ids = set()
    path = labels_dir(benchmark) / "human_verdicts.tsv"
    if path.exists():
        df = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE,
                         dtype=str, keep_default_na=False)
        ids |= set(df["id"])
    audit = LABELS_ROOT / f"audit_{benchmark}.tsv"
    if audit.exists():
        df = pd.read_csv(audit, sep="\t", quoting=csv.QUOTE_NONE,
                         dtype=str, keep_default_na=False)
        ids |= set(df["id"])
    return ids


def human_confirmed(benchmark):
    """(text, intent) for pool messages whose label a human confirmed correct.

    Three files record verdicts and all three are read, because they were
    written by different review rounds and none is a superset of the others:
    human_verdicts.tsv judged the LABELLER's category on 100 messages, the
    audit judged the MODEL's prediction on a different 120, and verdict_pairs
    accumulates every (message, category) pair judged in any round.

    Only verdict == "ok" is usable. A "bad" tells us the category was wrong
    but not what the right one is -- the notes sometimes say, but only 15 of
    120 audit rows carry one, so recovering them is hand work rather than a
    rule. A message approved for two different categories in different rounds
    is dropped: that is a contradiction, not extra data.

    These rows are worth more than their count suggests. They were selected
    for review BECAUSE they were confusable, so they put certain labels
    exactly where the pseudo-labels are noisiest -- and unlike holdout they
    are drawn from the pool, so they match test.tsv's length distribution.
    """
    votes = {}

    def record(df, col):
        for r in df.itertuples():
            if getattr(r, "verdict") != "ok":
                continue
            votes.setdefault(r.id, set()).add(getattr(r, col))

    path = labels_dir(benchmark) / "human_verdicts.tsv"
    if path.exists():
        record(pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE,
                           dtype=str, keep_default_na=False), "assigned")
    audit = LABELS_ROOT / f"audit_{benchmark}.tsv"
    if audit.exists():
        record(pd.read_csv(audit, sep="\t", quoting=csv.QUOTE_NONE,
                           dtype=str, keep_default_na=False), "assigned")
    pairs = LABELS_ROOT / f"verdict_pairs_{benchmark}.tsv"
    if pairs.exists():
        record(pd.read_csv(pairs, sep="\t", quoting=csv.QUOTE_NONE,
                           dtype=str, keep_default_na=False), "category")

    clean = {i: next(iter(v)) for i, v in votes.items() if len(v) == 1}
    pool = load_pool_unlabeled(benchmark)
    out = pool[pool["id"].isin(clean)].copy()
    out["intent"] = out["id"].map(clean)
    return out[["id", "text", "intent"]]


def load_soft_labels(path, intents, temperature=1.0):
    """id -> full probability vector over `intents`, from a .soft.jsonl file.

    The file stores only the top-k intents per row (the tail is numerically
    irrelevant and would multiply the file size by five), so the vector is
    rebuilt here with zeros elsewhere. Order comes from `intents`, i.e. from
    intents.txt, which is the same ordering the model's head uses -- a vector
    built in any other order would train the student to predict the wrong
    classes while reporting a perfectly normal loss.

    TEMPERATURE. A teacher fitted on hard labels is overconfident -- ours has a
    median confidence of 0.995, so its distribution says almost nothing its own
    argmax did not. Softening it is the standard treatment, and it is exact
    here despite only probabilities being saved: p_i is proportional to
    exp(z_i), so p_i**(1/T) is proportional to exp(z_i/T), which is precisely
    what temperature scaling the logits would have produced. The student must
    be softened by the same T and the loss multiplied by T**2 (DistilTrainer
    does both), or the gradients shrink with T instead of the target.
    """
    index = {name: i for i, name in enumerate(intents)}
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pairs = []
            for name, p in row["top"]:
                j = index.get(name)
                if j is None:
                    raise SystemExit(
                        f"{path}: intent {name!r} is not in intents.txt -- "
                        "this soft-label file belongs to a different benchmark"
                    )
                pairs.append((j, float(p)))
            if temperature != 1.0:
                pairs = [(j, p ** (1.0 / temperature)) for j, p in pairs]
                total = sum(p for _, p in pairs) or 1.0
                pairs = [(j, p / total) for j, p in pairs]
            vec = [0.0] * len(intents)
            for j, p in pairs:
                vec[j] = p
            out[str(row["id"])] = vec
    return out


def build_training_set(benchmark, model=None, cap_per_class=None,
                       drop_disagreed=False, exclude_human_judged=True,
                       labels_file=None, use_holdout=False, gold_weight=1,
                       use_judged=False, soft_labels_file=None,
                       soft_temperature=1.0, augment_file=None):
    """fewshot + pseudo-labelled pool, as a DataFrame of (text, intent, source).

    cap_per_class trims over-represented classes. Off by default: benchmark A's
    imbalance is genuine (verified -- rare classes score 99% recall in QA, they
    are simply rarer), and the leaderboard scores plain accuracy, so the natural
    prior is the correct training signal. The knob exists to test that
    empirically against holdout, not because rebalancing is assumed to help.

    use_holdout adds the 5-per-class holdout rows to training, doubling the gold
    data. They were held back to validate against, but holdout has failed every
    check we put it to: it reads ~17 points above the leaderboard, it saturates
    (747/750 on B, 380/385 on A), and two different architectures scored
    IDENTICALLY on it while differing by two points on the audit. The 120-message
    audit and the leaderboard replaced it, so the rows are better spent as the
    only certain labels in the project. Turning this on means holdout accuracy
    is no longer a validation number -- train.py stops reporting it as one.

    gold_weight repeats gold rows N times. Adding 385 clean rows to 9,013
    pseudo-labels barely shifts the balance: per class it is 10 certain examples
    against ~117 carrying ~12 errors. Repetition is what makes the clean anchor
    comparable to the noise, and it is a knob to measure rather than assume --
    too high and the model overfits 770 sentences.
    """
    intent2id, intents = label_maps(benchmark)
    fewshot = load_fewshot(benchmark)[["text", "intent"]].copy()
    fewshot["source"] = "gold"

    if use_holdout:
        held = load_holdout(benchmark)[["text", "intent"]].copy()
        held["source"] = "gold"
        fewshot = pd.concat([fewshot, held], ignore_index=True)

    if gold_weight > 1:
        fewshot = pd.concat([fewshot] * int(gold_weight), ignore_index=True)

    # labels_file lets a revised label set (e.g. a self-training pass) be
    # trained on without pretending it is the canonical labelling.
    source = Path(labels_file) if labels_file else find_pseudo_label_file(benchmark, model)
    pseudo = read_labels(source)
    pool = load_pool_unlabeled(benchmark)
    pseudo = pool.merge(pseudo, on="id")

    if exclude_human_judged:
        judged = human_judged_ids(benchmark)
        pseudo = pseudo[~pseudo["id"].isin(judged)]

    if drop_disagreed:
        flags = agreement_flags(benchmark, model)
        if flags.empty:
            raise SystemExit(
                "--drop-disagreed needs two pseudo-label files. Run the labeller "
                "again with a second --model."
            )
        pseudo = pseudo.merge(flags, on="id", how="left")
        pseudo = pseudo[pseudo["agree"].fillna(True)]

    # id is carried through so soft targets can be joined by it later. Gold rows
    # have no pool id and get an empty one; nothing downstream reads it except
    # the soft-target lookup, which only ever asks about pseudo rows.
    pseudo = pseudo[["id", "text", "intent"]].copy()
    pseudo["source"] = "pseudo"
    fewshot["id"] = ""

    # The judged rows come back as GOLD, not as pseudo-labels: a human read the
    # message and confirmed the category, which is the same standard as
    # labeled_examples.tsv. They stay excluded from the pseudo block above, so
    # nothing is duplicated and no row enters carrying a label a human rejected.
    #
    # Turning this on spends the project's only test-distributed validation set.
    # That is a deliberate trade: the audit saturated at 100/120 across eleven
    # model variants and stopped discriminating, so the rows are worth more as
    # certain training labels than as a gate that always reads the same.
    if use_judged:
        conf = human_confirmed(benchmark)[["text", "intent"]].copy()
        conf["source"] = "gold"
        conf["id"] = ""
        if gold_weight > 1:
            conf = pd.concat([conf] * int(gold_weight), ignore_index=True)
        pseudo = pd.concat([pseudo, conf], ignore_index=True)

    # Generated examples for the classes the pool barely covers. Their
    # labels are certain by construction -- we chose the category and asked
    # for text belonging to it -- so they enter as gold, not pseudo. This is
    # the only source in the project that ADDS information rather than
    # redistributing the labeller's; the organiser sanctioned it explicitly.
    if augment_file:
        aug = pd.read_csv(augment_file, sep='	', quoting=csv.QUOTE_NONE,
                          dtype=str, keep_default_na=False)[['text', 'intent']].copy()
        unknown_aug = set(aug['intent']) - set(intents)
        if unknown_aug:
            raise ValueError(f'augment file has unknown intents: {sorted(unknown_aug)[:5]}')
        aug['source'] = 'gold'
        aug['id'] = ''
        n_aug = len(aug)
        pseudo = pd.concat([pseudo, aug], ignore_index=True)
        print(f'  augmented: +{n_aug:,} generated rows across '
              f"{aug['intent'].nunique()} classes")

    train = pd.concat([fewshot, pseudo], ignore_index=True)

    # Soft targets ride along as a column so they survive the shuffle and the
    # per-class cap below. Gold rows get a one-hot vector: a human or the
    # provided examples said this category and nothing else, and KL against a
    # one-hot target is exactly cross-entropy, so the two kinds of row can share
    # one loss without special-casing either.
    if soft_labels_file:
        soft = load_soft_labels(soft_labels_file, intents, soft_temperature)
        onehot = {}

        def vector(row):
            if row["source"] == "gold":
                j = intent2id[row["intent"]]
                if j not in onehot:
                    v = [0.0] * len(intents)
                    v[j] = 1.0
                    onehot[j] = v
                return onehot[j]
            v = soft.get(row["id"])
            if v is None:  # a pseudo row the teacher never scored
                j = intent2id[row["intent"]]
                v = [0.0] * len(intents)
                v[j] = 1.0
            return v

        train["soft"] = train.apply(vector, axis=1)
        is_pseudo = train["source"] == "pseudo"
        covered = int((is_pseudo & train["id"].isin(soft)).sum())
        n_pseudo = int(is_pseudo.sum())
        print(f"  soft targets: {covered:,}/{n_pseudo:,} pseudo rows covered"
              + ("" if covered == n_pseudo else "  <- the rest fall back to one-hot"))

    unknown = set(train["intent"]) - set(intents)
    if unknown:
        raise ValueError(f"{benchmark}: training data has unknown intents {sorted(unknown)}")

    if cap_per_class:
        train = (
            train.sample(frac=1.0, random_state=0)  # shuffle so the cap is not id-ordered
            .groupby("intent", group_keys=False)
            .head(cap_per_class)
            .reset_index(drop=True)
        )

    train["label"] = train["intent"].map(intent2id)
    return train.sample(frac=1.0, random_state=0).reset_index(drop=True)


def load_human_validation(benchmark):
    """The human-verified validation set: pool messages a person confirmed.

    Only the APPROVED rows are usable as labelled data -- for the rejected
    ones we know the label was wrong but not what the right one is, so they
    make a gate for labelling changes (human_gate.py) but not a validation
    set for a classifier.

    Small (85 rows for A, roughly one per class) so it says nothing per-class.
    Its value is being the only set drawn from the SAME distribution as
    test.tsv: holdout is curated and capped at 60 characters, the pool and
    test are not. Accuracy here is the honest estimate; accuracy on holdout
    is the flattering one. Report both.
    """
    path = labels_dir(benchmark) / "human_verdicts.tsv"
    if not path.exists():
        return pd.DataFrame(columns=["id", "text", "intent"])
    df = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
    ok = df[df["verdict"] == "ok"].copy()
    return ok.rename(columns={"assigned": "intent"})[["id", "text", "intent"]]


def save_label_map(benchmark, out_dir, train_prior=None):
    """Persist the mapping next to the model. Inference is meaningless without it.

    Also stores the TRAINING class prior, because inference needs it and cannot
    compute it: classify.py must not import this module (it pulls in labeller,
    hence requests, breaking hard rule 3), so the prior has to travel with the
    model as plain data.

    It is needed because train and test priors differ. test.tsv is exactly
    3,080 = 77 x 40 for A and 4,500 = 150 x 30 for B -- one message per class
    per slot, i.e. uniform. Our pseudo-labelled pool is not: A runs 13.7:1 from
    rarest to commonest class. A model fitted to one prior and scored under
    another is biased toward the classes that were frequent in training, and
    subtracting log P_train(y) from the logits is the standard correction.
    """
    intent2id, intents = label_maps(benchmark)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "labels.json"
    payload = {"benchmark": benchmark, "intents": intents}
    if train_prior is not None:
        # aligned to `intents` by position, so inference needs no name lookup
        payload["train_prior"] = [float(train_prior[i]) for i in intents]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    from config import BENCHMARKS

    for b in BENCHMARKS:
        train = build_training_set(b)
        prior = class_prior(b)
        vc = train["intent"].value_counts()
        print(f"{b}: {len(train):,} training rows "
              f"({(train['source']=='gold').sum()} gold + {(train['source']=='pseudo').sum()} pseudo)")
        print(f"  classes {train['intent'].nunique()}  per-class min {vc.min()} median {int(vc.median())} max {vc.max()}")
        print(f"  prior: max {max(prior.values()):.4f}  min {min(prior.values()):.4f}  sums to {sum(prior.values()):.4f}")

        flags = agreement_flags(b)
        if not flags.empty:
            rate = flags["agree"].mean()
            print(f"  two labellers agree on {rate:.1%} of the pool "
                  f"({int((~flags['agree']).sum()):,} disagreements)")
            dropped = build_training_set(b, drop_disagreed=True)
            print(f"  --drop-disagreed would train on {len(dropped):,} rows "
                  f"({len(train)-len(dropped):,} fewer)")
        hv = load_human_validation(b)
        if len(hv):
            print(f"  human-verified validation set: {len(hv)} rows "
                  f"(excluded from training)")
