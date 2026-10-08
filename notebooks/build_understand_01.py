"""Generates notebooks/understand_01_data_and_agreement.ipynb.

Kept as a script so the notebook can be regenerated and diffed. Run once:
    python notebooks/build_understand_01.py
then open the .ipynb in VS Code.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

# ---------------------------------------------------------------------------
md("""
# Understanding the data and the agreement analysis

**Goal:** rebuild the findings yourself from the raw files, so you can explain
every number in the report and defend it. Nothing here calls an API or trains
a model — it is all pandas over committed files, and every cell runs in
seconds.

Work through it top to bottom. Each section has a short *why*, a code cell you
should read before running, and an **exercise** where you change something.
The exercises are the point; the code cells are just scaffolding.

Three messages to keep in mind throughout — you'll meet them in section 2:

| pool id | message | what the eight labellers did |
|---|---|---|
| 1 | *My card is due to expire* | all eight agree |
| 6 | *I have a card that was deactivated can I reactivate it?* | Sonnet ×3 say one thing, GPT ×3 say another |
| 46 | *Can I track history on funds?* | five different answers |
""")

code("""
import json, csv, itertools
from pathlib import Path
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
DATA = ROOT / 'data' / 'benchmark_a'
LABELS = ROOT / 'labels' / 'benchmark_a'
ART = ROOT / 'artifacts' / 'benchmark_a'

# The TSVs contain no quote characters, so we read them with QUOTE_NONE.
# (src/data.py does the same; this is why.)
read_tsv = lambda p: pd.read_csv(p, sep='\\t', quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
print('repo root:', ROOT)
""")

# ---------------------------------------------------------------------------
md("""
## 1. The data

Four files per benchmark. The only ground truth in the whole project is
`labeled_examples.tsv` — exactly 10 hand-labelled messages per category.
Everything else was labelled by an LLM.
""")

code("""
pool     = read_tsv(DATA / 'pool.tsv')
test     = read_tsv(DATA / 'test.tsv')
labeled  = read_tsv(DATA / 'labeled_examples.tsv')
intents  = (DATA / 'intents.txt').read_text(encoding='utf-8').split()

print(f'pool     {len(pool):>6,} rows   columns {list(pool.columns)}')
print(f'test     {len(test):>6,} rows   columns {list(test.columns)}')
print(f'labeled  {len(labeled):>6,} rows   columns {list(labeled.columns)}')
print(f'intents  {len(intents):>6}')
print()
print('one pool row:   ', pool.iloc[0].to_dict())
print('one labeled row:', labeled.iloc[0].to_dict())
""")

md("""
**Why the labelled examples are a subset of the pool.** The 770 gold rows
are *also* in the pool (same ids, same text). So the real labelling target is
pool-minus-labelled: 9,233 messages. Check it:
""")

code("""
gold_ids = set(labeled['id'])
pool_ids = set(pool['id'])
print('gold ids inside pool:', len(gold_ids & pool_ids), 'of', len(gold_ids))
print('pool minus gold:     ', len(pool_ids - gold_ids))
print('per-class gold count:', labeled['intent'].value_counts().unique())   # should be just [10]
""")

md("""
### The split — and why it isn't a percentage

Ten gold examples per class have three jobs competing for them: few-shot
examples *in the labelling prompt*, a QA set *for the labeller*, and a
validation set *for the classifier*. `src/make_splits.py` resolves this by
splitting **5 + 5 per class**, stratified and seeded:

- `fewshot.tsv` — may enter the LLM prompt
- `holdout.tsv` — **never** reaches the labeller

A random 50% split would leave some of the 77 classes with zero holdout rows,
and a class you can't measure is a class you can't fix. That's the whole reason
it's per-class.
""")

code("""
fewshot = read_tsv(ART / 'fewshot.tsv')
holdout = read_tsv(ART / 'holdout.tsv')
print('fewshot', len(fewshot), '| per class:', fewshot['intent'].value_counts().unique())
print('holdout', len(holdout), '| per class:', holdout['intent'].value_counts().unique())
print('overlap between them:', len(set(fewshot['id']) & set(holdout['id'])))
""")

md("""
> **Exercise 1.1** — The holdout rows should appear in *no* pseudo-label file,
> because the labeller was only ever run on pool-minus-labelled. Write a loop
> over `LABELS.glob('pseudo_labels.*.jsonl')` that counts how many holdout ids
> appear in each file. (Hint: each line is a JSON object with an `id` field —
> see the loader in section 2.)
>
> You'll find one file that breaks the pattern. Which one, and why is that
> acceptable? Then look at `build_system_prompt` in `src/labeller.py` (line
> ~185) and find a *different* subtlety: which labelling pass saw the holdout
> rows as **few-shot examples**, and what does that mean for QA-ing that pass?
""")

code("""
# your answer to exercise 1.1 here
""")

# ---------------------------------------------------------------------------
md("""
## 2. The labelling passes

Twenty files sit in `labels/benchmark_a/`. They are not twenty independent
labellings — most are derived. Sort them into two piles:

**Raw passes** — one LLM, one prompt configuration, one sweep over 9,233 messages:

| file | model | what varies |
|---|---|---|
| `…claude-sonnet-5.jsonl` | Sonnet 5 | default: 5-shot, rules v1, batch 50 |
| `…claude-sonnet-5.rules-v2.jsonl` | Sonnet 5 | + a disambiguation block in the prompt |
| `…claude-sonnet-5.shot10.jsonl` | Sonnet 5 | all 10 gold examples per class in the prompt |
| `…openai_gpt-5.1.jsonl` | GPT-5.1 | default |
| `…openai_gpt-5.1.rules-v2.jsonl` | GPT-5.1 | + disambiguation block |
| `…openai_gpt-5.1.b10.jsonl` | GPT-5.1 | batch size 10 instead of 50 |
| `…deepseek_deepseek-chat.jsonl` | DeepSeek | default |
| `…google_gemini-3.1-flash-lite.jsonl` | Gemini | default |

**Derived** — built from the above without calling any model:
`gpt3way` (majority vote of the three GPT passes), `gpt3way_unanimous` (only
rows where all three agreed), `consensus_*`, `handfixed` (Igor's 307 manual
decisions applied), `rerank*` (a second LLM call on disputed rows only), and
the `_limit*` files, which are **partial re-runs** of an identical config —
keep those in mind for section 3.

**What the labeller was actually asked.** Read the prompt — it's the thing
every one of these files is a response to.
""")

code("""
import sys; sys.path.insert(0, str(ROOT / 'src'))
from labeller import build_system_prompt, build_user_prompt

sp = build_system_prompt('benchmark_a')          # default: 5-shot, rules v1
print(sp[:900])
print('\\n... [', len(sp), 'characters in total — this is sent once per batch ]')
print()
print(build_user_prompt(pool.iloc[:3]))
""")

md("""
Notice three design decisions, because each one is a variable you later
measured:

1. Examples are **grouped under their intent** rather than listed flat, so the
   model sees the boundary between neighbouring categories.
2. *"If a message is ambiguous, pick the single closest category rather than
   refusing."* — the labeller is forced to choose. There is no "unsure."
3. Messages go in **batches of 50**, tagged with ids, and come back as one JSON
   object. `b10` changed that to 10 per call.
""")

md("""
### Load the eight raw passes into one table
""")

code("""
PASSES = {
    'sonnet v1':        'pseudo_labels.anthropic_claude-sonnet-5.jsonl',
    'sonnet rules-v2':  'pseudo_labels.anthropic_claude-sonnet-5.rules-v2.jsonl',
    'sonnet 10-shot':   'pseudo_labels.anthropic_claude-sonnet-5.shot10.jsonl',
    'gpt-5.1 v1':       'pseudo_labels.openai_gpt-5.1.jsonl',
    'gpt-5.1 rules-v2': 'pseudo_labels.openai_gpt-5.1.rules-v2.jsonl',
    'gpt-5.1 batch-10': 'pseudo_labels.openai_gpt-5.1.b10.jsonl',
    'deepseek':         'pseudo_labels.deepseek_deepseek-chat.jsonl',
    'gemini':           'pseudo_labels.google_gemini-3.1-flash-lite.jsonl',
}
FAMILY = {k: k.split(' ')[0] for k in PASSES}      # 'sonnet', 'gpt-5.1', 'deepseek', 'gemini'

def load_pass(filename):
    \"\"\"One JSONL file -> {id: intent}. Each line is {"id": "...", "intent": "...", "model": "..."}.\"\"\"
    out = {}
    with open(LABELS / filename, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            out[str(r['id'])] = r['intent']
    return out

votes = pd.DataFrame({name: load_pass(fn) for name, fn in PASSES.items()})
votes.index.name = 'id'
votes = votes.dropna()                             # keep only ids every pass labelled
print(votes.shape, '<- (messages, passes)')
votes.head()
""")

md("""
### The three messages
""")

code("""
text = dict(zip(pool['id'], pool['text']))
for i in ['1', '6', '46']:
    print(f'--- pool id {i}:  \"{text[i]}\"')
    print(votes.loc[i].to_string())
    print()
""")

md("""
> **Exercise 2.1 — the finding, by hand.** Pool id 6 is the pattern that
> matters: all three Sonnet passes agree with each other, all three GPT passes
> agree with each other, and the two blocks disagree. Count how many of the
> 9,233 messages have *exactly* that shape. Then count how many have the
> *opposite* shape — the three Sonnet passes disagree among themselves, or the
> three GPT passes do.
>
> Before you run it, write down which count you expect to be larger and why.
> Then check. (Hint: `votes[['sonnet v1','sonnet rules-v2','sonnet 10-shot']].nunique(axis=1)` gives
> the number of distinct Sonnet answers per row.)
""")

code("""
# your answer to exercise 2.1 here
""")

# ---------------------------------------------------------------------------
md("""
## 3. Agreement, from scratch

**Agreement between two passes** is nothing more than: the fraction of
messages where they gave the same label. Three lines.
""")

code("""
def agreement(a, b):
    \"\"\"Share of messages on which passes a and b gave the same intent, in %.\"\"\"
    same = (votes[a] == votes[b]).sum()
    return 100 * same / len(votes)

print(f\"sonnet v1  vs sonnet rules-v2 : {agreement('sonnet v1', 'sonnet rules-v2'):.2f}%\")
print(f\"sonnet v1  vs gpt-5.1 v1      : {agreement('sonnet v1', 'gpt-5.1 v1'):.2f}%\")
""")

md("""
Same model with a different prompt: ~94%. Different model: ~89%. That gap is
the result. Everything below is just making it precise.

### The full 8×8 matrix
""")

code("""
names = list(PASSES)
M = pd.DataFrame(index=names, columns=names, dtype=float)
for a in names:
    for b in names:
        M.loc[a, b] = float('nan') if a == b else agreement(a, b)
M.round(1)
""")

md("""
Read it as blocks. Rows 1–3 × columns 1–3 is Sonnet against itself (three
prompts). Rows 4–6 × columns 4–6 is GPT-5.1 against itself. Everything outside
those two blocks crosses model families.

### Within-family vs between-family
""")

code("""
within, between = [], []
for a, b in itertools.combinations(names, 2):        # each unordered pair once: 28 pairs
    (within if FAMILY[a] == FAMILY[b] else between).append(M.loc[a, b])

print(f'within-family  pairs: {len(within):>2}   mean agreement {sum(within)/len(within):.2f}%')
print(f'between-family pairs: {len(between):>2}   mean agreement {sum(between)/len(between):.2f}%')
""")

md("""
> **Exercise 3.1 — the separation test.** The report claims the two blocks
> separate *perfectly*: the **lowest** within-family agreement is higher than
> the **highest** between-family agreement. Verify it with `min(within)` and
> `max(between)`, and print which two passes each of those extremes belongs to.
>
> Then answer in one sentence: what does perfect separation let you do with
> eight *unlabelled* files whose model names have been stripped off?
""")

code("""
# your answer to exercise 3.1 here
""")

md("""
### The noise floor — the same model, the same prompt, twice

Within-family disagreement mixes two things: *prompt sensitivity* (the prompts
differ) and *sampling noise* (LLMs are stochastic even with identical input).
To separate them you need the same configuration run twice.

That's what `…gpt-5.1.b10_limit1000.jsonl` is: the batch-10 configuration
re-run on the first 1,000 messages, identical prompt, identical settings. Its
disagreement with the full `b10` file is sampling noise and nothing else.
""")

code("""
b10      = load_pass('pseudo_labels.openai_gpt-5.1.b10.jsonl')
b10_1000 = load_pass('pseudo_labels.openai_gpt-5.1.b10_limit1000.jsonl')
shared   = [i for i in b10_1000 if i in b10]
same     = sum(b10[i] == b10_1000[i] for i in shared)
floor    = 100 * (1 - same / len(shared))
print(f'{len(shared):,} shared messages, {same:,} identical  ->  disagreement {floor:.2f}%')
""")

md("""
### The decomposition

Now stack the three measurements. Each level *contains* the one below it.

| comparison | disagreement | what's in it |
|---|---|---|
| same model, same prompt, twice | **2.70%** | sampling noise |
| same model, different prompt | **5.91%** | + prompt sensitivity |
| different model family | **11.53%** | + model identity |
""")

code("""
lvl1 = floor
lvl2 = 100 - sum(within) / len(within)
lvl3 = 100 - sum(between) / len(between)
print(f'level 1  sampling noise         {lvl1:5.2f}%')
print(f'level 2  + prompt sensitivity   {lvl2:5.2f}%   (adds {lvl2-lvl1:+.2f})')
print(f'level 3  + model identity       {lvl3:5.2f}%   (adds {lvl3-lvl2:+.2f})')
print()
print(f'share of cross-family disagreement that is NOT sampling noise: {100*(lvl3-lvl1)/lvl3:.0f}%')
""")

md("""
> **Exercise 3.2 — say what 77% means.** Write the last number as a plain
> English sentence that a reviewer couldn't misread. Then answer: why does
> "not sampling noise" matter for whether *majority voting* can help? (Think
> about pool id 6 again: what would a vote across the three GPT passes return?)
>
> **Exercise 3.3 — how sure are we of the floor?** The 2.70% rests on 1,000
> messages. Compute its standard error with `sqrt(p*(1-p)/n)` where p = 0.027.
> Is the floor precise enough to support the gap between levels 1 and 2?
""")

code("""
# your answers to 3.2 and 3.3 here
""")

md("""
## What you can now explain

- what the data files are and why the split is per-class, not a percentage
- what a labelling pass is, what was varied between them, and what the model
  was actually asked
- how pairwise agreement is computed, and what the block structure means
- what the noise floor measures and why it needs an identical re-run
- the three-level decomposition and the arithmetic behind "77% systematic"

**Next notebook:** what "disagreement" is *not* (correctness), which is the
gap Xu et al. fill with human annotation — and the training runs that showed
which fixes fail.
""")

nb['cells'] = cells
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
out = 'notebooks/understand_01_data_and_agreement.ipynb'
nbf.write(nb, out)
print('wrote', out, 'with', len(cells), 'cells')
