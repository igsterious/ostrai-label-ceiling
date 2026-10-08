# The label ceiling

When you pay a large language model to label a training pool and then fine-tune a
small model on those labels, how good can the small model get? This repository holds
the code and the label sets behind one answer: **about three quarters of the
disagreement between LLM labellers is systematic**, so it does not average out, and
it transfers into the student as if it were ground truth.

The measurement needs no ground truth of its own. It compares labelling passes
against each other, with a repeat run of one configuration as the floor:

| What is compared | Disagreement | Rows |
|---|---|---|
| The same model, the same prompt, twice | 2.70% | 1,000 |
| The same model, the prompt reworded | 5.91% | 9,233 |
| Models from different families | 11.53% | 9,233 |

Subtract the floor and 8.83 of the 11.53 points are reproducible preference rather
than sampling noise. [FINDINGS.md](FINDINGS.md) has the full result, the model-family
separation, and the 42 fine-tuning runs it was tested against.

## Reproducing it

The agreement analysis reads only the committed label sets, so it needs no dataset
and no GPU:

```bash
python tools/agreement.py
```

That prints the pairwise agreement grid over the eight independent passes, the
within-family and between-family split, and the most frequently confused category
pairs.

## What is here, and what is not

The task was a summer-school leaderboard competition: sort short customer messages
into intents, two benchmarks, 77 and 150 categories, with ten labelled examples per
category as the only ground truth.

**Included.** The pipeline (`src/`), the analysis tools (`tools/`), the study
notebooks (`notebooks/`), and `labels/` — every pseudo-label set the project
produced. Those label sets are the evidence: they cost real API calls and a re-run
would not reproduce them, which is why they are committed rather than generated.

**Not included.** The organisers' `data/` directory. Their pool and test splits are
their construction to distribute, not ours, and no test row appears anywhere in this
repository. Benchmark A's label space and messages are [Banking77]'s and benchmark
B's are [CLINC150]'s, both public datasets, so the underlying corpora are available
from their own sources.

Most files under `labels/` are `id` → `intent` only. A few audit files also carry the
message text of sampled **pool** rows; `src/make_audit_set.py` shows the sampling.

[Banking77]: https://huggingface.co/datasets/PolyAI/banking77
[CLINC150]: https://huggingface.co/datasets/clinc/clinc_oos

## Layout

| Path | Contents |
|---|---|
| `labels/benchmark_a/` | Eight independent labelling passes, plus derived consensus, hand-corrected and generated sets |
| `labels/benchmark_b/` | Three labelling passes |
| `tools/agreement.py` | The agreement decomposition and the family separation |
| `src/labeller.py` | LLM pseudo-labelling over the pool, batched, few-shot prompted |
| `src/qa_labeller.py` | Scores the labeller against the held-out gate |
| `src/make_splits.py` | The seeded 5 + 5 split of the ten examples per class |
| `src/train.py`, `src/classify.py` | Student fine-tuning, and the single inference path |
| `tools/`, `src/` (rest) | The experiment machinery: consensus, reranking, augmentation, calibration, latency |
| `notebooks/` | The agreement study rebuilt from scratch, and the Colab training driver |

The eight passes on benchmark A are three model families crossed with prompt
revisions and batch sizes. Each `.jsonl` row records the model, prompt version and
shot count that produced it, so the grid is reconstructible from the files alone.

## Method notes worth knowing before reusing this

- **The ten examples per class had three competing jobs**: few-shot prompt content, a
  QA set to score the labeller, and validation for the student. A seeded, stratified
  5 + 5 split resolved it. The half that scores the labeller never enters its prompt.
- **The gate ran once.** The labeller was scored on the held-out half before any
  pseudo-label entered training, and that half was never used to tune the prompt,
  because it is also the student's validation set.
- **Accuracy was reported two ways**, weighted equally per class and by pool
  frequency, because the test prior was unknowable under the rules.

## Status

A short paper is in preparation. The mechanism is not novel and
[FINDINGS.md](FINDINGS.md) names the prior work it confirms; what is offered here is
the measurement with a floor under it, the family separation, and the downstream
test.

## License

doplň — not yet chosen. Until then, treat this as all rights reserved.
