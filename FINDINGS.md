# Findings

Measured on benchmark A (77 intents) over the 1,000 pool rows common to all eight
independent labelling passes. No gold labels are used anywhere in this measurement;
it compares labellers against each other.

## The decomposition

| Comparison | Disagreement | What the increment adds |
|---|---|---|
| Same model, same prompt, twice | 2.70% | Sampling noise, and nothing else |
| Same model, prompt reworded | 5.91% | + 3.21 prompt sensitivity |
| Different model families | 11.53% | + 5.62 model bias |

The repeat-run pair is what makes this a measurement rather than a ratio: it fixes a
zero point. Of the 11.53 points between families, 8.83 — **77%** — is reproducible for
a given model and prompt.

That distinction decides what can be fixed. Random error points in every direction
and cancels across a large pool. A reproducible preference points the same way every
time, so a student trained on 30,000 such labels learns it as truth.

## The family fingerprint

Agreement alone identifies which family produced a pass. Across all 28 pairs:

- weakest **within**-family pair: 93.43%
- strongest **between**-family pair: 90.21%
- pairs out of order: **0**

The groups do not overlap. Two runs of one model under different prompts resemble
each other more than two models under identical instructions.

## What it predicted, across 42 fine-tuning runs

The decomposition was measured first, then matched against the experiment record.
Interventions aimed at the random component moved nothing; one standard error on the
split these were read against is ±0.83.

| Intervention | What it attacks | Δ accuracy |
|---|---|---|
| Change the labeller's family | The systematic component | **+2.19** |
| 2.4× the student's parameters | Student capacity | +0.33 |
| Majority vote over three prompts, one model | The random component | +0.16 |
| Ensemble of students distilled to one | Variance | 0.00 |
| LLM-generated examples for 23 starved classes | Class coverage | −0.49 |
| 307 disputed rows re-decided by hand | Individual labels | −1.14 |
| Labeller re-ranks its own disputed rows | Its own bias | −1.78 |
| Self-training on the student's confident rows | Unlabelled data | −2.03 |

Two rows carry a second lesson. The generated examples failed for an additional
reason: they averaged 82 characters against the test set's 54, so the student could
learn that long and well-formed means rare class. The hand-correction failed because
on genuinely ambiguous neighbouring categories an annotator substitutes one judgement
for another rather than correcting an error.

One external check: the entry that placed above ours used an unrelated pipeline, a
much smaller and roughly six times faster model, and finished 0.45 away on benchmark
A. Two independent architectures landing inside the noise suggests a property of the
data rather than of either method.

## Prior work this confirms

The mechanism is not novel, and the paper in preparation says so in its introduction.

- **Liu, *Inter-Prompt Reliability*** (arXiv 2604.16413) uses the same pairwise
  agreement statistic over prompt variants and recommends majority voting across
  prompts. The +0.16 above is the quantified downstream counterpoint: the vote
  cancels the component it can reach and leaves the component shared by the variants
  being voted over. Voting buys reproducibility, not correctness.
- **Xu et al., *Enhancing LLM-Based Data Annotation with Error Decomposition***
  (arXiv 2601.11920, LAK26) separates model-specific from task-inherent error and
  estimates the task-inherent part through lightweight human annotation. The
  measurement here is disagreement-only: it needs no labels, and it cannot say which
  family is correct. Their method supplies that leg.
- The distillation half — a student inherits its teacher's bias, and ensembling does
  not remove systematic error — is long established.

## Limits

- One domain, mostly one benchmark. Benchmark B has three passes rather than eight,
  so the decomposition is not yet replicated there.
- The accuracy deltas were read against a competition split with ±0.83 of standard
  error, and they are not accompanied by paired significance tests. McNemar tests on
  the paired predictions, and confidence intervals on the agreement rates, are the
  obvious next step.
- Candidate selection during the competition was partly driven by readings from a
  visible leaderboard split of 1,232 rows, which carries roughly a point of noise per
  reading. That is a real selection effect on which candidates were pursued, and it
  is a caveat on the deltas, not on the agreement measurement.
- Whether the fingerprint is visible in the *student's* errors, and not only in the
  teacher's labels, has not been tested. That would close the transfer argument
  directly instead of by inference.

## Running this on your own data

The procedure costs three extra labelling passes over a small sample:

1. Label a few hundred items. Label the same items again with an identical
   configuration. The gap is your chance floor.
2. Label them again with the instructions reworded. The extra gap is prompt
   sensitivity.
3. Label them again with a model from a different family. The extra gap on top is the
   model's own contribution.

Whatever sits above the first number will not average out, however much data you
collect. If it is large, the lever is a different family of labeller, not a better
prompt and not a bigger student.
