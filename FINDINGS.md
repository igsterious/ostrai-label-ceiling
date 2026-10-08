# Findings

Measured on benchmark A (77 intents) over the 9,233 pool rows common to all eight
independent labelling passes. No gold labels are used anywhere in this measurement;
it compares labellers against each other.

## The decomposition

| Comparison | Disagreement | Rows | What the increment adds |
|---|---|---|---|
| Same model, same prompt, twice | 2.70% | 1,000 | Sampling noise, and nothing else |
| Same model, prompt reworded | 5.91% | 9,233 | + 3.21 prompt sensitivity |
| Different model families | 11.53% | 9,233 | + 5.62 model bias |

The first row is a separate measurement: `gpt-5.1` batch-10 against a 1,000-row
re-run of the identical configuration. The other two are the within-family and
between-family means of the full pairwise grid below.

The repeat-run pair is what makes this a measurement rather than a ratio: it fixes a
zero point. Of the 11.53 points between families, 8.83 — **77%** — is reproducible for
a given model and prompt.

That distinction decides what can be fixed. Random error points in every direction
and cancels across a large pool. A reproducible preference points the same way every
time, so a student trained on 30,000 such labels learns it as truth.

## The pairwise grid

Agreement between every pair of passes, in percent, over the 9,233 common rows.
`S` is claude-sonnet-5, `G` is gpt-5.1; `v1` and `v2` are prompt revisions, `10` is
ten-shot, `b10` is batch size ten.

| Pass | S·v1 | S·v2 | S·10 | G·v1 | G·v2 | G·b10 | DS | Gem |
|---|---|---|---|---|---|---|---|---|
| **sonnet-5 · v1** | — | *94.3* | *94.1* | 89.3 | 89.1 | 88.9 | 87.1 | 90.0 |
| **sonnet-5 · rules v2** | *94.3* | — | *93.5* | 89.1 | 89.3 | 88.7 | 86.7 | 90.2 |
| **sonnet-5 · 10-shot** | *94.1* | *93.5* | — | 89.2 | 89.0 | 88.6 | 86.7 | 90.1 |
| **gpt-5.1 · v1** | 89.3 | 89.1 | 89.2 | — | *95.6* | *93.7* | 86.2 | 89.7 |
| **gpt-5.1 · rules v2** | 89.1 | 89.3 | 89.0 | *95.6* | — | *93.4* | 85.8 | 89.6 |
| **gpt-5.1 · batch 10** | 88.9 | 88.7 | 88.6 | *93.7* | *93.4* | — | 86.1 | 89.4 |
| **deepseek-chat** | 87.1 | 86.7 | 86.7 | 86.2 | 85.8 | 86.1 | — | 87.5 |
| **gemini-3.1-flash** | 90.0 | 90.2 | 90.1 | 89.7 | 89.6 | 89.4 | 87.5 | — |

Italics mark the six same-family pairs. `deepseek-chat` and `gemini-3.1-flash` ran
once each, so they contribute none. The deepseek row is the lowest almost
everywhere, which is consistent with what its labels did downstream: −5.35 accuracy
when used to train the student.

Means: same-family **94.09** over 6 pairs, cross-family **88.47** over 22.

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
- **The noise floor is measured on one family.** The 2.70% comes from a gpt-5.1
  repeat run. The only other repeat available, deepseek-chat over 50 rows, disagrees
  with itself on 8.00% — too few rows to conclude anything (roughly ±4 points), but
  enough to say the floor may be model-specific rather than universal. If it is, the
  77% is a gpt-5.1 figure, not a constant, and a second full repeat per family is the
  cheapest way to find out.
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
