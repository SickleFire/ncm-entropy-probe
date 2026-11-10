# Level-wise Uncertainty and Failure in Next-Concept Modeling: A Sudoku-Hard Pilot Study

**Joshua Russel A. Sarmiento**  
*Independent Researcher*

## Abstract

On Sudoku-hard, one Next-Concept Modeling (NCM) checkpoint fails on 16 of 100 puzzles, and those failures first show up as early divergence from the encoder's codes, but forcing the correct coarse codes repairs only 5 of them. NCM generates text as a 12-level hierarchy of residual VQ codes. We report four observations from this single checkpoint. First, the encoder emits the same Level 4 codes for all 100 puzzles, so Level 4 has exactly zero entropy and carries no puzzle-specific information. Second, code sequences are not unique: no generated sequence equals the encoder's codes for its puzzle, yet all 84 correct generations decode to the correct grid. Third, mean entropy at middle levels predicts failure (AUC 0.88 at Level 6, chosen post hoc; 0.72 at Level 0), and ranking puzzles by Level 6 entropy puts 30 correct generations first (in-sample). Fourth, all 16 failures first diverge from the encoder's codes within Levels 0–3, and none of the 55 generations that match the encoder through Level 4 fail (Fisher exact p = 4.8 × 10⁻⁷, post hoc cut). A forced-prefix intervention shows that this divergence is mostly a marker of failure and not its cause: replacing the model's Level 0–3 codes with the encoder's and letting it regenerate the rest repairs 5 of the 16 failures (95% CI 11–59%), forcing Levels 0–10 repairs all 16, and forcing breaks none of the 84 correct generations. Eleven of 16 failures persist even when the coarse sketch is correct.

## 1. Introduction

Coarse-to-fine generators produce an output in levels: a short coarse code first, then successively finer residuals. NCM [1] predicts every position of a level in parallel, which makes it fast, and raises a question about where its errors come from. One hypothesis is that the coarse sketch fixes the global structure of the answer and so decides success, while fine levels only add detail. Another is that errors arise at middle levels, where the model is most unsure.

We test these on Sudoku-hard, where correctness is exact. We ask: *where in the hierarchy do failures show up, and how does that relate to the model's own uncertainty?* We combine three kinds of evidence: level-wise entropy of the model's code distributions, a comparison between generated codes and the encoder's codes, and a forced-prefix intervention.

Our contributions, stated at the strength the evidence supports:

1. **Descriptive.** The encoder emits the same Level 4 codes for every puzzle, and generated code sequences are not unique: none match the encoder's, yet 84 of 100 decode correctly.
2. **Correlational.** Mean entropy at middle levels predicts failure (Level 6 AUC 0.88, the best of the levels on the same data; Level 0 AUC 0.72). An entropy-ranked risk-coverage curve suggests a usable uncertainty signal that we have not validated out of sample.
3. **Observational.** All 16 failures first diverge from the encoder's codes within Levels 0–3, and 0 of 55 generations that match the encoder through Level 4 fail.
4. **Interventional.** Forcing the encoder's Level 0–3 codes repairs only 5 of 16 failures and breaks none of the 84 correct generations, so a correct coarse sketch does not prevent most failures.

This is a pilot study of one checkpoint on one task with 16 failures. Section 7 lists what we did not test.

## 2. Background and Setup

**NCM.** Next-Concept Modeling (NCM) [1] represents a sequence as a hierarchy of discrete codes from a multiscale residual vector quantizer and trains a transformer to predict that hierarchy one level at a time, with all positions inside a level predicted in parallel. A VQ-VAE encoder maps text to a continuous latent, the quantizer approximates the latent with a sum of per-level residual codes, and a decoder maps the summed latent back to text.

**Checkpoint and task.** We use one checkpoint trained on Sudoku (VQ-VAE: 14.9M parameters; transformer: 25.5M parameters) and evaluate it on 100 Sudoku-hard puzzles with greedy decoding. A generation is correct if the decoded string exactly matches the reference solution. 84 of the 100 generations are correct and 16 are failures.

**Hierarchy.** The quantizer has 12 levels (Level 0 to Level 11) with 1, 2, 4, 6, 9, 13, 18, 26, 37, 50, 67 and 89 positions, 322 codes in all. Level 11 has the full output length. Levels have different sizes, so quantities averaged over positions are not equally noisy across levels.

**Encoder codes.** To obtain reference codes, we tokenize the reference solution, append an end-of-sequence token, pad to the decoder length, run the VQ-VAE encoder, and quantize the latent into per-level codes. We call these the *encoder codes*. As a sanity check, decoding the encoder codes at full depth reproduced the reference in 15 of 15 checked puzzles.

**Free-running and teacher-forced.** In *free-running* generation the model conditions each level on its own earlier codes; this is how all reported accuracies are produced. In *teacher-forced* evaluation it conditions on the encoder codes of the earlier levels.

**Level-wise entropy.** For level k with T_k positions, mean entropy is the average over positions of the Shannon entropy (in nats) of the model's code distribution, computed on the logits of the free-running generation before the argmax:

```latex
H_k = \frac{1}{T_k} \sum_{t=1}^{T_k} H\big(p(\mathrm{code}_{k,t} \mid \mathrm{codes}_{<k}, \mathrm{puzzle})\big)
```

We score entropy as a predictor of correctness with the area under the ROC curve (AUC) of the negated entropy, with correct generations as the positive class.

## 3. Level-wise entropy and failure

**Entropy profile.** Mean entropy is highest at Level 0 (about 0.50 nats), is exactly 0.00 at Level 4 for all 100 puzzles, rises again over Levels 5–10 (about 0.35 at its highest), and collapses at Level 11.

**Level 4 is a constant code.** The zero entropy does not mean the model is unusually confident. The encoder itself emits the same Level 4 codes (all 9 positions) for every puzzle: both the encoder codes and the generated codes contain exactly one distinct Level 4 sequence across the 100 puzzles. There is nothing for the model to decide there. Level 1 is nearly as constrained, with 3 distinct sequences. Every level from 2 up has 99 or 100 distinct sequences, and Level 0 has 12 distinct encoder codes. We did not investigate why the tokenizer allocates a level this way; an under-used codebook level is one possibility.

**Entropy predicts failure.** We score mean entropy as a predictor of correctness (84 correct, 16 failures). Level 0 has AUC 0.72. Levels 2, 3, 5 and 6 average about 0.86 (excluding Level 4, whose AUC is 0.5 by construction), and the best single level is Level 6 at 0.88. With 16 failures the standard error of each AUC is roughly 0.06–0.07, so we cannot say that Level 0 is a weaker predictor than the middle levels; we did not compute a bootstrap interval or a paired test. Two further caveats apply. Level 0 has one position, whereas the middle levels average over 4 to 18, which reduces noise mechanically. And Level 6 is the maximum over the levels, chosen after seeing all of them, so 0.88 is optimistic.

**Risk-coverage.** Ranking the 100 puzzles by Level 6 entropy and keeping the most confident, the 30 most confident generations are all correct. A random ordering does this with probability 0.0018, or about 0.02 after allowing for the choice among 11 non-constant levels. Keeping the 70 most confident gives 95.7% accuracy (67/70; 95% CI 88–99%) against 84% overall. This evaluation is in-sample twice over: the level was chosen for its AUC on these puzzles, and the operating point can be read off the curve. It suggests a usable uncertainty signal, not a validated abstention rule.

## 4. Generated codes versus encoder codes

We compare each free-running generation with the encoder codes for the same puzzle, level by level.

**No generation reproduces the encoder's codes.** In all 100 puzzles at least one position differs. Among the 84 correct generations, up to 20.2% of positions at a level differ (Level 10; 17.5% at Level 9), and the decoded grid is still exactly right. Many code sequences therefore decode to the same output. If training used stochastic encoding, this may be partly by design; we did not check.

| Level | Positions | Mismatch, all 100 (%) | Mismatch, 84 correct (%) | Teacher-forced accuracy (%) | Distinct sequences, generated / encoder |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 21.0 | 19.0 | 79.0 | 10 / 12 |
| 1 | 2 | 2.0 | 1.2 | 98.5 | 3 / 3 |
| 2 | 4 | 6.8 | 4.8 | 94.0 | 100 / 99 |
| 3 | 6 | 6.8 | 3.6 | 94.2 | 100 / 100 |
| 4 | 9 | 0.0 | 0.0 | 100.0 | 1 / 1 |
| 5 | 13 | 6.4 | 3.8 | 94.2 | 100 / 100 |
| 6 | 18 | 10.7 | 6.6 | 92.7 | 100 / 100 |
| 7 | 26 | 12.5 | 9.6 | 93.2 | 100 / 100 |
| 8 | 37 | 14.6 | 11.2 | 93.6 | 100 / 100 |
| 9 | 50 | 20.9 | 17.5 | 92.2 | 100 / 100 |
| 10 | 67 | 22.9 | 20.2 | 91.7 | 100 / 100 |
| 11 | 89 | 5.3 | 3.4 | 98.6 | 100 / 100 |

*Table 1. Per-position mismatch between free-running generation and the encoder codes; accuracy of the model's teacher-forced prediction given the encoder's earlier codes; and the number of distinct per-puzzle code sequences. Level 0 depends only on the puzzle, so its mismatch is one minus its teacher-forced accuracy.*

**Where failures first diverge.** For each puzzle we find the first level at which the generated codes differ from the encoder's.

| First divergence | Correct (84) | Failures (16) |
| --- | --- | --- |
| Level 0 | 16 | 5 |
| Level 1 | 1 | 0 |
| Level 2 | 4 | 5 |
| Level 3 | 8 | 6 |
| Level 4 | 0 | 0 |
| Level 5 | 20 | 0 |
| Level 6 | 23 | 0 |
| Level 7 | 10 | 0 |
| Level 8 | 2 | 0 |

*Table 2. First level at which free-running generation differs from the encoder codes. Level 4 cannot be a first divergence because it is constant.*

All 16 failures first diverge at Level 0, 2 or 3. Of the correct generations, 29 first diverge at Levels 0–3 and 55 at Levels 5–8. If the 16 failures were spread at random over the puzzles, the probability that all fall among the 45 early-diverging ones is 4.8 × 10⁻⁷ (one-sided Fisher exact test). The cut at Level 4 is post hoc, though natural since Level 4 is constant.

Read the other way, no generation that matched the encoder through Level 4 failed (0 of 55; 95% upper bound 6.5%), while 16 of 45 early-diverging generations failed (36%; 95% CI 22–51%). Early divergence is close to necessary for failure in this sample but far from sufficient. As a diagnostic that needs the reference answer, "diverges by Level 3" has sensitivity 16/16 and specificity 55/84, an AUC of about 0.83, comparable to the entropy AUCs and unavailable at inference time.

## 5. Forced-prefix intervention

Forcing the encoder's Level 0–3 codes repairs only 5 of the 16 failures, so early divergence mostly marks failure and does not cause it. Observation alone cannot separate these, since both could come from a harder puzzle, so we intervene. For each puzzle we regenerate level by level, replace the model's codes at chosen levels with the encoder's codes, let the model generate the remaining levels conditioned on them, and decode. With no forcing, this level-by-level procedure reproduced the standard generation on all 100 puzzles. We also track the 29 early-diverging and 55 late-diverging correct generations, to check that forcing does not push the model off its training distribution.

| Levels forced | Codes forced (of 322) | All correct | Failures repaired (of 16) | Early-diverging correct kept (of 29) | Late-diverging correct kept (of 55) |
| --- | --- | --- | --- | --- | --- |
| None | 0 | 84/100 | 0 | 29 | 55 |
| 0 | 1 | 85/100 | 1 | 29 | 55 |
| 0–1 | 3 | 85/100 | 1 | 29 | 55 |
| 0–2 | 7 | 86/100 | 2 | 29 | 55 |
| 0–3 | 13 | 89/100 | 5 | 29 | 55 |
| 2 only | 4 | 86/100 | 2 | 29 | 55 |
| 3 only | 6 | 87/100 | 3 | 29 | 55 |
| 5–11 | 300 | 99/100 | 15 | 29 | 55 |
| 0–10 | 233 | 100/100 | 16 | 29 | 55 |
| All | 322 | 100/100 | 16 | 29 | 55 |

*Table 3. Accuracy when the encoder's codes replace the model's at the listed levels. Level 4 is constant, so forcing it changes nothing.*

Three results stand out.

- **A correct coarse sketch repairs a minority of failures.** Forcing Levels 0–3 repairs 5 of 16 (31%; 95% CI 11–59%), so 11 failures remain even though the model is given the encoder's coarse codes. Forcing Level 2 alone repairs 2 and Level 3 alone repairs 3.
- **Forcing later levels repairs almost everything, but this is confounded by size.** Levels 5–11 supply 300 of the 322 codes, against 13 for Levels 0–3. We did not run a size-matched comparison, such as forcing Level 5 alone (13 positions), so Table 3 does not show that later levels matter more per code. Forcing Levels 0–10 repairs all 16, so no failure needed an error at Level 11 on its own.
- **Forcing never broke a correct generation.** All 29 early-diverging and all 55 late-diverging correct generations stayed correct in every condition.

Forcing replaces the model's codes with one valid code path, the encoder's. Success shows that this path works. It does not show that the model's own early codes were wrong, since 29 correct generations also diverge early.

## 6. Discussion

Early divergence from the encoder's codes is a strong marker of failure in this sample (all 16 failures; none of 55 generations with a matching coarse prefix), but the intervention shows it is mostly not the cause: giving the model the encoder's Levels 0–3 repairs 5 of 16. Three accounts fit our data, and we did not run the tests that would separate them.

- **Shared difficulty.** Hard puzzles may make the model both diverge early and fail, so that early divergence is a symptom. We did not control for difficulty, for example the number of blank cells.
- **Errors at middle levels.** The 11 failures that survive a correct sketch must involve errors at Levels 5–11, since Level 4 is constant. Entropy predicts failure best at Level 6, so mid-level entropy may be a direct signal of where those errors arise, not a downstream symptom of coarse errors. A test would be whether Level 6 entropy still separates failures from successes among puzzles whose coarse levels are correct.
- **Compensating code paths.** Because code sequences are not unique, a difference from the encoder at one level is evidence of failure only when later levels do not compensate. In the 55 late-diverging correct generations they do.

Non-unique codes also complicate entropy. Part of the entropy at a level may reflect ambiguity among codes that decode equivalently, not uncertainty about the answer, and this is hardest to rule out at Levels 9 and 10, where mismatch with the encoder is highest.

For use, the two signals differ in availability. The encoder-based divergence needs the reference answer, so it is a diagnostic. Entropy is available at inference time, so a system could abstain on high Level 6 entropy, but that rule needs validation on held-out puzzles before it is trusted.

## 7. Limitations and Future Work

The results come from one checkpoint, one task, 100 puzzles and 16 failures, so every interval is wide and nothing here is established for other tasks or for open-ended text.

**Limitations**

- **Selection effects.** Level 6 was chosen as the best AUC among the levels, and the early/late cut was set after seeing the first-divergence table. Both are optimistic. The risk-coverage curve is in-sample and has no held-out validation.
- **No difficulty baseline.** We did not compare against simple predictors such as the number of blank cells, total entropy across levels, or minimum confidence. If one of them matches the entropy results or explains early divergence, the conclusions weaken. Entropy is also averaged over all code positions of a level, without restricting to positions that carry puzzle-specific content.
- **AUC uncertainty.** We computed no bootstrap interval or paired test for the gap between Level 0 and the middle levels, and the two sides average over different numbers of positions.
- **Forced-prefix design.** The forced codes are the encoder's code path, one of many valid ones. We ran no size-matched comparison between coarse and later levels, and the groups are small (16, 29, 55).
- **Stochastic encoding.** We did not check whether training used stochastic encoding. If it did, non-unique codes are partly by design and target noise may inflate entropy at every level.
- **Constant Level 4.** We did not examine codebook usage, so we cannot say whether Level 4 is an under-used codebook level or a learned rule.

**Future work**

- Add the difficulty and confidence baselines, and validate the risk-coverage curve on held-out puzzles with the level and threshold fixed in advance.
- Compare forcing at size-matched levels, for example Level 5 against Levels 0–3, to locate where failures are determined.
- Test whether mid-level entropy separates failures among puzzles whose coarse levels are already correct.
- Repeat on other checkpoints and on open-ended text.

## 8. Conclusion

On Sudoku-hard, one NCM checkpoint's failures coincide with early divergence from the encoder's codes, yet giving it the encoder's coarse codes repairs only 5 of 16, so early divergence marks failure more than it causes it. Mid-level entropy predicts failure and may offer an abstention signal pending held-out validation. The hierarchy also contains a constant Level 4 and non-unique code paths, both of which any entropy-based analysis has to account for. These are pilot results from one checkpoint; Section 7 lists what remains open.

## References

[1] NCM: Next-Concept Modeling. HazyResearch, 2026. arXiv:2610.14162.
