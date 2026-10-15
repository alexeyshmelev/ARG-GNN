# H047: test the training objective before another exchange operator

**Recommendation.** Wait for the frozen H046 comparison. If it also gives a small foreign-versus-own effect, use H047 as a zero-parameter diagnostic of endpoint allocation: continue the same full H042 body from the same pass-30 state under either the original all-query loss or a fixed 50:50 Present182/ancient504 loss. This is a falsifiable optimization hypothesis, not a proposed new architecture or a publication novelty claim.

**What the stored evidence says.** The eight-parent development contrasts are small and heterogeneous:

| Contrast | Present182 accuracy gain, percentage points | Parent SD, percentage points | Last five passes of treatment, percentage points |
|---|---:|---:|---:|
| H043 true partner versus own state | +0.01527 | 0.09198 | +0.29243 |
| H044 localized versus tree pooled | +0.00526 | 0.04432 | +0.08290 |
| H045 foreign episode versus own state | +0.00957 | 0.15266 | +0.07405 |

All arms selected pass 30. These last-five-pass differences are within-run trajectories, not causal estimates of what extra training will achieve. Nevertheless, the records do not demonstrate convergence or a saturated information ceiling. H045 TRAIN NLL also declined from 0.03597 at pass 25 to 0.03330 at pass 30 in its foreign arm. Absolute improvements over an earlier body confound continued training and added parameters; the paired operator contrasts are the relevant evidence for exchange.

H045 foreign accuracy is 96.8270% in present-day queries, versus 99.4799% in ancient queries. Its mean Present182 boundary F1 is 18.1558%, with 3,103 of 4,008 true boundaries missed and 5,063 extras across the eight parents. H044 localization gained only 0.00526 percentage points of accuracy while losing 3.13756 points of boundary F1. High span accuracy therefore leaves a substantial tract-event problem; it does not imply boundaries are resolved.

**Biological reading, with limits.** The simulated sources share deeper ancestry: Ana/CHG split within WA and WHG/EHG within NE. In H045, Ana↔CHG accounts for 41.71% of Present182 misclassified span, and the two sister-source pairs together account for 58.42%. The reference panel is uneven: Ana 50, CHG 28, WHG 102 and EHG 86 haplotypes. This makes source discrimination a plausible difficulty. It does not prove an irreducible reference limit: 41.58% of errors cross those sister pairs, and these aggregate summaries cannot identify recurrent hard loci, missing references, error-versus-callability, or error-versus-tract-length.

The body already uses reference codes, sample/node ages, donor allele evidence and branch geometry, two full tree sweeps and a genomic BiGRU. Another deterministic exchange of compressed states supplies no independent reference observation. It may improve computation, but cannot be assumed to recover evidence lost by 32-channel compression or incorrect inferred genealogies. Compression, optimization and the loss can fall short of the information in the inputs. These results distinguish none from Bayes non-identifiability.

**Specific H047 hypothesis.** Modern descendants have had more opportunity for post-admixture recombination, while many ancient queries are sampled closer to admixture. A model trained for equal query-bp risk across 686 queries may allocate its finite capacity differently from the Present182 endpoint. The intervention changes risk allocation while retaining all queries and reference evidence.

Let LP and LA be exact represented-span NLLs normalized separately within present and ancient query groups. The original objective is (182/686)LP+(504/686)LA. The treatment is 0.5LP+0.5LA, fixed before any new training. Both arms retain the identical full panel, observations, targets, forward computation, trainable 30,788-parameter H042 body, warm-start state, reset optimizer, parent order, 960 updates and evaluation barriers. Only loss aggregation changes. The original-loss continuation prevents confounding reweighting with extra training.

This hypothesis is deliberately uncertain. Despite its 26.53% nominal query weight, Present182 contributes about 67.37% of H045's equal-parent selected VAL NLL. That is not a TRAIN gradient measurement and provides no evidence of gradient starvation. Cohort counts alone cannot justify the intervention.

**Falsifier and interpretation.** Prospectively require at least +0.10 percentage points of equal-parent Present182 accuracy with no more than 1 point of mean boundary-F1 loss versus the matched continuation, reporting NLL, calibration, every ancient cohort and parent variability. Failure rejects this particular allocation hypothesis at this budget; it does not establish an information ceiling. Gains in both continuations with little paired difference favor continued adaptation over the weighting mechanism. A useful paired gain with identical inputs rules out an absolute ceiling at the old model's accuracy, but still does not isolate whether a larger body would do better. Existing eight VAL parents are development evidence; a contribution claim needs separately frozen independent confirmation.

**Novelty.** None is established. ARGMix already combines dated labeled references with TMRCA attention and discusses augmentation/adaptation (Methods 4.1, 4.3; Discussion). ARGformer explicitly uses reference-label contrastive training and inverse-frequency weighting (Section 2.1, Equations 4–5). Whitehouse et al. already combine tree graph encoding with genomic GRU context (Figure 1). Cohort-normalized risk differs from class reweighting but is a standard optimization intervention. Its value here is a controlled diagnostic and possibly a useful negative result, not a new GNN family. A credible later contribution would need a distinct biological mechanism, stress tests and independent evidence beyond these reused development parents.

All quantities trace to evidence.json and cached primary sources in citations.json. No H046 result, outer/empirical data, probability/GT arrays, model execution, implementation, GPU/probe/admission or Git action was used.
