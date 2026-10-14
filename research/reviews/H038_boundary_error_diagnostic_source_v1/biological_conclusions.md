# H038 Present182 boundary-error diagnostic

The diagnostic completed on both frozen selected H038 arms and reproduced all 16 original accuracy/boundary-score cells. All prediction seals and native readout geometry were verified before targets. No predictions or labels were modified. Values with ± are sample SDs across eight internal VAL simulations, ddof1, from one fitted model per arm.

| Frozen H038 arm | Precision | Recall | F1 | Hold-edge oracle recall ceiling |
|---|---:|---:|---:|---:|
| H038 aligned | 10.71±1.21% | 17.99±2.01% | 13.42±1.47% | 72.70±2.80% |
| H038 zero concordance | 8.93±1.04% | 17.39±1.94% | 11.79±1.33% | 72.70±2.80% |

**The main gap is transition recognition and tract continuity, with a substantial geometric limit.** Aligned detects only 17.99±2.01% of true boundaries against a 72.70±2.80% class-agnostic geometric oracle ceiling. Of 4,008 true changes, 1,094 (27.30%) have no allowed hold edge within 1 kb. Only 4 more are lost to one-to-one candidate competition. Of 3,289 observed misses, 2,195 (66.74%) still have a nearby candidate. The constant hold decoder limits possible recall, but it does not explain most observed misses.

**False short islands are important, and sub-kilobase spikes explain little.** Strict false internal A→B→A islands ≤10 kb account for 35.46±2.71% of extra boundaries, with deduplicated attribution. At ≤1 kb that fraction is only 4.12±1.20%. All predicted tracts ≤10 kb contain 3.83±0.73% of the wrong query-bp. Short-island errors matter much more for event counts than for base-pair accuracy, and they are not the whole error budget.

**Fragmentation extends beyond tiny islands.** Aligned predicts 6,729 changes versus 4,008 true changes (1.68×). The across-parent median interior predicted tract is 26.35±10.12 kb, versus 293.10±15.43 kb for uncensored true tracts. These are means±SD of per-parent medians, not pooled medians.

**Changing tolerance or blaming matcher competition would miss the main issue.** Only 6 of 6,010 extra predicted boundaries are within 1 kb of a true boundary that was already consumed by matching. Among positional matches, 85.69±5.22% have the correct ordered C4 transition. The original F1 ignores transition identity and can credit a nearby wrong transition, so it is slightly optimistic about biological transition fidelity. No alternative F1 or tolerance was selected.

**Useful next scientific conditions:** use independent TRAIN simulations to learn a tract-continuity or class-transition model, with explicit tests of preserving real short ancestry tracts. Study marker-aware within-hold decoding to address the approximately 27% geometric deficit, alongside training objectives that improve transition recognition where an edge already exists. Evaluate both hypotheses prospectively with the fixed scorer and a separately preserved outer evaluation. Do not apply validation-chosen smoothing or tune island cutoffs on these data.

The zero-concordance arm has the same geometric ceiling, lower precision and more strict false ≤10 kb islands. This is consistent with mutation concordance reducing fragmentation; it does not establish the cause of every false transition. Inferred-genealogy error and ancestry misclassification can also contribute.

The ceiling uses all scored-bin edges where native readout_hold changes, includes 100 bp snapping, and is a class-agnostic oracle. It does not fit an oracle predictor, impose the model’s transition constraints, or promise attainable recall. Endpoint-censored tracts, absent boundaries and undefined metrics follow the original scorer. Pooled event counts/histograms are labeled separately from equal-parent mean±SD. These H038 diagnostics do not assign boundary performance to H039 or H040 primary-only results.

Source/pins, handmade checks, all-seal preflight, per-parent compact JSON, CSV and the transition/tract plots are preserved beside this report.
