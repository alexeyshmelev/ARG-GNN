# H023: tile-boundary development diagnostic

The completed candidate's accuracy deficit relative to current-release ARGMix is larger near the artificial250kb tile boundaries. Age-D2 is lower by1.61–1.82 percentage points within25kb and0.57 points at50–125kb. Plain-D2 shows the same pattern. This exploratory association makes whole-ARG continuity or cross-tile state carry a useful next architectural hypothesis. It does not establish that boundaries caused the gap.

Root registered H023 before outcome inspection. Only the two already exposed H020 development VAL parents were scored. These are repeated-selection development results from two biological parents, with no significance, causal, independent superiority or TEST claim. Optimizer seeds101/202/303 are separate fits, not additional biological replicates. ARGMix uses its genuine current-release native seed50; this is not an April-paper reproduction.

## Frozen endpoint and accounting correction

The primary endpoint is the unchanged64 present-day query haplotypes, integrated over the original retained100bp prediction-bin support against every original complete external truth interval. Prediction probabilities, native postprocessing, targets, IDs, positions, models, selected epochs and checkpoints were not modified. There is no probability floor or posthoc renormalization. Ground truth was decoded only by the original ExternalTruthCapability after validating the existing prediction seal, then scored with the exact unchanged original intersections and full_span_score definitions in a private CPU-only capsule. All loaded scorer function/class AST hashes, input hashes and original fit source maps are recorded.

Distance is pointwise physical distance to the nearest of seven interior boundaries250000,500000,750000,1000000,1250000,1500000,1750000bp. The four prespecified bands remain[0,10000),[10000,25000),[25000,50000),[50000,125000]bp. Prediction/truth intersection spans are split exactly by these physical regions, rather than assigning whole bins by their centers.

Those four bands cannot cover the outer terminal regions where distance exceeds125000bp. Root recorded this accounting correction before outcomes were inspected. A separate>125000 terminal row preserves the remainder. Distances are not capped and0/2Mb are not added as artificial boundaries. All five accounting strata recover the original scores.

## Equal-parent and separate-fit results

Each fit is scored separately, its two parent metrics are averaged equally, and the three candidate fit means are then averaged. No probabilities are ensembled.

| Distance to nearest interior boundary | Age-D2 accuracy | Plain-D2 accuracy | ARGMix accuracy | Age−ARGMix | Plain−ARGMix |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–10 kb | 83.926190% | 83.739435% | 85.533945% | -1.607755 pp | -1.794511 pp |
| 10–25 kb | 83.839397% | 83.736133% | 85.664289% | -1.824892 pp | -1.928156 pp |
| 25–50 kb | 84.387697% | 84.346937% | 85.287156% | -0.899459 pp | -0.940219 pp |
| 50–125 kb | 85.198641% | 85.230650% | 85.770996% | -0.572355 pp | -0.540346 pp |
| >125 kb terminal accounting | 84.868243% | 85.051170% | 84.930365% | -0.062123 pp | +0.120805 pp |
| Overall original support | 84.783742% | 84.792314% | 85.553714% | -0.769973 pp | -0.761400 pp |

The age-D2 near0–10kb minus far50–125kb difference in accuracy gaps is-1.035400 percentage points. Plain-D2 gives-1.254165 points. This is descriptive. Both methods retain age-derived input features, so this comparison is not an age-free feature ablation.

## Parent-level outcomes

The mean age-D2 gap in VAL000 changes from−1.767094 points near0–10kb to−1.218370 at50–125kb. In VAL001 it changes from−1.448415 to+0.073659 points. Near-boundary deficits are present in both parents, while absolute performance and the far-region comparison differ. A remaining VAL000 deficit also prevents attributing the entire overall gap to boundaries.

| VAL parent | Distance | Age-D2 accuracy | Plain-D2 accuracy | ARGMix accuracy | Age−ARGMix | Plain−ARGMix |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| bio_paths_pilot_val_000 | 0–10 kb | 82.334468% | 82.222489% | 84.101562% | -1.767094 pp | -1.879074 pp |
| bio_paths_pilot_val_000 | 10–25 kb | 82.176255% | 82.055789% | 84.560841% | -2.384586 pp | -2.505052 pp |
| bio_paths_pilot_val_000 | 25–50 kb | 82.042415% | 82.099885% | 84.327955% | -2.285540 pp | -2.228070 pp |
| bio_paths_pilot_val_000 | 50–125 kb | 83.559153% | 83.577750% | 84.777522% | -1.218370 pp | -1.199772 pp |
| bio_paths_pilot_val_000 | >125 kb terminal accounting | 80.167589% | 80.470840% | 81.521146% | -1.353558 pp | -1.050307 pp |
| bio_paths_pilot_val_001 | 0–10 kb | 85.517913% | 85.256380% | 86.966328% | -1.448415 pp | -1.709948 pp |
| bio_paths_pilot_val_001 | 10–25 kb | 85.502540% | 85.416478% | 86.767738% | -1.265198 pp | -1.351260 pp |
| bio_paths_pilot_val_001 | 25–50 kb | 86.732979% | 86.593988% | 86.246357% | +0.486622 pp | +0.347631 pp |
| bio_paths_pilot_val_001 | 50–125 kb | 86.838129% | 86.883551% | 86.764470% | +0.073659 pp | +0.119080 pp |
| bio_paths_pilot_val_001 | >125 kb terminal accounting | 89.568897% | 89.631501% | 88.339585% | +1.229312 pp | +1.291916 pp |

## Existing selections and support

| Method | Seed | Existing selected epoch | Equal-parent overall accuracy |
| --- | ---: | ---: | ---: |
| candidate_age_d2 | 101 | 20 | 84.890559% |
| candidate_age_d2 | 202 | 14 | 84.707411% |
| candidate_age_d2 | 303 | 18 | 84.753255% |
| candidate_plain_d2 | 101 | 20 | 84.841034% |
| candidate_plain_d2 | 202 | 14 | 84.741146% |
| candidate_plain_d2 | 303 | 18 | 84.794763% |
| current_release_ARGMix | 50 | 20 | 85.553714% |

| Parent | Original retained-bin support (bp) | Four band query-bp denominators | Terminal accounting query-bp | Total primary query-bp |
| --- | --- | --- | ---: | ---: |
| bio_paths_pilot_val_000 | [300, 1.99968e+06] | 8.96e+06, 1.344e+07, 2.24e+07, 6.72e+07 | 1.59604e+07 | 1.2796e+08 |
| bio_paths_pilot_val_001 | [116, 1.9997e+06] | 8.96e+06, 1.344e+07, 2.24e+07, 6.72e+07 | 1.59734e+07 | 1.27973e+08 |

The four interior bands have identical query-bp denominators across parents. Only the terminal accounting row differs because the original common coverage and retained-center support differ. Original source truth IDs, inferred prediction IDs, query roster slots, physical coordinates and coverage were checked exactly across all14 prediction files.

Every band sum independently matches the unchanged original scorer. Per-parent original accuracy, NLL, Brier and infinite-NLL mass match their published H020 metadata within1e-12, and the aggregate accuracies recover84.7837418692% age-D2,84.7923141578% plain-D2 and85.5537143898% ARGMix. All observed selected probabilities here have finite NLL. The analysis retains the original explicit infinite-NLL representation if positive truth mass has zero probability; a handmade zero-probability case verifies that behavior.

## Audit boundary

summary.json stores all14 per-fit/per-parent outcomes, five-stratum accuracy/NLL/Brier/confusion counts, equal-parent separate-fit means, parent means, prediction/truth/source hashes, declared selected checkpoint hashes and selected epochs. registration_snapshot.json captures H023 and the ledger read before arrays. Twelve handmade geometry/infinity checks precede the actual scoring. receipt.json seals the compact artifacts and this document.

Execution used only CPU NumPy and compiled unchanged scorer definitions. No model construction, fitting, prediction generation, GPU/device query, installation, live budget/central-ledger mutation or Git action was performed. Root owns integration and any registration of a future architecture experiment.
