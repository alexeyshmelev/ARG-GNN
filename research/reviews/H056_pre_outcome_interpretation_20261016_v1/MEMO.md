# H056 frozen pre-outcome interpretation memo

This memo interprets the existing H056 protocol. It changes no hypothesis, cohort, window, weighting, predictor, threshold or analysis. It was prepared from source and metadata before this worker opened any H056 model output. Root reported that no H056 model outcomes existed at assignment. H055 outcomes were already known, so this is not a new preregistration made blind to H055. The original H056 protocol was frozen before H055 estimates were opened, but after H055 genotype acquisition.

## What is being tested

The fixed question is whether the existing H045 foreign_episode pass30 predictor assigns higher mean EHG and CHG probability to the selected later Danish people than to the selected earlier people, over the fixed chr20/21/22 GRCh37 windows [20,30) Mb. Both contrasts must be reported regardless of sign. The protocol has no effect-size threshold and makes no empirical ancestry-accuracy claim.

Use the same development-selected seed101 checkpoint, with no weight, temperature, calibration, reference-label or selection changes. Its descriptor records 31,812 parameters and 960 H045-stage updates. That count is not the total training history of its warm-started ancestry encoder. Application uses zero optimization steps.

The groups are selected Early Neolithic people versus selected Middle Neolithic B/Late Neolithic people. They do not span Denmark's initial introduction of farming. Earlier reported 95% date bounds cover 5311–5747 calBP1950 and later bounds cover 3851–4404 calBP1950. Country, archaeological period and dates do not determine individual ancestry. The prior positive direction was a plausible farmer-to-steppe-era extrapolation, not known ground truth for these exact people and windows.

## Fixed people and references

Earlier: NEO866, NEO945, NEO28, NEO942, NEO790. Later: NEO739, Gjerrild5, NEO735, NEO92, NEO870.

The whole inferred panel contains 18 diploid people, 36 haplotypes. Eight reference people contribute 16 reference haplotypes, with four haplotypes per class. Ten query people contribute 20 query haplotypes, all without ancestry targets.

| Assigned reference class | Fixed reference people |
|---|---|
| ANA | Bar8, Bon002 |
| CHG | KK1, SATP |
| WHG | Loschbour, Bichon |
| EHG | Sidelkino, NEO88 |

These are sparse labelled proxies. Their purity, genetic unrelatedness and equivalence to the simulated source populations have not been proved. They are not direct Yamnaya samples. Original mean calBP1950/28 dates enter the observed model inputs. Consequently a temporal contrast in output does not demonstrate ancestry inference independent of supplied age information.

## Exact primary weighting

For person i, chromosome c and source a, let b[t,c] be the genuine covered physical length of native hold t and L[c] its sum. The source-preserving application computes

P[i,c,a] = sum_t b[t,c] * (p[t,2i,a] + p[t,2i+1,a]) / (2 * L[c]).

It then pools chromosomes by L[c], giving P[i,a], averages people equally within each five-person era, and reports Delta[a] = mean_later P[i,a] - mean_earlier P[i,a]. EHG and CHG are the two primary components. Report all four components as registered secondary context.

This is physical-span weighting, not SNP weighting, equal-tree weighting, majority hard calls or whole-genome ancestry. Report each chromosome separately, its genuine covered interval and bp, and uncovered tails. Do not extrapolate beyond native support. All three fixed chromosomes and all fixed people are required. Missingness is inconclusive, never replaced by a new person, interval or older H40 prediction.

The source implementation at apply_h36_v1.py lines19–63 matches this order. Its native per-hold probabilities are FP32 and its small aggregate reductions use FP64. This is aggregation arithmetic, not another FP64 model-training experiment.

## Units and dependence

There are ten query people, five per era. The 20 haplotypes are paired within diploid people. Trees, SNPs and genomic bases are repeated, dependent observations and do not increase the biological sample size. All predictions also share reference people, imputation, an inferred panel and one trained model.

The three chromosome windows are repeated measurements of the same ten people, not three independent cohorts or three model-training replicates. Chromosome SD or leave-one-chromosome differences describe heterogeneity across these three windows, not a population-level confidence interval. The registered delete-one-person and delete-one-site summaries are the same ten deletion units because the ten people have ten distinct sites. No listed relatives does not establish genetic independence.

Use the already registered sensitivity summaries descriptively. Do not manufacture SNP-independent p-values, significance, effective sample sizes or Gaussian uncertainty from the many trees or bases. Sample variation among five people per era does not establish representative sampling of Denmark.

## Input and genealogy shifts

H056 preserves the learned equations and weights but changes the original 952-input simulated panel to H36/R16/Q20, a sparse empirical proxy panel. It uses up to10Mb of continuous recurrent history per chromosome versus2Mb training regions, with recurrent state reset only at chromosome boundaries. Panel frequencies, descendant fractions, clade episodes, observation quality and genomic-context distributions can change even if the parameter shapes and arithmetic remain compatible.

The fixed empirical native profile is pre-size Relate with effective haploid N30000, mutation rate1.25e-8,28years/generation and zero population-size iterations. The main simulated TRAIN inference condition is post-size, after five population-size estimation iterations. This is a disclosed genealogy/timing-feature domain shift. It has not been eliminated by source review or numerical interface qualification.

The native strict-GT reconstruction limitation stays preserved. Original phased VCF GT supplies model features while separately qualified node/edge/sample/date/support structure supplies graph geometry. Do not describe this as successful lossless native genotype reconstruction. Known imputation, phasing, proxy and genealogy uncertainty is not estimated by applying one frozen predictor once.

## Honest outcome rules

- Both pooled Delta[EHG] and Delta[CHG] strictly positive: the model assignments satisfy the two prespecified directions for this finite panel and covered support. This does not demonstrate calibrated mixture fractions, local ancestry accuracy, a useful effect size or an independently confirmed historical turnover. Report any opposite chromosome signs without suppressing them.
- One positive and the other zero or negative: mixed evidence. The joint two-component directional expectation is not met. Do not retain only the favorable class.
- Both zero or negative: the fixed positive model-assignment expectation is not met. This does not prove the historical pattern is absent, because model transfer, proxies, panel sampling and restricted support remain uncertain.
- Incomplete chromosome, person or model output: no complete primary H056 result. A partial descriptive artifact must not become the fixed three-window result.

There is no frozen magnitude cutoff. An arbitrarily small positive number is a positive sign, not evidence of a material biological effect. The four probabilities sum to one, so their contrasts sum to approximately zero. EHG and CHG changes are compositionally linked to ANA/WHG changes, not independent confirmations.

## What H055 can and cannot corroborate

H055 uses the same ten queries, same nominal windows and source imputed phased VCF archive, retaining GP expected dosages rather than phased GT. It also shares the Sidelkino, KK1 and Bar8 proxy people, all present in the H056 reference panel. Shared source genotype/imputation error and proxy choices make it dependent evidence.

Its fixed quantities are equal-qualified-SNP means of (mean_later dosage/2 - mean_earlier dosage/2) times (proxy dosage/2 - Bar8 dosage/2), for Sidelkino and KK1. It uses one complete13-person support set for both contrasts and pools chromosomes by qualified SNP count. H056 instead pools native physical spans and uses all eight labelled reference people through a nonlinear trained model. QC differs, including phased-call, GP-maximum, HS, external-frequency and ancestral-allele conditions in H056. Their SNP/support sets and weighting are not identical. A probability contrast is not numerically commensurate with f4 and its sign is not mathematically guaranteed to track these f4 products.

The preserved H055 audit records both contrasts negative on all three chromosomes, contrary to H055's fixed positive expectation. A positive H056 outcome would therefore be discordant with that observed proxy-affinity result. It cannot rescue or relabel H055 as a success, select a new proxy or certify that the neural model is correct. A negative H056 outcome would not fulfill H056's positive expectation even if signs agreed with H055. Agreement is at most descriptive consistency between dependent measurements, never independent ancestry truth or method superiority.

## Evidence and scope

The exact frozen protocol is ../H056_frozen_H045_Danish_application_registration_20261016_v1/protocol.root_frozen_v1.json, SHA256 9544bdbd499bddafeb0774c67e4e86ef25535d41dd2409e710c29b679619cd6e. The companion source_handoff.v1.json binds the inspected source/metadata references and locators.

Only source and JSON/TSV metadata were read. No genotype, tree, bank, checkpoint or prediction payload was opened or hashed. No target module was imported, no scientific computation or process/GPU probe ran, no new experiment was registered, and no existing scientific file was modified.
