# H031 fixed observed-variant signal diagnostic

H028 did not improve total development ancestry accuracy. This diagnostic asks whether its fixed younger-cohort SNP composite contains direct ancestry signal, whether it supplies correct evidence on errors of the frozen smoother, and whether the selected residual correction is too small or too limited to use that evidence. It is an information diagnostic on repeatedly used development data, not a new method or independent validation.

## Prediction and information boundary

Use the immutable H028 module/configuration and original observed VCF, sample manifests, pureR96 labels and dates only. Preserve alpha 0.5, categorical allele encoding, original-parent 10kb windows and original 1kb inverse-density weights. No new window, LD weighting, class prior tuning, probability floor, fitting or calibration is allowed.

Prepare both cohort unit fields through the unchanged H028 functions, including exact absent older Ana evidence. Seal the raw, uncentered younger four-class field for every original Q120 scoring center on both TRAIN and both VAL parents before any query truth array opens. Retain original phase, coordinates, sample identity joins and clipped scoring bins. No tree, model weight or previous prediction payload is needed for this stage.

An entirely uncalled younger window uses the observed younger donor-count prior [24,16,22,22]/84. This is a declared fallback only for no evidence, not a floor applied to a nonzero prediction. Store its mask and report fallback query-bp coverage in every stratum. The older field remains absent for Ana and never receives a synthetic donor. Its absence does not constrain the younger direct output or final residual ancestry classes.

Source-only summaries report original site counts/callability, physical-bin-weighted raw and within-exact-query-age centered variance, feature covariance eigenvalues, and effective available blocks. For each cohort/window/query, block mass is the sum of unchanged H028 site weights over called SNPs in each original 1kb block. Report effective blocks as (sum block mass)^2/sum squared block mass, with zero for no evidence. This describes information exposure and does not invent an LD model.

## Authorized scoring and conditional diagnosis

Only an active root H031 CPU input, reviewed source pins, payload authorization and an exclusive output reservation may execute. After all four new prediction seals exist, the unchanged original full-span scorer may read TRAIN truth for a loss-free diagnostic and both VAL truths after their seals. TRAIN has no optimizer, checkpoint selection or fit. Keep raw probability NLL, accuracy, confusion/per-class recall, boundaries, short-tract, dosage and calibration products on the same support. Infinite NLL remains explicit.

Conditional complementarity uses the immutable H028 frozen-smoother and selected genotype-residual prediction seals for the two VAL parents. No TRAIN baseline is invented or recomputed. Integrate against every truth interval exactly, including mixed-class mass within a 100bp bin. For Present64, all120 and the inherited age strata, report direct-field accuracy/evidence on baseline-correct and baseline-wrong query-bp, a joint correctness table, errors the direct field could correct but the selected residual leaves wrong, residual corrections and damage, and class-specific denominators.

Read the selected H028 genotype head only after the observation seals. Report allocated 32 parameters, identically absent-column coefficients and the at-most 28 data-responsive coefficients. Report per-channel coefficient scale, row-centered decision coefficient norms, centered-field covariance/rank, residual-logit RMS/range, and scale relative to the baseline top-two logit margin. Compare the reconstructed correction with existing sealed logits as a provenance/numerical diagnostic. Do not refit, rescale or choose a threshold from these results.

The joint-correctness union is an oracle diagnostic bound, never a deployable selector. Direct-class correctness is insufficient to prove useful calibrated complementary evidence. A weak direct field would argue against more work on this exact genotype representation. Direct signal with missed complementary errors would motivate a separately registered representation or fitting hypothesis, without establishing which change will help.

## Outputs and limits

Use one CPU execution and a maximum 300,000,000-byte exclusive output through the root V7 budget consumer adapter. Validate the complete V7 storage authority and initial ledger/journal pins, then use its bounded writer and reservation checks at preparation/output boundaries. No historical V4 consumer output or budget guard is called on this storage. Root pins the reviewed budget components at activation. Retain four compact prediction seals, one observation-preparation seal and compact source/score/conditional/weight reports. Do not persist allele arrays, centered feature cubes, old predictions or weights. The four uncompressed float32 probability arrays plus fallback masks total about164MB, before modest metadata overhead. The reservation enforces the actual cap.

Authoring reads source and metadata only. Weights, original VCF arrays, prior prediction arrays and truth arrays remain unopened until root source review, H031 registration and activation. No GPU, installation, broad scan or Git change. Temporal donor drift, SNP dependence, fixed frequency compression, reused VAL selection and the constrained linear head limit interpretation.
