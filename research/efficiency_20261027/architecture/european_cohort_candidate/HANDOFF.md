# European native-six cohort-input successor

The prepared condition is `AP_paper_scale_200Mb_native6_observed_cohort8_v1`.
Use `training_plan_DISABLED_v2.json`, `independent_validation_plan_v2.json`,
and `source_manifest_v1.json`. All execution remains disabled.

The new model receives eight public observed cohort indicators for every one
of the original 952 sampled haplotypes, including ancient query/self and
sampled ancestor occurrences. Present-day samples retain an all-zero cohort
vector, age zero and the original unknown-ancestry flag. The four pure ancestry
reference labels remain separate. BAA is still an unknown ancestry donor.
All genotypes, site pooling, trees, nodes, ages, two shared sweeps, own-state
episode coupling, cut memory and full bidirectional genomic history remain.
The original leaf input projection widens from 38 to 46 inputs, adding 256
weights: 32,647 parameters across the same 49 parameter tensors.

Fresh initialization uses the original seed 101 parameters with zero new
cohort columns; it loads no trained checkpoint. The bounded CPU check uses the
public 952-row TRAIN roster and two synthetic star-tree blocks. Existing
parameters and initial base/gate/degree values match exactly in this check.
All eight new cohort channels have nonzero finite gradients. This verifies
input wiring and initial algebra, not full-parent training or predictive parity.

The authorized coverage audit opens only the original three bounded TRAIN
label arrays after authenticating the original all-parent observation barrier,
registries, class order and target metadata. The original 50,000 rows cover
all six classes with counts 15,060; 9,716; 5,044; 10,180; 7,658; 2,342 in the
canonical six-class order. Each parent contains all six classes. Sampling,
weights, targets and the original three-parent support were not altered.
Coverage and cohort patterns do not establish the cause of prior accuracy.

The disabled fit plan preserves the original 50,000 rows per pass, 30 passes,
90 complete-parent updates, seed 101, FP32, AdamW settings, clipping, parent
order seed and fixed final epoch-30 checkpoint. The executor API is explicit:
`CohortStagedExecution`, `train_sampled_step`, and `predict_validation`.
A reviewed supervised fit driver and full-parent qualification remain pending.
The old native-six and direct-four conditions and all evidence remain intact.

Independent validation is prospective. One new 200 Mb parent is registered
as `ancestralpaths_european_validation_20261027_000`, using deterministic new
ancestry, mutation and Relate seed streams checked against all original eight
parent streams. It keeps the exact 476-diploid age roster and documented AP
demography, rates, Hudson/JC69 choices and five-iteration Relate inference.
It samples 10,000 rows: 2,000 per public query cohort, with the original
observed-SNP-rank algorithm. No TRAIN parent becomes VAL and no TEST data is
repurposed. Predictions must be sealed before opening VAL scoring targets.
Validation is a fixed-final-checkpoint diagnostic, not checkpoint selection.
One VAL parent does not estimate across-parent generalization variance.

`validation_generation_v2.py` is a visible plaintext copy of the canonical
scientific function with an explicit validation split and source provenance.
`validation_sampling.py` preserves the canonical algorithm with an explicit
VAL plan schema/split and quota checks. It avoids the historical workflow's
AST-to-executable source cloning. V1's missing ordinary json import was found
in the metadata-only check; V1 files and failure evidence are retained, and
V2 passes before any simulation runtime import or output creation.

Existing TRAIN generation artifacts occupy about 2.93 GB per parent, and
retained inference scopes occupy 11.20–11.62 GB. Those are retained-file sizes,
not peak scratch estimates. The original declared ceilings are 24 GiB for
generation and 96 GiB for inference, plus a 128 MiB metadata allowance here.
Each stage retains a 128 GiB host-family ceiling, with one-day generation and
seven-day inference wall ceilings and eight native threads. ROOT's 02:24 UTC
snapshot reserves 974,745,754,223 bytes under the 1 TB decimal project limit,
leaving 25,254,245,777 bytes. The prospective reservation does not fit. No
arbitrary smaller cap, data reduction, deletion, cache or launch was made.

No scientific TEST arrays, full European observation banks, new simulation,
inference or GPU training were opened/launched by this preparation. Source
review, current peer-aware admission, native runner review, runtime pinning and
full-parent GPU qualification remain required. No accuracy or efficiency
improvement is claimed.
