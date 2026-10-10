# Biological development status

The new biological panel contains two TRAIN and two independent VAL chromosomes,
each 2 Mb, inferred from observed genotypes with Relate 1.2.4 and five
population-size iterations. All methods use the same completed inferred-tree
manifest. Each parent has 96 pure-reference haplotypes, 64 unlabeled mixed
reference haplotypes and 120 queries. Present-day 64 queries are primary.
Ancient 56 queries are secondary. No TEST chromosome has been opened.

## Completed baseline

The original AncestralPaths TensorFlow Dense network completed five epochs for
seeds 101, 202 and 303. It fitted 323,731 real source-capped TRAIN rows. The reduced
cohort sampler keeps audited quota shortfalls rather than fabricating rows.
The architecture and six-path-to-four-source probability mapping are explicit.

| Development VAL endpoint | Mean across seeds and parents |
|---|---:|
| Present-day accuracy | 82.0603% |
| Present-day macro F1 | 0.791995 |
| Present-day Brier score | 0.285653 |
| Present-day ECE, 15 bins | 0.093995 |
| Ancient-query accuracy | 85.4973% |

Evidence: [native summary](../experiments/results/ancestralpaths_biological_pilot_v2/summary.json).
Raw softmax zeros and infinite NLL are preserved. A separately reviewed auxiliary
stable-logit readout completed without training, clipping or calibration. All six
reloads passed probability parity, with no C4 argmax change. Present-day direct-log
NLL is 0.819053. Original probabilities and metrics remain unchanged.

## Active matched comparisons

The existing age-anchor, plain-anchor and matched GRU models completed all 32
lossless observed whole-tree bank tiles. Every full-field/tensor roundtrip passed,
with all 280 sampled leaves and all 3,718 marginal trees retained. Preparation
accounted for 37.9685 GB. Seed 101 fitting is running through the qualified GPU launcher. Completed
epochs already provide checkpoint-selection NLL, but final selected-model
accuracy is not yet available.

All four native ARGformer within-ARG fits completed five MLM epochs and one
full-batch-4096 contrastive epoch. The genuine source optimizer probe used
46.7586 GB peak GPU memory and restored temporary weights exactly. The independently
audited common C4 scorer passed on both sealed VAL products.

| Adapted ARGformer development VAL endpoint | Unweighted parent mean |
|---|---:|
| Present-day accuracy | 35.8972% |
| Present-day macro F1 | 0.342940 |
| Present-day raw NLL | 1.378963 |
| Ancient-query accuracy | 36.2109% |

Evidence: [readout summary](../experiments/results/official_argformer_biological_readout_v1/summary.json).
This uses independent native within-ARG fitting and a fixed pure-reference
C4 embedding readout. It has different supervision from AP and our candidate,
and does not reproduce the original paper's task. Its weak endpoint needs a
representation/readout suitability and native fit-quality assessment before
rank interpretation. It cannot establish general superiority.

Current official ARGMix release c2481248 compiled successfully with a coherent
Torch 2.10/CUDA 12.9 runtime for its unchanged tch 0.23 source. The actual
Python-to-Rust TorchScript CPU admission passed. Narrow packaging/compiler
repairs and their earlier failures are retained. The biological integration
passed independent source audit. Actual GPU qualification, native biological
preparation, fitting and ancestry scores are still pending.
Evidence: [stock build](../experiments/results/argmix_native_abi210_execution_v1/v6/build/receipt.json),
[CPU admission](../experiments/results/argmix_native_cpu_admission_v6/completion.json),
[independent bridge audit](../research/reviews/official_argmix_biological_v1_independent/review_receipt.json).

The user confirmed that no older April archive is available. This current release
is a distinct native comparator, not an April-paper reproduction. Do not repeat
the archive search without a new lead.

The AncestralPaths source audit found that the frozen release implements the
Dense network despite the supplement using the word CNN. Its cited release and
public release listing did not identify an original CNN architecture or pretrained
checkpoint. An invented CNN cannot be labelled an exact reproduction. The completed
released Dense fits are retained.
Evidence: [source handoff](ancestralpaths_CNN_source_handoff_v1.md).

Three independent fresh-context reviewers rated the unchanged working draft
Major revision. The conditional theory was viewed as plausible, with demands for
complete biological comparisons, mechanism/capacity controls, precise prior-art
distinctions and resolution sensitivity.
Evidence: [meta-review](../research/reviews/anchor_kernel_fresh_review_v2/meta_review.md).

These are development results. The earlier Q64 engineering scores are from a
different cohort and cannot be ranked against this biological baseline.
Independent superiority, a final holdout result, and the real-data endpoint
remain unestablished. The manuscript stays a working methods draft.

## Resource record

The complete project-root scan ended at 2026-10-10T16:06:16Z and measured
161,936,973,824 bytes. It reserves 80 GB for native ARGMix and 130 GB for
concurrent/follow-up writes, leaving about 78 GB before the 450 GB bulk-write
threshold after those conservative reservations. The maximum is 500 GB.
Eligible GPUs are free physical devices 2 through 5. ARGformer released GPU2
for ARGMix admission and fitting. The candidate continues on GPU3, using the
prescribed process title and actual NVTOP evidence. CPU-only stages hide CUDA.
