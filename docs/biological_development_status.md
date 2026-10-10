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

The existing age-anchor, plain-anchor and matched GRU models are preparing
lossless observed whole-tree banks before seed 101 fitting. They use all 280
sampled leaves and all marginal trees. No candidate accuracy is available yet.

Native ARGformer is retrying the actual source batch-4096 optimizer probe after
a receipt-only OmegaConf container repair. Its previous full forward/backward
and optimizer step reached recording, without a completed restoration receipt.
Biological fitting has not yet completed.

Current official ARGMix release c2481248 is being installed with a coherent
Torch 2.10/CUDA 12.9 runtime for its unchanged tch 0.23 source. Packaging failures
are preserved and narrowly repaired. The user has no older April archive.
This release is a distinct native comparator, not an April-paper reproduction.

These are development results. The earlier Q64 engineering scores are from a
different cohort and cannot be ranked against this biological baseline.
Independent superiority, a final holdout result, and the real-data endpoint
remain unestablished. The manuscript stays a working methods draft.

## Resource record

The complete project-root scan ended at 2026-10-10T13:58:32Z and measured
66,237,984,768 bytes. Subsequent jobs reserve their planned writes below the
450 GB bulk-write threshold and the 500 GB maximum. Eligible project GPU jobs
remain restricted to free physical GPUs 2 and 3, with the prescribed process
title and actual NVTOP evidence. CPU-only stages hide CUDA explicitly.
