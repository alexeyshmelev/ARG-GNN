# Bounded TreeLoom learning diagnostic

Both registered arms completed 400 ordinary minibatch updates. The new model learns on this small sample. The cross-tree branch has not yet demonstrated a reliable advantage.

| Frozen v1 arm | Fitting accuracy | Development accuracy | Development macro F1 | Development cross-entropy | Training wall time |
|---|---:|---:|---:|---:|---:|
| Without cross-tree communication | **79.69%** | 68.75% | 60.21% | **0.5786** | **249.56 s** |
| With leaf and ancestral-group communication | 77.34% | **70.31%** | **65.82%** | 0.5824 | 348.73 s |

The development accuracy difference is exactly one additional correct prediction among 64 targets. The cross-tree model improves macro F1 and slightly worsens cross-entropy. Neither superiority nor the value of ancestral-group transfer is established. No sample SD is reported because there is only one development chromosome and one optimizer seed. These scores cannot be compared directly with the earlier five-chromosome CNN/TreeLoom benchmark.

## Fixed comparison

Targets were selected uniformly by canonical example ID, independently of labels, before export. There are 128 fitting draws from original TRAIN000/001 and 64 development draws from original TRAIN002. The registered policy preserves duplicates, although these particular draws happened to be unique. Each example includes the focal tree and four coordinate draws in a clipped 50 kb neighborhood. Every selected tree is complete, with all observed SNPs and all 952 leaves. Complete tree intervals and their SNP pools can extend outside that coordinate neighborhood.

Both arms use the same frozen model source, initial parameters, context draws, target order, input metadata and optimizer configuration. Their stored initialization and batch-order hashes match. Each receives 1,600 target presentations in 400 batches of four. The fixed final checkpoint is evaluated, with no development selection, early stopping or TEST access. All used gradients, losses and updated parameters remained finite. Learned neural tensors use FP32. Small geometric kernels use double precision.

The observations retain cohort categories, reference roles, ages and genotypes. Labels are stored separately and never enter the model's observation API. The two models intentionally differ in communication. The no-cross arm also skips a redundant second tree sweep but still encodes the matched context packet. Its wall time is not the fastest possible one-tree baseline.

## Interpretation and next decisions

Initial development accuracy was 10.94% for both arms. The learning curves and fixed final evaluations establish that the bounded data and optimizer pipeline can learn. They do not explain how much of the old full-parent model's failure was due to metadata, architecture or optimizer schedule, because this prototype changes several of those factors.

Some classes remain difficult. The cross-tree model identifies 2 of 15 development CHG_nonBAA targets and 1 of 6 WHG targets. Small class counts make estimates unstable. A larger development evaluation and adequate fitting support are necessary before deciding whether the architecture is competitive. A leaf-only control, matched distance scales/capacity, and disrupted ancestral correspondence are required before attributing a benefit to the proposed partition transfer. The direct metadata/SNP route also needs a separate control.

Training used two CPU threads and two preparation workers per arm. Mean wall time per update was 0.624 seconds without cross-tree communication and 0.872 seconds with it. Main-process peak RSS was 3.17 and 4.24 GB respectively, excluding preparation workers. Measured loader waiting occupied about 11.2% and 8.0% of training wall time. Worker preparation runs concurrently, so its accumulated time must not be added to main-process wall time. These are concurrent CPU diagnostic measurements, not a GPU throughput claim or a final scaling result.

The original simulation data, native baselines and production queue remain unchanged. A separately versioned focal executor is under review. It must preserve the predictor while removing computation that cannot affect requested focal outputs. A separate sliding graph-convolution research line investigates simultaneous messages within and across tree layers without recurrence.

[Exact metrics and source hashes](root/learning_summary.json) · [Registered protocol](root/pilot_protocol.json) · [Data verification](data/diagnostic_packet_validation.json) · [Independent harness review](review/HARNESS_REVIEW.json)
