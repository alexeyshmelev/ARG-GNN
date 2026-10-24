# TreeLoom scalability research

## Objective and fixed scientific conditions

Develop a method that improves useful local ancestry inference per unit of time and memory. Keep the existing simulation core, frozen observed datasets, reference/query identities and TRAIN/validation/TEST splits unchanged. Frozen TreeLoom and native competitors remain comparison arms. Preserve healthy jobs and completed updates. All writes remain in this repository, with GPUs 3, 4 and 5 eligible and a 1 TB project ceiling.

A new architecture is not an exact execution repair. Label it separately and measure predictive quality. Generic vectorization, hash-consing, low rank and temporal message passing are established tools, not novelty claims.

## Established starting measurements

- American TreeLoom saved 94 complete-parent updates. Attempt 2 added 36 updates. Preparing 20 compact parents took 11.65 hours. Sixteen warm updates had a median interval of 557.39 seconds. Last-ten phase means were 191.08 seconds context forward, 249.76 seconds trunk backward, 41.53 seconds decoder and 44.34 seconds layout construction.
- Attempt 2 failed at a host RAM reservation check, not a CUDA OOM. Attempt 3 resumes the same saved state at update 94 with more host RAM headroom. No new update is claimed until its progress receipt appears.
- European TreeLoom diagnostic recorded 59 of 90 updates with a 24.09-minute recent mean. A short device sample averaged 25.3 percent GPU activity. This is not a lifetime utilization measurement.
- GST has its own repeated tree-construction issue. That issue is not evidence of the same defect in TreeLoom.

## Parallel work and issue ledger

| ID | Question / deliverable | Owner | State |
|---|---|---|---|
| S01 | Which current operations actually repeat, and how much time/memory do they cost? | Execution worker | 128 real TRAIN trees measured, further full-parent profiling open |
| S02 | Reuse immutable execution layouts without changing TreeLoom equations | Execution worker | Prototype verified, integration open |
| S03 | Can identical computational states be shared across trees under the actual feature dependencies? | Model worker | Both censuses and saved-checkpoint spectrum complete, first-sweep scope only |
| S04 | Is a new factorized or approximate operator justified by measured redundancy? | Root + model worker | Exact categorical SNP encoding selected first, low rank deferred after outlier-error check |
| S05 | What is established prior art and what would constitute a defensible scientific contribution? | Independent worker | Primary-source review complete, generic sharing novelty rejected |
| S06 | Measure forward/backward, transfer, preprocessing, peak RAM/VRAM and end-to-end training throughput | Root | Representative pilot pending code review |
| S07 | Compare ancestry accuracy, per-class recall/F1, calibration and tract boundaries on unchanged data | Root | Pending a qualified candidate and substantive fit |
| S08 | Apply capacity-only failed ARGMix bridge repair without losing completed Relate output | Root + ARGMix worker | Applied at revision 5, bridge pending host RAM admission |
| S09 | Evaluate the site MLP once per observed allele and gather to haplotypes with the original pooling | Execution worker | Prototype and independent source review passed, production integration open |

## Method decisions

Exact reuse requires identical dependencies, not merely the same node ID or topology. Current TreeLoom injects per-interval SNP features and span, mutation support, parent/sibling context, structural flags and genomic history. These can invalidate numerical reuse even when a subtree is unchanged. Never cache learned states across optimizer updates without recomputation.

Measure structural redundancy separately from valid computational reuse. If exact reuse is insufficient, investigate separating shared genealogical computation from position-specific evidence. Any such separation is a new model and may sacrifice useful interactions. Do not choose low rank merely because it is fashionable. A candidate needs a defined operator, assumptions, complexity, a preserved-information argument and a measurable error or expressivity tradeoff.

## Evaluation and decision rules

1. Register each hypothesis and use the existing TRAIN data for profiling. Synthetic examples test indexing, gradients and invariances only.
2. For exact changes, check arrays and FP32 outputs/gradients at practical tolerances. Preserve objective, optimizer schedule, all trees and target weighting. Avoid an exhaustive FP64 campaign.
3. For a new model, fix the input/output contract, parameter count and training/validation selection before testing. Fit on TRAIN, select on validation, then evaluate TEST once the candidate is frozen. Test data must not choose architecture, approximation rank or context size.
4. Compare both a fixed data/optimization budget and a fixed wall-time budget. Report preprocessing separately and amortized across epochs. One GPU occupancy sample is not efficiency evidence.
5. Report SNP accuracy and macro/per-class metrics, negative log likelihood where calibrated posteriors exist, switch/boundary performance and accuracy stratified by distance to ancestry transitions. Focal labels alone cannot establish dense boundary quality. Use dense truth only in evaluation, never as model input.
6. A useful candidate must show a measured end-to-end time or memory improvement while retaining competitive validation quality. Register a numerical noninferiority margin and boundary metric before any comparative accuracy run. If quality drops materially, revise or reject it, do not hide the result.
7. Characterize scaling against tree count, edge changes, sample count and sequence length on identical frozen slices. Do not change simulations to make a candidate appear faster.
8. Publish concise measurements and negative findings. Claim superiority or novelty only after evidence supports them.

## First measured decisions

The first 128 American TRAIN trees contain 317,897 expanded active-node occurrences and 681 observed SNPs. Dense native conversion was slightly faster than a sparse remap, so the sparse remap is rejected as an American speed optimization. The global rank/stamp workspace is only 1.71 MB for this parent.

A reusable immutable layout plan preserves the original indices and reduction order. On the same slice, index storage falls from 12.67 MB to 6.34 MB. The original rebuild takes 25.5 ms, the first cached-plan construction 32.9 ms and subsequent bound reuse 56.9 microseconds. The first-sweep layout parity check passed on four real trees. These are bounded CPU component results, not a full-parent GPU speedup or a deployed change.

The conservative strict-source census finds no guaranteed upward-state sharing after including interval/SNP/gate dependencies. Primitive-feature canonicalization is a separate, less conservative question. A saved-checkpoint spectrum diagnostic will test whether position-dependent message variation is compressible before selecting an approximate operator.

References: [execution report](execution/REPORT.md), [model design](model/design.json), [scientific review](review/shared_computation_scientific_review.txt). No new ancestry accuracy, superiority or architectural novelty is established.

## Primitive-input census and approximation decision

The stricter source-ID result above is deliberately conservative. Canonicalizing identical primitive inputs and ordered operations finds 114,794 unique upward states among 317,897 occurrences, or 63.89 percent duplicate occurrences. This is predominantly leaf/input sharing. At most 0.895 percent of internal upward states can be shared under this key. It does not establish an equally large speedup or cover outside states, second sweeps or recurrence.

The observed site encoder has 1,089,600 site/haplotype feature rows but only 445 distinct primitive feature vectors on the measured slice. Evaluate once per site/allele, then gather to the original haplotypes and retain original chunk sum/max/count pooling. This exact categorical operator is selected for qualification first. It preserves early mutation-conditioned lineage reasoning and needs no simulation or training-objective change. Global hash-consing is unnecessary for the initial implementation.

At the saved 94-update checkpoint, a shared rank-eight basis captures 98.11 percent of within-subtree upward-state variation on this slice. However, relative vector error reaches 16.89 percent at the worst occurrence (6.52 percent at the 99th percentile). The physical-span-weighted variant has a still larger maximum. These are hidden-vector errors, not ancestry errors, but they rule out claiming that high average spectral capture guarantees safety. Post-hoc projection also does not avoid computing the original states. A low-rank operator remains a conditional research idea, not the selected replacement model.

The next scientific question is whether the categorical exact execution path yields a meaningful end-to-end benefit and whether any remaining bottleneck justifies an explicitly controlled approximation. Keep genotype interaction, readout resolution and simulation conditions separate in the design.

## Categorical site encoder result

The isolated prototype evaluates 768 category rows instead of 409,600 haplotype rows in the measured 256-site chunk, then restores the same haplotype ordering and chunk pooling. It retains the original early SNP conditioning. At saved checkpoint 94, FP32 sum/max outputs were identical on the measured chunk and maximum parameter-gradient error was 2.35e-7. A small missing-call/padding/tie case also passed.

On four CPU threads, warm chunk forward improved from 189.3 ms to 107.0 ms (1.77x). The single forward/backward probe improved from 525.6 ms to 265.9 ms (1.98x). The 681-site slice forward improved from 578.7 ms to 234.3 ms (2.47x). These numbers do not measure full-parent GPU training, preprocessing or ancestry accuracy. No live source was changed. See [categorical encoder report](categorical_site/REPORT.md).

Remaining engineering work is safe executor integration, GPU/end-to-end profiling, and reduction of full-parent preparation/expanded-bank costs. Remaining scientific work is a justified new operator if exact improvements are insufficient, followed by matched predictive and boundary evaluation. No new architecture or superiority claim is established by this optimization.

Latest benchmark snapshot: American TreeLoom has completed update 95 after restoration at 94. European TreeLoom has 61 update receipts. Both continue without a scientific change. The ARGMix conversion repair is applied but waits for legitimate host RAM growth reservations to become available.
