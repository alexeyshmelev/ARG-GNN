# TreeLoom efficiency redesign, 27 October 2026

The owner requests accurate, resource-efficient training with a target of at least 80 percent sustained useful GPU utilization. Execution and architecture changes are authorized. Simulations, observed inputs, benchmark support and competitor algorithms remain fixed. GPU activity alone is insufficient evidence of efficiency.

## Acceptance criteria

- Measure sustained GPU activity over complete preparation and training intervals, useful targets per second, time per optimizer update, peak CPU/GPU memory, and storage. Exclude diagnostic busy loops and unrelated processes from attribution.
- Exact execution changes retain model function, gradients and optimizer behavior within declared FP32 tolerances, including missing calls, empty groups, boundaries and ordering.
- Architectural changes receive new names and checkpoints. Compare predictive accuracy, calibration and ancestry boundaries on identical TRAIN/VAL support with equal disclosed training budgets. Do not use TEST results for tuning.
- An 80 percent utilization target is an engineering target to measure, not an unverified promise. A faster exact implementation can be useful even if a device counter decreases.
- Preserve reference implementations, running healthy jobs and historical results. Schedule any own-job transition only at a durable checkpoint. No changes to unrelated processes.

## Ownership

- Root: live telemetry, complete-pipeline profiling, integration, resource scheduling, acceptance decisions and Git publication.
- Execution agent: current source audit, exact bottleneck fixes, parity and gradient tests.
- Architecture agent: full ARGformer/GST papers and repository audit, scalable model design and novelty assessment.
- Fidelity agent: full ARGMix/AncestralPaths papers and repository audit, biological invariants and validation protocol.

## Work tracker

- [x] Establish current GPU3 process identity and original training call.
- [x] Delegate three independent areas with explicit ownership.
- [ ] Record complete training/preparation activity and diagnose major idle causes.
- [x] Inspect literature and source coverage, marking unavailable material explicitly.
- [ ] Implement and validate exact changes against the existing model.
- [ ] Qualify a scalable model only if architectural bottlenecks remain.
- [ ] Measure representative end-to-end performance under unchanged observations.
- [ ] Compare accuracy on fixed validation support where model function changes.
- [ ] Integrate accepted changes, publish measurements and residual limitations.

## Corrected baseline observations

The initial check read predecessor progress preserved during migration. The actual running task is attempt5 in the current job file, using the qualified October26 four-process encoder inside one parent-prefetch thread. Current source and progress are recorded in execution/active_attempt_audit.json. On inspection it had completed257updates and cached17of20parents. Preparation of the next parent overlapped training successfully in the latest four updates.

GPU3 utilization over120one-second samples averaged10percent, median5percent and maximum58percent. No sample reached80percent. This is a two-minute window, not yet a whole-training duty cycle. The preceding historical26–31minute preparation times must not be attributed to this current optimized run.

The current training still uses the original staged sweeps and pageable CPU staging. Previously validated categorical site factoring and immutable layout reuse are not active in it. The exact level-local sweep candidate addresses costly full-buffer autograd updates without changing equations. It must pass paired GPU and complete-parent checks before integration.

The guard for the new isolated profile derives from the previously reviewed project profile supervisor, with the current campaign state and study directory. Its own output allowance is64MiB and its project admission threshold995GB, below the user's1TB ceiling. It respects existing worker growth and controls only its own newly created session. No campaign or author jobs are changed.

## Measured decisions after repeated GPU testing

The first level-local sweep failed to improve warm GPU time and was rejected. Removing its redundant original topology transfers reduced API submissions but increased route payload versus the original algorithm. The second version also remained slower in eight balanced GPU rounds and is not selected for deployment.

The context-only residency candidate passed strict paired FP32 checks on 128 observed trees and all 49 gradients, updated parameters and Adam moments. Its median diagnostic update was 0.590 seconds versus 0.638 for original execution, with zero changed predicted classes among the 22 original focal records. Five differential decoder edge cases also passed exactly, including 1,025 intervals across the 1,024-step recurrent checkpoint boundary, noncontiguous inputs, duplicate targets, ragged tiles and zero-target tiles. Invalid coordinates and empty targets were rejected. These results do not establish full-parent speed, memory fit, sustained utilization or held-out accuracy.

Full-chromosome original CPU1, original CPU8 and integrated CPU8 profiling finished with actual exit zero. Update times were603.71,511.41and467.07seconds. This is one same-process first call per arm on a shared GPU. The runner used its declared loose numerical gate, and omitted an argmax gate, so it is not full strict prediction-preservation evidence. Production has not been changed. The new Transport architecture reduced small-test allocation with bounded partitions, but was much slower than its monolithic implementation. That tiny-partition path is not selected. Full-parent resource qualification and paired training/validation remain necessary before any accuracy claim.

The fidelity review finished the substantive ARGMix and AncestralPaths source files. Full papers and scientific supplemental text were read, with missing/garbled graphic extraction and archival metadata uncertainty explicitly retained. ARGformer scientific/configuration/documentation source coverage is now closed and hash-verified. GST substantive Python coverage is closed, with saved configuration and result provenance review also recorded. Unavailable source families and incomplete paper graphics remain explicit. Source coverage does not imply reproduction completeness.

## Exact blockwise backward qualification

The explicit VJP executor returns parameter contributions through autograd and retains the complete bidirectional genomic history. It changes the lifetime of gradients across tree blocks. It does not truncate history, remove targets or reuse learned state between updates. On the actual128-tree slice, four blocks, all1600sample columns and22registered targets,16paired calls passed strict FP32 checks of logits, loss, all49gradients, updated parameters and optimizer moments, with zero changed predicted classes. Six balanced warm rounds measured1.330seconds integrated versus1.120seconds explicit. Full TRAIN000 qualification completed with actual exit zero. All 10,000 original targets and all 49 gradients, parameters and optimizer states passed the strict comparator against integrated CPU8. Loss was identical and no predicted class changed. Update time was 424.363 seconds versus 511.194 seconds. Peak allocated GPU memory was 6.617 GB versus 52.064 GB. This is one fixed-order comparison on a shared GPU. This is engineering evidence, not a new ancestry score.

A separately named composition keeps final context and its decoder VJP on GPU. CPU chain-rule and strided-input tests pass. The composition passed 24 strict observed-prefix GPU comparisons with zero changed classes. Full-parent composition has not been qualified. Its full transient context concatenation requires twice the context bytes plus scratch. The production worker is preserved until a representative numerical, memory and throughput result justifies a checkpoint transition.


## Current integration queue

1. Completed: strict full-chromosome explicit backward comparison. The independent review confirms unchanged observations, all targets, complete history and optimizer update within FP32 tolerances.
2. In progress: larger tree blocks of 128, 512 and 1,024 on the same 1,153-tree observed prefix. The first attempt failed while serializing a NumPy integer in logging metadata, before CUDA. Preserve the failure and repair only the metadata emission in a versioned successor.
3. In progress: staged Transport GPU qualification. The architecture agent is preparing a paired TRAIN/VAL pilot using the staged executor. Changed-model accuracy remains untested.
4. Prepared next: strict full-parent comparison of the selected final executor directly against original execution. Select block size and context placement from measured timing and memory, preserving all original supervision and history.
5. In preparation: immutable production successor, memory admission and checkpoint transition. Current benchmark jobs remain active. No diagnostic result has been deployed yet.

The full explicit backward run's shared-device stage means were 14.0 percent for first layout construction, 44.8 percent for context forward, 86.6 percent for decoding and 71.7 percent for trunk backward. Device sharing prevents attributing these values wholly to TreeLoom. The sustained useful-utilization target remains unmet. Increasing activity is insufficient unless useful throughput and predictive accuracy are retained.
