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
- [ ] Inspect literature and source coverage, marking unavailable material explicitly.
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

Full-chromosome original CPU1, original CPU8 and integrated CPU8 profiling remains active. A separate integrated-versus-resident full-parent qualification is being prepared. Production has not been changed. The new Transport architecture passed an observed-tree smoke test and an initial exact block-execution comparison, but still needs full-parent resource checks and paired training/validation before any accuracy claim.

The fidelity review finished the substantive ARGMix and AncestralPaths source files. Full papers and scientific supplemental text were read, with missing/garbled graphic extraction and archival metadata uncertainty explicitly retained. ARGformer/GST full papers and core scientific code were read, but a fresh semantic reread of every inventoried file is not yet claimed.
