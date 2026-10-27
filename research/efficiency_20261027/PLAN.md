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
- [x] Implement and qualify an exact executor on a complete chromosome. Further candidates remain separate.
- [ ] Qualify a scalable model only if architectural bottlenecks remain.
- [ ] Measure sustained production performance, including cache rebuilding, under unchanged observations.
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

1. Complete: CPU8, explicit backward and 512-tree blocks passed the strict full TRAIN000 comparison against original CPU1/128. Time was 380.415 versus 620.745 seconds, with 7.909 versus 51.733 GB allocated GPU memory. All original targets, full genomic history, 49 gradients and all optimizer state tensors passed. No predicted class changed. One fixed-order full call per arm on a shared GPU is qualification evidence, not steady production throughput.
2. Deployed: attempt7 continues from immutable checkpoint268 on GPU3. All original600updates, TRAIN parent order, optimizer moments and RNG are retained. Attempt6 failed before restoration because a resource-check class loaded through python -m has a different Python identity. The v3 wrapper removes that identity assumption and retains task/scope/envelope checks. A real module-launch regression passed. All predecessor history and the failed startup are preserved.
3. In progress: verify the first durable successor update269 and measure production utilization across complete preparation/training cycles. Restart lost the20-parent warm pool. Record this cost and do not present the isolated speed ratio as whole-training speedup.
4. Closed without a learning pilot: full TreeLoom-Transport resource diagnostic completed, update1122.875seconds, forward132.900/backward989.275seconds. This changed architecture is substantially slower than the selected exact executor and has no predictive validation.
5. In preparation: forward/backward fusion, followed by one bounded learned CPU/CUDA timeline. The immutable input-pack diagnostic completed with exact values and six positive paired transfer savings, but the full-parent extrapolation is only about1.7percent. Close that candidate without standalone full qualification or a production restart. Keep it separate from the fusion comparison.
6. Prepared, disabled: European cohort8/native6 correction retains all public observed cohort metadata currently given to the CNN. Preserve the previous scores. The original3TRAIN parents and90updates remain fixed. A new independent200MbVAL simulation needs a120GiB reservation that does not fit the current1TB campaign accounting. TEST is not a validation substitute. The direct four-class CNN is a separate implemented condition that has not trained.
7. Research-only: compact interval/basis representations have measured American size advantages but may smooth ancestry boundaries. Relate's European converter assigns fresh internal nodes to each local tree, so American ancestral-node identity and compression assumptions do not transfer automatically. No approximation or novelty claim is accepted without training/validation and literature evidence.

## Open tickets

- E01 closed: checkpoint269 is durable on original TRAIN001, with all10,000targets and473.03seconds measured. All49Adam counters independently verified at269, with original settings, schedule and metadata.
- E02: measure complete-cycle utilization and useful throughput, including input preparation. The80percent target remains unmet.
- E03: qualify fixed-input reuse only if its cold-inclusive microbenchmark shows repeatable savings.
- E04: assess exact first/forcing forward and backward fusion before creating another large diagnostic.
- A01: correct European public-cohort inputs and obtain independent validation within the storage limit.
- B01: finish the American, GST, ARGMix and direct-four comparisons. Report ancestry metrics only after the appropriate common scoring stages.

Closed tickets: full exact512qualification, actual-module startup regression, checkpoint268capture, successor submission, full Transport resource test. No competitor algorithm, simulation, scoring rule or original scientific training budget changed.
