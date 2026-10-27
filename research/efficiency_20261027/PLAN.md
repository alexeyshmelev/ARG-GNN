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
3. Verified: updates269 and270 completed. The 26.9-minute production observation across cache preparation and training averaged18.35percent GPU utilization, with4.70percent of samples at least80percent. The first complete training call alone averaged31.36percent. These are device counters, not independent per-process attribution. Cache rebuilding still creates idle gaps. The80percent target remains unmet.
4. Closed without a learning pilot: full TreeLoom-Transport resource diagnostic completed, update1122.875seconds, forward132.900/backward989.275seconds. This changed architecture is substantially slower than the selected exact executor and has no predictive validation.
5. Prefix qualification passed: exact first/forcing boundary fusion reduced median unprofiled diagnostic time4.86244to3.52936seconds on1153trees, two timed calls per arm. All49gradients/parameters and147Adam state tensors passed with zero changed classes. Full-parent qualification is prepared. V1 profiler activity attribution is invalid because CUDA user annotation ranges were counted as kernels. Preserve raw evidence and replace the classification through a profiler-only follow-up. No numerical calls need repeating. The immutable input-pack diagnostic completed with exact values and six positive paired transfer savings, but the full-parent extrapolation is only about1.7percent. Close that candidate without standalone full qualification or a production restart. Keep it separate from the fusion comparison.
6. Prepared, disabled: European cohort8/native6 correction retains all public observed cohort metadata currently given to the CNN. Preserve the previous scores. The original3TRAIN parents and90updates remain fixed. A new independent200MbVAL simulation needs a120GiB reservation that does not fit the current1TB campaign accounting. TEST is not a validation substitute. The direct four-class CNN is a separate implemented condition that has not trained.
7. Research-only: compact interval/basis representations have measured American size advantages but may smooth ancestry boundaries. Relate's European converter assigns fresh internal nodes to each local tree, so American ancestral-node identity and compression assumptions do not transfer automatically. No approximation or novelty claim is accepted without training/validation and literature evidence.

## Open tickets

- E01 closed: checkpoint269 is durable on original TRAIN001, with all10,000targets and473.03seconds measured. All49Adam counters independently verified at269, with original settings, schedule and metadata.
- E02 active: first complete-cycle observation is recorded. Measure warm production separately and assess two-parent preparation. The80percent target remains unmet.
- E03 closed: six positive paired transfer savings, but full-parent extrapolation is only1.72percent. No full qualification or production restart justified.
- E04 active: real prefix numerical checks passed and27.4percent median diagnostic saving observed. Prepare one full-parent pair. GPU-resident upward states and direct gradients are a separate next candidate.
- E05 active: repair profiler CUDA-event classification. V1activity percentages/gaps/kernel rankings invalidated. Timings and numerical qualification remain valid.
- E06 active: assess two-parent CPU preparation with unchanged parent order and shared memory admission. Avoid a restart merely to rebuild the same warm cache.
- A01: correct European public-cohort inputs and obtain independent validation within the storage limit.
- B01: finish the American, GST, ARGMix and direct-four comparisons. Report ancestry metrics only after the appropriate common scoring stages.

Closed tickets: full exact512qualification, actual-module startup regression, checkpoint268capture, successor submission, full Transport resource test. No competitor algorithm, simulation, scoring rule or original scientific training budget changed.

## Latest accepted evidence and remaining work

- Full boundary fusion is qualified: 324.49 to 298.47 seconds on the original complete chromosome. All 49 gradients, 49 updated parameters and 147 Adam tensors pass strict FP32 tolerances, with no changed predicted classes. One fixed-order pair, no steady-state speed claim.
- The profiler classification repair is complete. Actual physical events exclude CUDA annotation ranges. Fused-prefix kernel activity is 36.76 percent and activity including copies is 64.56 percent. The former 95–97 percent attribution is invalidated.
- Upward/direct-gradient residency passed prefix numerical checks and gave a 14.0 percent lower measured median than boundary fusion. Full-parent qualification is running on GPU5, with a 74 GiB device budget and no production changes.
- Two-parent preparation has passed synthetic exact codec, RNG, scheduling and concurrency tests. A real-parent comparison and independent review are required before integration.
- A fully frozen European cohort8 TRAIN-only fit does not require independent VAL for checkpoint selection because it uses the fixed final epoch without selection. The earlier disabled policy must be explicitly superseded. Full-parent resource/gradient qualification and driver review remain. Independent VAL remains mandatory for subsequent architecture selection.
- Do not repeatedly restart production for incremental improvements. Combine qualified changes into one checkpoint transition and include cache rebuilding in the measured cost.

- Adaptive v1 is not ready for real qualification: independent review found that separate phase and child-credit admission can overcommit the declared host envelope. V2 must account for both atomically. A larger explicitly admitted successor RAM budget is permissible because the inherited 592 GiB cap is operational, not scientific.

## Full residency accepted, preparation qualification running

- E04 full residency prerequisite passed. Original TRAIN000, all 77,472 trees and 10,000 targets retained. The paired full call fell from 318.3155 to 237.0062 seconds. All 49 gradients, 49 parameters and 147 Adam tensors passed the strict comparator with zero changed predicted classes. Peak allocated GPU memory was 59.85 GB and reserved memory 64.80 GB. Independent completion review is UPWARD_RESIDENCY_FULL_V2_COMPLETION_REVIEW.json.
- E05 closed. Corrected physical activity distinguishes kernels, copies and annotation ranges. Shared-device activity in the new full candidate averaged 69.09 percent and does not establish exclusive TreeLoom utilization or the 80 percent target.
- E06 adaptive v2 fixed the joint phase/child-credit admission bug. Source and real-comparison launcher reviews passed. The CPU-only two-parent comparison is running under root/runs/two_parent_preparation_v1. No production transition yet.
- American production checkpoint 274 is durable. The successor must restore the latest checkpoint available at transition, including optimizer and RNG, rather than any earlier qualification checkpoint.
