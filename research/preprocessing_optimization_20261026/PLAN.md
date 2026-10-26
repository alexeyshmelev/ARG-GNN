# Exact preprocessing optimization

Owner request: accelerate the benchmark while preserving its scientific logic, numerical outputs and native competitor implementations. GPU authority remains physical 3, 4 and 5. All files and computation stay inside this repository, with the 1 TB project limit.

## Work packages

- Tree preparation: compiled exact layout, forest, clade and cut operations. Compare the current reference routes. Preserve node and episode order and floating reductions.
- Observed loading: vectorized VCF parsing with the original fallback and unchanged validations. Compare complete retained-site joins on real observed TRAIN data.
- Independent review: adversarial cases, scientific fairness, source identity, memory and rollout assessment.
- Integration: full observed-parent comparison, logits, gradients and optimizer equivalence, measured stage timing, authenticated execution bindings and checkpoint-preserving deployment where qualified.

## Acceptance gates

1. Original files, authors' models, simulations, loss definitions, class order and evaluation support remain unchanged.
2. Every deterministic preprocessing array must match byte for byte. Error behavior, empty, isolated, unary and multifurcating trees, exact clade continuity, missing, multiallelic and phased alleles, field order and physical boundaries are covered.
3. Identical prepared inputs enter the unchanged model. Check finite logits, loss, gradients, clipping and optimizer state from the same saved state and RNG. Any floating tolerance must be stated, not presented as bitwise identity.
4. Time and memory are measured on real complete observations. Stage speedups alone do not justify claims about full training throughput.
5. Independent review resolves material findings before rollout. Reject changes that weaken baseline methods or alter the biological problem.
6. Live author runs stay intact. Our worker can migrate only using genuine owned-process completion or cancellation evidence and a durable optimizer checkpoint. No fabricated task states or edits to an in-memory controller plan.

## Current status

Implementation and release qualification are complete. `integration/release.json` records `passed_release` with twelve authenticated proof roles. American production migration remains pending review of the RAM592 configuration and subsequent live verification. Qualified does not mean deployed.

The European parser and one-parent prediction lookahead wrapper are qualified components and are not deployed. Parser qualification includes the complete actual TRAIN000 VCF. Lookahead qualification covers scheduling, cleanup and fixed-logit product contracts, not a full real prediction run. The currently healthy European worker remains intact.

## Completed proof roles

| Release role | Evidence and completed result |
|---|---|
| `parser_parity` | `observed/qualification.json`: 11,173 genotype cases, 1,750 VCF cases and 17 complete-loader contract cases. Exact arrays, error behavior and callback boundaries. |
| `parser_adversarial` | `review/parser_adversarial.json`: 2,586 independent adversarial VCF cases passed. |
| `tree_parity` | `tree/parity.json`: 782 layouts, 210 episode sequences, 210 cut plans and 25 encoded banks. Every array byte and metadata value matches. |
| `tree_adversarial` | `review/tree_adversarial.json`: 30 complete sequences and six malformed downward-level cases match the reference acceptance or rejection. |
| `native_kernel_safety` | `tree/private_safety.json`: 13 invalid link, root, sample, cycle, shape and group cases reject without hangs. |
| `full_observed_parent` | `observed/full_train_qualification.json`: 1,335,446 retained SNPs across 952 haplotypes match exactly. Original 422.10 s, candidate 46.41 s, 9.10 times parsing speedup. |
| `full_american_parent` | `integration/full_parent_qualification.json`: complete registered TRAIN000 reference, compiled serial and process4 bundles match in tensor bytes, shapes, dtypes and biological metadata. Only pointer and version stamps are excluded from semantic comparison. |
| `gpu_numerical_parity` | `integration/gpu_parity_final.json`: real 128-tree TRAIN slice, 1,600 haplotypes, 681 sites and 22 original supervised focals. Unchanged FP32 model, exact prepared inputs, finite outputs, gradients, clipping and AdamW agreement at `atol=1e-6`, `rtol=1e-5`. An unchanged-reference repeat controls for GPU rounding variation. |
| `resume_contract` | `integration/resume_contract.json`: authenticated bindings, rollback, bounded history restoration, exclusive output paths, preserved model and 49 Adam states, RNG and remaining fixed 600-update schedule. These are continuation contract tests, not evidence of a live migration. |
| `independent_review` | `review/release_review.json`: final source identities, scientific invariants, process ordering, resource admission and migration safety reviewed. No unresolved release findings in that recorded source state. |
| `process_parity` | `tree/process_parity.json`: 1, 2, 4 and 8 workers, ordered output, unchanged RNG, transport accounting, shrinking chunks, oversized complete-tree serial fallback and exception cleanup. |
| `full_process_parent` | `tree/full_process_parent.json`: final four-worker complete-parent run passed, 606 chunks, exactly four workers, native and Torch pools at one thread, no worker CUDA initialization and no surviving workers. |

## Accepted execution configuration and measurements

Accepted: four spawned CPU processes, one native thread per worker, at most 128 complete trees and 64 MiB of charged input plus output per chunk. Oversized individual trees retain the unchanged serial route after pending work drains. The process route requests an additional 8 GiB reservation and admits 6.625 GiB of runtime and transport allowances. Worker current RSS is checked against 1.5 GiB at tree boundaries. These are sampled guards and reservations, not OS-enforced memory limits.

| Complete American TRAIN000 observed preparation | Seconds | Speedup |
|---|---:|---:|
| Current benchmark implementation | 1,521.36 | 1.00 |
| Compiled serial implementation | 1,007.66 | 1.51 |
| Compiled implementation with four processes | 769.01 | 1.98 |

The four-process run reduced complete observed preparation time by 49.5 percent. Its sampled simultaneous parent-plus-descendant RSS peaked at 58,835,169,280 bytes, or 54.79 GiB. The independent maximum descendant sum was 2,677,768,192 bytes. Independent parent and descendant maxima must not be added and called a simultaneous peak. Timing occurred on a shared host and includes worker startup and final guards. It excludes the separate qualification hash pass.

Rejected: the thread-pool encoder was slower than compiled serial on actual observed input. Eight processes gave only a small additional gain on the 4,096-tree component test, so four processes were selected for lower resource cost. No claim is made that more threads or workers always improve throughput.

## Remaining deployment work

- [ ] Finish independent review of the proposed American RAM592 migration configuration and revalidate affected execution bindings.
- [ ] Migrate only the owned American worker through the reviewed scheduler path, preserving a durable optimizer checkpoint and historical evidence.
- [ ] Verify actual resumed training advances, with the approved source identities, process count, resource reservation and original training schedule.
- [ ] Record measured production timing and GPU utilization separately from qualification timings, then update the deployment section.
- [ ] Consider the European parser and lookahead wrapper only through a legitimate future continuation or repair. Preserve the healthy current worker and qualify the real prediction path before any lookahead throughput claim.

The orchestrator owns deployment, the main README and Git. This plan does not authorize changing another user's processes, changing native competitor code, rewriting active task state or bypassing the scheduler.

## Preserved history and limits

Initial status on 2026-10-26 had all component and full-parent gates open. Material review findings were fixed before the final release, with earlier artifacts retained. These include restored downward-level rejection guards, the inherited `ru_maxrss` worker-startup failure, and a completed comparison whose report failed on NumPy scalar serialization. The reporting harness now tests strict scalar conversion and atomic publication. The failed logs and partial artifact remain evidence, rather than being represented as successful releases.

The source-pin audit of the four live workers passed: GST 71 references, European TreeLoom 171, native ARGformer 15 and American TreeLoom 145. This optimization changes only our preprocessing execution. Frozen models, simulations, sample order, losses, training budgets, metrics and native competitors remain unchanged.

Exact preprocessing identity is established on the qualified complete parents and extensive edge cases. The GPU diagnostic is a bounded real training slice with explicit tolerances and an unchanged-reference control. It is neither a new ancestry accuracy result nor proof of a bitwise identical entire training trajectory. Episode continuity and cut metadata retain serial work. Full training speedup, permanent GPU saturation and the absence of future optimization needs are not established.

## Production migration update

- Closed: independent 592 GiB resource review. The full cache and both working reserves remain unchanged.
- Closed: pidfd API compatibility repair, tested and independently reviewed. The first launcher failed before any signal, and its evidence remains recorded.
- Closed: real owned worker exit, checkpoint capture at update 241, preserved history, legitimate scheduler repair and successful checkpoint/optimizer/RNG restoration on GPU 3.
- Closed: four CPU preparation workers observed in production. Native compute pools use one thread per worker. An additional OS support thread is not an additional compute pool.
- Closed: post-migration audit of all 257 source pins and PID/start identities of the three other active workers.
- Pending: first complete production preparation and optimizer update. The benchmark queue remains active, with no failed or blocked stages after migration.

Operational evidence: `review/resource592_migration_review.json`, `review/pidfd_recovery_review.json`, `review/post_migration_source_audit.json` and `integration/migration/`. Historical archives and checkpoints remain on the server.
