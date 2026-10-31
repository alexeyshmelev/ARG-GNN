# TreeLoom execution audit and exact candidates, 27 October 2026

The live workload is limited by more than observation preparation. Its current four-process encoder generally prepares the next parent before the preceding optimizer update finishes. CPU staging, complete-chromosome tensor operations, many small GPU operations, pageable transfers, and repeated layout construction remain relevant. No result here establishes 80 percent sustained useful GPU utilization.

The first level-local candidate passed numerical checks but did **not** improve warm GPU slice runtime. It must not be adopted on the basis of its CPU kernel speedup. The minimal-transfer level-local follow-up also failed to improve GPU slice runtime. Context-only GPU residency passed strict slice checks and remains a full-parent qualification candidate. Windowed history pooling remains an unqualified isolated candidate. The root agent is separately measuring the full parent with one versus eight CPU intra-operation threads before deciding what to deploy.

## Actual current call and corrected provenance

The original campaign plan is not the active attempt definition. The running attempt is `.runtime/benchmarking/paper_campaign_resume_20261026_v1/jobs/paper_american_treeloom_fit.attempt5.json`, whose callable is `research.preprocessing_optimization_20261026.integration.treeloom_resume.run_task`. It resumes the original optimizer state at update 241 through `research/repairs/TreeLoom_lazy_compact_prefetch_20261024_v1/training.py`.

The current output is `experiments/results/paper_american_20261024/treeloom_fit/native/exact_preprocessing_from_update241_optimization_attempt5`. The older `resume_20261026_update238` logs are historical. The initial audit of that older location was corrected before this report. `active_attempt_audit.json` preserves a timestamped current-attempt snapshot, not a perpetually live status.

At that snapshot the current attempt had completed update 257 and retained 17 lossless compact parents, charged at 239,112,856,652 bytes. Updates 254–257 took 404–442 seconds in context forward, 389–416 seconds in trunk backward, 47–52 seconds in layout construction, and 46–62 seconds in the decoder. Each next-parent ready event preceded the preceding update completion. These phase times are existing production receipts, not controlled paired measurements.

The `CPU_workers: 1` prefetch event counts a parent-prefetch thread. The active `install_tree(preparation_workers=4)` hook runs four spawned encoder processes inside that thread. It replaces observed encoding, episode preparation and cut preparation only. It does not activate the earlier categorical encoder/layout integration or alter the selected tree sweep, history, decoder or optimizer. Another whole-parent process pool is therefore not presently justified by the observed warm pipeline. Four versus eight encoding processes and threads versus processes were already tested in the October 26 preparation work.

## Execution source coverage

The following relevant call path was read rather than inferred from filenames:

- Active attempt JSON, the October 26 resume callable, release hooks, four-process encoder and preparation report.
- Original lazy training loop and compact parent pool, including its once-only preparation, restore and memory admission behavior.
- `runtime_v2.py`, `static_model_v2.py`, `step_v2.py`, `staged_execution_v2.py`, `layout.py`, `decoder_v2.py` and `gru_v2.py` in the current American runtime.
- `anonymous_clade_gnn_v1.py` sweep equations and both original PyG message operators in `mutation_routed_pyg_gnn_v1.py`.
- `stable_leaf_history_v2.py`, `parallel_history_window.py`, and the original cut-memory mean.
- Prior October 24 layout, categorical-site, cache, executor, CPU qualification and GPU profiling evidence. The stopped historical full-parent profile is not treated as completed evidence.
- Current campaign `ResourceCheck` and progress-reporting code.

`source_inventory.json` records hashes for these files and the isolated candidate sources. This is a focused training-path audit, not a claim to have reviewed every repository file or every compiler/library implementation.

## Exact level-local sweep

The original sweep repeatedly overwrites complete node buffers with `index_copy_` at each depth. The candidate stores learned upward and outside values in level-local tensors and restores original node order once. Original child order, PyG operations, normalization, recipient exclusion, root outside state and parameters are retained. No learned state is reused between updates.

`level_sweep.py` contains the numerical kernel and immutable routing plan. Routing is packed into one contiguous CPU buffer for one device transfer per sweep. `executors.py` composes this kernel with the previously qualified categorical site encoder and immutable layout cache. It owns its additional CPU plans until backward completes and exposes explicit memory admission and a release operation. It does not modify live sources or monkeypatch the model.

The final packed-source CPU tests pass on singleton, unary, multifurcating, disconnected-forest and permuted-index cases and on real TRAIN000 blocks of 4, 32 and 128 trees. The 128-tree block has 317,897 nodes. Outputs, base/gate gradients and all ten connected sweep parameter gradients match exactly. Two-sweep checkpoint tests also match exactly. Final packed-source median total CPU time is 8.4891 seconds original versus 5.4285 seconds candidate, or 1.564 times, on one thread. The timing uses real topology and seeded synthetic base/gate values, not complete learned inputs. `cpu_results.json` records these final-source results. The earlier unpacked result is preserved separately.

`test_executor.py` invokes actual first and second executor methods at checkpoint 241, including categorical sites, clade and episode residuals, cut-memory injection and non-reentrant checkpoint recomputation. It preserves all 1,600 haplotypes on four real trees. Output, all 35 connected parameter gradients and memory-input gradients match exactly. Each closure runs twice. This CPU wiring test excludes chromosome history, GRU and optimizer.

Root's GPU slice comparison then tested the actual 128-tree packet, 681 SNPs and 22 original TRAIN focals from checkpoint 241 through update 242. All 49 gradients and AdamW state checks passed. Warm original times were 0.6288 and 0.6265 seconds, integrated categorical/layout times 0.5938 and 0.7160 seconds, and level-local times 0.6904 and 0.8565 seconds. The candidate adds about 0.083 seconds of plan construction per call and raises peak allocated device memory from about 1.99 GB integrated to 2.08 GB. The original is about 2.47 GB. These shared-device slice observations show no warm throughput advantage for level-local execution. They do not invalidate the earlier qualified categorical/layout result under a different partition and runtime condition, or establish a full-parent benefit.

GPU evidence: `research/efficiency_20261027/root/runs/slice_gpu5_v1/data/completed.json` and its individual call receipts. Those are root-owned results.

## Minimal-transfer follow-up

`executors_v2.py` adds a separate `level_local_v2` arm. The first candidate inherited `_block`, which transferred all original upward/downward index tuples even though the new sweep used its packed plan. The follow-up transfers only fields consumed by base construction, the new sweep and leaf readout. It leaves previous candidate files unchanged.

The CPU accounting test now measures 83 unused topology-array API calls, of which 81 are nonempty, and 10,131,256 bytes per 128-tree block visit. Four explicit CPU test visits remove 332 calls and 40,525,024 bytes relative to the first level-local candidate. The separate GPU diagnostic will count actual forward/recomputation visits rather than assume four.

This is not a net byte saving versus the original sweep. The new packed route is 25,362,680 bytes per 128-tree sweep, replacing 10,131,256 bytes of original topology. Against original/integrated execution, v2 therefore reduces API calls but increases route payload by 15,231,424 bytes per visit. The earlier approximately 24.6 GB whole-parent figure described removal of duplicated transfers from the first level-local candidate only, and was a slice extrapolation rather than a measured whole-parent total. Plan construction remains timed.

The actual first/second checkpoint CPU test passes exactly again, with 35 connected parameter gradients. `executor_v2_cpu_results.json` records this result. The later eight-repeat GPU comparison also passes strict numerical parity, but its median update time is 0.6948 seconds versus 0.608 seconds integrated and 0.638 seconds original. The level-local v2 arm remains rejected for throughput.

## Windowed history pooling

`history_pool.py` retains the original directional recurrence and its custom backward implementation. It only replaces the complete CPU `.5*(left+right)` and cut-group mean with device windows. Cut groups are disjoint across holds, so every original contributing row remains in its original group. The analytic mean VJP reconnects both complete directional histories. There is no history reset, truncated gradient or approximation.

At 77,472 holds, a single FP32 [holds,1600,8] directional array is 3,966,566,400 bytes. Both directions remain CPU outputs in this first implementation. Windowed pooling removes full-size CPU leaf addition/scaling and scatter-mean temporaries, but adds transfers of both directional windows to the device and reduced means back to CPU. It trades CPU work for device work and transfers. It does not yet eliminate every large history buffer. Original addition backward may already share its two input gradients, so no duplicate-gradient saving is claimed.

Five pooling tests, including 1,025 holds crossing a 1,024-hold window, pass bitwise on CPU. Four tests of the complete original bidirectional recurrence at all 1,600 haplotypes and different windows pass exact output, forcing gradient and ell-theta gradient parity. `history_cpu_results.json` records the evidence. GPU scatter accumulation may introduce ordinary FP32 differences and requires a paired tolerance check.

`history_executor.py` exposes `history_composed`, which combines the previously qualified categorical/layout path with windowed history pooling and the **original** tree sweep. It intentionally excludes the unproven level-local sweep. This is a standalone profiling choice, not a live binding. GPU correctness, total throughput and full-parent memory peaks remain pending.

## CPU thread and transfer diagnosis

Production currently hardcodes one PyTorch intra-operation thread and one inter-operation thread. The numerical path nevertheless allocates, concatenates, copies and differentiates CPU chromosome tensors. The complete [77472,1600,32] leaf context is 15,866,265,600 bytes. Decoder initialization creates another full CPU gradient, then copies strided haplotype tiles into it. The trunk concatenates forcing and context tensors, scans complete first-sweep buffers and reduces full history on CPU.

A fixed parent setting of eight or sixteen intra-operation threads preserves equations, parameters and complete sequence dependence. Parallel FP32 reductions may change rounding, so all-gradient and optimizer-state comparisons remain necessary. Preparation children explicitly force their own Torch/native pools to one thread. The scheduler must reserve the actual parent CPU threads plus preparation children; it must not merely change a display field. Existing running training is unchanged.

Production also performs a periodic process/child PSS query, output-scope scan and receipt write every 30 seconds. The standalone profile uses cheaper statm-based RSS checks. Root measured one current-production PSS read at 363.24 GB PSS and 2.656 seconds. At a nominal 30-second cadence that single read can consume about 8.9 percent of a wall-clock interval; output scans and other work are separate. It is meaningful overhead but does not explain roughly 90 percent GPU idleness alone. It must not be silently attributed to model speed if a standalone run is substantially faster than production.

The decoder's outer receipt says it avoids decoder recomputation, but the installed `gru_v2.chunked_scan_context` checkpoints each 1,024-hold GRU chunk. Consequently recurrent chunks do recompute during backward. The full sequence remains connected. This distinction matters when interpreting timing; no receipt or live source was changed here.

## Next controlled decisions

Root launched a complete TRAIN000 profile comparing original CPU1, original CPU8 and integrated CPU8, retaining all 77,472 trees, all 1,600 haplotypes and all 10,000 original targets. The current source files remain frozen while that profile runs. Its completion and results were not yet available when this report was written.

After that comparison, qualify `level_local_v2` and `history_composed` only as new explicit conditions if the measured residual costs justify them. Use the identical saved checkpoint 241, optimizer and RNG; preserve all original inputs, labels, weighting, clipping and FP32 rules. Measure plan construction, preparation/restore, forward, history, decoder, backward, optimizer, process RSS/PSS, peak device allocation, copy counts and sustained GPU duty separately. Warm paired orders should alternate. CPU component timing and a device busy sample are not substitutes for full-parent useful updates per hour.

No source in this directory is production activation authority. Root owns integration, source sealing, scheduling and any future checkpoint transition. This worker launched no GPU work, changed no running job and performed no Git operations.


## New comparison runner and verified accounting

`compare_minimal_transfer.py` is an isolated four-arm runner: original, integrated categorical/layout, level_local_v2, and root's `context_gpu` implementation in `root/context_residency.py`. It uses the same authenticated 128-tree observed packet, actual 22 TRAIN focals, checkpoint 241, optimizer state and RNG as the previous root protocol. It never writes training checkpoints or disk caches.

Two complete warmup rounds precede eight timed rounds. Each arm appears twice in each of the four order positions. A separate final diagnostic update per arm records actual `_block` invocations, per-field tensor calls and payloads, and CUDA event intervals around block and packed-route transfers. Optional Torch profiling is confined to that diagnostic pass. Event intervals include host launch gaps and pageable staging, so they are not isolated copy-engine times. Timed calls use the unchanged candidate factories without instrumentation.

Every call compares logits and loss at atol 1e-6 / rtol 1e-5; all 49 gradients, post-update parameters and Adam moments at atol 1e-6 / rtol 1e-4; and requires every optimizer step to become 242. Receipts include loss deltas and argmax disagreements. Plan construction remains inside each update. Overall protocol time and time through the optimizer are recorded separately, with state capture, cleanup and preparation boundaries stated.

`test_minimal_transfer_runner.py` passed on CPU. It checked balanced order, rejection of an out-of-tolerance gradient, actual block-return inventories for original/integrated/minimal/resident recipes, exact agreement of the eight retained fields, and no CUDA initialization. `runner_cpu_semantics.json` is the receipt. Four explicit CPU visits test accounting only; they are not claimed as GPU checkpoint measurements. No new GPU launch was performed by this worker.

## Memory feasibility and phased residency

`memory_feasibility.json` records analytical sizes for the complete 77,472-hold, 1,600-haplotype parent. One FP32 32-channel context is 15,866,265,600 bytes (14.7766 GiB). Its VJP has the same size. One 8-channel directional history is 3,966,566,400 bytes.

Putting the approximate 26 GB all-node upward state, both context arrays, and both directional histories on the GPU together already requires about 65.67 GB before forcing, readout, pooling temporaries, topology, episode arrays, checkpoint scratch or allocator reserve. That subtotal does not establish safe fit on an 80 GB GPU. Merely moving every existing tensor is not an admitted design.

Root's narrower context-residency path leaves the observation bank, first-sweep upward states, forcing and complete history on CPU. It keeps only second-sweep leaf context and the complete decoder VJP on CUDA. The context list plus its concatenation temporarily requires two context-size arrays; after that list is released, context plus VJP also requires two. Those large arrays total 31,732,531,200 bytes, or 29.5532 GiB. Under a 64 GiB soft budget this leaves about 34.45 GiB for all remaining allocations, before accounting for other processes on the physical GPU.

This phased design avoids the complete CPU context concatenation and zero-gradient/scatter path, as well as decoder context transfers, without transferring the whole observation bank or altering full-sequence dependence. It still pays the original first-sweep/history CPU staging costs and the complete GPU concatenation/copy cost. Bounded sweep/autograd scratch, GRU full-sequence tiles and checkpoint activations, global episode tensors, parameters, library allocations and allocator fragmentation remain unmeasured. Actual retained checkpoint tensors and the full-parent peak must be measured; the arithmetic is a feasibility proposal, not a memory guarantee or a deployment claim. GPU parity and full-parent throughput remain prerequisites.


## Completed four-arm slice comparison and new full-parent qualification

Root completed `root/runs/transfer_gpu4_v2/data/completed.json`. All 44 comparisons passed the stricter tolerances and all had zero argmax disagreements. Eight timed repeats per arm gave median update times of 0.637965 seconds original, 0.607751 seconds integrated, 0.694836 seconds level_local_v2, and 0.590110 seconds context_gpu. Ranges overlap materially; this is restricted 128-hold evidence, not established full-parent throughput.

The separate transfer diagnostic observed four actual block invocations per update. Original/integrated transferred 364 tensor API outputs, 356 nonempty, totaling 108,193,696 bytes. Minimal v2 transferred 32 block tensors totaling 67,668,672 bytes plus four packed routes totaling 101,450,720 bytes. Its combined block-and-route payload was therefore 169,119,392 bytes, greater than the original. Context residency preserves the original block recipe and changes the downstream context boundary.

`compare_full_context.py` now prepares the complete TRAIN000 parent with the same qualified four-process encoder used in `root/compare_host_parallelism.py`, before any CUDA initialization. It compares integrated CPU8 with context_gpu CPU8. One actual 128-hold update per arm warms libraries in-process; separate per-arm 8 GiB in-memory layout caches ensure both first full-parent calls start cold. An optional `--warm-repeats 1` adds full calls in reverse order. The default single full call per arm is correctness/memory qualification plus an initial timing observation, not a stable throughput estimate.

Every full call restores checkpoint 241, AdamW and RNG, preserves all 77,472 holds, 1,600 haplotypes and 10,000 original targets, and checks all logits, loss, 49 parameter gradients, post-update parameters and Adam states. Any argmax disagreement rejects qualification even if logits are within tolerance. Per-category metrics and partial receipts are written before rejecting a mismatch.

The runner records synchronized allocated/reserved high-water marks for model transfer, layout, context construction, decoder forward/backward, trunk backward, gradient capture, clipping/optimizer and state capture. `stage_memory.jsonl` is written incrementally. A run-wide maximum is retained across per-stage peak resets. Physical free-memory values are boundary snapshots; ROOT still owns continuous external monitoring and concurrent-job admission. The resident arm requires room for both complete context arrays plus at least 8 GiB of scratch reserve, then checks actual peaks against its explicit soft budget.

`full_runner_cpu_semantics.json` records passed CPU tests for the 49-gradient/147-Adam-state-tensor comparator, rejection of mismatches and within-tolerance argmax changes, allocator interval accounting, and transient soft-budget rejection using a fake allocator. No CUDA, observations, checkpoint or labels were loaded by that test. Full-parent execution has not been launched by this worker.

The existing original CPU1/CPU8/integrated CPU8 full profile remains the source for host-parallelism comparisons. Its standalone timing differs from production, so neither a different physical GPU nor cheaper resource checking is silently treated as a causal speedup. ROOT reported that concurrent full-profile host reservation was near the available-memory margin; this runner supplies no implicit admission.


## Explicit VJP and corrected memory assessment

The later full-parent allocator observation supersedes the earlier context-residency arithmetic as an admission decision. The original CPU1 call peaked at 52,854,567,936 allocated GPU bytes. Two complete 205,369,272-node FP32 state or adjoint arrays total 52,574,533,632 bytes, only 280,034,304 bytes less. This strong size correspondence supports the hypothesis that the global copy graph can enqueue many first-sweep GPU cotangents before bounded recomputation consumes them. It is not allocation-stack proof. The named finite-episode scalar/index allocations are much too small to explain the measured peak.

Adding both complete context buffers to that original peak would require 84,587,099,136 bytes. Ordinary context residency was therefore deferred pending a bounded derivative schedule. Earlier feasibility arithmetic in this report remains historical analysis, not a memory guarantee.

The separately frozen explicit_vjp.py uses original first/forcing/history/second equations under no-grad and retains CPU numerical boundaries. Backward differentiates every second block, one complete bidirectional history, then each forcing/first block in reverse block order. Every detached boundary is reconnected. All 49 original parameter identities are saved for version checks. Parameter VJPs return through autograd; the executor never writes .grad, clips, or steps an optimizer. The implementation supports first-order training on the frozen nonstochastic trunk; it does not provide stochastic replay or higher-order derivatives.

Its real 128-hold CPU test covered four blocks and eight history windows, with bitwise context equality, maximum parameter-gradient error 7.45e-9, and intact decoder gradient sentinels. ROOT subsequently reported strict actual decoder/49-gradient/AdamW/zero-argmax PASS for all 16 GPU slice calls. Six timed medians were approximately 1.330 seconds integrated and 1.120 seconds explicit, with allocated peaks approximately 0.700 and 0.686 GB. This four-block comparison must not be directly compared with earlier single-block timings. The receipt is root/runs/explicit_slice_gpu5_v1/data/completed.json. ROOT owns the separately launched full-parent comparison.

The explicit schedule still retains large host boundaries: saved up/out total 52,574,533,632 bytes and direct upward adjoints can add 26,287,266,816 bytes. The observation bank, layout, forcing, readout, history and other allocations require RSS/PSS measurement.

## Resident-context composition

The separate explicit_resident_vjp.py changes final second-sweep context placement while preserving the explicit derivative schedule. First states, forcing, full history, readout and their adjoints remain CPU boundaries. The context and incoming decoder VJP stay on the model device.

Its actual 128-hold CPU test used a noncontiguous external VJP: context was bitwise equal and all 49 gradient tensors matched with maximum error 2.33e-10. A second test used 33-hold blocks and a 29-hold tail with zero strided VJP; trunk gradients were exactly zero and decoder sentinels survived. These tests do not qualify actual GPU residency or the focal decoder.

One full context is 15,866,265,600 bytes. The context block list and concatenation coexist transiently at 31,732,531,200 bytes; later context plus decoder VJP has the same subtotal. Local graphs, GRU tiles, library workspace and allocator reserve are additional. Boundary samples inside the executor are not high-water marks. Exact derivative and lifetime analysis is in explicit_resident_static_analysis.json.

The frozen compare_explicit_resident_vjp.py compares original CPU8, integrated CPU8 and the composition using strict logits/loss, all 49 pre-clip gradients, post-update parameters, all AdamW states and zero argmax disagreement. It performs one clip and one optimizer step from 241 to 242. Six timed slice permutations place every arm in every position twice. Full admission requires at least 32 GiB context allowance plus at least 8 GiB scratch, available in both the declared soft budget and current physical free memory. The 40 GiB floor is necessary arithmetic, not an OOM guarantee.

CPU runner checks passed for all-state comparison, balanced order, budget rejection, stage clocks, controlled OOM receipts and preservation of the original OOM if subsequent telemetry fails. There is no automatic retry or smaller-workload fallback. ROOT owns GPU launches and concurrent reservations.

Keeping first states on GPU is a separate unimplemented design. Immediate forcing after each first block could release outside state early and retain only the full upward state. Releasing upward blocks as context is produced avoids overlap of the entire upward state with the entire context. Avoiding CPU snapshots, however, requires replaying the first sweep for second-block derivatives and again after the complete history adjoint. That increases first-sweep evaluations per update from two to three. Residency alone does not remove depth-wise host launches or prove high useful GPU duty.

## Larger immutable block variant

The separate large_layout_plan.py, large_layout_cache.py and large_block_executors.py extend only the admitted execution partition to 1,024 holds and bind a separate finite immutable cache. All original validation, provenance, mutation, lease and memory checks remain. Site encoding, both sweeps, forcing, finite-episode handling, complete history and explicit VJP methods are inherited unchanged. The original global 256-site chunk boundaries remain intact.

Exact full-parent metadata gives 606, 152, 76 and 38 blocks for 128, 512, 1,024 and 2,048 holds respectively. Their largest node counts are 347,010, 1,386,454, 2,767,224 and 5,521,491. One node-by-32 FP32 buffer is therefore 44.4, 177.5, 354.2 and 706.8 MB. The 2,048-hold case is analysis only; new source rejects partitions above 1,024. These buffer sizes do not bound activations, backward state, workspace or fragmentation.

CPU layout/frozen-sweep tests passed on 1,091 synthetic unary/binary holds at 128, 512 and 1,024 holds per block, including a 67-hold tail. Outputs and input gradients were exact; maximum parameter-gradient error was 4.10e-8. Immutable backing, active leases and mutation rejection were checked. A separate actual 128-hold checkpoint-241 trunk test compared four 32-hold blocks with one block under the 1,024 cap: context error was 8.49e-7 and maximum gradient error 3.49e-9, within strict tolerances. The latter test does not itself exercise more than 128 observed holds.

The frozen compare_large_blocks.py prepares an authenticated original TRAIN prefix of 1,024 or 1,153 holds before CUDA initialization. All haplotypes, observed sites and selected original TRAIN focals are retained. Every arm uses identical complete bidirectional restricted-prefix history and CPU context, with explicit VJP blocks of 128, 512 or 1,024 holds. The default performs one qualification update per arm; six balanced timed rounds are optional. A 1,153-hold slice additionally tests observed 1,024+129 partitioning and history carry across the first 1,024-hold window. CPU runner focal-boundary/duplicate selection, inherited-method, schedule and OOM checks passed.

The separately named compare_large_blocks_deferred.py adds an optional GPU predecessor gate after CPU preparation. It requires the exact owned launch/request and matching actual_wait receipt, successful return code, absent original PID/start identity and an independently absent process group/session. It then rechecks current physical GPU free memory against the declared budget plus 2 GiB. No CUDA initialization, process signaling or automatic retry occurs while waiting. CPU gate tests rejected seven invalid completion cases and verified that free-memory querying follows the exit receipt.

The four preparation workers and bounded IPC require a separate 8 GiB reservation in ROOT's peer-aware admission, in addition to parent RSS. This worker launched no GPU workload and modified no frozen dependency. Real activation scaling, full-update strict qualification and useful GPU-duty measurements remain required before deployment.
