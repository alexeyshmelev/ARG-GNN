# Paper-based benchmark rebuild

The user authorized this rebuild on 24 October 2026. The objective is a fair comparison using reference implementations and paper simulation recipes, with explicit reasonable reconstructions where details are absent. Exact reproduction of published numbers is not implied. Earlier results remain development evidence and are reused only after condition and artifact verification.

## Completion criteria for this work session

- Every included method has an executable data, fit, prediction and scoring route, with explicit dependencies.
- Shared observations, reference/query identities, independent simulation parents and held-out evaluation support are fixed before fitting.
- Necessary paper/source conflicts are resolved with a named condition and a written rationale, preserving the unchanged release control.
- The complete queue passes dependency and interface checks. First substantive jobs demonstrate increasing optimizer counters and healthy throughput.
- GPU jobs use physical GPUs 3, 4 and 5 only. Project processes have the requested title. Other users' processes remain untouched.
- Existing evidence, source snapshots, failures and valid ongoing work are preserved. New code and concise receipts are pushed.

## Parallel ownership

| Workstream | Owner | Deliverable | State |
|---|---|---|---|
| ARGMix | ARGMix worker | Frozen 1,611-person recipe, 29-node extraction, paper attention/raw-distance corrections, paired training and scoring routes | Implementation complete; production execution/results open |
| AncestralPaths | AP worker | Historical CNN complete; authenticated 476-person reuse, static TreeLoom prediction and shared scoring interfaces | CNN fit/predictions complete; remaining predictions/scores open |
| ARGformer / FLARE | ARGformer worker | Corrected native ARGformer, explicit PCA10 reconstruction, 15 authenticated FLARE outputs and scratch lifecycle | Implementation and CPU qualification complete; corrected GPU fits/results open |
| GST and TreeLoom integration | AP worker and root | GST-LAI-8 ancestry adapter and frozen full-parent TreeLoom routes with reviewed resume state | Code and bounded CPU qualification complete; migration/launch/performance open |
| Queue, performance, storage, publication | Root | Explicit campaign DAG, authenticated source closure, storage reservations, real progress evidence, README and Git | Master v2 launched; remaining startup/performance/publication open |

## Issue ledger

| ID | Issue | Resolution / remaining work | Status |
|---|---|---|---|
| B01 | Release experiments were conflated with paper comparisons | Preserve as named controls, build separate paper-oriented arms | In progress |
| B02 | ARGMix release representation differs | Implemented 25 donors, three ancestors and query; stable raw-TMRCA selection, capped raw distances, corrected paper attention and explicit author-key sensitivity source. CPU geometry/gradient checks passed | Closed: implementation only; production/results remain open |
| B03 | ARGMix data scenario differs | Frozen 14-parent 1,611-person chr22 recipe, BAA excluded, 0.001 allele-error reconstruction and explicit unavailable-paper assumptions. Authenticated completed TRAIN0 observations reused; bounded native inference route implemented | Closed: recipe/data implementation only; production/results remain open |
| B04 | AncestralPaths current Dense network differs from paper CNN | Historical 19,078-parameter CNN recovered, CPU checks passed, full 50,000-example fit completed with 8,335 updates | Resolved, five TEST predictions completed |
| B05 | ARGformer training defaults differ from paper | Corrected named native arm implements paper temperature, literal MASK, retained extant-token pooling and actual inverse-frequency class weights; CPU and author-loader checks passed | Closed: implementation only; corrected full fits/results remain open |
| B06 | ARGformer ancestry classifier unreleased | Implemented explicit PCA10 reconstruction using training reference labels and nearest class centroids; original row identities preserved. Hard calls are reported without posterior claims | Closed: reconstruction implementation only; held-out results remain open |
| B07 | GST omitted from active comparison | Implemented GST-LAI-8 with unchanged author GraphSetConv and an explicit ancestry input/output adapter on shared LAI data. Original GST synthetic and molecular tasks were not reproduced; no original-task or isolated-layer attribution claim | Closed: LAI-adapted implementation only; launch/performance/results remain open |
| B08 | Old TreeLoom steps are very expensive | Implemented plaintext affine-prefix history, connected full-sequence GRU tiling and complete-parent decoder VJP. Frozen 49-tensor model and all Adam moments retained; CPU parity, strict continuation and cumulative checkpoint-history review passed | Closed: implementation/CPU qualification only; migration and measured GPU performance remain open |
| B09 | Storage near conservative projected capacity | One 895.59 GB measurement plus conservative live-writer growth yields a 916.08 GB existing-project reservation; campaign operational ceiling is 990 GB. Scope bounds and sequential fresh-corpus retirement are implemented | Open: live admission/storage monitoring |
| B10 | Queue coverage and current scores hard to inspect | Reviewed 251-task master v2 launched after preserved setup failure; ARGMix first stage completed with genuine wait, and next preparation admitted. Complete-parent/model progress, README and publication remain | Open: remaining startup/progress verification and publication |
| B11 | Historical instructions contain superseded GPU assignments | Current user authority is GPUs 3, 4, 5 | Resolved |
| B12 | Benchmark outputs not yet complete | Publish only measured held-out metrics with replicate SD, do not turn training loss into accuracy | Open |

## Reuse decisions

- American observations, native focal registry and FLARE outputs are candidates for reuse after identity checks. ARGformer release checkpoints are not reused as paper-oriented checkpoints because masking and pooling change.
- ARGMix TRAIN0 completed observations are reusable candidates. Older top-level partial files and unfinished Relate outputs are not completed inference.
- European 476-person simulations and sampled supports require manifest checks before CNN reuse.
- Prior scores remain historical development results until matched conditions are demonstrated.

## Execution sequence

1. Inspect active jobs and preserve healthy compatible work.
2. Implement disjoint method routes and freeze reconstruction choices.
3. Independently review adapters and test synthetic shape, label, split and metric contracts.
4. Construct the full executable campaign with storage/resource accounting and real full budgets.
5. Launch eligible stages in parallel on GPUs 3, 4, 5 and CPU preprocessing workers.
6. Verify actual progress, record any failure, repair within the same declared condition, and publish receipts.

The queue being prepared and launched is different from all benchmarks being completed. Final comparisons remain pending until real predictions and scoring finish.

## First executed correction

The historical AncestralPaths CNN completed all five epochs and 8,335 optimizer updates on the fixed 50,000-example training set. Genuine process exit0 and owned-session drain are recorded. Five independent TEST prediction products also completed. Training accuracy is not a held-out ancestry comparison. Source package and representative fit were not substituted by a smoke test.

## Earlier integration snapshot (24 October 2026)

This snapshot records the earlier integration state. The current implementation status follows below.

- Historical CNN completed all six fit/prediction tasks. Held-out scores await TreeLoom and GST predictions.
- The corrected ARGformer delivery contains 58 executable preparation/training/prediction stages. All 15 existing FLARE outputs passed reuse checks.
- The ARGMix delivery contains paired training and prediction routes for ARGMix and frozen TreeLoom. CPU geometry, gradients and two-tree recurrence checks passed.
- GST-LAI-8 uses the unchanged author GraphSetConv with an explicitly new ancestry input/output adapter. This is an LAI comparison, not a reproduction of the original GST task.
- One completed project-capacity measurement found 895.59 GB, with live-writer growth still requiring reservations. No repeated full-project scan is planned.
- Existing live jobs are preserved. Old controllers may finish them but cannot dispatch further pending stages.
- Full master integration, scratch lifecycle, GPU startup and progress verification remain unfinished.


## Current implementation and integration status

The implementation-only tickets B02, B03, B05, B06, B07 and B08 are closed. This records completed code and the stated bounded CPU qualification. It does not record completed production fits, a GPU speed improvement, reproduced published numbers or final held-out comparisons. Launch, performance, capacity monitoring and results remain open.

- ARGMix is frozen with 150 executable jobs, the 14-parent recipe, paired ARGMix/TreeLoom predictions and an all-predictions barrier before evaluation truth joins. The AP native6 static prediction factory retains the original saved-fit checks and product schemas.
- Corrected American ARGformer has 58 preparation/training/prediction tasks, 15 authenticated reusable FLARE outputs, static AST-identical author path functions and bounded scratch retirement. The two reused native corpora remain protected.
- American TreeLoom retains all 49 FP32 trainable tensors, 32,132 parameters and the 600 complete-parent optimizer-update schedule. Independent review passed. CPU continuation from an actual saved 58-update model preserved all 49 Adam states; serialized update59 followed by reload matched uninterrupted update60 bitwise on the bounded synthetic case. This establishes continuation correctness on that case; production GPU speed and full-parent memory remain unmeasured.
- The latest inspected inert master draft contains 251 tasks: 162 CPU, 82 GPU and seven external evidence nodes. Scientific join review passed with explicit remaining integration conditions. The ARGMix/AP builders now include the manifest, qualified external history and original dependency closure; 149 unique source references matched their hashes across all 150 ARGMix tasks and the AP prediction task.
- GST-LAI-8 is explicitly LAI-adapted. Original GST synthetic and molecular benchmark tasks have not been reproduced. Its production resource estimates and held-out LAI results still require execution.
- Root is executing the American TreeLoom migration from an immutable durable optimizer boundary. The final admitted campaign reference, actual process waits, startup progress and first full-parent timing must be recorded before launch verification can close. No launch or speed claim is inferred from the inert plan or CPU receipts.
- The existing capacity reservation and scope limits govern admission. Fresh corrected-corpus retirement remains conditional on complete predictions and successful outer waits; historical evidence and retained final models, logs, vocabularies and PCA artifacts remain preserved.

Evidence: `argmix/READY.json`, `argmix/paper_american_independent_review.json`, `american/status.json`, `american/treeloom_speed/model_transition_smoke_v5/completed.json`, `integration/master_scientific_join_review_v1.json`, `integration/capacity_reservation.json`, and `configs/benchmarks/paper_gst_lai_8_20261024_final.json` from the project root where applicable.

The implementation snapshot above predates the launch attempt recorded below. README and Git have not been modified by this ledger task.


## First master launch and preserved setup failure

The 251-task master campaign was launched at 2026-10-23T22:47:52.347813+00:00 as controller PID 193320, from `configs/benchmarks/paper_campaign_20261024_v1.json` (SHA256 `2b88e41c780987c11b541af87ec28c02a855e46776fce2d65130d268b0209d50`). The launch receipt records pushed commit `71886ba01eaf1ef11116c72950f1459f9ef07b61` and physical GPUs 3, 4 and 5. Launching this queue did not complete the benchmarks.

The first ARGMix stage, `paper_argmix_published_recon_train_000.generated`, was admitted as PID 193868. It exited1 because the argv execution path had not created its required output parent directory. `paper_argmix_generator._fresh_directory` correctly rejected the missing parent. Storage and RAM did not prevent admission. The scheduler recorded a real wait and no surviving owned process group; its 149 dependent ARGMix stages became blocked. The failed generated scope contained only the 1,143-byte `actual_wait.json`, without scientific payloads.

Root deliberately ended only this newly launched queue and its four other owned task sessions to prepare a coordinated retry. The final first-attempt state has six external successes, five failed tasks, 149 blocked tasks, 91 pending tasks and zero running tasks. The original ARGMix setup failure is distinguished from the four controller-ended task attempts. Root reports zero remaining sessions from this attempt and confirms that older legacy jobs were untouched.

At 2026-10-23T22:52:41.290463+00:00, Root archived five exact initial output scopes under `experiments/results/paper_campaign_20261024/initial_setup_failure`, preserving 2,647,671 bytes. `integration/initial_attempt_archive.json` records original paths, archived paths and wait evidence; its SHA256 is `fa3837ece3f0592bf22e958f58000598d690d3b20b137c82c0e24bfb4abfb905`. Independent path checks found all five archive directories present and their original task leaves absent. The source first-attempt state still matches the archive manifest reference. Original logs and state remain evidence.

The fixed ARGMix recipe and DAG namespace is retained for a deliberate retry; the failed generated leaf has been archived rather than overwritten. The coordinator is reviewing parent-directory creation, attempt-specific job/log paths, cumulative log/archive storage accounting and failed-branch recovery hooks before the second launch. Startup verification, complete-parent progress, GPU throughput and held-out scores remain open. No v2 launch is claimed in this ledger entry.


## Second master launch and first persisted ARGMix completion

Root launched the reviewed 251-task v2 campaign at 2026-10-23T22:58:08.776105+00:00, controller PID 198309, from `configs/benchmarks/paper_campaign_20261024_v2.json` (SHA256 `9f0ee40f25e9def145a85318bc723903b4ef96bd7c85060685c7f959ce3ccb9b`), with pushed commit `654703aca3dac1c53bfab117552435dc3c534517` verified in the launch receipt. The scientific task definitions are unchanged from v1. The new baseline adds exactly 2,836,426 bytes: 2,647,671 bytes of archived outputs and 188,755 bytes of retained first-attempt records. Independent review verified all 269 task source refs and six controller source refs, fresh task leaves before launch and empty first-attempt sessions. The 19 repair fixtures and exact argv-parent/cleanup fixture passed. See `integration/master_v2_independent_review.json`.

By 2026-10-23T23:01:06.481447+00:00, ARGMix TRAIN0 adoption had completed in 40.48 seconds with genuine exit0, actual wait and no surviving owned process group. The persisted observation product has 311,599 sites, 3,222 haplotypes, 1,611 diploids and 2,972 query haplotypes; observed inputs contain no genealogy or founder truth. Historical data were authenticated and adapted without regeneration, repeated allele error or changes to the originals. Its `generation.json` SHA256 is `f66b4765295109874420df8c9047d88468571485a3ea737f944f0ad04031a04e`; the task retained 128,223,252 bytes including its log at completion.

The next `published_recon_train_000.relate_inputs` preparation stage also completed with exit0, actual wait and an empty owned process group; its `relate_work/commands.json` SHA256 is `807dbc4b9f6c9eebcd86d220fed7b1d70dbe721e729307673874aa5dba0df160`. Native Relate then started as PID 200743. The recorded ARGMix snapshot contains two succeeded stages, one running stage, 147 pending stages and no errors. `argmix/master_v2_startup_progress_receipt.json` records the exact observed task rows and immutable completion/wait references. This establishes recovery of the missing-parent setup failure and persisted observation reuse. Full native inference, model optimization, GPU throughput, complete held-out prediction barriers and final scores remain open.

## Initial production acceptance, corrected v2

The complete 251-stage comparison DAG is launched and initial substantive training is verified. The scientific task definitions are unchanged from the independently reviewed first plan. All 269 scientific source pins and six controller source pins passed the final check. The first setup failure is preserved in the attempt archive and is not represented as a scientific model failure.

| Ticket | Result | Status |
|---|---|---|
| B13 | Missing argv output parent repaired. Exact scheduler fixture completed with real exit zero and no surviving workers. | Closed |
| B14 | Explicit failed-branch repair requests retain previous task definitions, logs and attempts, charge archived storage, reject running/succeeded task edits and preserve healthy workers. Nineteen fixtures and independent review passed. | Closed |
| B15 | Both corrected ARGformer MLM runs saved native 500-update checkpoints with 256,000 sampled paths each, then advanced beyond 1,200 logged batches. No traceback. | Initial training accepted |
| B16 | ARGMix adopted the authenticated TRAIN0 observations and prepared native Relate inputs. Both actual process waits returned zero. Native Relate is running. | Initial data stages accepted |
| B17 | TreeLoom retains the authenticated 58-update state. Its new process is rebuilding the process-local CPU bank before resuming update 59. GST is reading its complete observed VCFs before starting eight data-loader workers. | Open, preparation advancing |
| B18 | Full paired predictions, matched held-out metrics, replicate standard deviations and final model ranking. | Open, queued |

The GPUs are assigned only within physical 3, 4 and 5. GPU 4 reached 94–100 percent utilization with two corrected native training jobs. Utilization of GPUs 3 and 5 varies while the new TreeLoom and GST jobs prepare CPU observations alongside retained legacy jobs. Continuous full utilization of all three GPUs is not yet demonstrated. Preparation file offsets and callbacks are advancing, so CPU preparation is not labelled as completed training.

No held-out score or new superiority claim is inferred from the training counters. The queue being built, launched and initially healthy does not mean that all benchmarks have completed. Source, protocol reconstruction, progress and failure receipts remain separate and preserved.
