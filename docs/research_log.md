# Research log

## 2026-10-10: initialization
- Cloned the empty user repository to /workspace-SR003.nfs2/ARG-GNN.
- Confirmed seven A100-SXM4-80GB devices. Project eligibility is restricted to free GPUs 2–5, with at most two active.
- Confirmed conda 24.11.3 at /home/jovyan/miniconda3/bin/conda.
- Started independent architecture, ARG benchmark and prior-art investigations.
- Research status: hypothesis formation. No training or validated model results yet.
- Initial evidence: GST already interleaves graph encoding and set feedback. Positional encoding alone would be insufficient evidence of novelty.
- Initial reproducibility finding: current ARGMix code differs from its April 2026 manuscript, requiring pinned implementations and explicit reproduction scope.

## Infrastructure and initial implementation
- Dedicated conda environment installed with CUDA PyTorch 2.6.0, msprime, tskit and NVTOP. Disabled user-site packages to prevent shared-environment leakage.
- Resource guards restrict jobs to free GPUs 2–5 and at most two concurrent leases.
- A small GPU 2 probe confirmed the requested process title via NVIDIA process reporting. NVTOP verification will be captured during a training job.
- Native PyTorch graph models, independent temporal controls and migration-derived simulation labels implemented.
- End-to-end CPU correctness tests passed.
- Versioned oracle pilot protocol frozen before model training. Its role is numerical/engineering validation and headroom assessment, not a final novelty claim.

## Completed oracle engineering pilot
- Two training seeds completed seven model/control variants each on the frozen 16/4/8 parent-replicate panel.
- NVTOP captured both GPU process names on physical GPUs 2 and 3.
- The nearest-TMRCA comparator scored 98.10% on the uniform held-out grid. The initial neural encoders were much weaker and often predicted nearly constant classes. This pilot does not establish a useful proposed-model gain.
- Next step: diagnose reference-information reachability, add stronger genealogy-conditioned local controls, and assess the separate four-source ancient-European generator.
- A fresh-context code audit led to fixes for switch diagnostics, checkpoint uniqueness, provenance checks, storage guards, and inherited GPU leases.
- Corrected full-resolution diagnostics changed 7 of 28 cases, from 749 to 759 switches in total. Original labels/tensors were preserved and their hashes verified.

## Fixed-anchor plain pilot and control integration, 2026-10-10

The nine-model plain inferred Q64 pilot completed on the reused 16 TRAIN/four VAL parents with seed101. Full depth2 selected NLL is 1.02564669, full depth1 1.03080856, scalar depth2 1.03870764, diagonal depth2 1.03976517 and gain-matched diagonal depth2 1.04072184. This is exploratory selection evidence. There is no final ancestry test or established novel superiority. The complete score and per-parent ledger is in research/reviews/anchor_plain_pilot_complete_v1.json.

Capacity/source-feedback and split-age interventions, canonicalization and direct-core Chickenpox adaptation are integrated in new independent modules. Their last combined integration verification passed 173 tests. Their training remains pending. The exhaustive label-free age-panel audit establishes non-scalar geometry and about94–95% physical-span source coverage, with no predictive implication. The fresh-development/confirmation protocol is a proposal requiring concrete execution and analysis locks. No confirmatory parents have been generated.

Relate authorization was reaffirmed by the user: “I reviewed, use it.” Credentials remain external to scientific artifacts.

## Fresh-development fitting and analysis implementation, 2026-10-10

The independent train_anchor_development.py runner restricts production to the declared64TRAIN/16VAL/TEST0 cohort and exactly eight plain/age × depth1/2 × learning-rate configurations, with three fixed training seeds. It verifies concrete dataset/configuration hashes and selects equally weighted parent NLL from stable float64 raw logits. It stores raw logits and compact dataset bindings. Production fitting has not started. The new confirmatory_metrics.py supplies synthetic-verified stable NLL, explicit missing-class support, calibration/dosage metrics, paired-parent t/bootstrap analysis, Holm adjustment and one-to-one physical boundary matching. The combined new suite passed27tests in6.01s. No confirmatory data were generated or opened.

## Completed anchor capacity/source controls, 2026-10-10

The frozen three-model capacity/source study completed. Noncontextual depth2 selected NLL is 1.0260340316324352, very close to full depth2 1.025646690096779. Static local depth1/depth2 scores are 1.1185405579115626/1.1175823589461271. These are one-seed selected scores on the same four reused validation parents. They supply no independent confirmation and little evidence for query-source feedback itself. See research/reviews/anchor_capacity_pilot_complete_v1.json.

GPU3 was released and assigned the frozen age full-depth1/full-depth2 and matched scalar-age full-depth1/full-depth2 priority block. GPU2 continues reference controls. No other GPU is occupied by the project.

## Fresh development preparation freeze and comparator integration, 2026-10-10

The complete new-tool integration suite passed123tests in12.31s. Independent fresh-context review reproduced a path escape through unrestricted model IDs in the new candidate and temporal runners. Root fixed both isolated modules and added absolute/traversal/control-character rejection, then33focusedtests passed in7.36s. Active jobs and shared preparation sources remain unchanged. The audit found no current CPU-preparation blocker. The temporal pilot remains a pilot, so a separate exact-hash, fixed-grid/seed adapter is being implemented before fresh-development training.

Preparation freeze SHA256 is5e19ff1650397ec657f5d90c341e4a223cf624154736f38f62908e42a0ff7ee1. It declares64TRAIN/16VAL/TEST0 and verifies240 new unique streams against618prior manifests/metadata. Conservative projected storage is102.6GB including reserves. The fixed-source preparation was reverified by root before launch. No final ancestry data or outcomes are available.

## Independent audit resolved and fresh simulation started, 2026-10-10

Fresh-context audit found a second concrete provenance defect: live configuration edits could misbind saved hashes to settings used by the runner. Root changed the two isolated new runners to capture config bytes/digest once, reuse them for protocol/checkpoints/predictions/results, and reject drift before completion. The combined adversarial suite passed36tests in8.31s. The auditor independently reran original reproducers and confirmed both path and config-hash findings resolved. Formal findings/reproducers remain in research/reviews/development_integration_audit_v1. No current sealed preparation blocker was identified.

CPU preparation --stage all was launched with the exact frozen preparation SHA. Oracle development parents are being generated under masters20261101/20261102 and fixed64TRAIN/16VAL/TEST0. The GPU2reference andGPU3age priority studies continue. Confirmatory inputs/labels/predictions are still absent, and no predictive superiority or novel contribution is established.

## Completed age priority and same-anchor temporal pilots, 2026-10-10

The four-model age priority block completed. Age fullD2 selected NLL1.02331655 versus matched scalar-age fullD2 1.02500114 is a small difference on reused fourVALparents/seed101. The ten-model same-anchor comparison also completed: GRUD2 NLL1.03207406, TransformerD2 1.04161990, best fixedreference smoother1.30781556. Neural temporal fitting took62.0/60.0s with28,308/47,700parameters versus the candidate10,420parameters and substantially longer kernel fitting. These are developmental selection scores, no independent confirmation or established novelty.

Fresh input-reachability audit confirms the candidate propagates every interveninggeometry hold, while anchor temporal baselines see source/readout-containinggraphs. CanonicalGRU and earlier BoundaryGRU/BoundaryTransformer process allretained observed trees. RawARGMix fulltree/localattention has no cross-tree aggregation. The untrained synthetic counterfactual verifies information paths only. Its facts remain in research/reviews/canonical_development_audit_v1. Further required all-record coverage controls and close prior-art searches will be frozen before final-test generation.

## Concise hypothesis ledger and competitor status, 2026-10-10

Added research/experiment_decisions.json and a generated human/machine ledger plus scalar result CSV. Eighteen questions record methods, concise positive/negative/partial conclusions, original evidence and explicit repeat reasons. The initial refresh indexes127 reported model results and239 JSON receipts. Old studies retain their original task/selection scope. No cross-study leaderboard is inferred. The updater reads no prediction tensors, data or sealed confirmation banks.

Same-Q64 GST completed at selected NLL1.123058/accuracy49.9237%, and GraphSSM continues. Both ARGMix equation adaptations completed, while whole-tree Transformer continues. Actual NVTOP verified physicalGPU2/PID631921 andGPU3/PID746712 titles. All80 fresh oracle parents are complete, with inferred-parent preparation still running. Native author bridge checks are ARGMix9, ARGformer13 and AncestralPaths37 CPU component/compatibility tests. No native biological training or real-data hypothesis result follows from those checks.

The user reaffirmed concise complete logging and regular GitHub pushes. New substantive runs consult/register the ledger. Unchanged answered pilots are not repeated. Paper results remain selected by scientific relevance, with complete failures/partial runs preserved separately.

## Native TensorFlow runtime and available ARGMix revision, 2026-10-10

An isolated AncestralPaths CPU runtime installed successfully, then passed actual TensorFlow2.15.1 source-native Dense architecture forward (26,630 parameters), one optimizer update on12 handmade random examples and exact .keras checkpoint replay. This is a transparent compatibility adapter, not unchanged author scripts or biological training. Full receipt: experiments/results/ancestralpaths_native_runtime_v1/receipt.json. NumPy1.26.4/tskit0.6.4/msprime1.3.4 are isolated from the main runtime.

The bounded ARGMix revision search found no publicly available April-compatible source in the inspected resources. The user confirmed no older archive. The next native benchmark uses unchanged c248 current-release Transformer+GRU, explicitly distinguished from April v1. Search receipts and absent-data requirements are preserved.

The initial result index contains127 scalar records, including three synthetic admission timing receipts. Its record-kind column now separates scientific performance rows from engineering runtime records. No training experiment was repeated to make this logging correction.

## Official ARGformer base runtime preparation, 2026-10-10

Reviewed and hash-verified the source-only full author-lock candidate, preserving all60 author files and eight prior artifacts. Root launched a separate442-conda/29-PyPI exact base stack installation, with an additional pinned process-title helper. Recipe copies are tracked under configs/official_argformer_native_environment_v1. CPU-only installation captures stage exits and complete logs, and admits30GB storage headroom before bulk writes. FlashAttention compilation, actual native12-layer forward/backward and biological fits remain pending. Scheduler-preserving qualification is being implemented separately, with no change to the existing launcher.
