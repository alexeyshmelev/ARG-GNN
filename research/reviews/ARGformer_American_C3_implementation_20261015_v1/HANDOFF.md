# American C3 reconstruction handoff

The five CLI modules implement observation-only simulation/inference artifacts, private pulse truth, native full-path preparation, native top1 sealing and common external scoring. Configurations here are inactive. Root owns admissions, biological execution, GPU leases and Git. This task ran no simulator, inference, training or prediction. The only numerical audit used four handmade leaves and two sites in memory.

## CLI and activation

Run from /workspace-SR003.nfs2/ARG-GNN after source scripts/project_env.sh. Metadata-only commands create no scientific arrays:

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .runtime/env/bin/python -B -m arggnn.american_c3_simulate_v1 --runtime-report
    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .runtime/envs/biological_tsinfer_v1/bin/python -B -m arggnn.american_c3_observe_v1 --runtime-report
    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .runtime/env/bin/python -B -m arggnn.american_c3_paths_v1 --runtime-report
    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .runtime/env/bin/python -B -m arggnn.american_c3_evaluate_v1 --runtime-report

No flags print inactive defaults and source closure. Every executable stage requires an actual root admission and SHA. Root substitutes these task-specific shell variables:

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .runtime/env/bin/python -B -m arggnn.american_c3_simulate_v1 --execute --admission "$C3_QUAL_ADMISSION" --expected-admission-sha256 "$C3_QUAL_ADMISSION_SHA"
    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .runtime/env/bin/python -B -m arggnn.american_c3_simulate_v1 --execute --admission "$C3_FULL30MB_ADMISSION" --expected-admission-sha256 "$C3_FULL30MB_ADMISSION_SHA"
    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .runtime/envs/biological_tsinfer_v1/bin/python -B -m arggnn.american_c3_observe_v1 --execute --admission "$C3_INFER_ADMISSION" --expected-admission-sha256 "$C3_INFER_ADMISSION_SHA"
    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .runtime/env/bin/python -B -m arggnn.american_c3_paths_v1 --execute --admission "$C3_PATHS_ADMISSION" --expected-admission-sha256 "$C3_PATHS_ADMISSION_SHA"
    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .runtime/env/bin/python -B -m arggnn.american_c3_evaluate_v1 --execute --admission "$C3_SEAL_ADMISSION" --expected-admission-sha256 "$C3_SEAL_ADMISSION_SHA"
    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .runtime/env/bin/python -B -m arggnn.american_c3_evaluate_v1 --execute --admission "$C3_SCORE_ADMISSION" --expected-admission-sha256 "$C3_SCORE_ADMISSION_SHA"

Qualification config simulation_qualification100kb_inactive.json is 100 kb and 64 haplotypes, not the paper benchmark. Full config simulation_full30mb_inactive.json is 30 Mb and 1600 haplotypes. Root creates activated copies with execution_enabled true, distinct parent IDs and registry seeds. Never change an admitted source/config in place.

The common CLI directly requires root_authorizer root_agent, execution_authorized true, GPU_authorized false, job_id, fresh canonical output_directory, exact source_ref/config_ref/runtime_ref/storage_adapter_ref/marker_repair_authority_ref, ordered source_closure_refs and the existing capacity-v2 storage record. Output parent must already exist. Runtime metadata must have exact interpreter prefix and CUDA_VISIBLE_DEVICES empty. Supplied package versions are rechecked. Reserve every actual common/producer/helper/config/runtime/adapter ref plus existing capacity-v2 marker/repair/historical/launcher requirements.

The common CLI calls existing configure_v2 and validate_storage_v9, preserving existing resource authority. Explicit writes check the existing guard before the planned write and after serialization. Partial failed outputs are preserved. No overwrite, implicit truncation or reduced fallback corpus is allowed.

## Scientific interfaces

Simulation resolves the catalog's named AFR/EUR/ASIA/ADMIX populations and overrides mutation/recombination rates to 1.25e-8/1e-8. It records and traces actual 12-generation migrations across every true-tree and pulse boundary. Labels are 0 EUR, 1 EAS, 2 AFR. No majority or nearest-label fallback exists.

Private producer artifacts are truth/simulated_true_with_migrations.trees and truth/c3_pulse_ancestry.npz. The truth schema is intervals_bp, query_sample_indices, source_classes_0based, parent_id and class_order. Original genotype-column index plus parent is stable across true/inferred namespaces. Raw node IDs are namespace specific. Producer completion and truth refs are not model inputs.

Observed genotypes, positions, allele strings, roster and manifest contain no query ancestry truth. Known source-reference classes are intentional training labels. The sanitized tree strips migrations, populations, individuals, provenances and global/node/edge/site/mutation metadata. Simulation topology is marked ORACLE, distinct from published inferred genealogy.

Inference consumes only observed genotypes/coordinates/alleles/roster. The dedicated existing runtime has tsinfer 0.5.1 and tsdate 0.2.7; these are absent from the shared simulation runtime. SampleData max_file_size bounds the LMDB store. All sites and samples must be retained. Verification compares allele strings, allowing valid inferred allele-code reordering. New inferred node IDs retain original sample indices.

Current tsinfer.infer and tsdate.date defaults are disclosed reconstruction choices. Installed tsdate defaults to variational_gamma with allow_unary false. There is no silent preprocessing, trimming or alternate-option fallback if dating rejects an input.

## Native full paths and retrieval

The materializer SHA-verifies native prepare_data_pretrain.py and compiles three unchanged author function AST definitions. It does not execute author hardcoded top-level I/O. Every sample and tree is processed in native order. Deduplication uses complete leaf-to-root tuples before native 128-token model truncation. Native vocabulary and seed-42 90/10 splitting are unchanged.

MLM sees every observed path. Contrastive views contain only known reference-source paths with native list-format labels.pkl. All views use the full observed vocabulary. Query embedding views contain exactly the prespecified 10000 unique admixed paths. Focals are chosen without truth by an independent seed, with first-occurrence midpoint coordinate explicitly marked as a reconstruction assumption.

Native raw_{train,val}_sequences_and_vocab.pkl files have row_order_metadata.npz sidecars carrying full-path IDs, sample node IDs, original sample indices, parent and classes. Reference metadata includes observed reference classes.

native_commands_inactive.json links audited native child argv and unchanged-numerics I/O-only YAML drafts. Root binds actual per-parent inputs/checkpoints/outputs through its admitted GPU launcher. Preserve include_splits train,val, pooling head_norm and max_samples 0. Corpus alone uses use_split_labels. Retrieval preserves native topk 1. Release temperature 0.30 versus paper default 0.05 is disclosed.

Evaluator native_top1_seal accepts exact preparation/focal refs, native corpus/query embeddings and native retrieve.py results, with no private truth. It validates complete row counts, splits, pooling, extant IDs, labels and native top1 indices/scores. Complete raw-path sidecars establish identity. Native scores/indices are saved unchanged, then retrieved reference labels become one-hot C3 calls. It implements no replacement retrieval algorithm.

## Common scores and GNN bridge

Sealed prediction NPZ exact fields: record_ids int64[N], sample_nodes int64[N], sample_indices int64[N], positions_bp0 float64[N], parent_id scalar unicode, class_order unicode[3], probabilities float32[N,3]. Identities/coordinates/order must exactly equal the registry. Probabilities stay raw and finite, nonnegative and normalized within serialization tolerance. No floor, calibration or renormalization. Native one-hot calls use the same schema.

Score mode validates every prediction before opening external private truth. It joins original sample index and parent using half-open intervals, then emits class precision/recall/F1, macro-F1, accuracy and 3x3 confusion with zero-division 0. A SNP registry can use the same identity schema under explicitly registered SNP support.

gnn_bridge_contract_inactive.json is a requirements artifact, not implemented/trained GNN output. It requires fresh C3 initialization, 3 emissions, 3x3 transitions and 6 directed off-diagonal states. Preserve message-passing, mutation-support and temporal equations. European C4 checkpoint relabeling is prohibited. Root must implement/pin actual GNN sources/configs and use the same observation/focal interface and parent splits.

## Storage, audit and limits

storage_and_runtime_bounds.json supplies count formulas, producer caps and RAM planning estimates. Full simulation bound 5896320256 bytes under configured counts, cap 6500000000. Path preparation bound 13904435456, cap 16000000000. Exceeding counts stops the stage, without changing the experiment. RAM figures are estimates, not measured/enforced limits. Complete native reference embeddings can reach about 54 GB at the maximum path count; author code stores input tensors and builds lists before concatenation. Vocabulary-dependent model/optimizer checkpoints need separate root admission.

parent_seed_registry_inactive.json separates 20 TRAIN, 5 VAL and 10 TEST independent simulation parents before any path/window sampling. It is not a 35-parent bulk storage reservation. Existing 1 TB total allowance and 950 GB scheduling threshold remain authoritative.

handmade_cpu_audit_receipt.json records the single permitted handmade audit. All five modules compile. No biological inference/native output was executed or claimed here. source_closure_receipt.json freezes actual implementation/config refs.

Exact Table1 numerical reproduction remains unestablished: original sample units, seeds, mutation-model details, exact inference versions/options, focal coordinate rule and PCA classifier are unresolved. The 1600 sample counts explicitly mean haploid in this reconstruction. No new scientific result, score or production timing is claimed.
