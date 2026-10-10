"""Inactive native ARGformer bridge for observed tsinfer/tsdate development data.

All public CLI modes are source/JSON metadata only. Native functions are compiled
from pinned source into an isolated namespace, never substituted with toy models.
Root activation, representation/corpus qualification and resource admission are
separate prerequisites. No query ancestry target enters any training route.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
CONFIG=ROOT/"configs/official_argformer_tsinfer_biological_v2.json"
SOURCE=ROOT/"src/arggnn/official_argformer_tsinfer_biological_v2.py"
FIT="src/arggnn/train_official_argformer_biological.py"
RETRIEVAL="scripts/score_official_argformer_retrieval_biological_v1.py"
NATIVE_FUNCTIONS=("sha","canonical","json_new","observed_path","load_frozen",
 "prepared_inputs","cache_loss","cache_adapter","object_tensor_manifest",
 "native_runtime","source_model","assert_backbone","audit_loader","audit_callback",
 "train_phase","memory_or_parity","readout","verify_encoder_initializer",
 "execute_native","bootstrap")
RETRIEVAL_FUNCTIONS=("corpus_roles","validate_support","lexicographic_topk",
 "exhaustive_neighbors","hard_votes","project_grid")
FIXED_CONFIG={'kind': 'official_argformer_tsinfer_biological_v2', 'status': 'inactive_source_only_template_root_qualification_and_scheduling_pending', 'execution_enabled': False, 'condition': 'native_ARGformer_current_tsinfer_tsdate_observed_heterochronous_development_transfer_v2', 'observation_condition': 'current_tsinfer_tsdate_observed_heterochronous_task_transfer_v2', 'native_helper_condition': 'official_documented_heterochronous_tsinfer_tsdate_task_transfer_v1', 'manifest_contract': {'path': 'data/biological_paths_execution_pilot_v1/tsinfer_tsdate_observed_v2/combined_observed_manifest.json', 'kind': 'biological_paths_tsinfer_observed_v2', 'status': 'four_observed_development_parents_inferred1000', 'rescaling_intervals': 1000, 'phase_paths': {'canary': 'canary/manifest.json', 'remaining': 'remaining/manifest.json'}, 'wrapper_rows_preserved': True, 'sha256': 'cbcabd3346802ad87a5b4aa0ea584b1ace8cfcf5ec06a04da4c827f51c3fb44c'}, 'parent_ids': ['bio_paths_pilot_train_000', 'bio_paths_pilot_train_001', 'bio_paths_pilot_val_000', 'bio_paths_pilot_val_001'], 'phase_parents': {'canary': ['bio_paths_pilot_train_000'], 'remaining': ['bio_paths_pilot_train_001', 'bio_paths_pilot_val_000', 'bio_paths_pilot_val_001']}, 'class_order': ['ANA', 'CHG', 'WHG', 'EHG'], 'pure_reference_haplotypes': 96, 'mixed_reference_haplotypes': 64, 'query_haplotypes': 120, 'total_observed_haplotypes': 280, 'source_commit': 'f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596', 'source_unchanged': True, 'sequence_length': 128, 'skip_extant_tokens': True, 'path_split_train_fraction': 0.9, 'path_split_seed': 42, 'pretraining': {'epochs': 5, 'seed': 17, 'logical_batch': 512, 'microbatch': 64, 'train_masking_probability': 0.3, 'validation_masking_probability': 0.15, 'precision': 'amp_bf16', 'optimizer_and_scheduler': 'released_mlm.yaml unchanged'}, 'contrastive': {'epochs': 1, 'seed': 36, 'logical_batch': 4096, 'microbatch': 4096, 'default_execution': 'full_batch', 'gradient_cache_chunk': 64, 'precision': 'fp32', 'temperature': 0.3, 'phantom_classifier_num_labels': 3, 'classifier_used_for_ancestry': False, 'reference_classes_in_loss': 4, 'class_weights': 'released uniform fallback', 'optimizer_and_scheduler': 'released_contrastive.yaml unchanged'}, 'gradient_cache_parity': {'real_paths': 16, 'chunk': 4, 'evaluation_mode_dropout_disabled': True, 'loss_absolute_tolerance': 1e-05, 'global_gradient_relative_l2_tolerance': 0.01, 'required_before_cache_fit': True}, 'readout': {'kind': 'explicit_downstream_C4_local_class_mean_exp_cosine', 'temperature': 0.3, 'embedding_batch': 128, 'coordinate_chunk': 32, 'fixed_grid_bin_bp': 100, 'center_formula_bp0': '50 + 100*grid_index', 'local_position': 'original_bp0 - local_to_original_offset_bp', 'use_only_declared_retained_fixed100bp_indexes': True, 'tail_extrapolation': False, 'calibration_fitted': False, 'calibration_status': 'pending_external_biological_VAL'}, 'fitting_regime': 'within_ARG_transductive; independent vocabulary/checkpoints per observed ARG', 'query_ancestry_targets_read': False, 'primary_readout': 'full_reference_corpus_cosine_top1_hard_votes', 'sensitivity_k': 5, 'retrieval': {'BLAS_threads': 4, 'GPU': False, 'NLL': 'raw_hard_vote_zero_retained_and_positive_truth_mass_at_zero_is_infinite', 'adapted_biological_task': True, 'calibration': False, 'class_balance': False, 'class_vote_ties': 'numpy_argmax_first_in_fixed_C4_order', 'corpus': 'every_unique_full_path_of_pure96_in_same_ARG_across_native_path_train_and_val', 'corpus_chunk': 8192, 'embedding_dim': 768, 'identical_embedding_rows': 'retained_if_full_path_ID_differs', 'max_exhaustive_pairs_per_parent': 40000000000, 'metric': 'dot_of_source_F_normalize_float32_vectors_eps1e-12', 'neighbor_count_saved': 6, 'neighbor_ties': 'exact_computed_score_ties_then_ascending_original_full_path_ID', 'normalization_epsilon': 1e-12, 'paper_task_reproduction': False, 'paper_top5_scope': 'explicit_archaic_and_Oceania_real_population_retrieval', 'path_multiplicity': 'one_per_full_path_ID_no_genomic_span_weighting', 'primary_endpoints': ['accuracy', 'macro_f1'], 'primary_k': 1, 'probability_floor': False, 'published_corpus_curation_and_tie_order': 'not_fully_resolved', 'published_simulated_LAI_k': 'unresolved_main_text_unspecified_source_default1', 'query_batch': 128, 'query_paths': 'unique_full_path_ids_needed_by_all_common_retained_query120_centers', 'score_dtype': 'float32', 'sensitivity_k': 5, 'span_weighting': False, 'storage_headroom_bytes': 2000000000, 'target_based_eligibility': False, 'torch_interop_threads': 1, 'torch_threads': 4, 'training': False, 'votes': 'unweighted_raw_neighbor_class_counts_divided_by_k'}, 'normalization_length_bp': 2000000, 'fixed_grid_bin_bp': 100, 'output_scopes': ['data/official_argformer_tsinfer_biological_v2', 'experiments/results/official_argformer_tsinfer_biological_v2', '.runtime/tmp/afbio_v2'], 'native_runtime_prefix': '.runtime/conda/envs/argformer_native_v1', 'storage': {'fresh_root_reservation_and_projection_required': True, 'existing_observed20GB_slot_not_borrowed': True, 'suggested_new_bytes': 20000000000, 'combined_all_stage_checkpoints_embeddings_prepared_cache_tmp_logs': True, 'stage_boundary_cap_not_OS_quota': True}, 'node_identity': {'preserve_native_ids_and_times': True, 'simplify_or_relabel': False, 'clade_or_local_number_identity_invention': False, 'namespace': 'parent_id + colon + observed_tree_SHA256', 'cross_ARG_checkpoint_transfer': False}, 'data_fidelity': {'current_dataset': 'AncientEurope four_source development family', 'current_parent_length_bp': 2000000, 'current_diploids': 140, 'paper_task': 'AmericanAdmixture_4B18', 'paper_samples': 1600, 'paper_parent_length_bp': 30000000, 'paper_classes': ['European', 'African', 'East_Asian'], 'original_paper_archives_bound': False, 'paper_task_numerical_reproduction': False, 'paper_original_data_task_preprocess_score_comparison_pending': True, 'released_YAML_is_not_proof_of_all_paper_budgets': True}, 'required_root_stages': ['metadata_binding', 'CPU_path_preparation_and_representation_qualification', 'exact_corpus_counts_and_budget_admission', 'fresh_GPU_memory_and_runtime_qualification', 'native_MLM5_then_contrastive1_fit', 'all_query_full_reference_retrieval_seal_before_external_truth', 'separate_original_paper_task_data_preprocess_score_contract'], 'frozen_files_sha256': {'src/arggnn/train_official_argformer_biological.py': 'edc290b9fbce093f4b844f7d07c4963c23bf24257f43c7d502c43a2753170b94', 'src/arggnn/official_argformer_bridge.py': 'd6d81a078cff1bb132bc5040a62db7e728c439472b8093eba41b24a4c1edab80', 'src/arggnn/native_argformer_exec.py': '84ed405379ac56511c2f4722c130e8b884226018bbf8e2949ea2bb4dceeb99d0', 'scripts/qualify_official_argformer_runtime.py': '8fe652f87848f4a296e3d88555998fb388f11ae5c04d0858095e70c34b10b31c', 'scripts/score_official_argformer_retrieval_biological_v1.py': '2f4fc18efa5e83e7c1c3c72ce01cd3783fa03c83dc62225a52b99733c0d56f37', 'scripts/score_official_argformer_biological_v1.py': '82e031a9b22bf469ed044000c87a345537c1255ee398705137fa47f6d72b68e4', 'configs/official_argformer_biological_pilot_v1.json': '84eaf8b6638f01f37b6519193b8e9d85880880123cdede002a17de80129f098e', 'configs/official_argformer_retrieval_biological_v1.json': '1a96eafabea2eafa2363005b2d96ae7915eb4394f029b7d24ff917efbf6398d7', 'configs/official_argformer_biological_readout_v1.json': '0758e9f2cdcf738d75ddcafdeb03a87d3be86f92836075f30cd2e55937efc96d', 'configs/biological_paths_tsinfer_observed_v2.json': '450a502066b74231b81f364ce47f4f927c1c28bf22be20f6d7a9abf945bd0953', 'scripts/infer_biological_paths_tsinfer_observed_v2.py': '9904dbbed05d38d2e1c3e03c7bc07f84376df309fff8fe4852ddeec229aac7b8', 'scripts/infer_biological_paths_tsinfer_v1.py': 'b1f1c9a50b77daa9a416e7fed04d06a1bbbc247644d63103b11a2fa80a4fb2e0', 'configs/biological_paths_tsinfer_v1.json': '85d8dc47f771826f3d23c42c55b847a2527f124540def1c91e5f51b200121e6c', 'src/arggnn/gpu_pool_v1.py': 'ae1e0494ab672cfa3017759460eb06a4b4789eaef54928496c3c80314adcc0e1', 'src/arggnn/resources.py': 'ffc7ac9ce1ebb19b6d7030b9db643d8de9a07179318a9bb1e593779fd3f6d111', '.runtime/author_sources/argformer_f7545be7bca0/yamls/mlm.yaml': 'df1cbb747a1bb6740fffd1e6eee5eb0ac3a1fa43d8fbbb15ce70d6b9cc7799b6', '.runtime/author_sources/argformer_f7545be7bca0/yamls/contrastive.yaml': '3b4f87e149a2b56d8a3efa2e35bd52ec94b61d2d3c768f15df67e82376ce4847', '.runtime/author_sources/argformer_f7545be7bca0/retrieve.py': '520f1961c396432a875a40be253a01648e7debf1fad8c9a274a7522d3809e7ea', 'data/literature/argformer_btag438_source_check_v1/body.txt': 'ca8b72e780bc885f8c53cc9d13bf7d147c0588742be2f1b31cebbf241640251b', 'data/literature/argformer_btag438_source_check_v1/source_receipt.json': '7bec3f9dde5848630ee2fddf8a4412daeec5d2ba52c88981e1e9f6d94ed51f87'}, 'observation_provenance': {'producer': {'path': 'scripts/infer_biological_paths_tsinfer_v1.py', 'sha256': 'b1f1c9a50b77daa9a416e7fed04d06a1bbbc247644d63103b11a2fa80a4fb2e0'}, 'biological_config': {'path': 'configs/biological_paths_tsinfer_v1.json', 'sha256': '85d8dc47f771826f3d23c42c55b847a2527f124540def1c91e5f51b200121e6c'}, 'runtime': {'path': 'experiments/results/biological_tsinfer_runtime_install_v1/runtime_receipt.json', 'sha256': '04876226b6f90fc803741b4257f17b9595374c58e1681e4c04b7d6d9ba4e6e8f'}, 'observed_asset_pins': {'path': 'research/reviews/biological_paths_tsinfer_v1_source_audit/observed_asset_pins_from_existing_relate_manifest.json', 'sha256': 'e7781b2a5b6ee9630d12db0edf9401fdba5ac3a9fe9099b69cd444e1c11c492d'}}, 'author_source': {'root': '.runtime/author_sources/argformer_f7545be7bca0', 'commit': 'f7545be7bca0200fb39aeb7f94a8e5f0b6bf5596', 'archive_sha256': '44c1b76f7806dfc3063d707eae1950601eb4caa3ebe2fa20277bb98e77f1890c', 'tree_sha256': '2295d86ace359ac1a7d0522c1001b6f2822b30cf9b16200517ea3473f234ca5b', 'files': 60}}

def exact(value,expected):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)==json.dumps(expected,sort_keys=True,separators=(",",":"),allow_nan=False)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical_path(value):
    p=Path(value)
    if ".." in p.parts or any("truth" in x.lower() or "oracle" in x.lower() for x in p.parts):
        raise ValueError("observed-only canonical paths required")
    p=p if p.is_absolute() else ROOT/p
    if p==ROOT or ROOT not in p.parents or p.resolve()!=p:
        raise ValueError("canonical confined project paths required")
    return p

def reference_shape(reference,expected=None):
    if not isinstance(reference,dict) or set(reference)!={"path","sha256"}:
        raise ValueError("exact path/SHA256 reference required")
    h=reference["sha256"]
    if not isinstance(h,str) or len(h)!=64 or any(x not in "0123456789abcdef" for x in h):
        raise ValueError("lowercase SHA256 required")
    p=canonical_path(reference["path"])
    if expected is not None:
        expected=canonical_path(expected)
        if reference["path"] not in {str(expected),str(expected.relative_to(ROOT))} or p!=expected:
            raise ValueError("expected parent/phase artifact reference required")
    return p

def json_metadata(reference,expected=None):
    p=reference_shape(reference,expected)
    if p.suffix!=".json" or not p.is_file() or p.is_symlink():
        raise ValueError("regular JSON metadata only; no array payload reads")
    if digest(p)!=reference["sha256"]:
        raise ValueError("metadata changed")
    return json.loads(p.read_text())

def policy(path=CONFIG):
    p=canonical_path(path)
    if p!=CONFIG:
        raise ValueError("fixed bridge config required")
    c=json.loads(p.read_text())
    if not exact(c,FIXED_CONFIG):
        raise ValueError("exact inactive bridge policy required")
    return c

def verify_author_source(c):
    authority=c["author_source"];root=canonical_path(authority["root"])
    receipt=json.loads((root/"ARG_GNN_SOURCE_RECEIPT.json").read_text())
    if receipt.get("commit")!=authority["commit"] or receipt.get("archive_sha256")!=authority["archive_sha256"]:
        raise ValueError("pinned author archive authority differs")
    records=[]
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise ValueError("author source symlinks forbidden")
        if p.is_file() and p.name!="ARG_GNN_SOURCE_RECEIPT.json":
            records.append({"path":p.relative_to(root).as_posix(),"sha256":digest(p)})
            if len(records)>authority["files"]:
                raise ValueError("author source inventory expanded")
    actual=hashlib.sha256(json.dumps(records,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if len(records)!=authority["files"] or actual!=authority["tree_sha256"]:
        raise ValueError("unchanged pinned author source inventory required")
    return {"commit":authority["commit"],"tree_sha256":actual,"files":len(records),"native_imported":False}

def verify_source(c):
    results={}
    for relative,expected in c["frozen_files_sha256"].items():
        p=canonical_path(relative)
        if p.is_symlink() or digest(p)!=expected:
            raise ValueError("frozen authority changed: "+relative)
        results[relative]=expected
    verify_author_source(c)
    return results

def function_nodes(relative,c,names):
    p=canonical_path(relative)
    if digest(p)!=c["frozen_files_sha256"][relative]:
        raise ValueError("pinned native function source changed")
    source=p.read_text();nodes={n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    return {name:copy.deepcopy(nodes[name]) for name in names}

def function_proof(c):
    out={}
    for relative,names in [(FIT,NATIVE_FUNCTIONS),(RETRIEVAL,RETRIEVAL_FUNCTIONS)]:
        out[relative]={name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
                       for name,n in function_nodes(relative,c,names).items()}
    return out

def native_capsule(c):
    """Compile genuine pinned definitions without importing native dependencies.

    This isolated namespace is inactive and is not a scheduler admission. An
    activation revision must attach stage authority before invoking side effects.
    Original module/config/globals are never loaded, rebound or mutated.
    """
    verify_source(c)
    fit=ast.parse(canonical_path(FIT).read_text())
    names=function_nodes(FIT,c,NATIVE_FUNCTIONS)
    imports=[copy.deepcopy(n) for n in fit.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    assignments=[copy.deepcopy(n) for n in fit.body if isinstance(n,ast.Assign)]
    tree=ast.Module(body=imports+assignments+list(names.values()),type_ignores=[])
    ast.fix_missing_locations(tree)
    ns={"__name__":"_inactive_tsinfer_argformer_native_capsule","__file__":str(SOURCE)}
    exec(compile(tree,str(canonical_path(FIT)),"exec"),ns)
    ns.update(ROOT=ROOT,CONFIG=CONFIG,SCRIPT=SOURCE)
    ns["policy"]=lambda expected_sha=None: execution_policy(c,expected_sha)
    pure={"sha","canonical","json_new","observed_path","load_frozen"}
    for name in set(NATIVE_FUNCTIONS)-pure:
        original=ns[name]
        def inactive(*args,_original=original,**kwargs):
            execution_policy(c,None)
            return _original(*args,**kwargs)
        ns[name]=inactive
    return types.MappingProxyType(ns)

def execution_policy(c,expected_sha=None):
    if not exact(c,FIXED_CONFIG) or c["execution_enabled"] is not True:
        raise RuntimeError("inactive source-only bridge; fresh root activation/qualification/scheduling required")
    if expected_sha!=digest(CONFIG):
        raise ValueError("root-reviewed activated config SHA256 required")
    return c

def phase_for(c,parent):
    found=[phase for phase,parents in c["phase_parents"].items() if parent in parents]
    if len(found)!=1:
        raise ValueError("exact configured parent/phase required")
    return found[0]

def validate_roster(c,rows,source_rows,mapping):
    if not isinstance(rows,list) or not isinstance(source_rows,list) or len(rows)!=280 or len(source_rows)!=280:
        raise ValueError("all280 source and inferred roster rows required")
    if len(mapping)!=280:
        raise ValueError("all280 explicit sample mappings required")
    ids=[]
    for index,(row,original,j) in enumerate(zip(rows,source_rows,mapping)):
        source_keys={"sample_index","sample_id","haplotype","observed_cohort","age_generations","role","observed_reference_class","observed_node_id"}
        if set(original)!=source_keys or set(j)!={"sample_index","inferred_tree_node_id"}:
            raise ValueError("exact observed-only source roster and mapping schema required")
        if any(type(x.get("sample_index")) is not int for x in [row,original,j]) or type(j["inferred_tree_node_id"]) is not int:
            raise ValueError("typed explicit native sample mapping required")
        if row.get("sample_index")!=index or original.get("sample_index")!=index or j.get("sample_index")!=index:
            raise ValueError("ordered sample indexes required")
        node=row.get("inferred_tree_node_id")
        if type(node) is not int or node<0 or j.get("inferred_tree_node_id")!=node or row.get("observed_node_id")!=node:
            raise ValueError("typed native sample IDs required")
        if row.get("source_observed_node_id")!=original.get("observed_node_id"):
            raise ValueError("source-to-native observed node lineage differs")
        expected=copy.deepcopy(original)
        expected["source_observed_node_id"]=expected.pop("observed_node_id")
        expected["observed_node_id"]=expected["inferred_tree_node_id"]=node
        if not exact(row,expected):
            raise ValueError("exact complete source/native roster lineage required")
        if (type(original.get("observed_node_id")) is not int or original["observed_node_id"]<0
            or not isinstance(row.get("sample_id"),str) or not row["sample_id"]
            or type(row.get("haplotype")) is not int or row["haplotype"] not in (0,1)):
            raise ValueError("explicit sample/haplotype/source IDs required")
        for key in ("sample_id","observed_cohort","haplotype","role","age_generations","observed_reference_class"):
            if row.get(key)!=original.get(key):
                raise ValueError("source/native roster lineage differs: "+key)
        age=row.get("age_generations")
        if isinstance(age,bool) or not isinstance(age,(int,float)) or not math.isfinite(age) or age<0:
            raise ValueError("actual finite sample ages required")
        role=row.get("role");label=row.get("observed_reference_class")
        if role=="pure_source_reference":
            if type(label) is not int or label not in range(4):
                raise ValueError("only pure reference C4 labels allowed")
        elif role not in {"query","mixed_reference_only"} or label is not None:
            raise ValueError("query/mixed labels cannot supervise native model")
        forbidden={"query_ancestry","true_ancestry","census_label","target","oracle"}
        if forbidden & set(row) or forbidden & set(original):
            raise ValueError("query ancestry/truth fields forbidden in preparation roster")
        ids.append(node)
    if len(set(ids))!=280:
        raise ValueError("unique native sample IDs required")
    counts={role:sum(r["role"]==role for r in rows) for role in ["pure_source_reference","mixed_reference_only","query"]}
    if counts!={"pure_source_reference":96,"mixed_reference_only":64,"query":120}:
        raise ValueError("exact96/64/120 observed roles required")
    if [sum(r["role"]=="pure_source_reference" and r["observed_reference_class"]==k for r in rows) for k in range(4)]!=[24]*4:
        raise ValueError("exact24 observed pure reference haplotypes per C4 class")
    return counts

def native_metadata(c,row,rows,source_rows):
    parent=row.get("parent_id");phase=phase_for(c,parent)
    native=row.get("frozen_helper_result")
    if (row.get("condition")!=c["observation_condition"] or row.get("source_helper_condition")!=c["native_helper_condition"]
        or not exact(row.get("rescaling_intervals"),1000) or row.get("native_paper_reproduction_claimed") is not False
        or not isinstance(native,dict) or native.get("parent_id")!=parent or native.get("condition")!=c["native_helper_condition"]
        or native.get("inferred_tree",{}).get("condition")!=c["native_helper_condition"]
        or native.get("status")!="qualified_observed_cpu_inference" or native.get("truth_or_oracle_read") is not False
        or native.get("no_input_variant_filtering") is not True or native.get("native_node_ids_and_times_preserved_during_sanitation") is not True):
        raise ValueError("exact observed wrapper/native parent and conditions required")
    work=Path(c["manifest_contract"]["path"]).parent/phase/parent
    for key,name in [("completion_provenance","inference.json"),("inferred_tree","observed.trees"),("observed_sample_manifest","observed_sample_manifest.json")]:
        reference_shape({k:native[key][k] for k in ["path","sha256"]},work/name)
    reference_shape(row["task_transfer_provenance"],work/"task_transfer_v2.json")
    original=native["original_observed_sample_manifest"]
    reference_shape(original,Path("data/biological_paths_execution_pilot_v1")/parent/"observed/sample_manifest.json")
    if native["input_observed_sha256"].get("sample_manifest.json")!=original["sha256"]:
        raise ValueError("source roster hash lineage differs")
    coordinate=native["coordinate_contract"]
    left,right=coordinate["coverage_original_bp0"]
    if (coordinate["original_parent_interval_bp0"]!=[0,2000000] or coordinate["normalization_length_bp"]!=2000000
        or isinstance(left,bool) or isinstance(right,bool) or not 0<=left<right<=2000000
        or coordinate["local_to_original_offset_bp"]!=left
        or coordinate["fixed_grid"]["total"]!=20000
        or coordinate["fixed_grid"]["center_formula_bp0"]!="50 + 100*grid_index"):
        raise ValueError("exact original physical coordinate/support contract required")
    expected=[i for i in range(20000) if left<=50+100*i<right]
    if not exact(coordinate["retained_fixed100bp_indexes"],expected) or not expected:
        raise ValueError("entire retained common support required")
    reference_shape({k:coordinate["fixed_grid"][k] for k in ["path","sha256"]})
    identity=native.get("connected_identity",{})
    if (type(identity.get("internal_reuse_occurrences")) is not int or identity["internal_reuse_occurrences"]<1
        or identity.get("uncovered_interval_inside_support_admitted") is not False):
        raise ValueError("connected shared native identity diagnostic required")
    adequacy=row["adequacy"]
    expected_directory=str(work)
    if adequacy.get("expected_parent")!=parent or adequacy.get("phase")!=phase or adequacy.get("native_parent_directory")!=expected_directory:
        raise ValueError("root wrapper parent/phase adequacy binding differs")
    for key in ["source_input_hashes_exact","original_genotypes_and_sample_ages_exact","strict_native_times_and_mutation_branches","entire_common_support_connected","native_ids_preserved_by_sanitation"]:
        if adequacy.get(key) is not True:
            raise ValueError("native adequacy metadata incomplete: "+key)
    if native["observed_sample_manifest"].get("rows")!=280:
        raise ValueError("exact280 observed roster count")
    for key,name in [("native_verification","native_verification.json"),("stage_journal","stages.json")]:
        reference_shape(adequacy[key],work/name)
    validate_roster(c,rows,source_rows,native["sample_index_to_inferred_tree_node_id"])
    return {"parent_id":parent,"phase":phase,"native":copy.deepcopy(native),
            "namespace":parent+":"+native["inferred_tree"]["sha256"],"roster_sha256":native["observed_sample_manifest"]["sha256"],
            "source_roster":copy.deepcopy(original),"coverage_original_bp0":[left,right],
            "retained_center_count":len(expected),"native_ID_preservation":True,
            "metadata_only_actual_tree_and_corpus_unqualified":True}

def validate_combined(c,manifest,rosters,source_rosters):
    contract=c["manifest_contract"]
    if (manifest.get("kind")!=contract["kind"] or manifest.get("status")!=contract["status"]
        or manifest.get("condition")!=c["observation_condition"]
        or not exact(manifest.get("rescaling_intervals"),1000)
        or manifest.get("truth_or_oracle_read") is not False or manifest.get("native_paper_reproduction_claimed") is not False
        or not exact(manifest.get("scope_parents"),c["parent_ids"])):
        raise ValueError("completed four-parent observed1000 combined metadata required")
    rows=manifest.get("replicates",[])
    if len(rows)!=4 or [r.get("parent_id") for r in rows]!=c["parent_ids"]:
        raise ValueError("four ordered distinct observed parent wrappers required")
    for field,source in [("implementation","scripts/infer_biological_paths_tsinfer_observed_v2.py"),("config","configs/biological_paths_tsinfer_observed_v2.json")]:
        reference_shape(manifest[field],source)
        if manifest[field]["sha256"]!=c["frozen_files_sha256"][source]:
            raise ValueError("combined observation source/config pin differs")
    for field,expected in c["observation_provenance"].items():
        if not exact(manifest.get(field),expected):
            raise ValueError("exact observed producer/runtime/input provenance required: "+field)
    phases=manifest.get("phase_manifests",{})
    if set(phases)!={"canary","remaining"}:
        raise ValueError("exact two phase manifests required")
    for phase,relative in contract["phase_paths"].items():
        reference_shape(phases[phase],Path(contract["path"]).parent/relative)
    return {row["parent_id"]:native_metadata(c,row,rosters[row["parent_id"]],source_rosters[row["parent_id"]]) for row in rows}

def preparation_entry(record):
    """Mechanical tsinfer source-roster field join. Original helper stays intact."""
    entry=copy.deepcopy(record["native"])
    entry["observed_sample_manifest"]["source_manifest"]=copy.deepcopy(record["source_roster"])
    return entry

def preparation_core(c):
    """Keep legacy extraction/ID vocabulary/path split/serialization code exact.

    The independent caller validates tsinfer metadata and supplies the explicit
    source-roster field join. The Relate admission prefix is intentionally omitted.
    This callable remains inaccessible from the inactive public CLI.
    """
    node=function_nodes(FIT,c,["prepare_paths"])["prepare_paths"]
    start=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="tree_path" for t in n.targets))
    body=copy.deepcopy(node.body[start:])
    args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=n) for n in ["entry","parent_id","output","expected_manifest_sha","config"]],vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[])
    core=ast.FunctionDef(name="_prepare_native_paths_core",args=args,body=body,decorator_list=[],returns=None,type_comment=None)
    module=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),core],type_ignores=[])
    ast.fix_missing_locations(module)
    capsule=dict(native_capsule(c))
    exec(compile(module,str(canonical_path(FIT)),"exec"),capsule)
    core=capsule["_prepare_native_paths_core"]
    def inactive(*args,**kwargs):
        execution_policy(c,None)
        return core(*args,**kwargs)
    return inactive,hashlib.sha256(ast.dump(ast.Module(body=body,type_ignores=[]),include_attributes=False).encode()).hexdigest()

def retrieval_core(c):
    """Unchanged reviewed full-corpus cosine/top-k/vote/project kernels."""
    nodes=function_nodes(RETRIEVAL,c,RETRIEVAL_FUNCTIONS)
    module=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),*nodes.values()],type_ignores=[])
    ast.fix_missing_locations(module)
    import time
    ns={"math":math,"time":time}
    exec(compile(module,str(canonical_path(RETRIEVAL)),"exec"),ns)
    return types.MappingProxyType(ns)

def corpus_budget(c,counts):
    required={"unique_full_paths","MLM_train","MLM_val","pure_reference_train","pure_reference_val","query_full_paths_scored"}
    if not isinstance(counts,dict) or set(counts)!=required or any(type(x) is not int or x<0 for x in counts.values()):
        raise ValueError("exact typed actual corpus counts required")
    if (counts["MLM_train"]+counts["MLM_val"]!=counts["unique_full_paths"]
        or counts["pure_reference_train"]+counts["pure_reference_val"]>counts["unique_full_paths"]
        or counts["MLM_train"]<512 or counts["MLM_val"]<1 or counts["pure_reference_train"]<4096
        or counts["pure_reference_val"]<1 or counts["query_full_paths_scored"]<1):
        raise ValueError("native source batch/split insufficient; no budget reduction")
    refs=counts["pure_reference_train"]+counts["pure_reference_val"]
    pairs=refs*counts["query_full_paths_scored"]
    if pairs>c["retrieval"]["max_exhaustive_pairs_per_parent"]:
        raise ValueError("full-corpus cost needs fresh explicit admission; never subsample baseline")
    return {"actual_counts":copy.deepcopy(counts),"MLM_optimizer_updates":5*(counts["MLM_train"]//512),
            "contrastive_optimizer_updates":counts["pure_reference_train"]//4096,
            "pure_reference_corpus_paths_all_train_plus_val":refs,
            "exhaustive_query_corpus_pairs":pairs,"MLM_epochs":5,"contrastive_epochs":1,
            "MLM_logical_batch":512,"MLM_microbatch":64,"contrastive_full_logical_and_microbatch":4096,
            "reference_class_balance_or_span_weighting":False,"query_truth_trains_any_model":False}

def metadata_report(c):
    contract=c["manifest_contract"]
    manifest=json_metadata({k:contract[k] for k in ["path","sha256"]},contract["path"])
    phases={}
    for phase,reference in manifest["phase_manifests"].items():
        phases[phase]=json_metadata(reference,Path(contract["path"]).parent/contract["phase_paths"][phase])
    rosters={};sources={}
    for row in manifest["replicates"]:
        parent=row["parent_id"];phase=phase_for(c,parent)
        candidates=[r for r in phases[phase]["replicates"] if r.get("parent_id")==parent]
        if len(candidates)!=1 or not exact(row,candidates[0]):
            raise ValueError("combined wrappers must preserve exact phase rows")
        wrapper=json_metadata(row["task_transfer_provenance"],Path(contract["path"]).parent/phase/parent/"task_transfer_v2.json")
        if not exact(wrapper,{k:v for k,v in row.items() if k!="task_transfer_provenance"}):
            raise ValueError("original wrapper provenance differs")
        native=row["frozen_helper_result"]
        rosters[parent]=json_metadata({k:native["observed_sample_manifest"][k] for k in ["path","sha256"]},
                                     Path(contract["path"]).parent/phase/parent/"observed_sample_manifest.json")
        sources[parent]=json_metadata(native["original_observed_sample_manifest"],
                                      Path("data/biological_paths_execution_pilot_v1")/parent/"observed/sample_manifest.json")
    records=validate_combined(c,manifest,rosters,sources)
    lock=json_metadata(c["observation_provenance"]["observed_asset_pins"])
    for parent,record in records.items():
        native=record["native"];pin=lock["parents"][parent]
        if native["input_observed_sha256"]!=pin["observed_files_sha256"] or native["coordinate_contract"]["coverage_original_bp0"]!=pin["comparison_coverage_original_bp0"]:
            raise ValueError("frozen exact input/support pins differ")
    return {"kind":"official_argformer_tsinfer_biological_v2_metadata_report",
            "status":"metadata_only_actual_tree_path_corpus_and_model_unqualified",
            "manifest":{"path":contract["path"],"sha256":contract["sha256"]},
            "parents":{p:{k:v for k,v in r.items() if k!="native"} for p,r in records.items()},
            "arrays_opened":False,"model_or_training_executed":False,
            "native_ID_preservation_claim":"verified root metadata only; actual tree/path qualification still required"}

def source_report(c):
    pins=verify_source(c)
    _,tail_hash=preparation_core(c)
    return {"kind":"official_argformer_tsinfer_biological_v2_source_report","status":"inactive_source_only_no_native_or_array_execution",
            "config_sha256":digest(CONFIG),"implementation_sha256":digest(SOURCE),"condition":c["condition"],
            "frozen_sources":pins,"native_function_ASTs":function_proof(c),
            "preparation_tail_AST_sha256":tail_hash,"recipe":{k:c[k] for k in ["pretraining","contrastive","sequence_length","fitting_regime","retrieval","data_fidelity"]},
            "native_imported":False,"actual_arrays_opened":False,"model_or_training_executed":False,"GPU":False,
            "paper_numerical_reproduction":False,"activation_prerequisites":c["required_root_stages"]}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=str(CONFIG))
    parser.add_argument("--metadata",action="store_true")
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args(argv)
    c=policy(args.config)
    if args.metadata:
        verify_source(c)
    if args.execute:
        execution_policy(c,None)
    print(json.dumps(metadata_report(c) if args.metadata else source_report(c),indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
