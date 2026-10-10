#!/usr/bin/env python
"""Source-only by default; separately admitted observed1000 canary then remaining."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
DEFAULT_CONFIG="configs/biological_paths_tsinfer_observed_v2.json"
CONDITION="current_tsinfer_tsdate_observed_heterochronous_task_transfer_v2"
FIXED_CONFIG={'kind': 'biological_paths_tsinfer_observed_v2',
 'status': 'source_only_parent_phase_artifact_binding_revision_v3_independent_rereview_and_root_admission_pending',
 'condition': 'current_tsinfer_tsdate_observed_heterochronous_task_transfer_v2',
 'fidelity': 'current_pinned_tsinfer_tsdate_family_task_transfer_not_exact_ARGformer_paper_numerical_reproduction',
 'producer': {'path': 'scripts/infer_biological_paths_tsinfer_v1.py',
              'sha256': 'b1f1c9a50b77daa9a416e7fed04d06a1bbbc247644d63103b11a2fa80a4fb2e0'},
 'biological_config': {'path': 'configs/biological_paths_tsinfer_v1.json',
                       'sha256': '85d8dc47f771826f3d23c42c55b847a2527f124540def1c91e5f51b200121e6c'},
 'runtime': {'path': 'experiments/results/biological_tsinfer_runtime_install_v1/runtime_receipt.json',
             'sha256': '04876226b6f90fc803741b4257f17b9595374c58e1681e4c04b7d6d9ba4e6e8f'},
 'observed_asset_pins': {'path': 'research/reviews/biological_paths_tsinfer_v1_source_audit/observed_asset_pins_from_existing_relate_manifest.json',
                         'sha256': 'e7781b2a5b6ee9630d12db0edf9401fdba5ac3a9fe9099b69cd444e1c11c492d'},
 'reduced_smoke_outcome': {'path': 'research/reviews/biological_paths_tsinfer_qualification_v2_root_admission/outcome.json',
                           'sha256': 'ece180db3f630483cfb0ed0d87a847dbc302e5baa7e7d81dc012e79608405541'},
 'reduced_smoke_certification': 'capability_smoke_only_not_same1000_interval_qualification',
 'scientific_recipe': {'rescaling_intervals': 1000,
                       'all_other_native_scientific_settings': 'exact_frozen_biological_config'},
 'phases': {'canary': ['bio_paths_pilot_train_000'],
            'remaining': ['bio_paths_pilot_train_001',
                          'bio_paths_pilot_val_000',
                          'bio_paths_pilot_val_001']},
 'phase_admission_policy': 'remaining_requires_separate_root_admission_and_same_source_runtime1000_canary_adequacy',
 'scope': 'four_original_biological_development_observation_parents_only_no_heldout_test_publication_predictions',
 'input_files': ['genotypes.vcf',
                 'sample_manifest.json',
                 'sample_order.txt',
                 'sample_ages.generations',
                 'population_labels.txt',
                 'site_coordinate_manifest.json',
                 'uniform_recombination_map.txt',
                 'scoring_coordinates.npz',
                 'observation_manifest.json'],
 'output_root': 'data/biological_paths_execution_pilot_v1/tsinfer_tsdate_observed_v2',
 'phase_launch_logs': {'canary': 'research/reviews/biological_paths_tsinfer_observed_v2_author/canary.launch.log',
                       'remaining': 'research/reviews/biological_paths_tsinfer_observed_v2_author/remaining.launch.log'},
 'storage': {'kind': 'readonly_persistent_ledger_with_observed20GB_output_caps_v2',
             'ledger_source': {'path': 'src/arggnn/gpu_pool_v1.py',
                               'sha256': 'ae1e0494ab672cfa3017759460eb06a4b4789eaef54928496c3c80314adcc0e1'},
             'snapshot': {'path': 'experiments/results/biological_shared_pool_storage_admission_v1/snapshot.json',
                          'sha256': '0826fcedefa40dd261999c4916b85ae11f28628e5cee9046b2e4f6bf868fa10d'},
             'ledger_path': '.runtime/gpup/storage_reservations.json',
             'external_reserve_bytes': 130000000000,
             'warning_bytes': 450000000000,
             'hard_limit_bytes': 500000000000,
             'reservation_key': 'biological_observed_tsinfer_v2/four-parent-observed-inference',
             'reserved_bytes': 20000000000,
             'local_write_cap_bytes': 20000000000,
             'readonly_runtime_reservation_key': 'biological_cpu_runtime_v1/isolated-runtime-and-handmade-qualification',
             'readonly_runtime_reserved_bytes': 8000000000,
             'stage_boundary_cap_not_OS_quota': True},
 'native_paper_reproduction_claimed': False,
 'truth_or_oracle_inputs_authorized': False,
 'heldout_test_or_publication_predictions_authorized': False,
 'silent_parameter_fallback': False,
 'artifact_join_contract': 'mandatory_expected_parent_phase_and_exact_canonical_native_artifacts_v3'}

def exact(value,expected):
    return json.dumps(value,sort_keys=True,separators=(",",":")) == json.dumps(expected,sort_keys=True,separators=(",",":"))

def load_producer():
    import hashlib
    sys.dont_write_bytecode=True
    path=ROOT/FIXED_CONFIG["producer"]["path"]
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=FIXED_CONFIG["producer"]["sha256"]:
        raise ValueError("frozen v3 producer changed")
    spec=importlib.util.spec_from_file_location("_frozen_tsinfer_producer_for_observed_v2",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # only standard-library definitions
    return module

def validate_config(base,c):
    if not exact(c,FIXED_CONFIG):
        raise ValueError("exact observed-v2 config required")
    biological=json.loads(base.checked_ref(c["biological_config"]).read_text())
    base.validate_config(biological)  # frozen1000 scientific contract
    if biological["dating"]["rescaling_intervals"]!=1000 or c["observed_asset_pins"]!=biological["observed_asset_pins"]:
        raise ValueError("original1000 scientific recipe and observation metadata required")
    base.checked_ref(c["runtime"])
    base.checked_ref(c["storage"]["ledger_source"])
    base.checked_ref(c["storage"]["snapshot"])
    smoke=json.loads(base.checked_ref(c["reduced_smoke_outcome"]).read_text())
    if (smoke.get("kind")!="root_tiny_native_reduced_rescaling_qualification_v2_outcome"
            or smoke.get("status")!="passed_reduced_numerical_fixture_only"
            or smoke.get("biological1000_setting_qualified") is not False
            or smoke.get("actual_observed_arrays_opened") is not False):
        raise ValueError("reduced fixture must be smoke evidence only, never biological1000 qualification")
    lock=json.loads(base.checked_ref(c["observed_asset_pins"]).read_text())
    if (lock.get("kind")!="biological_paths_tsinfer_v1_observed_asset_pins"
            or set(lock.get("parents",{}))!=set(base.PARENTS)):
        raise ValueError("exact original four-parent observation metadata lock required")
    base.checked_ref({k:v for k,v in lock["basis"].items() if k in {"path","sha256"}})
    for parent,pin in lock["parents"].items():
        if set(pin["observed_files_sha256"])!=set(base.FILES):
            raise ValueError("exact nine observed-file whitelist required")
        if any(not isinstance(h,str) or len(h)!=64 for h in pin["observed_files_sha256"].values()):
            raise ValueError("exact original observed-file SHA256 pins required")
        left,right=pin["comparison_coverage_original_bp0"]
        if base.grid_indexes(left,right)!=pin["comparison_retained_fixed100bp_indexes"]:
            raise ValueError("original common comparison support/grid metadata changed")
    return biological,lock

def canonical_path(base,path):
    lexical=Path(path).absolute()
    if lexical==ROOT or ROOT not in lexical.parents or lexical.resolve()!=lexical:
        raise ValueError("exact canonical project artifact path required; symlink/traversal denied")
    if base.confined(lexical)!=lexical:
        raise ValueError("artifact namespace changed under resolution")
    return lexical

def canonical_file(base,path):
    p=canonical_path(base,path)
    if not p.is_file() or p.is_symlink():
        raise ValueError("exact regular native artifact file required")
    return p

def artifact_join(base,reference,expected):
    expected=canonical_path(base,expected)
    allowed={str(expected),str(expected.relative_to(ROOT))}
    if (not isinstance(reference,dict) or set(reference)!={"path","sha256"}
            or reference.get("path") not in allowed):
        raise ValueError("native artifact reference must equal expected phase/parent path")
    canonical_file(base,expected)
    checked=base.checked_ref(reference)
    if checked!=expected:
        raise ValueError("native artifact resolved outside expected phase/parent path")
    return checked

def phase_target(base,c,mode):
    if mode not in c["phases"]:
        raise ValueError("configured canary/remaining phase required")
    return canonical_path(base,ROOT/c["output_root"]/mode)

def parent_directory(base,c,parent,phase):
    if phase not in c["phases"] or parent not in c["phases"][phase]:
        raise ValueError("expected parent must belong to configured phase")
    return canonical_path(base,phase_target(base,c,phase)/parent)

def storage_context(base,c,biological,receipt,mode):
    p=c["storage"]
    bindings={"kind":"biological_paths_tsinfer_observed_v2_storage_admission","mode":mode,
              "project_root":str(ROOT),"config":base.ref(ROOT/DEFAULT_CONFIG),
              "implementation":base.ref(Path(__file__)),"producer":c["producer"],
              "runtime":c["runtime"],"snapshot":p["snapshot"],
              "reservation_key":p["reservation_key"],"reserved_bytes":p["reserved_bytes"],
              "local_write_cap_bytes":p["local_write_cap_bytes"],
              "warning_bytes":p["warning_bytes"],"hard_limit_bytes":p["hard_limit_bytes"],
              "write_scopes":[c["output_root"],*c["phase_launch_logs"].values()],
              "readonly_runtime_reservation_key":p["readonly_runtime_reservation_key"],
              "readonly_runtime_reserved_bytes":p["readonly_runtime_reserved_bytes"]}
    if any(not exact(receipt.get(k),v) for k,v in bindings.items()):
        raise ValueError("mode-specific20GB storage receipt must bind exact source/runtime/all scopes")
    initial_sha=receipt.get("ledger_sha256_at_admission")
    if not isinstance(initial_sha,str) or len(initial_sha)!=64:
        raise ValueError("root-issued exact initial ledger SHA256 required")
    module=base.storage_module(biological)  # audited standard library, no GPU query

    class ObservedStorage(base.StorageAccounting):
        """Protocol subclass only; separate20GB policy, no base/global mutation."""
        def __init__(self):
            self.first=True
            self.high_water=0
            self.check_count=0
            self.check()

        def check(self):
            import fcntl
            with module.lock_file(ROOT,"scheduler.lock") as locked:
                fcntl.flock(locked,fcntl.LOCK_EX)
                snapshot=module.snapshot_record(ROOT,str(base.confined(p["snapshot"]["path"])),p["snapshot"]["sha256"])
                def available():
                    value=os.statvfs(ROOT)
                    return value.f_bavail*value.f_frsize
                ledger=module.StorageLedger(ROOT,snapshot,p["external_reserve_bytes"],p["warning_bytes"],available)
                if str(ledger.path)!=str(base.confined(p["ledger_path"])) or not ledger.path.is_file():
                    raise ValueError("existing persistent shared ledger required")
                before=base.sha(ledger.path)
                if self.first and before!=initial_sha:
                    raise ValueError("ledger changed before observed admission; renew root storage receipt")
                state=ledger._load()
                expected={"reserved_bytes":p["reserved_bytes"],"output":str(base.confined(c["output_root"]))}
                runtime={"reserved_bytes":p["readonly_runtime_reserved_bytes"],
                         "output":str(base.confined(biological["runtime_prefix"]))}
                if not exact(state["jobs"].get(p["reservation_key"]),expected):
                    raise ValueError("exact new20GB observed inference reservation/output required")
                if not exact(state["jobs"].get(p["readonly_runtime_reservation_key"]),runtime):
                    raise ValueError("exact retained readonly8GB runtime reservation required")
                account=ledger._check(state)  # includes EVERY controller/job and130GB external
                if base.sha(ledger.path)!=before:
                    raise ValueError("ledger changed during atomic read")
            if account["projected_bytes"]>=p["hard_limit_bytes"]:
                raise ValueError("500GB hard limit exceeded")
            current=base.local_scope_bytes(base.confined(c["output_root"]))
            current+=sum(base.local_scope_bytes(base.confined(path)) for path in c["phase_launch_logs"].values())
            self.high_water=max(self.high_water,current)
            if self.high_water>p["local_write_cap_bytes"] or self.high_water>p["reserved_bytes"]:
                raise ValueError("combined canary/remaining output/temp/cache/log20GB local cap exceeded")
            self.first=False
            self.check_count+=1
            return {**account,"ledger_sha256":before,"reservation_key":p["reservation_key"],
                    "reserved_bytes":p["reserved_bytes"],"local_current_bytes":current,
                    "local_high_water_bytes":self.high_water,"local_cap_bytes":p["local_write_cap_bytes"],
                    "check_count":self.check_count,"includes_canary_during_remaining":True,
                    "readonly_runtime_slot_retained":True,"snapshot_is_historical_not_fresh_measurement":True,
                    "cap_is_stage_boundary_sampled_not_OS_quota":True}
    return ObservedStorage()

def parent_adequacy(base,c,lock,result,*,expected_parent,phase):
    work=parent_directory(base,c,expected_parent,phase)
    if (result.get("parent_id")!=expected_parent
            or result.get("condition")!=base.CONDITION
            or result.get("inferred_tree",{}).get("condition")!=base.CONDITION):
        raise ValueError("inner native parent/helper condition must match expected outer/phase contract")
    pin=lock["parents"][expected_parent]
    if (result.get("status")!="qualified_observed_cpu_inference"
            or result.get("input_observed_sha256")!=pin["observed_files_sha256"]
            or result.get("native_node_ids_and_times_preserved_during_sanitation") is not True
            or result.get("no_input_variant_filtering") is not True
            or result.get("truth_or_oracle_read") is not False
            or result["coordinate_contract"]["coverage_original_bp0"]!=pin["comparison_coverage_original_bp0"]
            or result["coordinate_contract"]["retained_fixed100bp_indexes"]!=pin["comparison_retained_fixed100bp_indexes"]
            or result["connected_identity"]["internal_reuse_occurrences"]<1):
        raise ValueError("frozen observed genotype/date/site/support/identity output adequacy failed")
    completed=artifact_join(base,result["completion_provenance"],work/"inference.json")
    stored=json.loads(completed.read_text())
    if not exact(stored,{k:v for k,v in result.items() if k!="completion_provenance"}):
        raise ValueError("canary/native result differs from preserved frozen-helper completion")
    artifact_join(base,{k:result["inferred_tree"][k] for k in ["path","sha256"]},work/"observed.trees")
    artifact_join(base,{k:result["observed_sample_manifest"][k] for k in ["path","sha256"]},work/"observed_sample_manifest.json")
    # Frozen helper inventory joins every generated native file to this one parent.
    inventory=result.get("native_artifacts")
    required={"observed.trees","observed_sample_manifest.json","native_verification.json","stages.json"}
    if not isinstance(inventory,dict) or not required.issubset(inventory):
        raise ValueError("complete frozen-helper native artifact inventory required")
    for name,item in inventory.items():
        relative=Path(name)
        if (not isinstance(name,str) or relative.is_absolute() or name!=str(relative)
                or ".." in relative.parts or relative==Path(".")
                or not isinstance(item,dict) or set(item)!={"sha256","bytes"}
                or type(item["bytes"]) is not int or item["bytes"]<0):
            raise ValueError("canonical relative native artifact inventory required")
        expected=canonical_file(base,work/relative)
        if work not in expected.parents:
            raise ValueError("native artifact inventory escapes expected parent")
        artifact_join(base,{"path":str(expected.relative_to(ROOT)),"sha256":item["sha256"]},expected)
        if expected.stat().st_size!=item["bytes"]:
            raise ValueError("native artifact inventory byte count changed")
    verification_path=canonical_file(base,work/"native_verification.json")
    stages_path=canonical_file(base,work/"stages.json")
    verification=json.loads(verification_path.read_text())
    if (verification.get("exact_observed_genotypes_and_ages") is not True
            or verification.get("strict_native_times") is not True
            or verification.get("source_labels_passed_to_inference") is not False
            or verification.get("truth_or_oracle_tree_read") is not False
            or verification.get("all_original_sites_preserved")!=result["native_full_sites"]
            or verification.get("identity_support_original_bp0")!=pin["comparison_coverage_original_bp0"]
            or result["connected_identity"].get("uncovered_interval_inside_support_admitted") is not False):
        raise ValueError("native genotype/date/strict-time verification unavailable")
    stages=json.loads(stages_path.read_text())
    names=["modern_inferred","modern_preprocessed","modern_dated","modern_site_ages",
           "ancient_site_bounds","ancestors_generated","historical_proxy_ancestors",
           "ancestors_matched","all_samples_matched","all_preprocessed","all_dated"]
    trees={"modern_inferred","modern_preprocessed","modern_dated","ancestors_matched",
           "all_samples_matched","all_preprocessed","all_dated"}
    if [e.get("stage") for e in stages]!=names or any(e.get("status")!="complete" for e in stages):
        raise ValueError("all11 native1000 scientific stages must complete")
    for entry in stages:
        if entry["stage"] in trees:
            artifact_join(base,entry["output"],work/(entry["stage"]+".trees"))
        elif "output" in entry:
            raise ValueError("unexpected native tree output on non-tree stage")
    return {"expected_parent":expected_parent,"phase":phase,
            "native_parent_directory":str(work.relative_to(ROOT)),
            "source_input_hashes_exact":True,"original_genotypes_and_sample_ages_exact":True,
            "strict_native_times_and_mutation_branches":True,"entire_common_support_connected":True,
            "native_ids_preserved_by_sanitation":True,"rescaling_intervals":1000,
            "minimal_identity_reuse_diagnostic_only":True,"biological_accuracy_or_identity_demonstrated":False,
            "native_verification":base.ref(verification_path),"stage_journal":base.ref(stages_path)}

def wrapper_join(base,c,row,*,expected_parent,phase):
    work=parent_directory(base,c,expected_parent,phase)
    path=artifact_join(base,row["task_transfer_provenance"],work/"task_transfer_v2.json")
    stored=json.loads(path.read_text())
    if not exact(stored,{k:v for k,v in row.items() if k!="task_transfer_provenance"}):
        raise ValueError("wrapper provenance differs from preserved expected parent/phase row")
    return path

def canary_gate(base,c,lock,admission):
    review=json.loads(base.checked_ref(admission["canary_adequacy"]).read_text())
    manifest_path=artifact_join(base,admission["canary_manifest"],phase_target(base,c,"canary")/"manifest.json")
    m=json.loads(manifest_path.read_text())
    if (review.get("kind")!="biological_paths_tsinfer_observed_v2_root_canary_adequacy"
            or review.get("authorizer")!="root_agent"
            or review.get("status")!="passed_source_input_genotype_date_support_identity_canary1000"
            or review.get("canary_manifest")!=admission["canary_manifest"]
            or review.get("config")!=base.ref(ROOT/DEFAULT_CONFIG)
            or review.get("implementation")!=base.ref(Path(__file__))
            or review.get("runtime")!=c["runtime"]
            or review.get("producer")!=c["producer"]
            or review.get("biological_config")!=c["biological_config"]
            or review.get("observed_asset_pins")!=c["observed_asset_pins"]
            or review.get("biological1000_observed_canary") is not True
            or review.get("reduced_smoke_is_same1000_qualification") is not False):
        raise ValueError("separate root same-source/runtime1000 observed canary adequacy required")
    expected={"kind":"biological_paths_tsinfer_observed_v2_canary","status":"passed_train000_observed_canary1000",
              "mode":"canary","condition":CONDITION,"config":base.ref(ROOT/DEFAULT_CONFIG),
              "implementation":base.ref(Path(__file__)),"producer":c["producer"],
              "biological_config":c["biological_config"],"runtime":c["runtime"],
              "observed_asset_pins":c["observed_asset_pins"],"scope_parents":c["phases"]["canary"],
              "rescaling_intervals":1000,"reduced_smoke_only":True,"native_paper_reproduction_claimed":False,
              "truth_or_oracle_read":False,"heldout_test_or_publication_predictions":False}
    if any(not exact(m.get(k),v) for k,v in expected.items()) or len(m.get("replicates",[]))!=1:
        raise ValueError("same1000 TRAIN000 observation canary manifest required")
    if manifest_path!=phase_target(base,c,"canary")/"manifest.json":
        raise ValueError("canary receipt must bind configured fresh output namespace")
    row=m["replicates"][0]
    expected_parent=c["phases"]["canary"][0]
    if (row.get("parent_id")!=expected_parent or row.get("condition")!=CONDITION
            or row.get("source_helper_condition")!=base.CONDITION
            or not exact(row.get("rescaling_intervals"),1000)
            or row.get("native_paper_reproduction_claimed") is not False):
        raise ValueError("TRAIN000 canary outer parent/helper/phase contract required")
    adequacy=parent_adequacy(base,c,lock,row["frozen_helper_result"],expected_parent=expected_parent,phase="canary")
    if not exact(row.get("adequacy"),adequacy):
        raise ValueError("recorded canary adequacy differs from preserved native metadata")
    wrapper_join(base,c,row,expected_parent=expected_parent,phase="canary")
    return m

def admission(base,c,biological,lock,args,config_path):
    if not all([args.expected_config_sha256,args.expected_source_sha256,args.admission,args.expected_admission_sha256]):
        raise ValueError("observed modes require exact root config/source/admission SHA256 opt-in")
    if (base.sha(config_path)!=args.expected_config_sha256
            or base.sha(Path(__file__))!=args.expected_source_sha256):
        raise ValueError("root-reviewed observed harness/config changed")
    path=base.checked_ref({"path":args.admission,"sha256":args.expected_admission_sha256})
    a=json.loads(path.read_text())
    expected={"kind":"biological_paths_tsinfer_observed_v2_root_admission","authorizer":"root_agent",
              "mode":args.mode,"condition":CONDITION,"config":base.ref(config_path),
              "implementation":base.ref(Path(__file__)),"producer":c["producer"],
              "biological_config":c["biological_config"],"runtime":c["runtime"],
              "observed_asset_pins":c["observed_asset_pins"],"scope_parents":c["phases"][args.mode],
              "rescaling_intervals":1000,"reduced_smoke_outcome":c["reduced_smoke_outcome"],
              "reduced_smoke_only":True,"truth_or_oracle_inputs_authorized":False,
              "heldout_test_or_publication_predictions_authorized":False,
              "observed_development_genotypes_authorized":True,"silent_parameter_fallback":False}
    if any(not exact(a.get(k),v) for k,v in expected.items()):
        raise ValueError("root observed mode/source/input/1000 scope differs")
    base.checked_ref(a["experiment_registration"])
    if args.mode=="remaining":
        canary_gate(base,c,lock,a)
    elif "canary_adequacy" in a or "canary_manifest" in a:
        raise ValueError("canary admission may not borrow a qualification or earlier canary")
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise ValueError("CPU mode must explicitly hide CUDA")
    target=phase_target(base,c,args.mode)
    output=base.confined(c["output_root"])
    if target.exists() or target.is_symlink() or (args.mode=="canary" and output.exists()):
        raise ValueError("fresh versioned phase output required; repeats need new source/config/admission")
    runtime=json.loads(base.checked_ref(c["runtime"]).read_text())
    if (runtime.get("kind")!="biological_paths_tsinfer_v1_isolated_runtime"
            or runtime.get("prefix")!=biological["runtime_prefix"]
            or Path(sys.prefix).resolve()!=base.confined(biological["runtime_prefix"])
            or f"{sys.version_info.major}.{sys.version_info.minor}"!=biological["python_minor"]
            or base.sha(sys.executable)!=runtime.get("python_executable_sha256")
            or runtime.get("versions")!=base.VERSIONS):
        raise ValueError("frozen isolated runtime interpreter/versions differ")
    if base.runtime_inventory(biological["runtime_prefix"])!=runtime.get("files_sha256"):
        raise ValueError("complete frozen runtime inventory changed")
    for name,expected_version in base.VERSIONS.items():
        if base.importlib.metadata.version(name)!=expected_version:
            raise ValueError("installed distribution changed")
    storage=json.loads(base.checked_ref(a["storage"]).read_text())
    context=storage_context(base,c,biological,storage,args.mode)
    return a,path,context

def run_parent(base,c,biological,lock,parent,target,context,*,phase):
    parent_directory(base,c,parent,phase)
    if canonical_path(base,target)!=phase_target(base,c,phase):
        raise ValueError("native run target must equal configured expected phase")
    context.check()
    native=base.infer_parent(biological,parent,target,lock["parents"][parent],context)
    adequacy=parent_adequacy(base,c,lock,native,expected_parent=parent,phase=phase)
    # Preserve frozen-helper inference.json condition and bytes; wrapper is explicit.
    row={"parent_id":parent,"condition":CONDITION,"source_helper_condition":base.CONDITION,
         "frozen_helper_result":native,"adequacy":adequacy,
         "native_paper_reproduction_claimed":False,"rescaling_intervals":1000}
    out=target/parent/"task_transfer_v2.json"
    base.write_json(out,row,new=True)
    row["task_transfer_provenance"]=base.ref(out)
    wrapper_join(base,c,row,expected_parent=parent,phase=phase)
    context.check()
    return row

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=DEFAULT_CONFIG)
    parser.add_argument("--mode",choices=["source-only","canary","remaining"],default="source-only")
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--admission")
    parser.add_argument("--expected-admission-sha256")
    args=parser.parse_args()
    try:
        base=load_producer()
        config_path=base.confined(args.config)
        if config_path!=ROOT/DEFAULT_CONFIG:
            raise ValueError("fixed observed-v2 config path required")
        c=json.loads(config_path.read_text())
        biological,lock=validate_config(base,c)
        report={"kind":"biological_paths_tsinfer_observed_v2_source_check",
                "status":"source_only_observed_arrays_native_inference_unexecuted",
                "condition":CONDITION,"fidelity":c["fidelity"],"config":base.ref(config_path),
                "implementation":base.ref(Path(__file__)),"producer":c["producer"],
                "biological_config":c["biological_config"],"runtime":c["runtime"],
                "observed_asset_pins":c["observed_asset_pins"],"phases":c["phases"],
                "rescaling_intervals":1000,"reduced_smoke_only":True,
                "runtime_imported":False,"observed_arrays_opened":False,
                "native_pipeline_executed":False,"native_paper_reproduction_claimed":False}
        if args.mode=="source-only":
            print(json.dumps(report,indent=2))
            return
        a,admission_path,context=admission(base,c,biological,lock,args,config_path)
        os.environ["PYTHONDONTWRITEBYTECODE"]="1"
        sys.dont_write_bytecode=True
        output=base.confined(c["output_root"])
        if args.mode=="canary":
            output.mkdir(exist_ok=False)
        target=phase_target(base,c,args.mode)
        target.mkdir(exist_ok=False)
        for name in ("TMPDIR","TMP","TEMP","NUMBA_CACHE_DIR","XDG_CACHE_HOME","XDG_CONFIG_HOME","XDG_DATA_HOME"):
            p=target/"write_scopes"/name.lower()
            p.mkdir(parents=True,exist_ok=True)
            os.environ[name]=str(p)
        context.check()
        manifest={**{k:v for k,v in report.items() if k not in {"kind","status","runtime_imported","observed_arrays_opened","native_pipeline_executed","phases"}},
                  "kind":"biological_paths_tsinfer_observed_v2_"+args.mode,"status":"running",
                  "mode":args.mode,"scope_parents":c["phases"][args.mode],"replicates":[],
                  "admission":base.ref(admission_path),"storage":a["storage"],
                  "experiment_registration":a["experiment_registration"],
                  "truth_or_oracle_read":False,"heldout_test_or_publication_predictions":False,
                  "source_helper_condition":base.CONDITION,"native_pipeline_entered":False}
        if args.mode=="remaining":
            manifest.update(canary_manifest=a["canary_manifest"],canary_adequacy=a["canary_adequacy"])
        base.write_json(target/"manifest.partial.json",manifest,new=True)
        try:
            for parent in c["phases"][args.mode]:
                manifest.update(active_parent=parent,native_pipeline_entered=True)
                base.write_json(target/"manifest.partial.json",manifest)
                manifest["replicates"].append(run_parent(base,c,biological,lock,parent,target,context,phase=args.mode))
                base.write_json(target/"manifest.partial.json",manifest)
            if args.mode=="remaining":
                canary_gate(base,c,lock,a)  # preserve and recheck admitted canary primary artifacts
            manifest["status"]="passed_train000_observed_canary1000" if args.mode=="canary" else "remaining_three_observed_parents_inferred1000"
            if base.runtime_inventory(biological["runtime_prefix"])!=json.loads(base.checked_ref(c["runtime"]).read_text())["files_sha256"]:
                raise ValueError("frozen runtime changed during execution")
            manifest["storage_before_final_manifest"]=context.check()
        except Exception as error:
            manifest.update(status="failed_partial_preserved",error_type=type(error).__name__,error=str(error))
            base.write_json(target/"manifest.json",manifest)
            raise
        base.write_json(target/"manifest.json",manifest,new=True)
        try:
            context.check()
        except Exception as error:
            manifest.update(status="failed_storage_after_manifest_preserved",error_type=type(error).__name__,error=str(error))
            base.write_json(target/"manifest.json",manifest)
            raise
        print(json.dumps(base.ref(target/"manifest.json"),indent=2))
    except (ValueError,KeyError,RuntimeError) as error:
        parser.error(str(error))

if __name__=="__main__":
    main()
