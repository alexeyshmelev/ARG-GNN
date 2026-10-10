#!/usr/bin/env python
"""Source-only by default; separate reduced-rescaling handmade qualification."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
DEFAULT_CONFIG = "research/reviews/biological_paths_tsinfer_native_failure_v1/qualification_v2.json"
CONDITION = "handmade_tsinfer_tsdate_single_interval_rescaling_qualification_v2"
FIXED_CONFIG = {'kind': 'biological_paths_tsinfer_handmade_rescaling_qualification_v2',
 'status': 'source_only_independent_review_and_root_native_admission_pending',
 'condition': 'handmade_tsinfer_tsdate_single_interval_rescaling_qualification_v2',
 'biological_condition': 'official_documented_heterochronous_tsinfer_tsdate_task_transfer_v1',
 'producer': {'path': 'scripts/infer_biological_paths_tsinfer_v1.py',
              'sha256': 'b1f1c9a50b77daa9a416e7fed04d06a1bbbc247644d63103b11a2fa80a4fb2e0'},
 'biological_config': {'path': 'configs/biological_paths_tsinfer_v1.json',
                       'sha256': '85d8dc47f771826f3d23c42c55b847a2527f124540def1c91e5f51b200121e6c'},
 'runtime': {'path': 'experiments/results/biological_tsinfer_runtime_install_v1/runtime_receipt.json',
             'sha256': '04876226b6f90fc803741b4257f17b9595374c58e1681e4c04b7d6d9ba4e6e8f'},
 'previous_failed_scope_pins': {'path': 'research/reviews/biological_paths_tsinfer_native_failure_v1/preserved_qualification_scope_pins.json',
                                'sha256': '49b7f16ad30797c3e08feb44c5a1f84dcc64011edea8d748a5bfb00bd408501b'},
 'qualification_root': 'research/reviews/biological_paths_tsinfer_native_failure_v1/root_cpu_qualification_rescaling_v2',
 'qualification_launch_log': 'research/reviews/biological_paths_tsinfer_native_failure_v1/root_cpu_qualification_rescaling_v2.launch.log',
 'fixture': 'exact_frozen_producer_handmade_assets_6sites_8haplotypes_ages0_10_20',
 'qualification_dating': {'method': 'variational_gamma',
                          'max_iterations': 25,
                          'rescaling_intervals': 1,
                          'rescaling_iterations': 5,
                          'match_segregating_sites': False,
                          'singletons_phased': True,
                          'constr_iterations': None,
                          'min_branch_length': 1e-08,
                          'allow_unary': False,
                          'set_metadata': True,
                          'time_units': 'generations'},
 'qualification_scientific_delta': {'dating.rescaling_intervals': {'biological': 1000,
                                                                   'qualification': 1}},
 'certification': 'reduced_single_interval_rescaling_only_not_exact1000_interval_biological_qualification',
 'actual_observed_inference_authorized': False,
 'truth_or_oracle_inputs_authorized': False,
 'silent_retry_or_parameter_fallback': False,
 'include_preserved_v3_qualification_scopes_in_1GB_local_cap': True}

def load_producer():
    sys.dont_write_bytecode = True
    import hashlib
    path = ROOT/FIXED_CONFIG["producer"]["path"]
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != FIXED_CONFIG["producer"]["sha256"]:
        raise ValueError("frozen v3 producer changed")
    spec = importlib.util.spec_from_file_location("_frozen_tsinfer_producer_for_qualification_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # standard-library definitions only
    return module

def validate_config(base, c):
    if c != FIXED_CONFIG:
        raise ValueError("exact qualification-only v2 config required")
    biological_path = base.checked_ref(c["biological_config"])
    biological = json.loads(biological_path.read_text())
    base.validate_config(biological)  # original1000 contract, never the qualification copy
    base.checked_ref(c["runtime"])
    prior_path = base.checked_ref(c["previous_failed_scope_pins"])
    prior = json.loads(prior_path.read_text())
    expected_scopes = [biological["qualification_root"],
                       biological["storage_accounting"]["qualification_launch_log"]]
    if (prior.get("kind") != "biological_paths_tsinfer_failed_handmade_qualification_scope_pins_v1"
            or prior.get("scope_paths") != expected_scopes or not prior.get("files")):
        raise ValueError("exact preserved failed-v3 scopes required")
    if biological["dating"]["rescaling_intervals"] != 1000:
        raise ValueError("biological recipe must retain1000 intervals")
    intended = {**biological["dating"], "rescaling_intervals":1}
    if c["qualification_dating"] != intended:
        raise ValueError("only explicit qualification rescaling1000-to1 change allowed")
    effective = copy.deepcopy(biological)
    effective["dating"] = copy.deepcopy(c["qualification_dating"])
    effective["qualification_root"] = c["qualification_root"]
    effective["storage_accounting"]["qualification_launch_log"] = c["qualification_launch_log"]
    # The scientific delta is explicit in the pinned config and every manifest.
    for scope in expected_scopes:
        p = base.confined(scope)
        for target in [base.confined(c["qualification_root"]),
                       base.confined(c["qualification_launch_log"])]:
            if p == target or p in target.parents or target in p.parents:
                raise ValueError("fresh v2 scopes overlap preserved v3 outputs")
    return biological, effective, prior

def verify_prior_scopes(base, prior):
    tree, log = [base.confined(p) for p in prior["scope_paths"]]
    actual = {}
    def unreadable(error):
        raise ValueError("unreadable preserved qualification scope") from error
    for folder, dirs, names in os.walk(tree, followlinks=False, onerror=unreadable):
        for name in dirs + names:
            if (Path(folder)/name).is_symlink():
                raise ValueError("symlink in preserved qualification scope")
        for name in names:
            p = Path(folder)/name
            if not p.is_file():
                raise ValueError("nonregular preserved qualification output")
            actual[str(p.relative_to(ROOT))] = {"sha256":base.sha(p), "bytes":p.stat().st_size}
    if log.is_symlink() or not log.is_file():
        raise ValueError("preserved v3 launch log absent or symlinked")
    actual[str(log.relative_to(ROOT))] = {"sha256":base.sha(log), "bytes":log.stat().st_size}
    if actual != prior["files"]:
        raise ValueError("preserved failed-v3 artifacts changed")
    return True

def storage_context(base, c, effective, receipt, prior):
    if (receipt.get("harness_config") != base.ref(ROOT/DEFAULT_CONFIG)
            or receipt.get("harness_implementation") != base.ref(Path(__file__))
            or receipt.get("previous_failed_scope_pins") != c["previous_failed_scope_pins"]
            or receipt.get("qualification_write_scopes") != [
                c["qualification_root"], c["qualification_launch_log"], *prior["scope_paths"]]):
        raise ValueError("fresh root storage receipt must bind harness and all current/previous write scopes")
    class QualificationStorage(base.StorageAccounting):
        def __init__(self):
            self.combined_high_water = 0
            super().__init__(effective, receipt, c["runtime"])

        def check(self):
            result = super().check()  # original atomic readonly all-reservations guard
            prior_bytes = sum(base.local_scope_bytes(base.confined(p)) for p in prior["scope_paths"])
            combined = result["local_current_bytes"] + prior_bytes
            self.combined_high_water = max(self.combined_high_water, combined)
            if (combined > self.policy["qualify_local_write_cap_bytes"]
                    or self.installation_charge + self.combined_high_water > self.policy["qualify_reserved_bytes"]):
                raise ValueError("combined preserved-v3 and new-v2 qualification local cap exceeded")
            return {**result, "new_v2_local_current_bytes":result["local_current_bytes"],
                    "preserved_v3_local_bytes":prior_bytes, "local_current_bytes":combined,
                    "local_high_water_bytes":self.combined_high_water,
                    "local_scope_policy":"preserved_v3_tree_and_log_plus_new_v2_tree_and_log"}
    return QualificationStorage()

def admission(base, c, effective, prior, args, config_path):
    if not all([args.expected_config_sha256, args.expected_source_sha256,
                args.admission, args.expected_admission_sha256]):
        raise ValueError("native qualification requires exact root config/source/admission hashes")
    if (base.sha(config_path) != args.expected_config_sha256
            or base.sha(Path(__file__)) != args.expected_source_sha256):
        raise ValueError("root-reviewed qualification harness/config changed")
    path = base.checked_ref({"path":args.admission, "sha256":args.expected_admission_sha256})
    a = json.loads(path.read_text())
    fixed = {"kind":"biological_paths_tsinfer_handmade_rescaling_qualification_v2_root_admission",
             "authorizer":"root_agent", "mode":"qualify-cpu", "condition":CONDITION,
             "config":base.ref(config_path), "implementation":base.ref(Path(__file__)),
             "producer":c["producer"], "biological_config":c["biological_config"],
             "runtime":c["runtime"], "previous_failed_scope_pins":c["previous_failed_scope_pins"],
             "certification":c["certification"], "actual_observed_inference_authorized":False,
             "truth_or_oracle_inputs_authorized":False}
    if any(a.get(k) != v for k,v in fixed.items()):
        raise ValueError("root admission does not bind reduced handmade-only qualification")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("qualification must explicitly hide CUDA")
    target = base.confined(c["qualification_root"])
    if target.exists() or target.is_symlink():
        raise ValueError("fresh v2 qualification output required")
    verify_prior_scopes(base, prior)
    runtime = json.loads(base.checked_ref(c["runtime"]).read_text())
    if (runtime.get("kind") != "biological_paths_tsinfer_v1_isolated_runtime"
            or runtime.get("prefix") != effective["runtime_prefix"]
            or Path(sys.prefix).resolve() != base.confined(effective["runtime_prefix"])
            or f"{sys.version_info.major}.{sys.version_info.minor}" != effective["python_minor"]
            or base.sha(sys.executable) != runtime.get("python_executable_sha256")
            or runtime.get("versions") != base.VERSIONS):
        raise ValueError("frozen isolated runtime interpreter/versions differ")
    if base.runtime_inventory(effective["runtime_prefix"]) != runtime.get("files_sha256"):
        raise ValueError("complete frozen runtime inventory changed")
    for name,expected in base.VERSIONS.items():
        if base.importlib.metadata.version(name) != expected:
            raise ValueError("installed distribution changed")
    storage = json.loads(base.checked_ref(a["storage"]).read_text())
    context = storage_context(base, c, effective, storage, prior)
    return a, path, context

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=["source-only","qualify-cpu"], default="source-only")
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--admission")
    parser.add_argument("--expected-admission-sha256")
    args = parser.parse_args()
    try:
        base = load_producer()
        config_path = base.confined(args.config)
        if config_path != ROOT/DEFAULT_CONFIG:
            raise ValueError("only fixed qualification v2 config path is accepted")
        c = json.loads(config_path.read_text())
        biological, effective, prior = validate_config(base, c)
        report = {"kind":c["kind"]+"_source_check", "status":"source_only_native_unexecuted",
                  "condition":CONDITION, "config":base.ref(config_path),
                  "implementation":base.ref(Path(__file__)), "producer":c["producer"],
                  "qualification_dating":c["qualification_dating"],
                  "qualification_scientific_delta":c["qualification_scientific_delta"],
                  "certification":c["certification"], "biological_rescaling_intervals":1000,
                  "actual_observed_inference_authorized":False,
                  "runtime_imported":False, "native_pipeline_executed":False}
        if args.mode == "source-only":
            print(json.dumps(report, indent=2))
            return
        a, admission_path, context = admission(base,c,effective,prior,args,config_path)
        os.environ["PYTHONDONTWRITEBYTECODE"]="1"
        sys.dont_write_bytecode=True
        target=base.confined(c["qualification_root"])
        target.mkdir(exist_ok=False)
        for name in ("TMPDIR","TMP","TEMP","NUMBA_CACHE_DIR","XDG_CACHE_HOME","XDG_CONFIG_HOME","XDG_DATA_HOME"):
            p=target/"write_scopes"/name.lower()
            p.mkdir(parents=True,exist_ok=True)
            os.environ[name]=str(p)
        context.check()
        manifest={**{k:v for k,v in report.items() if k not in {"runtime_imported","native_pipeline_executed"}},
                  "kind":c["kind"],"status":"running","runtime":c["runtime"],
                  "native_pipeline_entered":False,"native_pipeline_completed":False,
                  "admission":base.ref(admission_path),"storage":a["storage"],
                  "handmade_only":True,"same1000_interval_biological_qualification":False,
                  "previous_failed_scope_pins":c["previous_failed_scope_pins"]}
        base.write_json(target/"manifest.partial.json",manifest,new=True)
        try:
            case=base.handmade_assets()  # exact six-site genotype fixture and dates
            base.write_json(target/"handmade_inputs.json",case,new=True)
            manifest["native_pipeline_entered"]=True
            base.write_json(target/"manifest.partial.json",manifest)
            dated=base.pipeline(effective,case,target,context)  # no global date rebinding
            manifest["native_pipeline_completed"]=True
            summary=base.qualification_summary(effective,case,dated)
            manifest.update(summary)
            manifest.update(runtime_imported=True,native_pipeline_executed=True)
            verify_prior_scopes(base,prior)
            if base.runtime_inventory(effective["runtime_prefix"]) != json.loads(base.checked_ref(c["runtime"]).read_text())["files_sha256"]:
                raise ValueError("frozen runtime changed during qualification")
            manifest["final_storage_before_manifest"]=context.check()
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

if __name__ == "__main__":
    main()
