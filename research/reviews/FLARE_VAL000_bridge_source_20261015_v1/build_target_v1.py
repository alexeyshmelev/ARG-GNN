"""Root-only late binding of actual original VAL000 FLARE target; JSON/source IO only."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,os
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/FLARE_VAL000_bridge_source_20261015_v1"
SOURCE=HERE+"/build_target_v1.py"
PINS_REF={'path': 'research/reviews/FLARE_VAL000_bridge_source_20261015_v1/frozen_refs.json', 'sha256': '8dbf8dd95db751f7b74ffe2b19b01e124d80939dbb8e1437f28d12e6fbd60086'}
BASE="experiments/results/American_C3_val000_full30mb_shared_v1"
CPU="research/reviews/prospective_980GB_capacity_source_v1"
PARENT="american_c3_val_000"
COUNTS=dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000)
CAPS=dict(convert=950000000,native=9000000000,seal=32000000)
ROLE="registered_heldout_full30Mb_comparison"
def need(ok,message):
    if not ok:raise ValueError(message)
def local(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    need(p.resolve()==p and p.is_relative_to(ROOT) and p!=ROOT,"Canonical project-only path")
    return p
def ref(value):
    p=local(value)
    need(p.suffix in (".json",".py") and p.stat().st_size<=262144,"Bounded JSON/source only; no payload hash")
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(record):
    p=local(record["path"])
    need(set(record)=={"path","sha256"} and p.suffix==".json" and p.stat().st_size<=131072
        and ref(p)==record,"Exact bounded JSON metadata")
    return json.loads(p.read_text())
def save(folder,name,value):
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    need(Path(name).name==name and len(data)<=65536,"Compact flat target metadata")
    fd=os.open(folder/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,"wb") as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    return ref(folder/name)
def pinned():
    pins=read(PINS_REF)["refs"]
    need(all(ref(r["path"])==r for r in pins.values()),"Every original source and existing metadata pin")
    handoff=read(pins["FLARE_source_handoff"])
    need(handoff["stage_caps_bytes"]==CAPS and all(ref(r["path"])==r for r in handoff["source_closure_refs"]),
        "Existing separately pinned980 FLARE route and frozen scientific sources")
    return pins
def terminal(stage,completion_ref,wait_ref):
    expected_dir={"simulate":"simulation","infer":"inference","native_paths":"native_paths"}[stage]
    need(completion_ref["path"]==BASE+"/"+expected_dir+"/completed.json"
        and wait_ref["path"]==CPU+"/admitted_"+stage+"_v1/actual_wait.json","Exact original VAL000 stage paths")
    done=read(completion_ref);wait=read(wait_ref)
    need(wait.get("actual_direct_worker_wait") is True and wait.get("returncode")==0
        and wait.get("stage")==stage and wait.get("parent_id")==PARENT
        and wait.get("completed_ref")==completion_ref and wait.get("owned_worker_group_ESRCH") is True
        and wait.get("operational_failure") is None,"Actual same-stage direct wait0 and owned-group drain required")
    minimal=read(wait["minimal_wait_ref"]);launch=read(wait["launch_ref"]);spec=read(wait["admission_ref"])
    need(minimal["actual_direct_worker_wait"] is True and minimal["returncode"]==0
        and minimal["pid"]==wait["pid"]==launch["pid"] and minimal["process_group"]==launch["process_group"]==wait["pid"]
        and minimal["stage"]==stage and minimal["admission_ref"]==launch["admission_ref"]==wait["admission_ref"],
        "Exact genuine direct-wait and actual Popen launch lineage")
    need(spec["root_authorizer"]=="root_agent" and spec["execution_authorized"] is True
        and spec["parent_id"]==PARENT and spec["split"]=="val" and spec["stage"]==stage
        and spec["output_directory"]==BASE+"/"+expected_dir
        and spec["query_ancestry_truth_payload_access_authorized"] is False,
        "Actual root original VAL000 pre-truth stage admission")
    return done,spec
def build(args):
    pins=pinned()
    def future(name,path):
        digest=getattr(args,"expected_"+name+"_sha256")
        need(digest and len(digest)==64 and all(c in "0123456789abcdef" for c in digest),"Actual SHA required: "+name)
        return dict(path=path,sha256=digest)
    infer_ref=future("inference_completion",BASE+"/inference/completed.json")
    infer_wait_ref=future("inference_wait",CPU+"/admitted_infer_v1/actual_wait.json")
    focal_ref=future("focal_completion",BASE+"/native_paths/completed.json")
    focal_wait_ref=future("focal_wait",CPU+"/admitted_native_paths_v1/actual_wait.json")
    sim,sim_spec=terminal("simulate",pins["simulation_completion"],pins["simulation_wait"])
    infer,infer_spec=terminal("infer",infer_ref,infer_wait_ref)
    focal,focal_spec=terminal("native_paths",focal_ref,focal_wait_ref)
    simulation=read(pins["simulation_config"]);contract=read(pins["shared_contract"])
    need(sim["status"]=="completed_C3_simulation_reconstruction" and sim["parent_id"]==PARENT
        and simulation["schema"]=="american_c3_simulation_reconstruction_v1"
        and simulation["parent_id"]==PARENT and simulation["split"]=="val"
        and simulation["sequence_length_bp"]==30000000 and simulation["sample_counts"]==COUNTS
        and simulation["sample_unit"]=="haploid" and simulation["recombination_rate"]==1e-8
        and (simulation["ancestry_seed"],simulation["mutation_seed"])==(31001,1031001)
        and sim_spec["config_ref"]==pins["simulation_config"],"Exact actual original independent full30Mb VAL000 simulation")
    need(contract["parent_id"]==PARENT and contract["split"]=="val" and contract["counts_haploid"]==COUNTS
        and contract["common_focal_count"]==10000 and contract["focal_sampling_seed"]==93001
        and contract["query_sample_indices"]==dict(left_inclusive=600,right_exclusive=1600)
        and contract["reference_sample_indices"]==dict(left_inclusive=0,right_exclusive=600)
        and contract["all_method_prediction_seals_required_before_common_truth_scoring"] is True
        and contract["GNN_VAL_optimization_forbidden"] is True,"One common R600/Q1000 and10k registry contract")
    need(infer["status"]=="completed_current_C3_inference"
        and infer_spec["source_ref"]==pins["original_inference_source"],"Actual unchanged inference producer completed")
    observation_ref=infer["observation_ref"];obs=read(observation_ref)
    need(obs["parent_id"]==PARENT and obs["split"]=="val" and obs["sequence_length_bp"]==30000000
        and obs["sample_unit"]=="haploid" and obs["counts_haploid"]==COUNTS
        and obs["class_order"]==["EUR","EAS","AFR"] and obs["query_ancestry_truth_in_inputs"] is False
        and obs["genealogy_kind"]=="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG"
        and obs["sites_and_sample_identities_verified"] is True and obs["source_observation_ref"]==sim["observation_ref"],
        "Exact shared observed identity/roles and all original sites")
    preprocess_ref=infer["unary_preprocessing_ref"]
    need(obs["unary_preprocessing_ref"]==preprocess_ref,"Same completed inference count provenance")
    prep=read(preprocess_ref)
    need(prep["schema"]=="american_c3_explicit_unary_collapse_v1"
        and prep["all_observed_sites_and_genotypes_verified"] is True and prep["all_sample_identities_preserved"] is True
        and len(prep["collapsed_sample_nodes"])==len(set(prep["collapsed_sample_nodes"]))==1600
        and prep["sequence_length_bp"]==30000000 and type(prep["num_sites"]) is int and prep["num_sites"]==sim["sites"]>0
        and all(prep[k] is False for k in ("filter_sites","filter_individuals","filter_populations","keep_unary",
            "trimming_performed","gap_splitting_performed","allow_unary_bypass")),"Actual retained-site/sample count from genuine producer metadata")
    native_cfg=read(focal_spec["config_ref"])
    need(focal_spec["source_ref"]==pins["original_native_paths_source"]
        and native_cfg["observation_ref"]==observation_ref and native_cfg["query_focal_count"]==10000
        and native_cfg["path_sampling_seed"]==93001
        and focal["status"]=="completed_native_C3_path_preparation" and focal["observation_ref"]==observation_ref
        and focal["parent_id"]==PARENT and focal["query_focal_count"]==10000
        and focal["class_order"]==["EUR","EAS","AFR"] and focal["query_ancestry_truth_opened"] is False
        and focal["contrastive_labels_reference_only"] is True and focal["query_embedding_view_prespecified_focals_only"] is True
        and focal["native_split_seed"]==42 and focal["native_train_fraction"]==.9
        and focal["focals_ref"]["path"]==BASE+"/native_paths/focals.npz",
        "Original actual same-observation query-only seed93001 shared10k focal registry")
    resource_ref=pins["pure_resource_source"]
    ms=importlib.util.spec_from_file_location("_FLARE_original_pure_bytes",ROOT/resource_ref["path"])
    resource=importlib.util.module_from_spec(ms);ms.loader.exec_module(resource)
    plan=resource.plan_v1(prep["num_sites"],600,1000,10000)
    need(plan["conversion_original_formula_bytes"]+524288<=CAPS["convert"]
        and plan["native_original_formula_with1GiB_tmp_bytes"]+524288<=CAPS["native"]
        and plan["seal_original_formula_without_optional_tract_TSV_bytes"]+524288<=CAPS["seal"],
        "Actual full-site original formulas must fit unchanged caps")
    folder=local(args.output_packet);need(not folder.exists(),"Fresh exclusive late-bound target packet")
    folder.mkdir(parents=True)
    proof=save(folder,"prerequisite_receipt.json",dict(schema="FLARE_VAL000_actual_prerequisite_receipt_v1",
        root_authorizer="root_agent",source_ref=ref(SOURCE),frozen_refs_ref=PINS_REF,
        simulation_completion_ref=pins["simulation_completion"],simulation_actual_wait_ref=pins["simulation_wait"],
        inference_completion_ref=infer_ref,inference_actual_wait_ref=infer_wait_ref,
        focal_completion_ref=focal_ref,focal_actual_wait_ref=focal_wait_ref,
        unary_preprocessing_ref=preprocess_ref,shared_contract_ref=pins["shared_contract"],
        one_shared_focal_registry_ref=focal["focals_ref"],payloads_opened=False,truth_opened=False,
        all_method_predictions_still_required_before_truth=True,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    count_ref=save(folder,"root_counts.json",dict(schema="American_C3_full30Mb_observation_count_root_v1",
        root_authorizer="root_agent",source_observation_ref=observation_ref,tree_ref=obs["tree_ref"],
        num_samples=1600,num_sites=prep["num_sites"],sequence_length=30000000,truth_opened=False,
        count_evidence_kind="actual_completed_inference_producer_JSON_metadata_no_tree_payload_opened",
        count_evidence_ref=preprocess_ref,prerequisite_receipt_ref=proof))
    target=save(folder,"target.json",dict(schema="FLARE_root_scientific_target_v1",parent_id=PARENT,
        scientific_role=ROLE,observation_ref=observation_ref,observation_completion_ref=infer_ref,
        focal_completion_ref=focal_ref,focal_registry_ref=focal["focals_ref"],count_manifest_ref=count_ref,
        simulation_config_ref=pins["simulation_config"],external_truth_ref=None,
        shared_evaluation_contract_ref=pins["shared_contract"],actual_prerequisite_receipt_ref=proof,
        source_only_target_metadata=True,score_status="pending_no_truth_opened"))
    return dict(status="actual_VAL000_target_bound_not_admitted",target_ref=target,count_manifest_ref=count_ref,
        shared_focal_registry_ref=focal["focals_ref"],sites=prep["num_sites"],samples=1600,reference_haplotypes=600,
        query_haplotypes=1000,focal_count=10000,resource_plan=plan,scientific_role=ROLE,score_status="pending_no_truth_opened")
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--build",action="store_true")
    p.add_argument("--output-packet",default=HERE+"/bound_target_v1")
    for name in ("inference-completion","inference-wait","focal-completion","focal-wait"):
        p.add_argument("--expected-"+name+"-sha256")
    args=p.parse_args()
    if not args.build:
        print(json.dumps(dict(status="source_only_late_binding_no_target_or_admission",
            prerequisite_stages=["actual_infer_wait0","actual_native_paths_wait0"],parent_id=PARENT,
            shared_reference_haplotypes=600,shared_query_haplotypes=1000,shared_focal_count=10000)));return 0
    print(json.dumps(build(args),sort_keys=True),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
