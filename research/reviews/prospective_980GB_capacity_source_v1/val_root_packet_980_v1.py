"""Specific original30Mb VAL000 root packet. Default mode is metadata-only.

This file admits no work unless root explicitly selects --admit. It never launches
a worker: --command emits exact unchanged worker argv/environment for root's
existing owned-process supervisor and genuine direct-wait/drain capture.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,importlib.util,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/prospective_980GB_capacity_source_v1"
HERE=ROOT/REL
SOURCE=REL+"/val_root_packet_980_v1.py"
PARENT="american_c3_val_000"
OUTPUT="experiments/results/American_C3_val000_full30mb_shared_v1"
STAGES=("simulate","infer","native_paths")
PRODUCERS={"simulate":"src/arggnn/american_c3_simulate_v1.py","infer":"src/arggnn/american_c3_observe_v3.py","native_paths":"src/arggnn/american_c3_paths_v1.py"}
TEMPLATES={"simulate":"simulation.inactive.json","infer":"inference.inactive.json","native_paths":"native_paths.inactive.json"}
DIRS={"simulate":"simulation","infer":"inference","native_paths":"native_paths"}
COMMON="src/arggnn/american_c3_common_v1.py"
AUTHOR=".runtime/author_sources/argformer_f7545be7bca0/src/data/prepare_data_pretrain.py"
SIM_RUNTIME="research/reviews/American_C3_root_execution_20261015_v1/admitted_full30mb_v2/runtime.json"
INFER_RUNTIME="research/reviews/American_C3_observation_root_execution_20261015_v3/inference_cpu_runtime_metadata.json"
OVERLAY="research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py"
POLICY="research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json"
REPAIR="research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json"
BUDGET="scripts/project_budget_v9.py"
LAUNCHER="scripts/launch_reserved_gpu_v6.py"
def require(ok,message):
    if not ok:raise ValueError(message)
def local(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    require(".." not in p.parts and p.resolve()==p,"Canonical own metadata path")
    p.relative_to(ROOT)
    return p
def ref(path):
    p=local(path)
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(record):
    require(isinstance(record,dict) and set(record)=={"path","sha256"} and ref(record["path"])==record,"Exact source/metadata SHA")
    raw=local(record["path"]).read_bytes()
    require(len(raw)<=1048576,"Bounded source/metadata JSON")
    return json.loads(raw)
def save(folder,name,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    require(len(raw)<=65536,"Compact root metadata")
    with (folder/name).open("xb") as stream:stream.write(raw)
    return ref(folder/name)
def frozen():
    value=read(ref(HERE/"frozen_refs.json"))["refs"]
    require(all(ref(p)==r for p,r in value.items()),"Frozen original source/runtime metadata changed")
    if (ROOT/POLICY).exists():
        policy_ref=ref(POLICY);policy=read(policy_ref)
        require(policy["schema"]=="prospective_980GB_credit_policy_v1" and policy["active"] is True
            and policy["source_ref"]==value[OVERLAY] and policy["scheduling_ceiling_bytes"]==980000000000
            and policy["warning_threshold_bytes"]==950000000000,"Exact root-activated prospective policy")
        value[POLICY]=policy_ref
    return value
def identity(stage):
    require(stage in STAGES,"Only three preparation stages")
    return HERE/("admitted_"+stage+"_v1"),OUTPUT+"/"+DIRS[stage],"American-C3-val000-full30mb-"+stage+"-shared-980-v1"
def upstream_observation(stage,completion_ref,wait_ref):
    require(stage in ("infer","native_paths") and completion_ref is not None and wait_ref is not None,"Genuine prior completion and direct wait required")
    prior="simulate" if stage=="infer" else "infer"
    expected=identity(prior)[1]+"/completed.json"
    require(completion_ref["path"]==expected,"Original VAL000 full30Mb stage lineage only")
    completed=read(completion_ref);wait=read(wait_ref)
    require(wait.get("returncode")==0 and wait.get("actual_direct_worker_wait") is True
        and wait.get("completed_ref")==completion_ref and wait.get("owned_worker_group_ESRCH") is True
        and wait.get("operational_failure") is None,"Root must bind genuine wait0 and group drain")
    expected_status="completed_C3_simulation_reconstruction" if stage=="infer" else "completed_current_C3_inference"
    require(completed["status"]==expected_status,"Successful unchanged upstream API required")
    observation_ref=completed["observation_ref"];m=read(observation_ref)
    require(m["parent_id"]==PARENT and m["split"]=="val" and m["sequence_length_bp"]==30000000
        and m["sample_unit"]=="haploid" and m["class_order"]==["EUR","EAS","AFR"]
        and m["counts_haploid"]==dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000)
        and m["query_ancestry_truth_in_inputs"] is False,"Exact full30Mb original VAL identity/observation")
    if stage=="native_paths":
        require(m["genealogy_kind"]=="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG"
            and m.get("sites_and_sample_identities_verified") is True,"All original sites and ordered samples retained")
    # Do not follow completed.private_truth_ref/private_true_tree_ref or any payload.
    return observation_ref
def configuration(stage,*,completion_ref=None,wait_ref=None,active=False):
    frozen()
    require(stage in STAGES,"Known stage")
    cfg=read(ref(HERE/TEMPLATES[stage]))
    if stage=="simulate":
        require(completion_ref is None and wait_ref is None,"Simulation has no previous scientific input")
        require(cfg["parent_id"]==PARENT and cfg["split"]=="val" and cfg["sequence_length_bp"]==30000000
            and (cfg["ancestry_seed"],cfg["mutation_seed"])==(31001,1031001),"Original registry VAL000 seeds and full extent")
    else:cfg["observation_ref"]=upstream_observation(stage,completion_ref,wait_ref)
    cfg["execution_enabled"]=active
    return cfg
def source_closure(stage,pins):
    paths=[COMMON,PRODUCERS[stage]]
    if stage=="infer":paths.append(PRODUCERS["simulate"])
    if stage=="native_paths":paths.append(AUTHOR)
    return [pins[p] for p in paths]
def stage_plan(stage):
    pins=frozen();packet,output,job=identity(stage)
    return dict(stage=stage,parent_id=PARENT,split="val",sequence_length_bp=30000000,job_id=job,
        output_directory=output,packet_directory=str(packet.relative_to(ROOT)),
        config_template_ref=ref(HERE/TEMPLATES[stage]),source_ref=pins[PRODUCERS[stage]],
        source_closure_refs=source_closure(stage,pins),runtime_ref=pins[INFER_RUNTIME if stage=="infer" else SIM_RUNTIME],
        shared_contract_ref=ref(HERE/"shared_contract.inactive.json"),forecast_ref=ref(HERE/"storage_forecast.json"),
        storage_adapter_ref=pins[OVERLAY],activation_policy_ref=pins.get(POLICY),
        upstream_actual_completion_and_wait0_drained_required=stage!="simulate",
        TEST_access=False,GNN_VAL_optimization=False,query_ancestry_truth_payload_access=False,
        native_model_transfer_from_TRAIN000=False,preparation_is_not_native_training=True)
def admit(stage,policy_ref,*,completion_ref=None,wait_ref=None):
    """Root-only explicit current-overlay admission; never called by source checks."""
    require(Path.cwd().resolve()==ROOT,"Repository execution only")
    pins=frozen();require(POLICY in pins and policy_ref==pins[POLICY],"Exact active root prospective980 policy")
    cfg=configuration(stage,completion_ref=completion_ref,wait_ref=wait_ref,active=True)
    packet,output,job=identity(stage)
    require(not packet.exists() and not local(output).exists(),"Fresh stage, no replay/resume")
    packet.mkdir()
    config_ref=save(packet,"config.json",cfg)
    loader=importlib.util.spec_from_file_location("_C3_VAL_root_current_guard",ROOT/OVERLAY)
    guard=importlib.util.module_from_spec(loader);loader.loader.exec_module(guard)
    guard.configure_v2(pins[REPAIR],policy_ref=policy_ref)
    with guard.budget.v4.mutex(ROOT):state=guard._state_locked(pins[BUDGET]["sha256"],pins[LAUNCHER]["sha256"],fresh=True)
    closure=source_closure(stage,pins);runtime=pins[INFER_RUNTIME if stage=="infer" else SIM_RUNTIME]
    consumers=[*closure,config_ref,runtime,ref(SOURCE),ref(HERE/"frozen_refs.json"),ref(HERE/"shared_contract.inactive.json"),
        ref(HERE/"storage_forecast.json"),ref(HERE/TEMPLATES[stage]),\
        pins["research/reviews/ARGformer_American_C3_implementation_20261015_v1/parent_seed_registry_inactive.json"],*state[3].values(),
        *[pins[p] for p in (OVERLAY,POLICY,REPAIR,"research/reviews/H039_missing_marker_repair_v1/marker_repair_v1.py",
          "research/reviews/H039_failed_scope_capacity_v1/failed_scope_capacity_v1.py",
          "research/reviews/H039_failed_scope_capacity_v2/failed_scope_capacity_v2.py","src/arggnn/project_budget_v9_consumer.py")]]
    if completion_ref is not None:consumers.extend([completion_ref,wait_ref])
    consumers=list({(r["path"],r["sha256"]):r for r in consumers}.values())
    require(len(consumers)<=64,"Existing64 direct source/JSON bound")
    now=datetime.now(timezone.utc);expiry=(now+timedelta(hours=22)).isoformat()
    request=dict(key="H041/"+job,output=str(ROOT/output),auxiliary_scopes=[str(packet/"worker_aux")],
        reserved_bytes=cfg["output_cap_bytes"],consumer_bindings=consumers)
    authority=guard.operation_draft_v1("reserve",request,expected_before=guard._pins(state[2]),bindings=state[3])
    authority.update(active=True,expires_at_utc=expiry)
    authority_ref=save(packet,"reserve_authority.json",authority)
    receipt=guard.apply_reserve_v1(authority_ref,expected_source_sha256=pins[BUDGET]["sha256"],expected_launcher_sha256=pins[LAUNCHER]["sha256"])
    save(packet,"reserve_receipt.json",receipt)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,accounting_mode=guard.MODE,
        job_id=job,output_directory=output,reserved_new_bytes=cfg["output_cap_bytes"],reservation=receipt["value"],
        expires_at_utc=expiry,completed_overlay_policy_ref=policy_ref,initial_ledger_pins=receipt["initial_ledger_pins"],
        budget_components=dict(implementation=pins[BUDGET],launcher=pins[LAUNCHER],consumer_adapter=pins[OVERLAY]))
    spec=dict(root_authorizer="root_agent",hypothesis_id="H041",job_id=job,parent_id=PARENT,split="val",stage=stage,
        execution_authorized=True,GPU_authorized=False,source_ref=pins[PRODUCERS[stage]],source_closure_refs=closure,
        config_ref=config_ref,runtime_ref=runtime,storage_adapter_ref=pins[OVERLAY],marker_repair_authority_ref=pins[REPAIR],
        completed_overlay_policy_ref=policy_ref,shared_evaluation_contract_ref=ref(HERE/"shared_contract.inactive.json"),
        driver_ref=ref(SOURCE),storage=storage,output_directory=output,created_at_utc=now.isoformat(),expires_at_utc=expiry,
        upstream_completion_ref=completion_ref,upstream_actual_wait_ref=wait_ref,
        query_ancestry_truth_payload_access_authorized=False,TEST_access_authorized=False,GNN_optimization_authorized=False)
    admission_ref=save(packet,"admission.json",spec)
    admitted_ref=save(packet,"admitted.json",dict(stage=stage,parent_id=PARENT,admission_ref=admission_ref,driver_ref=ref(SOURCE)))
    return dict(admitted_ref=admitted_ref,admission_ref=admission_ref,projected_bytes=receipt["projected_bytes"],
        warning_active=receipt["warning_active"],warning_message=receipt["warning_message"],
        remaining_to_human_limit_bytes=receipt["remaining_to_human_limit_bytes"])
def command(stage,admission_ref):
    """Print only. Root uses existing owned-session CPU supervisor and direct wait."""
    pins=frozen();spec=read(admission_ref);packet,output,job=identity(stage)
    require(admission_ref["path"]==str((packet/"admission.json").relative_to(ROOT))
        and spec["parent_id"]==PARENT and spec["stage"]==stage and spec["job_id"]==job
        and spec["source_ref"]==pins[PRODUCERS[stage]] and spec["execution_authorized"] is True
        and spec["GPU_authorized"] is False,"Exact root-created VAL stage admission")
    python=ROOT/(".runtime/envs/biological_tsinfer_v1/bin/python" if stage=="infer" else ".runtime/env/bin/python")
    return dict(argv=[str(python),"-B","-m","arggnn."+Path(PRODUCERS[stage]).stem,"--execute","--admission",admission_ref["path"],"--expected-admission-sha256",admission_ref["sha256"]],
        cwd=str(ROOT),environment=dict(CUDA_VISIBLE_DEVICES="",PYTHONPATH=str(ROOT/"src"),PYTHONDONTWRITEBYTECODE="1",
          TMPDIR=str(packet/"worker_aux/work"),TMP=str(packet/"worker_aux/work"),TEMP=str(packet/"worker_aux/work"),
          OMP_NUM_THREADS="8",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMBA_NUM_THREADS="8",
          ARGGNN_COMPLETED_CREDITS_POLICY_REF=json.dumps(spec["completed_overlay_policy_ref"],separators=(",",":"))),
        create_before_launch=[str((ROOT/output).parent),str(packet/"worker_aux/work")],
        supervisor_requirements="Root existing owned-session CPU supervisor;16MiB bounded RAM log;134217728-byte private-work cap;direct wait durably preserved before postprocessing;full owned group drain before boundary check;no deletion or automatic credit.",
        deadlines_seconds=dict(simulate=3600,infer=10800,native_paths=7200)[stage],
        native_RSS_ceiling_bytes=68719476736 if stage=="native_paths" else None)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    m=p.add_mutually_exclusive_group();m.add_argument("--plan",choices=STAGES);m.add_argument("--admit",choices=STAGES);m.add_argument("--command",choices=STAGES)
    p.add_argument("--policy");p.add_argument("--policy-sha256");p.add_argument("--upstream-completion");p.add_argument("--upstream-completion-sha256")
    p.add_argument("--upstream-wait");p.add_argument("--upstream-wait-sha256");p.add_argument("--admission");p.add_argument("--admission-sha256")
    a=p.parse_args()
    def record(path,digest):
        require(bool(path)==bool(digest),"Supply exact path and SHA together")
        return dict(path=path,sha256=digest) if path else None
    if a.admit:
        value=admit(a.admit,record(a.policy,a.policy_sha256),
          completion_ref=record(a.upstream_completion,a.upstream_completion_sha256),wait_ref=record(a.upstream_wait,a.upstream_wait_sha256))
    elif a.command:value=command(a.command,record(a.admission,a.admission_sha256))
    else:value=stage_plan(a.plan) if a.plan else dict(status="inactive_source_packet",stages=[stage_plan(s) for s in STAGES])
    print(json.dumps(value,sort_keys=True,indent=2))
if __name__=="__main__":main()
