"""Concrete full30Mb TRAIN001 observation over the unchanged reviewed CPU supervisor.

Imports only pinned metadata/helpers until ROOT selects admission or execution.
No new quota, accounting or lifecycle implementation. The original supervisor
execute(), owned gate, pidfd/subreaper, OS waits and cleanup remain unchanged.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import json
import sys

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/frozen_TRAIN001_019_preparation_pipeline_source_20261017_v2"
OLD="research/reviews/American_C3_VAL003004_current_Census_preparation_source_20261016_v1"
MANIFEST=REL+"/source_manifest.v1.json"
DRIVER=REL+"/worker_entry_v1.py"
WORKER=DRIVER
SOURCE=DRIVER
BRIDGE=DRIVER
SCIENCE=REL+"/source_manifest.v1.json"
MODULE=SOURCE[:-3].replace("/",".")
PARENT=PHASE=STAGE=None
ALLOWED_PARENTS=tuple("american_c3_train_%03d"%i for i in range(1,20))
SIM_BATCH_SOURCE="research/reviews/frozen_TRAIN002_019_batch_continuation_source_20261017_v3/worker_entry_v1.py"
SIM_BATCH_SOURCE_SHA="2c4a976949fe147238ba775fa6aee58bbb782a9cef2c76d65945a472eae6be73"
PATHS="src/arggnn/american_c3_paths_v1.py"
AUTHOR=".runtime/author_sources/argformer_f7545be7bca0/src/data/prepare_data_pretrain.py"
IDENTITY_SOURCE="src/arggnn/american_c3_gnn_targets_v1.py"
SELECTOR={"path":"research/reviews/Census_observation_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json","sha256":"fd05d8934ed29eed39f7e60121a33e48fbeb0106991e76690664ca96355692e4"}
CHILD={"path":"research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py","sha256":"72f6dc51844b2a261b581f5e69bd2a22982e6244c059f24549747a7b851faeeb"}
OLD_CONTRACT_SHA="507b364df4bb697d2f0e56617502de461305fa18bb4528e1e8204894579dcb71"
OLD_DRIVER_SHA="85d3da989e132c2ef972d0e4091608befc21c9e1926ca8f95b13d566409cda8b"
c=d=None
_guard=None
_memo=None
_active=None


def need(ok,message):
    if not ok:raise ValueError(message)


def _load_initial(relative,expected,name):
    import hashlib
    path=ROOT/relative
    need(hashlib.sha256(path.read_bytes()).hexdigest()==expected,"Pinned unchanged CPU framework source")
    sp=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(sp);sys.modules[name]=module;sp.loader.exec_module(module)
    return module


def initialize():
    global c,d
    if c is not None:return c,d
    c=_load_initial(OLD+"/root_contract_v1.py",OLD_CONTRACT_SHA,"_frozen30_train001_contract")
    c.REL=REL;c.SOURCE=SOURCE;c.DRIVER=DRIVER;c.WORKER=c.ENTRY=WORKER;c.MANIFEST=MANIFEST
    c.PARENT=PARENT;c.STAGE=STAGE;c.PHASE=PHASE;c.PARENTS=(PARENT,);c.STAGES=(STAGE,)
    c.PHASES=c.CPU_PHASES=(PHASE,);c.STATUSES={PHASE:stage_status(STAGE)}
    pool=c.read(c.ref(MANIFEST))["condition"]["synchronization_pool"]
    need(pool["slot_paths"]==[OLD+"/cpu_slot_0.lock",OLD+"/cpu_slot_1.lock"]
        and pool["source_contract_ref"]=={"path":OLD+"/root_contract_v1.py","sha256":OLD_CONTRACT_SHA}
        and pool["maximum_producers"]==2 and pool["synchronization_nodes_are_not_source_bindings"] is True
        and pool["runtime_empty_regular_inode_and_shared_OFD_exclusion_required"] is True,
        "Exact original shared synchronization pool bound by allowed source/JSON metadata")
    selected=c.read(SELECTOR)
    need(selected["source_ref"]==selected["selector_source_ref"]==CHILD,"Exact active ROOT current Census child")
    c.GUARD_REF=CHILD;c.POLICY_REF=selected["policy_ref"];c.SELECTOR_REF=SELECTOR
    c.source_paths=source_paths;c.validate_request=validate_request;c.phase_paths=phase_paths
    c.make_config=lambda request,caps:dict(request["scientific_config"])
    c.direct_roles=direct_roles;c.input_spec=input_spec
    c.original_guard=lambda:guard(False)
    c.consumer_guard=lambda:(guard(True),_memo)
    c.guard=guard
    # Preserve the existing two-producer OFD pool shared by original preparation.
    old_slot_source=(ROOT/OLD/"root_contract_v1.py").read_text()
    import ast
    node=next(n for n in ast.parse(old_slot_source).body if isinstance(n,ast.FunctionDef)and n.name=="cpu_slot")
    snippet="\n".join(old_slot_source.splitlines()[min([node.lineno]+[v.lineno for v in node.decorator_list])-1:node.end_lineno])
    snippet=snippet.replace('REL+"/cpu_slot_"','"'+OLD+'"+"/cpu_slot_"')
    exec(compile(snippet,OLD+"/root_contract_v1.py#unchanged-shared-slot","exec"),c.__dict__)
    d=_load_initial(OLD+"/root_driver_v1.py",OLD_DRIVER_SHA,"_frozen30_train001_driver")
    d.c=c
    for key,value in dict(ROOT=ROOT,HERE=REL,SOURCE=DRIVER,PIDFD=c.PIDFD,OWNED=c.OWNED,
            need=c.require,local=c.local,ref=c.ref,read=c.read,load=c.load,put=c.put,current=c.current,guard=guard).items():
        setattr(d,key,value)
    d.validate_completion=validate_completion
    return c,d


def source_paths():
    manifest=c.read(c.ref(MANIFEST))
    need(manifest["schema"]=="TreeLoom_TRAIN001_019_preparation_source_manifest_v1","Exact one-parent source closure")
    for r in manifest["refs"]:c.raw(r)
    return tuple(r["path"] for r in manifest["refs"])


def phase_paths(phase):
    need(_active is not None and phase==PHASE,"ROOT request selected before path use")
    return c.local(_active["packet_directory"]),_active["output_directory"]


def guard(consumer=True):
    global _guard,_memo
    need(type(consumer)is bool and _active is not None,"Explicit ROOT current selector lifetime")
    if _guard is None:
        module=c.load(CHILD,"_frozen30_train001_current_Census")
        _guard=module.load_selected_guard_v1(SELECTOR)
        need(_guard.MODE=="Census_observation_two_scope_retention_v1","Unchanged current child")
    if consumer and _memo is None:
        m=c.load(c.NAMESPACE_REF,"_frozen30_train001_readonly_namespace")
        _memo=m.install_consumer_v1(_guard.budget.v4,_guard.budget,consumer_role="read_only_consumer",
            expected_source_path=ROOT/c.V4_REF["path"],expected_source_sha256=c.V4_REF["sha256"])
    return _guard


def validate_request(record):
    global _active
    initialize();q=c.read(record);c.active(q,300)
    authority=validate_pipeline_authority(q["pipeline_authority_ref"])
    need(q.get("schema")=="TreeLoom_TRAIN_preparation_ROOT_request_v1"
        and q.get("active")is True and q.get("root_authorizer")=="root_agent"
        and q.get("execution_authorized")is True and q.get("GPU_authorized")is False
        and q.get("parent_id")==PARENT and q.get("stage")==STAGE and q.get("phase")==PHASE
        and q.get("fresh_unfinished_stage")is True and q.get("automatic_retry")is False
        and q.get("source_manifest_ref")==c.ref(MANIFEST) and q.get("storage_policy_selector_ref")==SELECTOR
        and q.get("query_truth_access_authorized")is False and q.get("GNN_optimization_authorized")is False
        and q.get("TEST_access_authorized")is False
        and q.get("current_Census_original_reserve_execute_authorized")is True
        and q.get("batch_control_input_ref")==authority["simulation_batch_control_input_ref"],
        "Exact finite ROOT original TRAIN observation/native preparation authority")
    review=c.read(q["source_review_ref"])
    need(review.get("verdict")=="SOURCE_PASS"and review.get("independent")is True
        and review.get("blockers")==[]and review.get("source_manifest_ref")==c.ref(MANIFEST),
        "Exact independent pipeline source review")
    condition=c.read(c.ref(MANIFEST))["condition"];caps=q["caps"]
    need(caps==condition["stage_caps"][STAGE]and all(type(v)is int and v>0 for v in caps.values())
        and caps["reservation_bytes"]>=condition["stage_templates"][STAGE]["output_cap_bytes"]+caps["aux_bytes"]
        and caps["cleanup_seconds"]==30 and caps["terminal_evidence_margin_seconds"]==150
        and caps["work_cutoff_margin_seconds"]==300 and caps["CPU_threads"]==8,
        "Original science primary plus complete finite charged AUX")
    capacity=q["actual_capacity"];c.active(capacity,300);ids=q.get("cpu_ids")
    need(capacity.get("root_authorizer")=="root_agent"and capacity.get("actual_current_observation")is True
        and capacity.get("source_only_estimate")is False and capacity.get("parent_id")==PARENT
        and isinstance(ids,list)and len(ids)==len(set(ids))==8 and all(type(i)is int and i>=0 for i in ids)
        and capacity.get("cpu_ids")==ids and capacity.get("admissible_new_bytes",0)>=caps["reservation_bytes"]
        and capacity.get("available_owned_RSS_bytes",0)>=caps["rss_bytes"],"Genuine fresh actual8CPU/storage/RSS capacity")
    need(isinstance(q.get("scope_id"),str)and q["scope_id"]
        and all(ch.isalnum()or ch in "_-"for ch in q["scope_id"]),"Fresh stage scope")
    need(datetime.fromisoformat(q["expires_at_utc"])>datetime.now(timezone.utc)
        +__import__("datetime").timedelta(seconds=caps["wall_seconds"]+150)
        and datetime.fromisoformat(q["expires_at_utc"])<=datetime.fromisoformat(authority["expires_at_utc"]),
        "Finite parent authority contains original work and real terminal evidence")
    need(q["runtime_ref"]==condition["stage_runtime_refs"][STAGE],"Exact qualified stage runtime")
    runtime=c.read(q["runtime_ref"])
    need(runtime["prefix"]==str(ROOT/(".runtime/envs/biological_tsinfer_v1"if STAGE=="infer"else".runtime/env"))
        and runtime["CUDA_VISIBLE_DEVICES"]=="","Original stage-specific CPU runtime")
    prior=predecessor_capsule(q["prior_capsule_ref"],condition)
    cfg=scientific_config(PARENT,STAGE,prior["observation_ref"])
    need(q["scientific_config"]==cfg and q["bindings"]==dict(parent_id=PARENT,stage=STAGE,
        scientific_config=cfg,prior_capsule_ref=q["prior_capsule_ref"])and q["native_MLM_dependency"]is False,
        "Original unchanged stage science and every genuine predecessor logical role")
    need(q["source_sha256"]=={p:c.ref(p)["sha256"]for p in source_paths()},"Complete source closure")
    prefix=REL+"/ROOT_actual_"+PARENT.removeprefix("american_c3_").upper()+"_"+STAGE+"_"
    output="experiments/results/frozen_TreeLoom_full30Mb_"+PARENT.removeprefix("american_c3_").upper()+"_"+STAGE+"_"
    need(q["packet_directory"].startswith(prefix)and q["output_directory"].startswith(output)
        and c.local(q["packet_directory"])!=ROOT and c.local(q["output_directory"])!=ROOT,"Fresh original whole-parent stage namespaces")
    need(_active is None or _active==q,"No lifetime or request swap")
    _active=q;c.CAPS=caps;c.RUNTIME_REF=q["runtime_ref"]
    return q,caps,runtime


def direct_roles(request_ref,q,cfgref,closure,runtime):
    need(closure==request_ref,"Actual complete immutable source/input request")
    return [c.ref(SOURCE),c.ref(MANIFEST),c.ref(OLD+"/root_contract_v1.py"),c.ref(OLD+"/root_driver_v1.py"),
        *[c.ref(p)for p in("src/arggnn/american_c3_observe_v3.py","src/arggnn/american_c3_simulate_v1.py",
         "src/arggnn/american_c3_common_v1.py",PATHS,AUTHOR,IDENTITY_SOURCE)],
        c.PIDFD,c.OWNED,c.NAMESPACE_REF,c.V4_REF,SELECTOR,CHILD,c.POLICY_REF,request_ref,runtime,
        q["source_review_ref"],cfgref,q["prior_capsule_ref"],c.ref(SIM_BATCH_SOURCE),q["batch_control_input_ref"],
        q["pipeline_authority_ref"]]


def input_spec(record):
    s=c.read(record);q,caps,_=validate_request(s["root_stage_request_ref"]);c.active(s,300)
    packet,output=phase_paths(PHASE);storage=c.read(s["refs"]["storage"])
    need(record==c.ref(packet/"input.json")and s["root_driver_ref"]==c.ref(DRIVER)
        and s["parent_id"]==PARENT and s["phase"]==PHASE and s["stage"]==STAGE
        and s["GPU_authorized"]is False and s["storage"]==storage and s["output_directory"]==output
        and s["runtime_ref"]==q["runtime_ref"]and s["created_at_utc"]==q["created_at_utc"]
        and s["expires_at_utc"]==q["expires_at_utc"]and s["caps_ref"]==s["root_stage_request_ref"]
        and s["caps_locator"]==["caps"],"Exact actual one-shot current-Census observation input")
    need(storage["reservation"]["reserved_bytes"]==caps["reservation_bytes"]
        and storage["completed_overlay_policy_ref"]==c.POLICY_REF
        and storage["budget_components"]["consumer_adapter"]==CHILD,"Exact current child whole-scope reservation")
    sources={p:c.ref(p)["sha256"]for p in source_paths()}
    need(s["source_sha256"]==sources==q["source_sha256"]
        and s["transitive_source_closure_ref"]==s["root_stage_request_ref"],
        "Complete actual ROOT request is immutable source/input closure")
    needed=direct_roles(s["root_stage_request_ref"],q,s["refs"]["config"],s["transitive_source_closure_ref"],s["runtime_ref"])
    need(len(storage["reservation"]["consumer_bindings"])<=64
        and all(r in storage["reservation"]["consumer_bindings"]for r in needed),
        "Every actual direct source and predecessor metadata role within original64 ceiling")
    need(c.read(s["refs"]["config"])==q["scientific_config"],"Verbatim original activated observation cfg")
    return s,q,dict(q["scientific_config"])


def validate_completion(value,q):
    need(value["status"]==stage_status(STAGE)and value.get("parent_id",PARENT)==PARENT,"Original stage completion")
    if STAGE=="infer":
        condition=c.read(c.ref(MANIFEST))["condition"]
        need(value["unary_preprocessing_policy"]==condition["stage_templates"]["infer"]["unary_preprocessing_policy"]
            and value["tsinfer_num_threads"]==8,"Unchanged full observed genotype inference")
        obs=observation_metadata(value["observation_ref"])
        prior=predecessor_capsule(q["prior_capsule_ref"],condition)
        need(obs["genealogy_kind"]=="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG"
            and obs["sites_and_sample_identities_verified"]is True and obs["source_observation_ref"]==prior["observation_ref"]
            and obs["migrations_and_internal_population_labels_removed"]is True,"Complete inferred observations, no simulated-tree input")
    else:
        need(value["observation_ref"]==q["scientific_config"]["observation_ref"]
            and value["query_focal_count"]==10000 and value["native_split_seed"]==42
            and value["native_train_fraction"]==.9 and value["query_ancestry_truth_opened"]is False
            and value["contrastive_labels_reference_only"]is True,"Unchanged original author native-path corpus and label-blind10000focals")


def observation_metadata(record):
    m=read(record)
    need(m["schema"]=="american_c3_observed_v1"and m["parent_id"]==PARENT and m["split"]=="train"
        and m["sequence_length_bp"]==30000000 and m["sample_unit"]=="haploid"and m["class_order"]==["EUR","EAS","AFR"]
        and m["counts_haploid"]==dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000)
        and m["query_ancestry_truth_in_inputs"]is False,
        "Original complete30Mb/1600 observation metadata; no ancestry truth features")
    return m

def stage_status(stage):
    need(stage in("infer","native_paths"),"Only original observation/native preparation")
    return "completed_current_C3_inference"if stage=="infer"else"completed_native_C3_path_preparation"

def registered_row(parent):
    need(parent in ALLOWED_PARENTS,"Only original TRAIN001..019, reuse TRAIN000 separately")
    rows=read(metadata_ref(MANIFEST))["condition"]["registry"]["parents"]
    matches=[r for r in rows if r["parent_id"]==parent]
    need(len(matches)==1 and matches[0]["split"]=="train","Original registered independent TRAIN parent")
    return dict(matches[0])

def scientific_config(parent,stage,observation_ref):
    row=registered_row(parent);condition=read(metadata_ref(MANIFEST))["condition"]
    cfg=dict(condition["stage_templates"][stage],execution_enabled=True,observation_ref=observation_ref)
    if stage=="native_paths":cfg["path_sampling_seed"]=row["path_sampling_seed"]
    return cfg

def validate_pipeline_authority(record):
    a=read(record);now=datetime.now(timezone.utc)
    start=datetime.fromisoformat(a["created_at_utc"]);end=datetime.fromisoformat(a["expires_at_utc"])
    need(start.tzinfo and end.tzinfo and start<=now<end and 0<(end-start).total_seconds()<=172800,
        "One finite nonfuture ROOT preparation authority up to48hours")
    need(a.get("schema")=="TreeLoom_TRAIN001_019_finite_preparation_authority_v1"
        and a.get("root_authorizer")=="root_agent"and a.get("active")is True and a.get("source_only")is False
        and a.get("source_manifest_ref")==metadata_ref(MANIFEST)
        and a.get("allowed_parents")==list(ALLOWED_PARENTS)
        and a.get("stages")==["infer","native_paths"]and a.get("TRAIN001_inference_reuse_required")is True
        and a.get("maximum_producers")==2 and a.get("maximum_open_unit_reservations")==2
        and a.get("stage_reservation_bytes")==dict(infer=13000000000,native_paths=16500000000)
        and a.get("admin_reservation_bytes")==33554432
        and a.get("automatic_retry")is False and a.get("GPU_authorized")is False
        and a.get("query_truth_access_authorized")is False and a.get("TEST_access_authorized")is False
        and a.get("GNN_optimization_authorized")is False and a.get("preserve_original_science_and_accounting")is True
        and a.get("reuse_original_simulation_accounting_coordinator")is True
        and a.get("new_unit_terminal_semantics")=="actual_direct_OS_wait_and_owned_drain_no_UI_terminal_claim",
        "Exact original science, shared slots and serialized ordinary accounting")
    sim=coordinator_support();control=read(a["simulation_batch_control_input_ref"])
    sim.validate_batch_authority(control["authority_ref"])
    need(datetime.fromisoformat(a["expires_at_utc"])<=datetime.fromisoformat(read(control["authority_ref"])["expires_at_utc"]),
        "Shared coordinator authority outlives all delegated mutations")
    return a

def select_parent_from_record(record):
    global PARENT,STAGE,PHASE
    value=read(record);q=value
    if value.get("schema")=="TreeLoom_TRAIN_preparation_admitted_input_v1":q=read(value["root_stage_request_ref"])
    parent=q.get("parent_id");stage=q.get("stage");registered_row(parent);stage_status(stage)
    need(not(parent=="american_c3_train_001"and stage=="infer"),"Reuse genuine TRAIN001 observation, never repeat")
    need(q.get("schema")=="TreeLoom_TRAIN_preparation_ROOT_request_v1"
        and q.get("source_manifest_ref")==metadata_ref(MANIFEST),"Exact selected parameterized request")
    need(PARENT in(None,parent)and STAGE in(None,stage)and c is None,"No context reuse across unit")
    PARENT=parent;STAGE=stage;PHASE=stage+"_"+parent.removeprefix("american_c3_")
    return parent

def coordinator_support():
    return _load_initial(SIM_BATCH_SOURCE,SIM_BATCH_SOURCE_SHA,"_original_simulation_v3_accounting_support")
def accounting_transaction(record):
    return coordinator_support().accounting_transaction(record)

def canonical_ref_shape(r):
    need(type(r)is dict and set(r)=={"path","sha256"}and isinstance(r["path"],str)
        and isinstance(r["sha256"],str)and len(r["sha256"])==64
        and all(ch in"0123456789abcdef"for ch in r["sha256"]),"Original immutable payload handle shape only")
    local(r["path"]);return r

def predecessor_capsule(record,condition):
    capsule=read(record)
    need(capsule.get("schema")=="TreeLoom_preparation_predecessor_capsule_v1"
        and capsule.get("parent_id")==PARENT and capsule.get("next_stage")==STAGE
        and capsule.get("query_truth_opened")is False,"Exact same-parent immutable original predecessor capsule")
    refs=capsule["refs"];values=capsule["metadata"]
    required={"input_ref","config_ref","completion_ref","wait_ref","outer_ref","observation_ref","roster_ref"}
    need(set(refs)==required and set(values)==required,"All seven predecessor logical roles retained")
    for key in required:need(read(refs[key])==values[key],"Every original immutable predecessor metadata authenticated, never replaced")
    inp=values["input_ref"];cfg=values["config_ref"];done=values["completion_ref"];wait=values["wait_ref"];outer=values["outer_ref"]
    predecessor_request=read(inp["root_stage_request_ref"])
    previous="simulate"if STAGE=="infer"else"infer"
    expected_manifest=(condition["original_simulation_batch_manifest_ref"]if previous=="simulate"else
        condition["original_inference_manifest_ref"]if PARENT=="american_c3_train_001"else metadata_ref(MANIFEST))
    expected_sources={r["path"]:r["sha256"]for r in read(expected_manifest)["refs"]}
    need(inp["source_sha256"]==predecessor_request["source_sha256"]==expected_sources
        and predecessor_request["source_manifest_ref"]==expected_manifest,
        "Exact complete immutable original predecessor manifest/source closure")
    need(inp["parent_id"]==done.get("parent_id",PARENT)==PARENT and inp["stage"]==previous
        and inp["phase"]==previous+"_"+PARENT.removeprefix("american_c3_")
        and inp["GPU_authorized"]is False and inp["query_truth_access_authorized"]is False
        and inp["TEST_access_authorized"]is False and inp["refs"]["config"]==refs["config_ref"]
        and cfg==read(inp["root_stage_request_ref"])["scientific_config"]
        and refs["input_ref"]==metadata_ref(str(pathlib_packet(inp)/"input.json"))
        and refs["completion_ref"]==metadata_ref(inp["output_directory"]+"/completed.json")
        and refs["wait_ref"]==metadata_ref(inp["worker_auxiliary_scope"]+"/actual_wait.json"),
        "Exact original actual admitted predecessor input/config/completion/wait")
    need(wait["schema"]=="American_C3_full_original_CPU_actual_wait_v1"and wait["input_ref"]==refs["input_ref"]
        and wait["completed_ref"]==refs["completion_ref"]and wait["parent_id"]==PARENT
        and wait["phase"]==inp["phase"]and wait["actual_direct_worker_wait"]is True
        and wait["actual_direct_launcher_wait"]is True and type(wait["actual_OS_wait_status"])is int
        and wait["actual_OS_wait_status"]==0 and type(wait["returncode"])is int and wait["returncode"]==0
        and wait["all_owned_processes_and_groups_drained"]is True and wait["operational_failure"]is None
        and wait["cleanup_errors"]==[]and all(x["returncode"]==0 for x in wait["adopted_waits"])
        and bool(wait["group_observations"])and all(x["absent"]is True for x in wait["group_observations"]),
        "Genuine actual workerwait0 and original complete owned drain")
    if outer.get("schema")=="ROOT_batch_unit_actual_OS_terminal_v1":
        need(outer["root_authorizer"]=="root_agent"and outer["actual_supervisor_OS_terminal_consumed"]is True
            and outer["actual_exit_code"]==0 and outer["UI_tool_terminal_claimed"]is False
            and outer["actual_worker_wait_ref"]==refs["wait_ref"]and outer["input_ref"]==refs["input_ref"]
            and outer["completed_ref"]==refs["completion_ref"]and outer["all_owned_processes_and_groups_drained"]is True,
            "Actual controller-consumed supervisorOS0, no enclosing UI claim")
    else:
        need(PARENT=="american_c3_train_001"and STAGE=="native_paths"
            and outer["root_authorizer"]=="root_agent"and outer["actual_terminal_consumed"]is True
            and outer["actual_exit_code"]==0 and outer["input_ref"]==refs["input_ref"]
            and outer["completed_ref"]==refs["completion_ref"]and outer["actual_worker_wait_ref"]==refs["wait_ref"]
            and outer["all_future_writes_permanently_revoked"]is True and outer["output_frozen"]is True,
            "Exact ROOT genuine existing TRAIN001 inferred observation terminal")
    accepted=read(capsule["close_accepted_ref"]);receipt=read(accepted["receipt_ref"])
    need(accepted["parent_id"]==PARENT and accepted["actual_direct_OS_wait0"]is True
        and accepted["all_owned_processes_and_groups_drained"]is True
        and receipt["key"]==inp["storage"]["reservation"]["key"],"Genuine ordinary accepted predecessor close")
    rev=read(capsule["permanent_revocation_ref"])
    need(rev["input_ref"]==refs["input_ref"]and rev["completed_ref"]==refs["completion_ref"]
        and(rev.get("all_original_launch_resume_retry_write_routes_permanently_revoked")is True
            or rev.get("all_future_writes_permanently_revoked")is True)
        and rev.get("direct_wait_ref",refs["wait_ref"])==refs["wait_ref"]
        and rev.get("actual_outer_terminal_ref",refs["outer_ref"])==refs["outer_ref"],"All predecessor write routes permanently revoked")
    m=observation_metadata(refs["observation_ref"])
    need(done["observation_ref"]==refs["observation_ref"]and m["roster_ref"]==refs["roster_ref"],
        "Exact same complete observed predecessor")
    roster=read(refs["roster_ref"])
    need(roster["parent_id"]==PARENT and roster["class_order"]==["EUR","EAS","AFR"]and len(roster["samples"])==1600,
        "Original complete ordered1600observed roster")
    offset=0;nodes=[]
    for pop,count,cls in [("AFR",200,2),("EUR",200,0),("ASIA",200,1),("ADMIX",1000,None)]:
        for j in range(count):
            r=roster["samples"][offset+j]
            need(r["sample_index"]==offset+j and r["sample_id"]==f"{pop}_{j//2:05d}"
                and r["haplotype"]==j%2 and r["age_generations"]==0
                and r["role"]==("query"if pop=="ADMIX"else"pure_source_reference")
                and r["observed_reference_class"]==cls and type(r["observed_node_id"])is int
                and r["observed_node_id"]>=0,"Original exact ordered observed sample identities")
            nodes.append(r["observed_node_id"])
        offset+=count
    need(len(set(nodes))==1600,"Original1600 unique stable observed sample nodes")
    if previous=="simulate":
        sim=coordinator_support();row=registered_row(PARENT)
        need(inp["root_driver_ref"]==metadata_ref(SIM_BATCH_SOURCE)
            and cfg==sim.scientific_config_for_parent(PARENT,True)
            and cfg["ancestry_seed"]==row["ancestry_seed"]and cfg["mutation_seed"]==row["mutation_seed"]
            and done["status"]=="completed_C3_simulation_reconstruction"
            and m["genealogy_kind"]=="simulated_true_topology_sanitized_ORACLE_not_published_inferred_ARG",
            "Unchanged registered full30Mb simulator, genotype inference excludes its topology")
    else:
        expected_source=condition["original_TRAIN001_inference_source_ref"]if PARENT=="american_c3_train_001"else metadata_ref(SOURCE)
        need(inp["root_driver_ref"]==inp["main_source_ref"]==expected_source
            and predecessor_request["source_sha256"].get(expected_source["path"])==expected_source["sha256"],
            "Exact genuine original TRAIN001v2 or exact pipeline inference source")
        need(done["status"]=="completed_current_C3_inference"
            and m["genealogy_kind"]=="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG"
            and m["sites_and_sample_identities_verified"]is True
            and cfg==dict(condition["stage_templates"]["infer"],execution_enabled=True,observation_ref=m["source_observation_ref"]),
            "Original all-site tsinfer/tsdate inference recipe, never oracle fallback")
    return dict(refs)

def pathlib_packet(inp):
    return Path(inp["worker_auxiliary_scope"]).parent

import argparse,hashlib,os,stat,resource,time,fcntl
context=sys.modules[__name__]
CENSUS_CHILD=CHILD
INFERENCE="src/arggnn/american_c3_observe_v3.py"
GENERATOR="src/arggnn/american_c3_simulate_v1.py"
COMMON="src/arggnn/american_c3_common_v1.py"

def local(value):
    p = Path(value)
    p = p if p.is_absolute() else ROOT / p
    need(p != ROOT and p.is_relative_to(ROOT) and ".." not in p.parts
         and p.resolve() == p, "Canonical contained repository path")
    return p

def metadata_ref(value):
    p = local(value)
    s = p.lstat()
    need(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and s.st_size <= 4 * 2**20,
         "Small regular immutable source or JSON metadata")
    return dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def read(record):
    need(isinstance(record, dict) and set(record) == {"path", "sha256"}
         and metadata_ref(record["path"]) == record, "Exact immutable source/metadata ref")
    return json.loads(local(record["path"]).read_text())

class _OriginalWriterCurrentGuard:
    """Delegate every original Writer boundary to the unchanged current guard."""
    def __init__(self, guard, resource_check):
        self.guard = guard
        self.resource_check = resource_check

    def budget_check_v9(self, *args, **kwargs):
        self.resource_check("before_original_Writer_current_Census_boundary")
        result = self.guard.budget_check_v9(*args, **kwargs)
        self.resource_check("after_original_Writer_current_Census_boundary")
        return result

def execute_in_admitted_context_v1(config_ref, *, spec, current_guard,current_guard_source_ref,resource_check):
    need(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site
        and os.environ.get("CUDA_VISIBLE_DEVICES")=="","Exact original CPU-only -B -s runtime")
    cfg=read(config_ref);q=read(spec["root_stage_request_ref"]);condition=read(metadata_ref(MANIFEST))["condition"]
    need(current_guard_source_ref==CENSUS_CHILD and metadata_ref(CENSUS_CHILD["path"])==CENSUS_CHILD
        and callable(resource_check)and callable(getattr(current_guard,"validate_storage_v9",None))
        and callable(getattr(current_guard,"budget_check_v9",None)),"Actual original current guard")
    prior=predecessor_capsule(q["prior_capsule_ref"],condition)
    need(spec.get("root_authorizer")=="root_agent"and spec.get("execution_authorized")is True
        and spec.get("GPU_authorized")is False and spec["source_ref"]==metadata_ref(SOURCE)
        and spec["config_ref"]==config_ref and cfg==scientific_config(PARENT,STAGE,prior["observation_ref"]),
        "Exact real ROOT context and unchanged full scientific stage config")
    required=[metadata_ref(p)for p in(SOURCE,INFERENCE,GENERATOR,COMMON,PATHS,AUTHOR,IDENTITY_SOURCE,SIM_BATCH_SOURCE)]
    required+=[config_ref,spec["runtime_ref"],current_guard_source_ref,q["prior_capsule_ref"],q["batch_control_input_ref"]]
    need(all(r in spec["storage"]["reservation"]["consumer_bindings"]for r in required),"All directly used sources and predecessor capsule bound")
    reserved=spec["storage"]["reservation"]["reserved_bytes"]
    need(type(reserved)is int and reserved==spec["storage"]["reserved_new_bytes"]and reserved>=cfg["output_cap_bytes"],
        "Whole scope charge without changing scientific primary cap")
    resource_check("before_original_scientific_current_Census_validation")
    current_guard.validate_storage_v9(spec,dict(cfg,output_cap_bytes=reserved),spec["storage"],
        source_ref=spec["source_ref"],config_ref=config_ref,runtime_ref=spec["runtime_ref"])
    from arggnn.american_c3_common_v1 import Writer,runtime_report
    runtime=read(spec["runtime_ref"])
    need(runtime.get("prefix")==str(Path(sys.prefix).resolve())and runtime.get("CUDA_VISIBLE_DEVICES")=="",
        "Original exact existing stage runtime")
    if "packages"in runtime:
        actual=runtime_report()["packages"]
        need(all(actual.get(k)==v for k,v in runtime["packages"].items()),"Original qualified scientific package metadata")
    if STAGE=="infer":
        from arggnn import american_c3_observe_v3 as science
        need(Path(science.__file__).resolve()==local(INFERENCE),"Exact unchanged original inference source")
    else:
        from arggnn import american_c3_paths_v1 as science
        need(Path(science.__file__).resolve()==local(PATHS),"Exact unchanged original native paths source")
    writer=Writer(spec,_OriginalWriterCurrentGuard(current_guard,resource_check))
    resource_check("before_original_full30Mb_scientific_execute")
    result=science.execute(cfg,writer)
    resource_check("after_original_full30Mb_scientific_execute")
    if STAGE=="native_paths":publish_focal_seal(read(result),writer)
    return result

def publish_focal_seal(done,writer):
    import ast,numpy as np
    record=done["focals_ref"];canonical_ref_shape(record);p=local(record["path"])
    need(p.stat().st_size<=2**20 and hashlib.sha256(p.read_bytes()).hexdigest()==record["sha256"],
        "Original bounded10000focal registry, no ancestry truth")
    text=local(IDENTITY_SOURCE).read_text();need(metadata_ref(IDENTITY_SOURCE)in
        read(metadata_ref(MANIFEST))["refs"],"Pinned original identity source")
    nodes=ast.parse(text).body
    keep=[n for n in nodes if(isinstance(n,ast.Assign)and any(isinstance(v,ast.Name)and v.id=="IDENTITY"for v in n.targets))
        or(isinstance(n,ast.FunctionDef)and n.name=="identity_sha")]
    need(len(keep)==2,"Exact original IDENTITY tuple and identity_sha helper")
    namespace=dict(np=np,hashlib=hashlib,json=json)
    exec(compile(ast.Module(body=keep,type_ignores=[]),IDENTITY_SOURCE+"#unchanged-focal-identity","exec"),namespace)
    with np.load(p,allow_pickle=False)as z:records={k:z[k]for k in namespace["IDENTITY"]}
    need(str(records["parent_id"])==PARENT and list(records["class_order"])==["EUR","EAS","AFR"]
        and len(records["record_ids"])==10000 and len(np.unique(records["record_ids"]))==10000
        and all(len(records[k])==10000 for k in("sample_nodes","sample_indices","positions_bp0")),
        "Original10000unique same-parent label-blind focal identities")
    identity=namespace["identity_sha"](records)
    return writer.json("focal_seal.json",dict(schema="TreeLoom_original_native_focal_identity_seal_v1",
        parent_id=PARENT,
        observation_ref=done["observation_ref"],focal_registry_ref=record,focal_identity_sha256=identity,
        focal_records=10000,query_truth_opened=False,identity_source_ref=metadata_ref(IDENTITY_SOURCE)))

def admit(request_ref):
    q=read(request_ref);entered=False
    try:
        with accounting_transaction(q["batch_control_input_ref"]):
            entered=True
            age=(datetime.now(timezone.utc)-datetime.fromisoformat(q["created_at_utc"])).total_seconds()
            if age>240:
                return dict(status="QUEUED_BEFORE_OPERATION",execution_started=False,accounting_operation_started=False,
                    immutable_request_refresh_required=True,reason="Actual coordinator acquired after capacity freshness margin")
            return admit_under_coordinator(request_ref)
    except ValueError as exc:
        if not entered and str(exc)=="Finite shared ROOT accounting coordinator wait":
            return dict(status="QUEUED_BEFORE_OPERATION",execution_started=False,accounting_operation_started=False,
                immutable_request_refresh_required=True,reason="Original finite coordinator acquisition window busy")
        raise

def admit_under_coordinator(request_ref):
    c,d=context.initialize()
    q,caps,_=c.validate_request(request_ref);packet,output=c.phase_paths(c.PHASE)
    c.require(not packet.exists()and not c.local(output).exists(),"Fresh exclusive packet/output")
    g=c.original_guard();implementation=c.ref(g.budget.SOURCE_REL);launcher=c.ref(g.budget.LAUNCHER_REL)
    with g.budget.v4.mutex(c.ROOT):
        state=g._state_locked(implementation["sha256"],launcher["sha256"],fresh=True)
    packet.mkdir();config_ref=c.save(packet,"config.json",c.make_config(q,caps))
    sources={p:c.ref(p)["sha256"]for p in c.source_paths()};closure=request_ref
    consumers=c.unique([*state[3].values(),*c.direct_roles(request_ref,q,config_ref,closure,q["runtime_ref"])])
    job="TreeLoom-frozen-full30Mb-"+PARENT.removeprefix("american_c3_").upper()+"-"+STAGE+"-"+q["scope_id"];aux=packet/"worker_aux"
    reserve=dict(key="H041/"+job,output=str(c.ROOT/output),auxiliary_scopes=[str(aux)],
        reserved_bytes=caps["reservation_bytes"],consumer_bindings=consumers)
    authority=g.operation_draft_v1("reserve",reserve,expected_before=g._pins(state[2]),bindings=state[3])
    authority.update(active=True,expires_at_utc=q["expires_at_utc"])
    ar=c.save(packet,"reserve_authority.json",authority)
    receipt=g.apply_reserve_v1(ar,expected_source_sha256=implementation["sha256"],expected_launcher_sha256=launcher["sha256"])
    rr=c.save(packet,"reserve_receipt.json",receipt)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
        accounting_mode=g.MODE,job_id=job,output_directory=output,reserved_new_bytes=caps["reservation_bytes"],
        reservation=receipt["value"],initial_ledger_pins=receipt["initial_ledger_pins"],expires_at_utc=q["expires_at_utc"],
        completed_overlay_policy_ref=c.POLICY_REF,
        budget_components=dict(implementation=implementation,launcher=launcher,consumer_adapter=c.GUARD_REF))
    sr=c.save(packet,"storage.json",storage)
    spec=dict(schema="TreeLoom_TRAIN_preparation_admitted_input_v1",status="active",root_authorizer="root_agent",
        execution_authorized=True,GPU_authorized=False,parent_id=c.PARENT,phase=c.PHASE,stage=c.STAGE,
        job_id=job,output_directory=output,worker_auxiliary_scope=str(aux.relative_to(c.ROOT)),
        root_stage_request_ref=request_ref,root_driver_ref=c.ref(context.DRIVER),main_source_ref=c.ref(context.WORKER),
        source_sha256=sources,transitive_source_closure_ref=closure,caps_ref=request_ref,caps_locator=["caps"],
        refs=dict(config=config_ref,runtime=q["runtime_ref"],storage=sr),runtime_ref=q["runtime_ref"],storage=storage,
        created_at_utc=q["created_at_utc"],expires_at_utc=q["expires_at_utc"],
        query_truth_access_authorized=False,TEST_access_authorized=False,GNN_optimization_authorized=False)
    ir=c.save(packet,"input.json",spec)
    c.save(packet,"admitted.json",dict(input_ref=ir,storage=storage,reserve_receipt_ref=rr,
        driver_ref=c.ref(context.DRIVER),consumer_binding_count=len(consumers),scientific_execution_started=False))
    return dict(input_ref=ir,consumer_binding_count=len(consumers),execution_started=False)

def worker(record,gate_fd):
    c,_=context.initialize()
    c.require(os.read(gate_fd,1)==b"G","No scientific work before actual owned launch gate")
    os.close(gate_fd)
    s,q,cfg=c.input_spec(record)
    c.require(os.environ.get("CUDA_VISIBLE_DEVICES")==""and os.getpid()==os.getpgrp()==os.getsid(0),
        "Actual owned CPU-only session")
    c.require(sys.dont_write_bytecode and sys.flags.no_user_site
        and Path(sys.prefix)==c.ROOT/(".runtime/envs/biological_tsinfer_v1"if STAGE=="infer"else".runtime/env")
        and set(os.sched_getaffinity(0))==set(q["cpu_ids"]),"Exact actual CPU interpreter/affinity")
    fd=int(os.environ["ARGGNN_C3_PREPARATION_SLOT_FD"]);identity=os.fstat(fd)
    locks=[c.ROOT/context.OLD/("cpu_slot_"+str(i)+".lock")for i in range(2)]
    paths=[p for p in locks if(p.lstat().st_dev,p.lstat().st_ino)==(identity.st_dev,identity.st_ino)]
    c.require(len(paths)==1 and identity.st_size==0,"Exact inherited original shared two-producer slot")
    with paths[0].open("r+")as probe:
        try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:
            fcntl.flock(probe,fcntl.LOCK_UN)
            raise ValueError("Original shared slot is not genuinely held")
    g=c.guard(True)
    started=time.monotonic()
    def resource_check(event):
        c.current(s,150)
        c.require(time.monotonic()-started<c.CAPS["wall_seconds"]-300,
            "Original finite scientific cutoff")
        c.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=c.CAPS["rss_bytes"],
            "Actual worker RSS; unchanged outer additionally enforces all owned RSS")
        c.require(os.environ.get("CUDA_VISIBLE_DEVICES")==""and set(os.sched_getaffinity(0))==set(q["cpu_ids"]),
            "Actual CPU-only affinity remains")
        return dict(event=event)
    # The worker is the complete actual context. No absent factory is required.
    bridge=context
    adapted=dict(s,source_ref=c.ref(context.BRIDGE),config_ref=s["refs"]["config"])
    return bridge.execute_in_admitted_context_v1(s["refs"]["config"],
        spec=adapted,current_guard=g,current_guard_source_ref=c.GUARD_REF,
        resource_check=resource_check)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument("--admit-ref");mode.add_argument("--execute-ref");mode.add_argument("--worker-ref")
    p.add_argument("--gate-fd",type=int);a=p.parse_args(argv)
    if not(a.admit_ref or a.execute_ref or a.worker_ref):
        print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",admission=False,execution=False)));return 0
    record=json.loads(a.admit_ref or a.execute_ref or a.worker_ref)
    select_parent_from_record(record)
    c,d=initialize()
    c.require(os.getcwd()==str(ROOT)and sys.dont_write_bytecode and sys.flags.no_user_site,
        "Exact existing ROOT -B -s interpreter and repository cwd")
    if a.admit_ref:result=admit(json.loads(a.admit_ref))
    elif a.worker_ref:
        c.require(a.gate_fd is not None,"Actual original owned gate")
        result=dict(completed_ref=worker(json.loads(a.worker_ref),a.gate_fd))
    else:
        record=json.loads(a.execute_ref);s,q,_=d.input_spec(record)
        c.require(set(q["cpu_ids"])<=os.sched_getaffinity(0),"Actual selected CPUs available to ROOT")
        os.sched_setaffinity(0,set(q["cpu_ids"]))
        with c.cpu_slot()as fd:
            d.SLOT_FD=fd;result=d.execute(record)
    print(json.dumps(result,sort_keys=True),flush=True)
    return result.get("returncode",0)
if __name__=="__main__":raise SystemExit(main())
