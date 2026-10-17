"""Concrete one-parent TRAIN001 simulator context over the reviewed CPU supervisor.

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
REL="research/reviews/frozen_benchmark_training_ready_20261017_v1/train001_simulate_v2"
OLD="research/reviews/American_C3_VAL003004_current_Census_preparation_source_20261016_v1"
MANIFEST=REL+"/train001_simulate_source_manifest.v1.json"
DRIVER=REL+"/train001_simulate_driver_v1.py"
WORKER=REL+"/worker_entry_v1.py"
SOURCE=REL+"/train001_simulate_context_v1.py"
BRIDGE=REL+"/american_full30mb_simulator_bridge_v1.py"
PARENT="american_c3_train_001"
PHASE="simulate_train_001"
STAGE="simulate"
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
    c.PHASES=c.CPU_PHASES=(PHASE,);c.STATUSES={PHASE:"completed_C3_simulation_reconstruction"}
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
    need(manifest["schema"]=="frozen_TRAIN001_simulator_source_manifest_v1","Exact one-parent source closure")
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
    initialize()
    q=c.read(record);c.active(q,300)
    need(q.get("schema")=="frozen_TRAIN001_full30Mb_simulator_ROOT_request_v1"
        and q.get("active")is True and q.get("root_authorizer")=="root_agent"
        and q.get("execution_authorized")is True and q.get("GPU_authorized")is False
        and q.get("parent_id")==PARENT and q.get("stage")==STAGE and q.get("phase")==PHASE
        and q.get("fresh_unfinished_parent")is True and q.get("automatic_retry")is False
        and q.get("source_manifest_ref")==c.ref(MANIFEST)
        and q.get("storage_policy_selector_ref")==SELECTOR
        and q.get("query_truth_access_authorized")is False
        and q.get("TRAIN_truth_creation_only")is True
        and q.get("TEST_access_authorized")is False,
        "Exact one-parent source-reviewed ROOT simulation authority")
    review=c.read(q["source_review_ref"])
    need(review.get("verdict")=="SOURCE_PASS"and review.get("independent")is True
        and review.get("blockers")==[]and review.get("source_manifest_ref")==c.ref(MANIFEST),
        "Independent review binds this exact concrete sealed simulator closure")
    directive=c.read(q["root_directive_ref"]);c.active(directive,300)
    need(directive.get("root_authorizer")=="root_agent"and directive.get("active")is True
        and directive.get("parent_id")==PARENT and directive.get("phase")==PHASE
        and directive.get("source_manifest_ref")==c.ref(MANIFEST)
        and directive.get("current_Census_original_reserve_execute_authorized")is True,
        "Explicit actual ROOT admission/execution directive")
    capacity=c.read(q["actual_capacity_ref"]);c.active(capacity,300)
    need(capacity.get("root_authorizer")=="root_agent"and capacity.get("actual_current_observation")is True
        and capacity.get("source_only_estimate")is False and capacity.get("parent_id")==PARENT,
        "ROOT actual current capacity evidence, never a source proposal")
    caps=c.read(q["caps_ref"])
    need(set(caps)=={"reservation_bytes","aux_bytes","private_work_bytes","rss_bytes","wall_seconds","cleanup_seconds","log_bytes",
         "terminal_evidence_margin_seconds","work_cutoff_margin_seconds","CPU_threads"}
        and all(type(v)is int and v>0 for v in caps.values())
        and caps["reservation_bytes"]>=6500000000+caps["aux_bytes"]
        and caps["cleanup_seconds"]==30 and caps["terminal_evidence_margin_seconds"]==150
        and caps["work_cutoff_margin_seconds"]==300 and caps["wall_seconds"]>600
        and caps["CPU_threads"]==8,"Explicit finite ROOT caps preserving original CPU lifecycle")
    ids=q.get("cpu_ids")
    need(isinstance(ids,list)and len(ids)==len(set(ids))==8
        and all(type(i)is int and i>=0 for i in ids)
        and capacity.get("cpu_ids")==ids
        and capacity.get("admissible_new_bytes",0)>=caps["reservation_bytes"]
        and capacity.get("available_owned_RSS_bytes",0)>=caps["rss_bytes"],"Actual bound8CPU/storage/RSS capacity")
    need(isinstance(q.get("scope_id"),str) and q["scope_id"]
        and all(ch.isalnum()or ch in "_-"for ch in q["scope_id"]),
        "One explicit fresh scope identifier")
    need(datetime.fromisoformat(q["expires_at_utc"])>
        datetime.now(timezone.utc)+__import__("datetime").timedelta(seconds=caps["wall_seconds"]+150),
        "Actual authority covers full declared wall plus terminal evidence")
    runtime=c.read(q["runtime_ref"])
    need(runtime["prefix"]==str(ROOT/".runtime/env")and runtime["CUDA_VISIBLE_DEVICES"]=="",
        "Actual existing simulation CPU runtime")
    cfg=q["scientific_config"]
    template=c.read(c.ref(REL+"/american_c3_train_001.simulation.inactive.json"))
    need(set(cfg)==set(template)and cfg.get("execution_enabled")is True
        and all(cfg[k]==v for k,v in template.items()if k!="execution_enabled"),
        "Unchanged full30Mb TRAIN001 simulation science")
    for key,prefix in [("packet_directory",REL+"/ROOT_actual_TRAIN001_"),("output_directory","experiments/results/frozen_TreeLoom_full30Mb_TRAIN001_")]:
        need(isinstance(q.get(key),str)and q[key].startswith(prefix)and c.local(q[key])!=ROOT,
             "Fresh declared one-parent namespaces")
    source_paths()
    need(_active is None or _active==q,"No request/cap/lifetime swap")
    _active=q;c.CAPS=caps;c.RUNTIME_REF=q["runtime_ref"]
    return q,caps,runtime


def direct_roles(request_ref,q,cfgref,closure,runtime):
    roles=[c.ref(SOURCE),c.ref(DRIVER),c.ref(WORKER),c.ref(BRIDGE),c.ref(MANIFEST),
        c.ref(OLD+"/root_contract_v1.py"),c.ref(OLD+"/root_driver_v1.py"),
        c.ref(OLD+"/cpu_slot_0.lock"),c.ref(OLD+"/cpu_slot_1.lock"),
        c.ref("src/arggnn/american_c3_simulate_v1.py"),c.ref("src/arggnn/american_c3_common_v1.py"),
        c.PIDFD,c.OWNED,c.NAMESPACE_REF,c.V4_REF,SELECTOR,CHILD,c.POLICY_REF,
        request_ref,q["caps_ref"],q["runtime_ref"],q["root_directive_ref"],q["actual_capacity_ref"],
        q["source_review_ref"],cfgref,closure,
        c.ref(REL+"/american_c3_train_001.simulation.inactive.json"),
        c.ref("research/reviews/ARGformer_American_C3_implementation_20261015_v1/parent_seed_registry_inactive.json"),
        c.ref("research/reviews/ARGformer_American_C3_implementation_20261015_v1/simulation_full30mb_inactive.json")]
    return roles


def input_spec(record):
    s=c.read(record);q,caps,_=validate_request(s["root_stage_request_ref"]);c.active(s,300)
    packet,output=phase_paths(PHASE);storage=c.read(s["refs"]["storage"])
    need(record==c.ref(packet/"input.json")and s["root_driver_ref"]==c.ref(DRIVER)
        and s["parent_id"]==PARENT and s["phase"]==PHASE and s["stage"]==STAGE
        and s["GPU_authorized"]is False and s["storage"]==storage
        and s["output_directory"]==output and s["runtime_ref"]==q["runtime_ref"]
        and s["created_at_utc"]==q["created_at_utc"]and s["expires_at_utc"]==q["expires_at_utc"],
        "Exact actual one-shot current-Census admitted simulator input")
    need(storage["reservation"]["reserved_bytes"]==caps["reservation_bytes"]
        and storage["completed_overlay_policy_ref"]==c.POLICY_REF
        and storage["budget_components"]["consumer_adapter"]==CHILD,
        "Unchanged full current child reservation and policy")
    sources={path:c.ref(path)["sha256"]for path in source_paths()}
    need(s["source_sha256"]==sources,"Complete source closure unchanged")
    closure=c.read(s["transitive_source_closure_ref"])
    need(closure["source_sha256"]==sources and closure["root_stage_request_ref"]==s["root_stage_request_ref"]
        and closure["bindings"]==q["bindings"]and closure["native_MLM_dependency"]is False,
        "Exact complete actual source/input closure")
    needed=direct_roles(s["root_stage_request_ref"],q,s["refs"]["config"],s["transitive_source_closure_ref"],s["runtime_ref"])
    need(len(storage["reservation"]["consumer_bindings"])<=64
        and all(r in storage["reservation"]["consumer_bindings"]for r in needed),
        "Every actual direct role under unchanged64 ceiling")
    need(c.read(s["refs"]["config"])==q["scientific_config"]
        and q["bindings"]=={"parent_id":PARENT,"scientific_config":q["scientific_config"]},
        "Exact activated immutable simulator configuration")
    return s,q,dict(q["scientific_config"])


def validate_completion(value,q):
    need(value["status"]=="completed_C3_simulation_reconstruction"and value["parent_id"]==PARENT
        and value["sample_unit"]=="haploid"and value["class_order"]==["EUR","EAS","AFR"],
        "Original full simulator completion and independent parent")
    obs=c.read(value["observation_ref"])
    need(obs["parent_id"]==PARENT and obs["sequence_length_bp"]==30000000
        and obs["counts_haploid"]=={"AFR":200,"EUR":200,"ASIA":200,"ADMIX":1000}
        and obs["query_ancestry_truth_in_inputs"]is False,
        "Complete30Mb/all1600 observation; no cropping or context loss")
