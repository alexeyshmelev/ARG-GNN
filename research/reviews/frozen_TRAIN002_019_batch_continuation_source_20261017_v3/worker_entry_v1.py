"""Parameterized original full30Mb TRAIN002..019 simulator over the reviewed CPU supervisor.

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
import contextlib

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/frozen_TRAIN002_019_batch_continuation_source_20261017_v3"
OLD="research/reviews/American_C3_VAL003004_current_Census_preparation_source_20261016_v1"
MANIFEST=REL+"/source_manifest.v1.json"
DRIVER=REL+"/worker_entry_v1.py"
WORKER=DRIVER
SOURCE=DRIVER
BRIDGE=DRIVER
SCIENCE="research/reviews/frozen_benchmark_training_ready_20261017_v1/train001_simulate_v5/scientific_condition.v5.json"
MODULE=SOURCE[:-3].replace("/",".")
ALLOWED_PARENTS=tuple(f"american_c3_train_{i:03d}"for i in range(2,20))
PARENT=None
PHASE=None
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


def validate_batch_authority(record):
    value=read(record);now=datetime.now(timezone.utc)
    start=datetime.fromisoformat(value["created_at_utc"]);end=datetime.fromisoformat(value["expires_at_utc"])
    need(start.tzinfo and end.tzinfo and start<=now<end
        and 0<(end-start).total_seconds()<=172800,"Finite nonfuture ROOT batch authority up to48hours")
    need(value.get("schema")=="ROOT_TRAIN002_019_finite_batch_authority_v1"
        and value.get("root_authorizer")=="root_agent"and value.get("active")is True
        and value.get("execute_registered_simulations_and_ordinary_closes")is True
        and value.get("source_manifest_ref")==metadata_ref(MANIFEST)
        and value.get("allowed_parents")==list(ALLOWED_PARENTS)
        and value.get("maximum_producers")==2
        and value.get("per_parent_reservation_bytes")==7000000000
        and value.get("maximum_open_unit_reservations")==2
        and value.get("batch_admin_reservation_bytes")==33554432
        and value.get("automatic_retry")is False and value.get("GPU_authorized")is False
        and value.get("TRAIN_truth_creation_only")is True
        and value.get("TEST_access_authorized")is False and value.get("query_truth_scoring_authorized")is False
        and value.get("preserve_original_science_and_accounting")is True
        and value.get("ROOT_exclusive_initial_admin_admission_until_coordinator_registered")is True
        and value.get("new_unit_terminal_semantics")=="actual_direct_OS_wait_and_owned_drain_no_UI_terminal_claim",
        "Explicit exact ROOT autonomous future-unit authority")
    return value


@contextlib.contextmanager
def accounting_transaction(control_input_ref):
    """Shared finite OFD coordinator for ROOT and future accounting mutations."""
    control=read(control_input_ref)
    need(control.get("schema")=="ROOT_TRAIN002_019_batch_admitted_input_v1"
        and control.get("source_manifest_ref")==metadata_ref(MANIFEST),
        "Genuine exact admitted batch control input")
    batch=validate_batch_authority(control["authority_ref"])
    record=control["coordinator"]
    need(set(record)=={"path","st_dev","st_ino"}
        and record["path"]==batch["run_directory"]+"/accounting_coordinator.lock"
        and type(record["st_dev"])is int and type(record["st_ino"])is int,
        "Actual batch coordination inode recorded after genuine admin admission")
    p=local(record["path"]);before=p.lstat()
    need(stat.S_ISREG(before.st_mode)and before.st_nlink==1 and before.st_size==0
        and(before.st_dev,before.st_ino)==(record["st_dev"],record["st_ino"]),
        "Original recorded empty singly linked coordinator")
    fd=os.open(p,os.O_RDWR|os.O_NOFOLLOW)
    end=min(time.monotonic()+300,time.monotonic()+datetime.fromisoformat(batch["expires_at_utc"]).timestamp()-time.time()-150)
    try:
        actual=os.fstat(fd)
        need((actual.st_dev,actual.st_ino,actual.st_size,actual.st_nlink)==
            (record["st_dev"],record["st_ino"],0,1),"Same no-follow coordinator OFD")
        while True:
            need(time.monotonic()<end,"Finite shared ROOT accounting coordinator wait")
            try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(.1)
        after=p.lstat()
        need((after.st_dev,after.st_ino,after.st_size,after.st_nlink)==
            (record["st_dev"],record["st_ino"],0,1),"Coordinator path still names held original inode")
        yield
    finally:os.close(fd)


def registered_row(parent):
    need(isinstance(parent,str)and parent in ALLOWED_PARENTS,
        "Only missing original registered TRAIN002..019 are selectable")
    condition=read(metadata_ref(SCIENCE))
    rows=[r for r in condition["original_registry"]["parents"]if r["parent_id"]==parent]
    need(len(rows)==1 and rows[0]["split"]=="train","One original registered TRAIN parent")
    return dict(rows[0])


def scientific_config_for_parent(parent,active=False):
    need(type(active)is bool,"Explicit scientific activation flag")
    row=registered_row(parent);condition=read(metadata_ref(SCIENCE))
    return dict(condition["original_full30mb_template"],parent_id=parent,split="train",
        ancestry_seed=row["ancestry_seed"],mutation_seed=row["mutation_seed"],execution_enabled=active)


def select_parent_from_record(record):
    global PARENT,PHASE
    # Small authenticated JSON only, before original context or guard is loaded.
    value=read(record);parent=value.get("parent_id");registered_row(parent)
    request=value
    if value.get("schema")=="frozen_TRAIN002_019_current_Census_simulator_input_v1":
        request=read(value["root_stage_request_ref"])
        need(request.get("parent_id")==parent and value.get("phase")=="simulate_"+parent.removeprefix("american_c3_")
            and value.get("stage")=="simulate","Actual input retains exact request parent identity")
    need(request.get("schema")=="frozen_TRAIN002_019_full30Mb_simulator_ROOT_request_v1"
        and request.get("parent_id")==parent and request.get("source_manifest_ref")==metadata_ref(MANIFEST),
        "Exact concrete parameterized ROOT request")
    need(PARENT is None or PARENT==parent,"Parent cannot change within one interpreter")
    need(c is None or c.PARENT==parent,"Private original context cannot be reused for another parent")
    PARENT=parent;PHASE="simulate_"+parent.removeprefix("american_c3_")
    return parent


def initialize():
    global c,d
    if c is not None:return c,d
    need(PARENT in ALLOWED_PARENTS and PHASE=="simulate_"+PARENT.removeprefix("american_c3_"),
        "Exact parent selected from authenticated request/input before context initialization")
    c=_load_initial(OLD+"/root_contract_v1.py",OLD_CONTRACT_SHA,"_frozen30_train_continuation_contract")
    c.REL=REL;c.SOURCE=SOURCE;c.DRIVER=DRIVER;c.WORKER=c.ENTRY=WORKER;c.MANIFEST=MANIFEST
    c.PARENT=PARENT;c.STAGE=STAGE;c.PHASE=PHASE;c.PARENTS=(PARENT,);c.STAGES=(STAGE,)
    c.PHASES=c.CPU_PHASES=(PHASE,);c.STATUSES={PHASE:"completed_C3_simulation_reconstruction"}
    pool=c.read(c.ref(SCIENCE))["synchronization_pool"]
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
    d=_load_initial(OLD+"/root_driver_v1.py",OLD_DRIVER_SHA,"_frozen30_train_continuation_driver")
    d.c=c
    for key,value in dict(ROOT=ROOT,HERE=REL,SOURCE=DRIVER,PIDFD=c.PIDFD,OWNED=c.OWNED,
            need=c.require,local=c.local,ref=c.ref,read=c.read,load=c.load,put=c.put,current=c.current,guard=guard).items():
        setattr(d,key,value)
    d.validate_completion=validate_completion
    return c,d


def source_paths():
    manifest=c.read(c.ref(MANIFEST))
    need(manifest["schema"]=="frozen_TRAIN002_019_simulator_source_manifest_v1","Exact one-parent source closure")
    for r in manifest["refs"]:c.raw(r)
    return tuple(r["path"] for r in manifest["refs"])


def phase_paths(phase):
    need(_active is not None and phase==PHASE,"ROOT request selected before path use")
    return c.local(_active["packet_directory"]),_active["output_directory"]


def guard(consumer=True):
    global _guard,_memo
    need(type(consumer)is bool and _active is not None,"Explicit ROOT current selector lifetime")
    if _guard is None:
        module=c.load(CHILD,"_frozen30_train_continuation_current_Census")
        _guard=module.load_selected_guard_v1(SELECTOR)
        need(_guard.MODE=="Census_observation_two_scope_retention_v1","Unchanged current child")
    if consumer and _memo is None:
        m=c.load(c.NAMESPACE_REF,"_frozen30_train_continuation_readonly_namespace")
        _memo=m.install_consumer_v1(_guard.budget.v4,_guard.budget,consumer_role="read_only_consumer",
            expected_source_path=ROOT/c.V4_REF["path"],expected_source_sha256=c.V4_REF["sha256"])
    return _guard


def validate_request(record):
    global _active
    initialize()
    q=c.read(record);c.active(q,300)
    need(q.get("schema")=="frozen_TRAIN002_019_full30Mb_simulator_ROOT_request_v1"
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
    directive=q;c.active(directive,300)
    batch=validate_batch_authority(q["batch_authority_ref"])
    need(q["source_review_ref"]==batch["source_review_ref"]
        and q["scope_id"]==batch["scope_id"]+"_"+PARENT.removeprefix("american_c3_")
        and q["cpu_ids"]in batch["CPU_lanes"]
        and datetime.fromisoformat(q["expires_at_utc"])<=datetime.fromisoformat(batch["expires_at_utc"]),
        "Per-unit authority and exact CPU/source/scope remain inside finite batch")
    need(directive.get("root_authorizer")=="root_agent"and directive.get("active")is True
        and directive.get("parent_id")==PARENT and directive.get("phase")==PHASE
        and directive.get("source_manifest_ref")==c.ref(MANIFEST)
        and directive.get("current_Census_original_reserve_execute_authorized")is True,
        "Explicit actual ROOT admission/execution directive")
    control=c.read(q["batch_control_input_ref"])
    need(control["authority_ref"]==q["batch_authority_ref"]and control["source_manifest_ref"]==c.ref(MANIFEST)
        and control["schema"]=="ROOT_TRAIN002_019_batch_admitted_input_v1"
        and control["storage"]["reservation"]["reserved_bytes"]==33554432,
        "Same genuine32MiB batch control input and coordinator binding")
    capacity=q["actual_capacity"];c.active(capacity,300)
    need(capacity.get("root_authorizer")=="root_agent"and capacity.get("actual_current_observation")is True
        and capacity.get("source_only_estimate")is False and capacity.get("parent_id")==PARENT,
        "ROOT actual current capacity evidence, never a source proposal")
    caps=q["caps"]
    need(set(caps)=={"reservation_bytes","aux_bytes","private_work_bytes","rss_bytes","wall_seconds","cleanup_seconds","log_bytes",
         "terminal_evidence_margin_seconds","work_cutoff_margin_seconds","CPU_threads"}
        and all(type(v)is int and v>0 for v in caps.values())
        and caps==c.read(c.ref(MANIFEST))["caps_proposed"]
        and caps["reservation_bytes"]==7000000000 and caps["reservation_bytes"]>=6500000000+caps["aux_bytes"]
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
    template=scientific_config_for_parent(PARENT,False)
    need(set(cfg)==set(template)and cfg.get("execution_enabled")is True
        and all(cfg[k]==v for k,v in template.items()if k!="execution_enabled"),
        "Unchanged full30Mb registered TRAIN002..019 simulation science")
    for key,prefix in [("packet_directory",REL+"/ROOT_actual_"+PARENT.removeprefix("american_c3_").upper()+"_"),("output_directory","experiments/results/frozen_TreeLoom_full30Mb_"+PARENT.removeprefix("american_c3_").upper()+"_")]:
        need(isinstance(q.get(key),str)and q[key].startswith(prefix)and c.local(q[key])!=ROOT,
             "Fresh declared one-parent namespaces")
    need(q["source_sha256"]=={p:c.ref(p)["sha256"]for p in source_paths()}and q["native_MLM_dependency"]is False,
        "Complete actual source closure inside immutable ROOT request")
    need(_active is None or _active==q,"No request/cap/lifetime swap")
    _active=q;c.CAPS=caps;c.RUNTIME_REF=q["runtime_ref"]
    return q,caps,runtime


def direct_roles(request_ref,q,cfgref,closure,runtime):
    return [c.ref(SOURCE),c.ref(MANIFEST),c.ref(SCIENCE),
        c.ref(OLD+"/root_contract_v1.py"),c.ref(OLD+"/root_driver_v1.py"),
        c.ref("src/arggnn/american_c3_simulate_v1.py"),c.ref("src/arggnn/american_c3_common_v1.py"),
        c.PIDFD,c.OWNED,c.NAMESPACE_REF,c.V4_REF,SELECTOR,CHILD,c.POLICY_REF,
        request_ref,q["runtime_ref"],q["batch_authority_ref"],q["batch_control_input_ref"],c.ref(REL+"/batch_ROOT_v1.py"),c.ref(REL+"/close_unit_ROOT_v1.py"),
        q["source_review_ref"],cfgref,closure]


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
    need(s["transitive_source_closure_ref"]==s["root_stage_request_ref"]and s["caps_ref"]==s["root_stage_request_ref"]
        and s["caps_locator"]==["caps"],"Actual complete request holds caps and source closure")
    closure=c.read(s["transitive_source_closure_ref"])
    need(closure["source_sha256"]==sources and s["transitive_source_closure_ref"]==s["root_stage_request_ref"]
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

import argparse,hashlib,os,stat,resource,time,fcntl
context=sys.modules[__name__]
CENSUS_CHILD=CHILD
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

def execute_in_admitted_context_v1(config_ref, *, spec, current_guard,
                                    current_guard_source_ref, resource_check):
    """Direct callable for an existing ROOT-owned worker after its actual gate.

    spec/current_guard are the actual current-Census admission and exact child.
    ROOT's guard validates the complete existing authority, storage, source,
    config and runtime binding. The original Writer calls the same guard before
    each planned write and after each serialization. No old adapter is loaded.
    """
    need(Path.cwd() == ROOT and sys.dont_write_bytecode and sys.flags.no_user_site
         and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
         "Exact repository CPU-only -B -s interpreter")
    cfg = read(config_ref)
    need(current_guard_source_ref == CENSUS_CHILD
         and metadata_ref(CENSUS_CHILD["path"]) == CENSUS_CHILD
         and callable(resource_check)
         and callable(getattr(current_guard, "validate_storage_v9", None))
         and callable(getattr(current_guard, "budget_check_v9", None)),
         "Actual immutable current-Census child guard; no legacy fallback")
    need(spec.get("root_authorizer") == "root_agent"
         and spec.get("execution_authorized") is True
         and spec.get("GPU_authorized") is False
         and spec.get("source_ref") == metadata_ref(SOURCE)
         and spec.get("config_ref") == config_ref
         and cfg.get("execution_enabled") is True
         and cfg.get("sequence_length_bp") == 30000000,
         "Genuine ROOT CPU source/config authority for one complete30Mb parent")
    condition=read(metadata_ref(SCIENCE))
    registry=condition["original_registry"]
    matches = [r for r in registry["parents"] if r["parent_id"] == cfg.get("parent_id")]
    need(len(matches) == 1 and matches[0]["split"] == cfg.get("split") == "train"
         and matches[0]["parent_id"]==PARENT and PARENT in ALLOWED_PARENTS
         and matches[0]["ancestry_seed"] == cfg.get("ancestry_seed")
         and matches[0]["mutation_seed"] == cfg.get("mutation_seed"),
         "Original full-length missing TRAIN parent only; do not rerun completedTRAIN000")
    template=condition["original_full30mb_template"]
    mutable = {"parent_id", "split", "ancestry_seed", "mutation_seed", "execution_enabled"}
    need(set(cfg) == set(template) and all(cfg[k] == v for k, v in template.items() if k not in mutable),
         "Only original registered parent identity/seeds and actual activation may differ")
    required = [metadata_ref(p) for p in (SOURCE, GENERATOR, COMMON)]
    required += [config_ref, spec["runtime_ref"], current_guard_source_ref]
    bindings = spec["storage"]["reservation"]["consumer_bindings"]
    need(all(r in bindings for r in required), "Exact real source/config/runtime/current-child direct bindings")
    resource_check("before_original_simulator_current_Census_validation")
    # Guard authority covers the original primary plus separate charged AUX.
    # This operational view is never passed to the scientific simulator.
    reserved=spec["storage"]["reservation"]["reserved_bytes"]
    need(type(reserved) is int and reserved==spec["storage"]["reserved_new_bytes"]
         and reserved>=cfg["output_cap_bytes"],
         "Exact admitted whole-scope cap; original primary cap remains unchanged")
    storage_cfg=dict(cfg,output_cap_bytes=reserved)
    current_guard.validate_storage_v9(
        spec, storage_cfg, spec["storage"], source_ref=spec["source_ref"],
        config_ref=config_ref, runtime_ref=spec["runtime_ref"])
    # These original imports do not select/configure any storage adapter.
    from arggnn import american_c3_simulate_v1 as simulator
    from arggnn.american_c3_common_v1 import Writer, runtime_report
    runtime = read(spec["runtime_ref"])
    need(runtime.get("prefix") == str(Path(sys.prefix).resolve())
         and runtime.get("CUDA_VISIBLE_DEVICES") == "",
         "Actual ROOT-qualified original scientific CPU interpreter")
    if "packages" in runtime:
        actual = runtime_report()["packages"]
        need(all(actual.get(k) == v for k, v in runtime["packages"].items()),
             "Exact qualified scientific package metadata")
    resource_check("before_original_full30Mb_scientific_execute")
    writer = Writer(spec, _OriginalWriterCurrentGuard(current_guard, resource_check))
    result = simulator.execute(cfg, writer)
    resource_check("after_original_full30Mb_scientific_execute")
    return result

def admit(request_ref):
    q=read(request_ref)
    with accounting_transaction(q["batch_control_input_ref"]):
        return admit_under_coordinator(request_ref)


def admit_under_coordinator(request_ref):
    c,d=context.initialize()
    q,caps,_=c.validate_request(request_ref);packet,output=c.phase_paths(c.PHASE)
    c.require(not packet.exists()and not c.local(output).exists(),"Fresh exclusive packet/output")
    g=c.original_guard()
    implementation=c.ref(g.budget.SOURCE_REL);launcher=c.ref(g.budget.LAUNCHER_REL)
    with g.budget.v4.mutex(c.ROOT):
        state=g._state_locked(implementation["sha256"],launcher["sha256"],fresh=True)
    packet.mkdir()
    config_ref=c.save(packet,"config.json",c.make_config(q,caps))
    sources={path:c.ref(path)["sha256"]for path in c.source_paths()}
    closure=request_ref
    consumers=c.unique([*state[3].values(),*c.direct_roles(request_ref,q,config_ref,closure,q["runtime_ref"])])
    job="TreeLoom-frozen-full30Mb-"+PARENT.removeprefix("american_c3_").upper()+"-simulate-"+q["scope_id"]
    aux=packet/"worker_aux"
    reserve=dict(key="H041/"+job,output=str(c.ROOT/output),auxiliary_scopes=[str(aux)],
        reserved_bytes=caps["reservation_bytes"],consumer_bindings=consumers)
    authority=g.operation_draft_v1("reserve",reserve,expected_before=g._pins(state[2]),bindings=state[3])
    authority.update(active=True,expires_at_utc=q["expires_at_utc"])
    ar=c.save(packet,"reserve_authority.json",authority)
    receipt=g.apply_reserve_v1(ar,expected_source_sha256=implementation["sha256"],expected_launcher_sha256=launcher["sha256"])
    rr=c.save(packet,"reserve_receipt.json",receipt)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
        accounting_mode=g.MODE,job_id=job,output_directory=output,
        reserved_new_bytes=caps["reservation_bytes"],reservation=receipt["value"],
        initial_ledger_pins=receipt["initial_ledger_pins"],expires_at_utc=q["expires_at_utc"],
        completed_overlay_policy_ref=c.POLICY_REF,
        budget_components=dict(implementation=implementation,launcher=launcher,consumer_adapter=c.GUARD_REF))
    sr=c.save(packet,"storage.json",storage)
    spec=dict(schema="frozen_TRAIN002_019_current_Census_simulator_input_v1",
        status="active",root_authorizer="root_agent",execution_authorized=True,GPU_authorized=False,
        parent_id=c.PARENT,phase=c.PHASE,stage=c.STAGE,job_id=job,output_directory=output,
        worker_auxiliary_scope=str(aux.relative_to(c.ROOT)),root_stage_request_ref=request_ref,
        root_driver_ref=c.ref(context.DRIVER),main_source_ref=c.ref(context.WORKER),
        source_sha256=sources,transitive_source_closure_ref=closure,caps_ref=request_ref,caps_locator=["caps"],
        refs=dict(config=config_ref,runtime=q["runtime_ref"],storage=sr),
        runtime_ref=q["runtime_ref"],storage=storage,
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
        and Path(sys.prefix)==c.ROOT/".runtime/env"
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

def inactive_template(parent):
    """Only static source/JSON reads. Result is expressly not active authority."""
    registered_row(parent)
    template=read(metadata_ref(REL+"/ROOT_request.template.inactive.v1.json"))
    template.update(parent_id=parent,phase="simulate_"+parent.removeprefix("american_c3_"),
        source_manifest_ref=metadata_ref(MANIFEST),scientific_config=scientific_config_for_parent(parent,False))
    template["bindings"]=dict(parent_id=parent,scientific_config=template["scientific_config"])
    template["source_sha256"]={r["path"]:r["sha256"]for r in read(metadata_ref(MANIFEST))["refs"]}
    return template

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument("--admit-ref");mode.add_argument("--execute-ref");mode.add_argument("--worker-ref")
    mode.add_argument("--inactive-template-parent")
    p.add_argument("--gate-fd",type=int);a=p.parse_args(argv)
    if a.inactive_template_parent:
        result=inactive_template(a.inactive_template_parent)
        print(json.dumps(result,sort_keys=True,indent=2));return 0
    if not(a.admit_ref or a.execute_ref or a.worker_ref):
        print(json.dumps(dict(status="SOURCE_ONLY_INACTIVE",admission=False,execution=False)));return 0
    record=json.loads(a.admit_ref or a.execute_ref or a.worker_ref)
    select_parent_from_record(record)
    c,d=initialize()
    c.require(os.getcwd()==str(ROOT)and sys.dont_write_bytecode and sys.flags.no_user_site,
        "Exact existing ROOT -B -s interpreter and repository cwd")
    if a.admit_ref:result=admit(record)
    elif a.worker_ref:
        c.require(a.gate_fd is not None,"Actual original owned gate")
        result=dict(completed_ref=worker(record,a.gate_fd))
    else:
        s,q,_=d.input_spec(record)
        c.require(set(q["cpu_ids"])<=os.sched_getaffinity(0),"Actual selected CPUs available to ROOT")
        os.sched_setaffinity(0,set(q["cpu_ids"]))
        with c.cpu_slot()as fd:
            d.SLOT_FD=fd;result=d.execute(record)
    print(json.dumps(result,sort_keys=True),flush=True)
    return result.get("returncode",0)

if __name__=="__main__":raise SystemExit(main())
