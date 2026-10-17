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
REL="research/reviews/frozen_TRAIN001_observe_ready_20261017_v1"
OLD="research/reviews/American_C3_VAL003004_current_Census_preparation_source_20261016_v1"
MANIFEST=REL+"/source_manifest.v1.json"
DRIVER=REL+"/worker_entry_v1.py"
WORKER=DRIVER
SOURCE=DRIVER
BRIDGE=DRIVER
SCIENCE=REL+"/source_manifest.v1.json"
MODULE=SOURCE[:-3].replace("/",".")
PARENT="american_c3_train_001"
PHASE="infer_train_001"
STAGE="infer"
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
    c.PHASES=c.CPU_PHASES=(PHASE,);c.STATUSES={PHASE:"completed_current_C3_inference"}
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
    need(manifest["schema"]=="frozen_TRAIN001_observation_source_manifest_v1","Exact one-parent source closure")
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
    need(q.get("schema")=="frozen_TRAIN001_full30Mb_observation_ROOT_request_v1"
        and q.get("active")is True and q.get("root_authorizer")=="root_agent"
        and q.get("execution_authorized")is True and q.get("GPU_authorized")is False
        and q.get("parent_id")==PARENT and q.get("stage")==STAGE and q.get("phase")==PHASE
        and q.get("fresh_unfinished_stage")is True and q.get("automatic_retry")is False
        and q.get("source_manifest_ref")==c.ref(MANIFEST) and q.get("storage_policy_selector_ref")==SELECTOR
        and q.get("query_truth_access_authorized")is False and q.get("GNN_optimization_authorized")is False
        and q.get("TEST_access_authorized")is False
        and q.get("current_Census_original_reserve_execute_authorized")is True,
        "Exact independently reviewed one-parent ROOT observation authority")
    review=c.read(q["source_review_ref"])
    need(review.get("verdict")=="SOURCE_PASS"and review.get("independent")is True
        and review.get("blockers")==[]and review.get("source_manifest_ref")==c.ref(MANIFEST),
        "Independent review binds exact sealed observation closure")
    condition=c.read(c.ref(MANIFEST))["condition"];caps=q["caps"]
    need(caps==condition["caps_proposed"]and all(type(v)is int and v>0 for v in caps.values())
        and caps["reservation_bytes"]>=condition["inference_template_inactive"]["output_cap_bytes"]+caps["aux_bytes"]
        and caps["cleanup_seconds"]==30 and caps["terminal_evidence_margin_seconds"]==150
        and caps["work_cutoff_margin_seconds"]==300 and caps["CPU_threads"]==8,
        "Exact finite original lifecycle and complete primary plus AUX allowance")
    capacity=q["actual_capacity"];c.active(capacity,300)
    need(capacity.get("root_authorizer")=="root_agent"and capacity.get("actual_current_observation")is True
        and capacity.get("source_only_estimate")is False and capacity.get("parent_id")==PARENT,
        "Inline actual ROOT capacity evidence, never a source proposal")
    ids=q.get("cpu_ids")
    need(isinstance(ids,list)and len(ids)==len(set(ids))==8 and all(type(i)is int and i>=0 for i in ids)
        and capacity.get("cpu_ids")==ids and capacity.get("admissible_new_bytes",0)>=caps["reservation_bytes"]
        and capacity.get("available_owned_RSS_bytes",0)>=caps["rss_bytes"],"Actual bound8CPU/storage/RSS capacity")
    need(isinstance(q.get("scope_id"),str)and q["scope_id"]
        and all(ch.isalnum()or ch in "_-"for ch in q["scope_id"]),"One fresh explicit scope")
    need(datetime.fromisoformat(q["expires_at_utc"])>datetime.now(timezone.utc)
        +__import__("datetime").timedelta(seconds=caps["wall_seconds"]+150),
        "ROOT authority covers full declared wall and terminal evidence")
    need(q["runtime_ref"]==condition["inference_runtime_ref"],"Exact existing qualified inference runtime")
    runtime=c.read(q["runtime_ref"])
    need(runtime["prefix"]==str(ROOT/".runtime/envs/biological_tsinfer_v1")and runtime["CUDA_VISIBLE_DEVICES"]=="",
        "Original existing biological tsinfer/tsdate CPU runtime")
    prior=simulation_prerequisites(q["prior"],condition)
    cfg=dict(condition["inference_template_inactive"],execution_enabled=True,observation_ref=prior["observation_ref"])
    need(q["scientific_config"]==cfg and q["bindings"]==dict(parent_id=PARENT,scientific_config=cfg,simulation_prerequisites=prior)
        and q["native_MLM_dependency"]is False,"Verbatim original inference recipe and exact actual predecessor")
    need(q["source_sha256"]=={p:c.ref(p)["sha256"]for p in source_paths()},"Complete source closure inside actual request")
    for key,prefix in [("packet_directory",REL+"/ROOT_actual_TRAIN001_"),
            ("output_directory","experiments/results/frozen_TreeLoom_full30Mb_TRAIN001_infer_")]:
        need(isinstance(q.get(key),str)and q[key].startswith(prefix)and c.local(q[key])!=ROOT,
            "Fresh canonical one-parent observation namespaces")
    need(_active is None or _active==q,"No request/caps/lifetime swap")
    _active=q;c.CAPS=caps;c.RUNTIME_REF=q["runtime_ref"]
    return q,caps,runtime


def direct_roles(request_ref,q,cfgref,closure,runtime):
    need(closure==request_ref,"Actual request is complete source/input closure; no hidden extra file")
    return [c.ref(SOURCE),c.ref(MANIFEST),c.ref(OLD+"/root_contract_v1.py"),c.ref(OLD+"/root_driver_v1.py"),
        c.ref("src/arggnn/american_c3_observe_v3.py"),c.ref("src/arggnn/american_c3_simulate_v1.py"),
        c.ref("src/arggnn/american_c3_common_v1.py"),c.PIDFD,c.OWNED,c.NAMESPACE_REF,c.V4_REF,
        SELECTOR,CHILD,c.POLICY_REF,request_ref,runtime,q["source_review_ref"],cfgref,
        *[q["prior"][key]for key in ("input_ref","config_ref","completion_ref","wait_ref","outer_ref","observation_ref","roster_ref")]]


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
    condition=c.read(c.ref(MANIFEST))["condition"]
    need(value["status"]=="completed_current_C3_inference"
        and value["unary_preprocessing_policy"]==condition["inference_template_inactive"]["unary_preprocessing_policy"]
        and value["tsinfer_num_threads"]==8,"Original unchanged current inference completion")
    obs=observation_metadata(value["observation_ref"])
    need(obs["genealogy_kind"]=="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG"
        and obs["sites_and_sample_identities_verified"]is True
        and obs["source_observation_ref"]==q["prior"]["observation_ref"]
        and obs["unary_preprocessing_policy"]==value["unary_preprocessing_policy"]
        and obs["migrations_and_internal_population_labels_removed"]is True,
        "All original30Mb/1600 samples and observed sites; simulated topology is not inferred feature")


def observation_metadata(record):
    m=c.read(record)
    need(m["schema"]=="american_c3_observed_v1"and m["parent_id"]==PARENT and m["split"]=="train"
        and m["sequence_length_bp"]==30000000 and m["sample_unit"]=="haploid"and m["class_order"]==["EUR","EAS","AFR"]
        and m["counts_haploid"]==dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000)
        and m["query_ancestry_truth_in_inputs"]is False,
        "Original complete30Mb/1600 observation metadata; no ancestry truth features")
    return m

def simulation_prerequisites(prior,condition):
    need(set(prior)=={"input_ref","config_ref","completion_ref","wait_ref","outer_ref","observation_ref","roster_ref"},
        "All seven genuine predecessor metadata refs required")
    s=c.read(prior["input_ref"]);cfg=c.read(prior["config_ref"])
    done=c.read(prior["completion_ref"]);wait=c.read(prior["wait_ref"]);outer=c.read(prior["outer_ref"])
    expected=condition["simulation_stage"]
    need(s["schema"]=="frozen_TRAIN001_current_Census_simulator_input_v1"
        and s["root_authorizer"]=="root_agent"and s["execution_authorized"]is True and s["GPU_authorized"]is False
        and s["parent_id"]==PARENT and s["stage"]=="simulate"and s["phase"]=="simulate_train_001"
        and s["root_driver_ref"]==expected["worker_source_ref"]and s["main_source_ref"]==expected["worker_source_ref"]
        and s["source_sha256"]==expected["source_sha256"]and s["refs"]["config"]==prior["config_ref"]
        and prior["input_ref"]==c.ref(Path(s["worker_auxiliary_scope"]).parent/"input.json")
        and prior["config_ref"]==c.ref(Path(s["worker_auxiliary_scope"]).parent/"config.json")
        and prior["completion_ref"]==c.ref(s["output_directory"]+"/completed.json")
        and prior["wait_ref"]==c.ref(s["worker_auxiliary_scope"]+"/actual_wait.json")
        and prior["outer_ref"]==c.ref(Path(s["worker_auxiliary_scope"]).parent/"outer_terminal.root_actual_v1.json"),
        "Exact actual v5 simulator input/source/config/completion/wait/terminal paths")
    need(cfg==dict(condition["simulation_config_inactive"],execution_enabled=True),
        "Original full30Mb TRAIN001 registered ancestry21002/mutation1021002 science")
    need(done["status"]=="completed_C3_simulation_reconstruction"and done["parent_id"]==PARENT
        and done["sample_unit"]=="haploid"and done["class_order"]==["EUR","EAS","AFR"]
        and done["observation_ref"]==prior["observation_ref"]and type(done["sites"])is int
        and 0<done["sites"]<=cfg["max_sites"],"Actual original complete simulation metadata")
    need(wait["schema"]=="American_C3_full_original_CPU_actual_wait_v1"
        and wait["input_ref"]==prior["input_ref"]and wait["completed_ref"]==prior["completion_ref"]
        and wait["parent_id"]==PARENT and wait["phase"]=="simulate_train_001"
        and wait["actual_direct_worker_wait"]is True and wait["actual_direct_launcher_wait"]is True
        and wait["actual_OS_wait_status"]==0 and wait["returncode"]==0
        and wait["all_owned_processes_and_groups_drained"]is True and wait["operational_failure"]is None
        and wait["cleanup_errors"]==[]and all(x["returncode"]==0 for x in wait["adopted_waits"])
        and bool(wait["group_observations"])and all(x["absent"]is True for x in wait["group_observations"]),
        "Genuine same-parent simulator OS wait0 and complete owned drain, never predicted")
    need(outer["root_authorizer"]=="root_agent"and outer["actual_terminal_consumed"]is True
        and outer["actual_exit_code"]==0 and outer["input_ref"]==prior["input_ref"]
        and outer["completed_ref"]==prior["completion_ref"]and outer["direct_wait_ref"]==prior["wait_ref"]
        and outer["all_prior_launch_resume_write_capabilities_revoked"]is True
        and outer["simulation_outputs_frozen"]is True,
        "ROOT actually consumed simulator outer0 and froze predecessor writes; no close or credit inferred")
    m=observation_metadata(prior["observation_ref"])
    need(prior["observation_ref"]==c.ref(s["output_directory"]+"/observed/manifest.json")
        and m["roster_ref"]==prior["roster_ref"]==c.ref(s["output_directory"]+"/observed/roster.json")
        and m["genealogy_kind"]=="simulated_true_topology_sanitized_ORACLE_not_published_inferred_ARG"
        and m["migrations_and_internal_population_labels_removed"]is True,
        "Original sanitized observation; original inference reads genotypes, never simulated tree")
    # Structural refs only. No arrays, alleles, true tree or ancestry truth read here.
    for key,name in [("genotypes_ref","genotypes_int8.npy"),("positions_ref","positions_bp0.npy"),("alleles_ref","site_alleles.jsonl")]:
        r=m[key]
        need(set(r)=={"path","sha256"}and r["path"]==s["output_directory"]+"/observed/"+name
            and isinstance(r["sha256"],str)and len(r["sha256"])==64
            and all(ch in "0123456789abcdef"for ch in r["sha256"]),"Original observed payload ref shape only")
    roster=c.read(prior["roster_ref"])
    need(roster["parent_id"]==PARENT and roster["class_order"]==["EUR","EAS","AFR"]
        and len(roster["samples"])==1600,"Complete original1600 observed roster metadata")
    offset=0;nodes=[]
    for pop,count,cls in [("AFR",200,2),("EUR",200,0),("ASIA",200,1),("ADMIX",1000,None)]:
        for j in range(count):
            r=roster["samples"][offset+j]
            need(r["sample_index"]==offset+j and r["sample_id"]==f"{pop}_{j//2:05d}"
                and r["haplotype"]==j%2 and r["age_generations"]==0
                and r["role"]==("query"if pop=="ADMIX"else "pure_source_reference")
                and r["observed_reference_class"]==cls and type(r["observed_node_id"])is int
                and r["observed_node_id"]>=0,"Original genotype columns, diploids, references and queries")
            nodes.append(r["observed_node_id"])
        offset+=count
    need(len(set(nodes))==1600,"Stable unique original sample identities")
    return dict(prior)

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

def execute_in_admitted_context_v1(config_ref, *, spec, current_guard,
                                    current_guard_source_ref, resource_check):
    """Original inference callable only after real admitted owned gate."""
    need(Path.cwd()==ROOT and sys.dont_write_bytecode and sys.flags.no_user_site
        and os.environ.get("CUDA_VISIBLE_DEVICES")=="","Exact CPU-only -B -s interpreter")
    cfg=read(config_ref)
    need(current_guard_source_ref==CENSUS_CHILD and metadata_ref(CENSUS_CHILD["path"])==CENSUS_CHILD
        and callable(resource_check)and callable(getattr(current_guard,"validate_storage_v9",None))
        and callable(getattr(current_guard,"budget_check_v9",None)),"Actual unchanged current child guard")
    condition=read(metadata_ref(MANIFEST))["condition"]
    need(spec.get("root_authorizer")=="root_agent"and spec.get("execution_authorized")is True
        and spec.get("GPU_authorized")is False and spec.get("source_ref")==metadata_ref(SOURCE)
        and spec.get("config_ref")==config_ref and cfg.get("execution_enabled")is True,
        "Genuine exact admitted original observation authority")
    prior=simulation_prerequisites(read(spec["root_stage_request_ref"])["prior"],condition)
    need(cfg==dict(condition["inference_template_inactive"],execution_enabled=True,observation_ref=prior["observation_ref"]),
        "Original all-site/all1600 full30Mb inference recipe passed verbatim")
    required=[metadata_ref(p)for p in (SOURCE,INFERENCE,GENERATOR,COMMON)]+[config_ref,spec["runtime_ref"],current_guard_source_ref]
    need(all(r in spec["storage"]["reservation"]["consumer_bindings"]for r in required),
        "Original inference/common/sanitize/config/runtime/child direct bindings")
    resource_check("before_original_inference_current_Census_validation")
    reserved=spec["storage"]["reservation"]["reserved_bytes"]
    need(type(reserved)is int and reserved==spec["storage"]["reserved_new_bytes"]and reserved>=cfg["output_cap_bytes"],
        "Whole-scope operational cap; scientific inference cap unchanged")
    current_guard.validate_storage_v9(spec,dict(cfg,output_cap_bytes=reserved),spec["storage"],
        source_ref=spec["source_ref"],config_ref=config_ref,runtime_ref=spec["runtime_ref"])
    from arggnn import american_c3_observe_v3 as inference
    from arggnn.american_c3_common_v1 import Writer,runtime_report
    need(Path(inference.__file__).resolve()==local(INFERENCE),"Exact original observe_v3 module origin")
    runtime=read(spec["runtime_ref"])
    need(runtime.get("prefix")==str(Path(sys.prefix).resolve())and runtime.get("CUDA_VISIBLE_DEVICES")=="",
        "Actual qualified original inference CPU interpreter")
    if "packages"in runtime:
        actual=runtime_report()["packages"]
        need(all(actual.get(k)==v for k,v in runtime["packages"].items()),"Exact existing scientific package metadata")
    resource_check("before_original_full30Mb_observe_execute")
    result=inference.execute(cfg,Writer(spec,_OriginalWriterCurrentGuard(current_guard,resource_check)))
    resource_check("after_original_full30Mb_observe_execute")
    return result

def admit(request_ref):
    c,d=context.initialize()
    q,caps,_=c.validate_request(request_ref);packet,output=c.phase_paths(c.PHASE)
    c.require(not packet.exists()and not c.local(output).exists(),"Fresh exclusive packet/output")
    g=c.original_guard();implementation=c.ref(g.budget.SOURCE_REL);launcher=c.ref(g.budget.LAUNCHER_REL)
    with g.budget.v4.mutex(c.ROOT):
        state=g._state_locked(implementation["sha256"],launcher["sha256"],fresh=True)
    packet.mkdir();config_ref=c.save(packet,"config.json",c.make_config(q,caps))
    sources={p:c.ref(p)["sha256"]for p in c.source_paths()};closure=request_ref
    consumers=c.unique([*state[3].values(),*c.direct_roles(request_ref,q,config_ref,closure,q["runtime_ref"])])
    job="TreeLoom-frozen-full30Mb-TRAIN001-infer-"+q["scope_id"];aux=packet/"worker_aux"
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
    spec=dict(schema="frozen_TRAIN001_current_Census_observation_input_v1",status="active",root_authorizer="root_agent",
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
        and Path(sys.prefix)==c.ROOT/".runtime/envs/biological_tsinfer_v1"
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
