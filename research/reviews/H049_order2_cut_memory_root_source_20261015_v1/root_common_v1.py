"""Pinned H042 root utilities plus H049 metadata planning. Import is inert."""
from pathlib import Path
import hashlib,importlib.util,json,os
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H049_order2_cut_memory_root_source_20261015_v1"
PREVIOUS="research/reviews/H042_GPU45_clade_flags_root_source_v1/root_driver.py"
PREVIOUS_SHA="5b87957f3667187b97172b00871f3cc5c85981810179554578e9779bcca9d0b0"
if hashlib.sha256((ROOT/PREVIOUS).read_bytes()).hexdigest()!=PREVIOUS_SHA:
    raise ValueError("Frozen H042 root utility source changed")
_spec=importlib.util.spec_from_file_location("_H049_frozen_root_patterns",ROOT/PREVIOUS)
previous=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(previous)
require,ref,read,save,unique=(
    previous.require,previous.ref,previous.read,previous.save,previous.unique)
REPAIR_REF,AUTH_REF,TEMPLATE_REF,CORE,BASE=(
    previous.REPAIR_REF,previous.AUTH_REF,previous.TEMPLATE_REF,previous.CORE,previous.BASE)
GUARD_REF={'path': 'research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py', 'sha256': 'fc71030865e0a3d5da31120d4daa2caa0b2d39ec7dd212b8c46c476b5a866d56'}
POLICY_REF={'path': 'research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json', 'sha256': 'eb3408004a508136aa434c0a2e02a9d14a5598a96b8483c264c93026f9326018'}
LAUNCHER_REF={'path': 'scripts/launch_reserved_gpu_authorized_GPU45_980_v1.py', 'sha256': '0ca3c59e49f5fd55f5a798c99a4a8a6613e71a90bcb3f89a04eb66e7555046ca'}

def policy_env():
    env=os.environ.copy()
    env["ARGGNN_COMPLETED_CREDITS_POLICY_REF"]=json.dumps(POLICY_REF,separators=(",",":"))
    env["PYTHONPATH"]=os.pathsep.join([str(ROOT),str(ROOT/"scripts"),str(ROOT/"src"),env.get("PYTHONPATH","")])
    return env

MEMO_REF=dict(path="research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py",
              sha256="c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364")
V4_REF=dict(path="src/arggnn/project_budget_v4.py",
            sha256="a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47")
def cap_v1(cfg,name):
    value=cfg.get(name)
    require(type(value) is int and value>0,"Root must bind reviewed resource: "+name)
    return value

def load_guard_v1(*,consumer=False):
    """Fresh processes: mutations use original v4; consumer checks may use memo2."""
    require(ref(MEMO_REF["path"])==MEMO_REF and ref(V4_REF["path"])==V4_REF,"Pinned memo2/originalv4 required")
    for r in (GUARD_REF,POLICY_REF,LAUNCHER_REF,REPAIR_REF,AUTH_REF):
        require(ref(r["path"])==r,"Exact active980 IO authority")
    ms=importlib.util.spec_from_file_location("_H049_eval980_guard",ROOT/GUARD_REF["path"])
    guard=importlib.util.module_from_spec(ms);ms.loader.exec_module(guard)
    guard.configure_v2(REPAIR_REF,policy_ref=POLICY_REF)
    require(guard.MODE=="prospective_980gb_v1","Actual980 storage mode")
    if not consumer:
        require(all(getattr(guard.budget.v4,name).__module__==guard.budget.v4.__name__
                    for name in ("canonical","file_bytes","mutex")),
                "Reserve/close mutations require unmemoized original v4 functions")
        return guard,None
    spec=importlib.util.spec_from_file_location("_H049_memo2",ROOT/MEMO_REF["path"])
    memo=importlib.util.module_from_spec(spec);spec.loader.exec_module(memo)
    adapter=memo.install(guard.budget.v4,expected_source_path=ROOT/V4_REF["path"],
                         expected_source_sha256=V4_REF["sha256"])
    return guard,adapter

from research.reviews.H049_order2_cut_memory_root_source_20261015_v1 import cut_recipe_codec_v1 as codec
from research.reviews.order2_cut_memory_implementation_20261015_v1 import integration_hooks_v1 as hooks

def count_plan_v1(manifest,phase,preparation,cfg):
    require(manifest.get("schema")=="root_H043_original40_observed_count_manifest_v1"
        and manifest.get("counts_only_no_labels_or_model_metrics") is True
        and manifest.get("outer_included") is False
        and set(manifest["parents"])==set(hooks.TRAIN+hooks.VAL),"Exact genuine original40 counts")
    require(phase in ("prepare","train","evaluate")
        and set(preparation["parents"])==set(hooks.TRAIN+hooks.VAL),"Original all40 marker preparation")
    ids=hooks.TRAIN+hooks.VAL if phase=="prepare" else (hooks.TRAIN if phase=="train" else hooks.VAL)
    raw={};recipes=prediction=grid_total=marker_total=durable=0
    for pid in ids:
        row=manifest["parents"][pid];entry=preparation["parents"][pid];pcount=entry["counts"]
        t,n,h=(row[k] for k in ("native_holds","native_nodes","haplotypes"))
        s,p=pcount["sites"],pcount["scoring_centers"]
        require(h==952 and pcount["haplotypes"]==h and pcount["native_holds"]==t
            and pcount["queries"]==686,"Exact original complete native counts")
        b=hooks.bounds_from_counts_v1(n,t,s,p)
        compact=codec.compact_upper_v1(n,t)
        marker=entry["marker_tensor_bytes"]
        require(type(marker) is int and marker>0,"Actual marker metadata bytes")
        recipes+=b["total_recipe_JSON_planning_allowance_bytes"]
        prediction+=18*16*p*686 if phase=="evaluate" else 0
        grid_total+=b["extra_retained_tensor_upper_bytes"]+marker;marker_total+=marker
        durable+=compact["compressed_NPZ_upper_bytes"]+compact["recipe_JSON_planning_bytes"]+2*8192
        raw[pid]=dict(b,marker_tensor_bytes=marker,compact_cut_recipe=compact,
            grid_plus_marker_cache_upper_bytes=b["extra_retained_tensor_upper_bytes"]+marker,
            original_grid_logits_or_target_bytes=16*p*686)
    checkpoint=(18*(cfg["checkpoint_file_cap_bytes"]+8192) if phase=="train"
        and type(cfg.get("checkpoint_file_cap_bytes")) is int else (None if phase=="train" else 0))
    known=(durable if phase=="prepare" else recipes)
    log_raw=cfg.get("log_cap_bytes")
    serialized_log=(6*log_raw+65536+16384 if type(log_raw) is int and log_raw>0 else None)
    overhead=[cfg.get("fixed_metadata_cap_bytes"),serialized_log]
    complete=(known+checkpoint+sum(overhead)
        if checkpoint is not None and all(type(v) is int and v>0 for v in overhead) else None)
    return dict(phase=phase,parent_count=len(ids),parents=raw,
        all_phase_grid_plus_marker_cache_upper_bytes=grid_total,
        maximum_parent_grid_plus_marker_cache_upper_bytes=max(r["grid_plus_marker_cache_upper_bytes"] for r in raw.values()),
        all_phase_marker_tensor_bytes=marker_total,
        original_all40_retained_coordinate_bytes=preparation["retained_coordinate_bytes"],
        original_all40_retained_binding_JSON_bytes=preparation["retained_binding_JSON_bytes"],
        maximum_named_cut_preparation_work_estimate_bytes=max(r["additional_cut_preparation_work_estimate_bytes"] for r in raw.values()),
        maximum_named_grid_cut_preparation_work_estimate_bytes=max(r["named_preparation_allowance_bytes"] for r in raw.values()),
        maximum_parent_persistent_GPU_grid_cut_bytes=max(r["GPU_persistent_grid_upper_bytes"] for r in raw.values()),
        maximum_parent_marker_tensor_bytes=max(r["marker_tensor_bytes"] for r in raw.values()),
        maximum_original_parent_target_FP64_plus_FP32_bytes=max(48*r["grid_centers"]*686 for r in raw.values()),
        all144_original_grid_probability_bytes=prediction,
        compact_prepare_durable_upper_bytes=durable if phase=="prepare" else 0,
        observation_recipes_JSON_planning_bytes=recipes,
        checkpoint_files=18 if phase=="train" else 0,
        raw_model_and_two_Adam_moment_tensor_bytes_per_checkpoint=3*32446*4,
        checkpoint_archive_allowance_bytes=checkpoint,
        raw_log_cap_bytes=log_raw,serialized_root_log_and_wait_planning_bytes=serialized_log,
        known_output_subtotal_bytes=known,total_output_scope_upper_bound_bytes=complete,
        output_cap_bytes=cfg.get("output_cap_bytes"),resource_caps_reviewed=cfg.get("root_resource_caps_reviewed",False),
        preparation_RSS_and_full_model_GPU_allocator_unqualified=True,
        all_dense_probability_arrays_are_RAM_only=True)


def current_RSS_bytes_v1():
    parts=Path("/proc/self/statm").read_text().split()
    require(len(parts)>=2 and parts[1].isdigit(),"Current-process Linux RSS")
    return int(parts[1])*os.sysconf("SC_PAGE_SIZE")


def preparation_check_v1(pid,bank,declared,cfg,counts,preparation,*,rss_reader=current_RSS_bytes_v1):
    p=preparation["parents"][pid]["counts"]
    actual=hooks.parent_bounds_v1(bank,range(p["scoring_centers"]))
    require(declared==actual,"Actual complete native/grid bounds")
    row=counts["parents"][pid]
    require(row["native_nodes"]==actual["native_nodes"]
        and row["native_holds"]==actual["native_holds"]
        and row["haplotypes"]==952 and p["queries"]==686
        and p["sites"]==actual["sites"],"Frozen genuine counts equal loaded native bank")
    marker=preparation["parents"][pid]["marker_tensor_bytes"]
    require(actual["extra_retained_tensor_upper_bytes"]+marker<=cap_v1(cfg,"grid_cache_cap_bytes")
        and actual["GPU_persistent_grid_upper_bytes"]<=cap_v1(cfg,"GPU_grid_cap_bytes"),
        "One complete cut/grid/marker parent fits root-reviewed caps")
    scratch=actual["named_preparation_allowance_bytes"];rss=rss_reader()
    require(scratch<=cap_v1(cfg,"grid_preparation_cap_bytes")
        and rss+scratch<=cap_v1(cfg,"max_cpu_rss_bytes"),"Prospective named preparation allowance")
    return dict(parent_id=pid,current_RSS_bytes=rss,preparation_planning_allowance_bytes=scratch,
        projected_RSS_planning_bytes=rss+scratch,preparation_estimate_is_not_an_RSS_bound=True,
        GPU_persistent_grid_upper_bytes=actual["GPU_persistent_grid_upper_bytes"],
        full_model_GPU_autograd_peak_unqualified=True)

def validate_train_lineage_v1(reader,spec):
    if spec["phase"]!="evaluate":return
    from arggnn import train_marker_order2_cut_memory_chain_v1 as worker
    train=reader.decode_json(spec["train_completion_ref"])
    launch=reader.decode_json(spec["train_launch_ref"]);wait=reader.decode_json(spec["train_actual_wait_ref"])
    require(train.get("schema")=="H049_three_arm_train_completion_v1"
        and train.get("status")=="completed_fixed30pass_three_arm_TRAIN"
        and train.get("hypothesis_id")=="H049" and train.get("arms")==list(worker.ARMS)
        and train.get("steps")=={a:960 for a in worker.ARMS}
        and train.get("VAL_truth_read") is False and train.get("outer_truth_read") is False
        and train.get("same_VAL_is_development_evidence") is True,"Actual complete three_arm H049 TRAIN")
    require(wait.get("actual_direct_launcher_wait") is True and wait.get("returncode")==0
        and wait.get("logging_error") is None and wait.get("supervisor_error") is None
        and wait.get("error") is None and wait.get("owned_group_drained") is True
        and wait["completed_ref"]==spec["train_completion_ref"] and wait["launcher_pid"]==launch["launcher_pid"]
        and launch["input_ref"]==train["input_ref"] and launch["phase"]==wait["phase"]=="train"
        and launch["physical_gpu"]==wait["physical_gpu"] and launch["physical_gpu"] in (4,5),
        "Exact successful same-launcher actual TRAIN wait/lineage")

def validate_eval_train_io_v1(reader,spec):
    require(spec["phase"]=="evaluate","Evaluation lineage only")
    validate_train_lineage_v1(reader,spec)
    from arggnn import train_marker_order2_cut_memory_chain_v1 as worker
    train=reader.decode_json(spec["train_completion_ref"])
    launch=reader.decode_json(spec["train_launch_ref"])
    expected_input=HERE+"/admitted_train/launch_input.json"
    require(train["input_ref"]["path"]==expected_input
        and spec["train_launch_ref"]["path"]==HERE+"/admitted_train/launch.json"
        and spec["train_actual_wait_ref"]["path"]==HERE+"/admitted_train/actual_wait.json",
        "Only this fresh H049 TRAIN admission, launcher and actual wait")
    old=reader.decode_json(train["input_ref"])
    train_output="experiments/results/H049_GPU%d_order2_cut_memory_train_seed101_v1"%launch["physical_gpu"]
    require(spec["train_completion_ref"]["path"]==train_output+"/completed.json"
        and train["config_ref"]==old["refs"]["config"] and old["phase"]=="train"
        and old["physical_gpu"]==launch["physical_gpu"] and old["output_directory"]==train_output
        and old["hypothesis_id"]=="H049" and old["producer_ref"]==train["producer_ref"]
        and train["producer_ref"]==ref(worker.SOURCE)
        and train["preparation_completion_ref"]==old["preparation_completion_ref"],
        "Exact original H049 input/config/runtime/preparation and frozen producer")
    groups=train["checkpoint_groups"]
    require(set(groups)=={str(p) for p in worker.PASSES},"Six original H049 checkpoint groups")
    for at,arms in groups.items():
        require(set(arms)==set(worker.ARMS),"All3 matched arms per group")
        for arm,r in arms.items():
            require(set(r)=={"path","sha256"} and r["path"]==train_output+"/"+arm+".pass_%02d.pt"%int(at)
                and len(r["sha256"])==64 and all(c in "0123456789abcdef" for c in r["sha256"]),
                "Exact generated checkpoint refs; payloads verified by scientific reader at execution")
    return dict(train_input_ref=train["input_ref"],train_config_ref=train["config_ref"],
        train_runtime_ref=old["refs"]["runtime"],train_checkpoint_groups=groups)

