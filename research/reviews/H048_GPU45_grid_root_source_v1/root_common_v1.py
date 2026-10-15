"""Pinned H042 root utilities plus H048 metadata planning. Import is inert."""
from pathlib import Path
import hashlib,importlib.util,json,os
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H048_GPU45_grid_root_source_v1"
PREVIOUS="research/reviews/H042_GPU45_clade_flags_root_source_v1/root_driver.py"
PREVIOUS_SHA="5b87957f3667187b97172b00871f3cc5c85981810179554578e9779bcca9d0b0"
if hashlib.sha256((ROOT/PREVIOUS).read_bytes()).hexdigest()!=PREVIOUS_SHA:
    raise ValueError("Frozen H042 root utility source changed")
_spec=importlib.util.spec_from_file_location("_H048_frozen_root_patterns",ROOT/PREVIOUS)
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
OUTPUT_CAP=100000000
LOG_CAP=4000000
RSS_CAP=512000000000
CUDA_CAP=50000000000

def load_guard_v1(*,consumer=False):
    """Fresh processes: mutations use original v4; consumer checks may use memo2."""
    require(ref(MEMO_REF["path"])==MEMO_REF and ref(V4_REF["path"])==V4_REF,"Pinned memo2/originalv4 required")
    for r in (GUARD_REF,POLICY_REF,LAUNCHER_REF,REPAIR_REF,AUTH_REF):
        require(ref(r["path"])==r,"Exact active980 IO authority")
    ms=importlib.util.spec_from_file_location("_H048_eval980_guard",ROOT/GUARD_REF["path"])
    guard=importlib.util.module_from_spec(ms);ms.loader.exec_module(guard)
    guard.configure_v2(REPAIR_REF,policy_ref=POLICY_REF)
    require(guard.MODE=="prospective_980gb_v1","Actual980 storage mode")
    if not consumer:
        require(all(getattr(guard.budget.v4,name).__module__==guard.budget.v4.__name__
                    for name in ("canonical","file_bytes","mutex")),
                "Reserve/close mutations require unmemoized original v4 functions")
        return guard,None
    spec=importlib.util.spec_from_file_location("_H048_memo2",ROOT/MEMO_REF["path"])
    memo=importlib.util.module_from_spec(spec);spec.loader.exec_module(memo)
    adapter=memo.install(guard.budget.v4,expected_source_path=ROOT/V4_REF["path"],
                         expected_source_sha256=V4_REF["sha256"])
    return guard,adapter


def count_plan_v1(manifest,phase,preparation):
    from arggnn import train_marker_grid_reset_chain_v1 as worker
    require(manifest.get("schema")=="root_H043_original40_observed_count_manifest_v1"
        and manifest.get("counts_only_no_labels_or_model_metrics") is True
        and manifest.get("outer_included") is False
        and set(manifest["parents"])==set(worker.TRAIN+worker.VAL),"Exact original40 observation-only counts")
    require(phase in ("train","evaluate") and set(preparation["parents"])==set(worker.TRAIN+worker.VAL),
        "Exact phase and original all40 preparation")
    ids=worker.TRAIN if phase=="train" else worker.VAL
    raw={};recipes=prediction=grid_total=marker_total=0
    for pid in ids:
        row=manifest["parents"][pid];entry=preparation["parents"][pid];prepared=entry["counts"]
        t,n,h=(row[k] for k in ("native_holds","native_nodes","haplotypes"))
        s,p=prepared["sites"],prepared["scoring_centers"];q=686
        require(all(type(v) is int and v>0 for v in (s,t,n,h,p)) and h==952
            and prepared["haplotypes"]==h and prepared["native_holds"]==t
            and prepared["queries"]==q and p<=20000 and s<=2000000,
            "Actual complete952/686 native and original100bp dimensions")
        extra=2*n+20*p*q+36*p-28
        flag_plan=2058*n+128*h
        named=extra+flag_plan+96*s+128*p*32+512*8192*32
        marker=entry["marker_tensor_bytes"]
        require(type(marker) is int and marker>0,"Genuine metadata-counted original marker payload")
        recipes+=131072;prediction+=12*16*p*q if phase=="evaluate" else 0
        grid_total+=extra+marker;marker_total+=marker
        raw[pid]=dict(native_nodes=n,native_holds=t,haplotypes=h,queries=q,sites=s,grid_centers=p,
            extra_retained_tensor_upper_bytes=extra,full_grid_CPU_features_bytes=20*p*q,
            GPU_persistent_grid_upper_bytes=extra-20*p*q,
            one_GPU_feature_query_chunk_bytes=4*p*32*5,
            named_preparation_allowance_bytes=named,marker_tensor_bytes=marker,
            grid_plus_marker_cache_upper_bytes=extra+marker,
            original_grid_logits_or_target_bytes=16*p*q,recipe_JSON_upper_bound_bytes=131072)
    checkpoint=12*1048576 if phase=="train" else 0;fixed=16777216
    projected=recipes+checkpoint+LOG_CAP+fixed
    require(projected<=OUTPUT_CAP,"Count-derived compact output bound exceeds100MB")
    return dict(phase=phase,parent_count=len(ids),parents=raw,
        all_phase_grid_plus_marker_cache_upper_bytes=grid_total,all_phase_marker_tensor_bytes=marker_total,
        all96_original_grid_probability_bytes=prediction,recipes_JSON_upper_bound_bytes=recipes,
        checkpoint_archive_allowance_bytes=checkpoint,bounded_log_bytes=LOG_CAP,
        fixed_seals_metrics_history_filesystem_overhead_bytes=fixed,
        total_output_scope_upper_bound_bytes=projected,output_cap_bytes=OUTPUT_CAP,
        full_model_GPU_peak_and_backbone_allocator_qualification="first_actual_scheduled_step_pending")

def current_RSS_bytes_v1():
    parts=Path("/proc/self/statm").read_text().split()
    require(len(parts)>=2 and parts[1].isdigit(),"Current-process Linux RSS required")
    return int(parts[1])*os.sysconf("SC_PAGE_SIZE")

def preparation_check_v1(pid,bank,declared,cfg,counts,preparation,*,rss_reader=current_RSS_bytes_v1):
    from arggnn import train_marker_grid_reset_chain_v1 as worker
    prepared=preparation["parents"][pid]["counts"]
    actual=worker.grid_upper_bounds_v1(bank,range(prepared["scoring_centers"]))
    require(declared==actual,"Declared bound must equal actual original bank/grid dimensions")
    row=counts["parents"][pid]
    require(row["native_nodes"]==actual["native_nodes"] and row["native_holds"]==actual["native_holds"]
        and row["haplotypes"]==prepared["haplotypes"]==952
        and prepared["sites"]==actual["sites"] and prepared["queries"]==actual["queries"]==686,
        "Actual dimensions differ from frozen count/preparation metadata")
    extra=actual["extra_retained_tensor_upper_bytes"];marker=preparation["parents"][pid]["marker_tensor_bytes"]
    require(extra+marker<=cfg["grid_cache_cap_bytes"]
        and actual["GPU_persistent_grid_upper_bytes"]<=cfg["GPU_grid_cap_bytes"],
        "One complete grid payload must fit root cache/GPU caps")
    scratch=actual["named_preparation_allowance_bytes"]
    require(scratch<=cfg["grid_preparation_cap_bytes"],"Named grid+flag preparation bound exceeds cap")
    rss=rss_reader()
    require(type(rss) is int and rss>=0 and rss+scratch<=RSS_CAP,
        "Current RSS plus prospective preparation allowance exceeds512GB")
    return dict(parent_id=pid,current_RSS_bytes=rss,preparation_planning_allowance_bytes=scratch,
        projected_RSS_planning_bytes=rss+scratch,full_grid_CPU_features_bytes=actual["full_grid_CPU_features_bytes"],
        GPU_persistent_grid_upper_bytes=actual["GPU_persistent_grid_upper_bytes"],
        one_GPU_feature_query_chunk_bytes=actual["one_GPU_feature_query_chunk_bytes"],
        full_model_GPU_autograd_peak_unqualified=True,full_position_history_and_queries_preserved=True)

def validate_train_lineage_v1(reader,spec):
    if spec["phase"]!="evaluate":return
    from arggnn import train_marker_grid_reset_chain_v1 as worker
    train=reader.decode_json(spec["train_completion_ref"])
    launch=reader.decode_json(spec["train_launch_ref"]);wait=reader.decode_json(spec["train_actual_wait_ref"])
    require(train.get("schema")=="H048_paired_train_completion_v1"
        and train.get("status")=="completed_fixed30pass_paired_TRAIN"
        and train.get("hypothesis_id")=="H048" and train.get("arms")==list(worker.ARMS)
        and train.get("steps")=={a:960 for a in worker.ARMS}
        and train.get("VAL_truth_read") is False and train.get("outer_truth_read") is False
        and train.get("same_VAL_is_development_evidence") is True,"Actual complete paired H048 TRAIN")
    require(wait.get("actual_direct_launcher_wait") is True and wait.get("returncode")==0
        and wait.get("logging_error") is None and wait.get("supervisor_error") is None
        and wait["completed_ref"]==spec["train_completion_ref"] and wait["launcher_pid"]==launch["launcher_pid"]
        and launch["input_ref"]==train["input_ref"] and launch["phase"]==wait["phase"]=="train"
        and launch["physical_gpu"]==wait["physical_gpu"] and launch["physical_gpu"] in (4,5),
        "Exact successful same-launcher actual TRAIN wait/lineage")

def validate_eval_train_io_v1(reader,spec):
    require(spec["phase"]=="evaluate","Evaluation lineage only")
    validate_train_lineage_v1(reader,spec)
    from arggnn import train_marker_grid_reset_chain_v1 as worker
    train=reader.decode_json(spec["train_completion_ref"])
    launch=reader.decode_json(spec["train_launch_ref"])
    expected_input=HERE+"/admitted_train/launch_input.json"
    require(train["input_ref"]["path"]==expected_input
        and spec["train_launch_ref"]["path"]==HERE+"/admitted_train/launch.json"
        and spec["train_actual_wait_ref"]["path"]==HERE+"/admitted_train/actual_wait.json",
        "Only this fresh H048 TRAIN admission, launcher and actual wait")
    old=reader.decode_json(train["input_ref"])
    train_output="experiments/results/H048_GPU%d_grid_reset_train_seed101_v1"%launch["physical_gpu"]
    require(spec["train_completion_ref"]["path"]==train_output+"/completed.json"
        and train["config_ref"]==old["refs"]["config"] and old["phase"]=="train"
        and old["physical_gpu"]==launch["physical_gpu"] and old["output_directory"]==train_output
        and old["hypothesis_id"]=="H048" and old["producer_ref"]==train["producer_ref"]
        and train["producer_ref"]==ref(worker.SOURCE)
        and train["preparation_completion_ref"]==old["preparation_completion_ref"],
        "Exact original H048 input/config/runtime/preparation and frozen producer")
    groups=train["checkpoint_groups"]
    require(set(groups)=={str(p) for p in worker.PASSES},"Six original H048 checkpoint groups")
    for at,arms in groups.items():
        require(set(arms)==set(worker.ARMS),"Both matched arms per group")
        for arm,r in arms.items():
            require(set(r)=={"path","sha256"} and r["path"]==train_output+"/"+arm+".pass_%02d.pt"%int(at)
                and len(r["sha256"])==64 and all(c in "0123456789abcdef" for c in r["sha256"]),
                "Exact generated checkpoint refs; payloads verified by scientific reader at execution")
    return dict(train_input_ref=train["input_ref"],train_config_ref=train["config_ref"],
        train_runtime_ref=old["refs"]["runtime"],train_checkpoint_groups=groups)
