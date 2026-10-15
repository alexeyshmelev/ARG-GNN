"""Pinned H042 root utilities plus H046 metadata planning. Import is inert."""
from pathlib import Path
import hashlib,importlib.util,json,os
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H046_GPU45_flank_eval980_source_v1"
PREVIOUS="research/reviews/H042_GPU45_clade_flags_root_source_v1/root_driver.py"
PREVIOUS_SHA="5b87957f3667187b97172b00871f3cc5c85981810179554578e9779bcca9d0b0"
if hashlib.sha256((ROOT/PREVIOUS).read_bytes()).hexdigest()!=PREVIOUS_SHA:
    raise ValueError("Frozen H042 root utility source changed")
_spec=importlib.util.spec_from_file_location("_H046_frozen_root_patterns",ROOT/PREVIOUS)
previous=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(previous)
require,ref,read,save,unique=(
    previous.require,previous.ref,previous.read,previous.save,previous.unique)
REPAIR_REF,AUTH_REF,TEMPLATE_REF,CORE,BASE=(
    previous.REPAIR_REF,previous.AUTH_REF,previous.TEMPLATE_REF,previous.CORE,previous.BASE)
GUARD_REF={'path': 'research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py', 'sha256': 'fc71030865e0a3d5da31120d4daa2caa0b2d39ec7dd212b8c46c476b5a866d56'}
POLICY_REF={'path': 'research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json', 'sha256': 'eb3408004a508136aa434c0a2e02a9d14a5598a96b8483c264c93026f9326018'}
LAUNCHER_REF={'path': 'scripts/launch_reserved_gpu_authorized_GPU45_980_v1.py', 'sha256': '0ca3c59e49f5fd55f5a798c99a4a8a6613e71a90bcb3f89a04eb66e7555046ca'}
TRAIN_INPUT_REF={'path': 'research/reviews/H046_GPU45_flank_root_source_v2/admitted_train/launch_input.json', 'sha256': '938bc89b753d08db66a4b1141cdc551f192e4d3a7952f215d5b866ee2bd3f752'}
TRAIN_CONFIG_REF={'path': 'research/reviews/H046_GPU45_flank_root_source_v2/config_inactive.json', 'sha256': '13d1b2fa0503a452c3bee7ba3b47bf3d702ade18a796c0f4c6e562e6f129f5a8'}
TRAIN_LAUNCH_REF={'path': 'research/reviews/H046_GPU45_flank_root_source_v2/admitted_train/launch.json', 'sha256': 'a899af3d485339b26e1873eb13d7db0b514c4886b668a5e81127c29e8d20cd0e'}
TRAIN_RUNTIME_REF={'path': 'research/reviews/H046_GPU45_flank_root_source_v2/admitted_train/runtime.json', 'sha256': '2a3071f159b1193a29e7e0335754e8f049a9a720a2f6692f7b93af8c2934d007'}
TRAIN_OUTPUT='experiments/results/H046_GPU5_flank_train_seed101_v2'

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
    ms=importlib.util.spec_from_file_location("_H046_eval980_guard",ROOT/GUARD_REF["path"])
    guard=importlib.util.module_from_spec(ms);ms.loader.exec_module(guard)
    guard.configure_v2(REPAIR_REF,policy_ref=POLICY_REF)
    require(guard.MODE=="prospective_980gb_v1","Actual980 storage mode")
    if not consumer:
        require(all(getattr(guard.budget.v4,name).__module__==guard.budget.v4.__name__
                    for name in ("canonical","file_bytes","mutex")),
                "Reserve/close mutations require unmemoized original v4 functions")
        return guard,None
    spec=importlib.util.spec_from_file_location("_H046_memo2",ROOT/MEMO_REF["path"])
    memo=importlib.util.module_from_spec(spec);spec.loader.exec_module(memo)
    adapter=memo.install(guard.budget.v4,expected_source_path=ROOT/V4_REF["path"],
                         expected_source_sha256=V4_REF["sha256"])
    return guard,adapter

def count_plan_v1(manifest,phase,preparation):
    from arggnn import train_continuous_clade_flank_v1 as worker
    require(manifest.get("schema")=="root_H043_original40_observed_count_manifest_v1"
        and manifest.get("counts_only_no_labels_or_model_metrics") is True
        and manifest.get("outer_included") is False
        and set(manifest["parents"])==set(worker.TRAIN+worker.VAL),
        "Exact unchanged original40 observation-only counts required")
    require(phase in ("train","evaluate") and set(preparation["parents"])==set(worker.TRAIN+worker.VAL),
        "Exact phase and original all40 marker preparation")
    ids=worker.TRAIN if phase=="train" else worker.VAL
    raw={};recipes=0;prediction=0
    for pid in ids:
        row=manifest["parents"][pid];prepared=preparation["parents"][pid]["counts"]
        t,n,h=(row[k] for k in ("native_holds","native_nodes","haplotypes"))
        s,p=prepared["sites"],prepared["scoring_centers"]
        require(all(type(v) is int and v>0 for v in (s,t,n,h,p)) and h==952
            and prepared["haplotypes"]==h and prepared["native_holds"]==t
            and prepared["queries"]==686 and p<=20000 and s<=2000000,
            "Actual full952 source-native/SNP/original-grid dimensions")
        # H046 recipes contain scalar dimensions and fixed-size hashes, with
        # no per-tree/member/grid lists. Generous bounded per-parent JSON.
        recipe=131072;recipes+=recipe
        prediction+=12*16*p*686 if phase=="evaluate" else 0
        raw[pid]=dict(native_holds=t,native_nodes=n,haplotypes=h,sites=s,grid_centers=p,
            compact_payload_upper_bound_bytes=24*n+4*t+40*p,
            GPU_flank_payload_upper_bound_bytes=24*n+4*t+16*p,
            CPU_checkpoint_local_feature_upper_bound_bytes=40*p*686,
            original_grid_logits_or_target_bytes=16*p*686,
            recipe_JSON_upper_bound_bytes=recipe)
    checkpoint=12*1048576 if phase=="train" else 0
    fixed=16777216
    projected=recipes+checkpoint+LOG_CAP+fixed
    require(projected<=OUTPUT_CAP,"Count-derived output bound exceeds root100MB cap")
    return dict(phase=phase,parent_count=len(ids),recipes_JSON_upper_bound_bytes=recipes,
        checkpoint_archive_allowance_bytes=checkpoint,bounded_log_bytes=LOG_CAP,
        fixed_seals_metrics_history_filesystem_overhead_bytes=fixed,
        total_output_scope_upper_bound_bytes=projected,output_cap_bytes=OUTPUT_CAP,
        all96_original_grid_probability_bytes=prediction,parents=raw)


def current_RSS_bytes_v1():
    parts=Path("/proc/self/statm").read_text().split()
    require(len(parts)>=2 and parts[1].isdigit(),"Current-process Linux RSS required")
    return int(parts[1])*os.sysconf("SC_PAGE_SIZE")

def preparation_check_v1(pid,bank,declared,cfg,counts,preparation,*,rss_reader=current_RSS_bytes_v1):
    from arggnn import train_continuous_clade_flank_v1 as worker
    prepared=preparation["parents"][pid]["counts"]
    actual=worker.flank_upper_bounds_v1(bank,prepared["scoring_centers"])
    require(declared==actual,"Declared bound must equal actual original bank/grid dimensions")
    row=counts["parents"][pid]
    require(all(row[k]==actual[k] for k in ("native_holds","native_nodes","haplotypes"))
        and prepared["sites"]==actual["sites"]
        and prepared["native_holds"]==actual["native_holds"]
        and prepared["haplotypes"]==actual["haplotypes"] and prepared["queries"]==686,
        "Actual original dimensions differ from frozen counts/preparation")
    n,t,h,p=(actual[k] for k in ("native_nodes","native_holds","haplotypes","grid_centers"))
    raw=24*n+4*t+40*p;gpu=24*n+4*t+16*p
    require(actual["compact_flank_payload_upper_bound_bytes"]==raw,"24N+4T+40P compact bound")
    require(raw<=min(cfg["flank_payload_cap_bytes"],cfg["flank_cache_cap_bytes"])
        and gpu<=cfg["GPU_flank_cap_bytes"],"Complete flank payload exceeds root cap")
    require(actual["preparation_named_array_allowance_bytes"]<=cfg["flank_preparation_cap_bytes"],
        "Named preparation arrays exceed root cap")
    # Conservative Python forest/container planning plus named arrays. This is
    # neither a total allocator/autograd proof nor actual GPU qualification.
    scratch=2058*n+128*h+raw+64*n+actual["preparation_named_array_allowance_bytes"]
    require(scratch<=cfg["flank_preparation_cap_bytes"],"Conservative preparation planning exceeds cap")
    rss=rss_reader()
    require(type(rss) is int and rss>=0 and rss+scratch<=RSS_CAP,
        "Current RSS plus preparation allowance exceeds512GB")
    return dict(parent_id=pid,current_RSS_bytes=rss,preparation_planning_allowance_bytes=scratch,
        projected_RSS_planning_bytes=rss+scratch,raw_bound_bytes=raw,
        CPU_checkpoint_local_feature_upper_bound_bytes=actual["CPU_all_checkpoint_local_feature_bytes"],
        original_grid_logits_or_target_bytes=actual["original_grid_logits_or_target_bytes"],
        all96_uniform_parent_grid_probability_upper_bound_bytes=actual["all96_original_grid_probability_bytes"],
        full_logits_materialized=True,actual_backbone_autograd_peak_unmeasured=True)


def validate_train_lineage_v1(reader,spec):
    if spec["phase"]!="evaluate":return
    train=reader.decode_json(spec["train_completion_ref"])
    launch=reader.decode_json(spec["train_launch_ref"]);wait=reader.decode_json(spec["train_actual_wait_ref"])
    require(train.get("schema")=="H046_paired_train_completion_v1"
        and train.get("status")=="completed_fixed30pass_paired_TRAIN"
        and train.get("hypothesis_id")=="H046" and train.get("arms")==["foreign_flank","own_state"]
        and train.get("steps")=={"foreign_flank":960,"own_state":960}
        and train.get("VAL_truth_read") is False and train.get("outer_truth_read") is False
        and train.get("same_VAL_is_development_evidence") is True,"Actual paired fixed H046 TRAIN completion required")
    require(wait.get("actual_direct_launcher_wait") is True and wait.get("returncode")==0
        and wait.get("logging_error") is None and wait.get("supervisor_error") is None
        and wait["completed_ref"]==spec["train_completion_ref"]
        and wait["launcher_pid"]==launch["launcher_pid"] and launch["input_ref"]==train["input_ref"]
        and launch["phase"]=="train" and wait["phase"]=="train"
        and launch["physical_gpu"]==wait["physical_gpu"] and launch["physical_gpu"] in (4,5),
        "Exact successful same-launcher actual TRAIN wait/lineage required")


def validate_eval_train_io_v1(reader,spec):
    """Validate immutable original TRAIN JSON and checkpoint references without payload reads."""
    require(spec["phase"]=="evaluate","This successor admits evaluation only")
    validate_train_lineage_v1(reader,spec)
    require(spec["train_completion_ref"]["path"]==TRAIN_OUTPUT+"/completed.json"
        and spec["train_launch_ref"]==TRAIN_LAUNCH_REF
        and spec["train_actual_wait_ref"]["path"]==
            "research/reviews/H046_GPU45_flank_root_source_v2/admitted_train/actual_wait.json",
        "Actual original GPU5 TRAIN terminal lineage only")
    train=reader.decode_json(spec["train_completion_ref"])
    old=reader.decode_json(TRAIN_INPUT_REF)
    require(train["input_ref"]==TRAIN_INPUT_REF and train["config_ref"]==TRAIN_CONFIG_REF
        and old["phase"]=="train" and old["physical_gpu"]==5
        and old["output_directory"]==TRAIN_OUTPUT and old["refs"]["runtime"]==TRAIN_RUNTIME_REF
        and train["preparation_completion_ref"]==old["preparation_completion_ref"],
        "Unchanged original TRAIN input/config/runtime/preparation")
    groups=train["checkpoint_groups"]
    require(set(groups)=={"5","10","15","20","25","30"},"Original six TRAIN checkpoint groups")
    for at,arms in groups.items():
        require(set(arms)=={"foreign_flank","own_state"},"Original paired checkpoint arms")
        for arm,r in arms.items():
            require(set(r)=={"path","sha256"}
                and r["path"]==TRAIN_OUTPUT+"/"+arm+".pass_%02d.pt"%int(at)
                and len(r["sha256"])==64 and all(c in "0123456789abcdef" for c in r["sha256"]),
                "Exact original checkpoint references, verified by unchanged scientific reader at execution")
    return dict(train_input_ref=TRAIN_INPUT_REF,train_config_ref=TRAIN_CONFIG_REF,
        train_runtime_ref=TRAIN_RUNTIME_REF,train_checkpoint_groups=groups)
