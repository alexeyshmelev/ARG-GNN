"""Pinned H042 root utilities plus H046 metadata planning. Import is inert."""
from pathlib import Path
import hashlib,importlib.util,json,os
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/H046_GPU45_flank_root_source_v1"
PREVIOUS="research/reviews/H042_GPU45_clade_flags_root_source_v1/root_driver.py"
PREVIOUS_SHA="5b87957f3667187b97172b00871f3cc5c85981810179554578e9779bcca9d0b0"
if hashlib.sha256((ROOT/PREVIOUS).read_bytes()).hexdigest()!=PREVIOUS_SHA:
    raise ValueError("Frozen H042 root utility source changed")
_spec=importlib.util.spec_from_file_location("_H046_frozen_root_patterns",ROOT/PREVIOUS)
previous=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(previous)
require,ref,read,save,unique,policy_env=(
    previous.require,previous.ref,previous.read,previous.save,previous.unique,previous.policy_env)
GUARD_REF,POLICY_REF,LAUNCHER_REF,REPAIR_REF,AUTH_REF,TEMPLATE_REF,CORE,BASE=(
    previous.GUARD_REF,previous.POLICY_REF,previous.LAUNCHER_REF,previous.REPAIR_REF,
    previous.AUTH_REF,previous.TEMPLATE_REF,previous.CORE,previous.BASE)
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
    guard=previous.load_guard()
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
