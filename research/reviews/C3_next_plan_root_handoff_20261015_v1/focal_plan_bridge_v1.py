"""Exact native focal-only + existing PyG plan bridge. Default metadata only.

Root alone admits/executes this CPU source through the existing active980 guard.
No MLM corpus or native fit is required. No private truth capability is accepted.
"""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,hashlib,importlib.util,json,os,sys,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
from arggnn.american_c3_common_v1 import require,ref,read,bound,Writer,runtime_report,CLASSES
REL="research/reviews/C3_next_plan_root_handoff_20261015_v1"
HERE=ROOT/REL;SOURCE=REL+"/focal_plan_bridge_v1.py"
TRAIN=tuple("american_c3_short_train_%03d"%i for i in range(20))
REGISTRY_REF=dict(path="research/reviews/American_C3_short_train2mb_wave_source_v1/parent_registry.inactive.json",sha256="c75c111bc48f94d164fe9d6dfe4f8796290b03d1093aca88602049ed47f80e37")
GUARD_REF=dict(path="research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py",sha256="fc71030865e0a3d5da31120d4daa2caa0b2d39ec7dd212b8c46c476b5a866d56")
POLICY_REF=dict(path="research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json",sha256="eb3408004a508136aa434c0a2e02a9d14a5598a96b8483c264c93026f9326018")
REPAIR_REF=dict(path="research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json",sha256="f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db")
MEMO_REF=dict(path="research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py",sha256="c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364")
V4_REF=dict(path="src/arggnn/project_budget_v4.py",sha256="a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47")
CORE="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e"
BASE="a8049bc103afb1cbbbce6f8fc3707a21de9613b23034599641b688994fc5caa3"
CAP=33554432;RSS_CAP=512000000000;DEADLINE=7200
GENEALOGY="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG"

def save(folder,name,value):
    require(Path(name).name==name,"Flat root metadata")
    with (folder/name).open("x") as f:json.dump(value,f,indent=2);f.write("\n")
    return ref(folder/name)

def parent_metadata(parent):
    require(parent in TRAIN,"Canonical short TRAIN only")
    row=next(r for r in read(REGISTRY_REF)["parents"] if r["parent_id"]==parent)
    index=TRAIN.index(parent)
    origin="American_C3_short_train2mb_wave_supervision_v2" if index<14 else "American_C3_short_train2mb_wave_980_source_v1"
    wait_path="research/reviews/"+origin+"/admitted/"+parent+"/infer/actual_wait.json"
    wait_ref=ref(wait_path);wait=read(wait_ref)
    require(wait["parent_id"]==parent and wait["stage"]=="infer" and wait["returncode"]==0
        and wait["actual_direct_worker_wait"] is True and wait["worker_process_group_drained"] is True
        and wait["operational_failure"] is None,"Actual successful inferred-observation wait/drain")
    completed_ref=wait["completed_ref"]
    require(completed_ref["path"]=="experiments/results/American_C3_short_train2mb_wave_v2/"+parent+"/inference/completed.json",
        "Actual wave_v2 completion")
    completed=read(completed_ref);observation_ref=completed["observation_ref"];m=read(observation_ref)
    require(completed["status"]=="completed_current_C3_inference" and m["parent_id"]==parent
        and m["split"]=="train" and m["sequence_length_bp"]==2000000
        and m["sample_unit"]=="haploid" and m["counts_haploid"]=={"AFR":200,"EUR":200,"ASIA":200,"ADMIX":1000}
        and m["class_order"]==list(CLASSES) and m["genealogy_kind"]==GENEALOGY
        and m["query_ancestry_truth_in_inputs"] is False and m["sites_and_sample_identities_verified"] is True,
        "Exact inferred short TRAIN; ORACLE and heldout inputs denied")
    return dict(parent_id=parent,observation_ref=observation_ref,inference_completion_ref=completed_ref,
        inference_actual_wait_ref=wait_ref,path_sampling_seed=row["path_sampling_seed"],registry_ref=REGISTRY_REF)

def guard_module(*,consumer):
    for r in (GUARD_REF,POLICY_REF,REPAIR_REF,MEMO_REF,V4_REF):bound(r)
    loader=importlib.util.spec_from_file_location("_C3_focal_plan_actual980",bound(GUARD_REF))
    m=importlib.util.module_from_spec(loader);loader.loader.exec_module(m)
    m.configure_v2(REPAIR_REF,policy_ref=POLICY_REF)
    require(m.MODE=="prospective_980gb_v1","Current980 guard only")
    if consumer:
        loader=importlib.util.spec_from_file_location("_C3_focal_plan_memo2",bound(MEMO_REF))
        memo=importlib.util.module_from_spec(loader);loader.loader.exec_module(memo)
        memo.install(m.budget.v4,expected_source_path=ROOT/V4_REF["path"],expected_source_sha256=V4_REF["sha256"])
    else:
        require(all(getattr(m.budget.v4,n).__module__==m.budget.v4.__name__ for n in ("canonical","file_bytes","mutex")),
            "Original unmemoized root mutation functions")
    return m

def admit(parent):
    require(Path.cwd().resolve()==ROOT and os.environ.get("CUDA_VISIBLE_DEVICES")=="","Root explicit CPU repository")
    meta=parent_metadata(parent);packet=HERE/("admitted_"+parent)
    output="experiments/results/C3_short_TRAIN_focal_plans_v1/"+parent
    require(not packet.exists() and not (ROOT/output).exists(),"Fresh parent scope; no replay")
    manifest_ref=ref(HERE/"source_manifest.json");manifest=read(manifest_ref)
    for r in manifest["source_refs"]:bound(r)
    cfg=dict(schema="C3_short_TRAIN_exact_native_focal_and_plan_v1",execution_enabled=True,
        output_cap_bytes=CAP,metadata=meta,max_parent_trees=100000,max_raw_path_nodes=1024,
        query_focal_count=10000,max_owned_RSS_bytes=RSS_CAP,deadline_seconds=DEADLINE)
    packet.mkdir();cfg_ref=save(packet,"config.json",cfg);runtime_ref=save(packet,"runtime.json",runtime_report())
    m=guard_module(consumer=False)
    with m.budget.v4.mutex(ROOT):state=m._state_locked(CORE,BASE,fresh=True)
    view,bindings=state[2],state[3]
    consumers=[cfg_ref,runtime_ref,manifest_ref,*manifest["source_refs"],REGISTRY_REF,
        meta["inference_actual_wait_ref"],meta["inference_completion_ref"],meta["observation_ref"],*bindings.values()]
    consumers=list({(r["path"],r["sha256"]):r for r in consumers}.values())
    require(len(consumers)<=64,"Exact existing64 direct-ref limit")
    now=datetime.now(timezone.utc);expiry=(now+timedelta(hours=3)).isoformat()
    job="C3-short-native-focals-plan-"+parent+"-v1"
    request=dict(key="H041/"+job,output=str(ROOT/output),auxiliary_scopes=[],reserved_bytes=CAP,consumer_bindings=consumers)
    authority=m.operation_draft_v1("reserve",request,expected_before=m._pins(view),bindings=bindings)
    authority.update(active=True,expires_at_utc=expiry)
    authority_ref=save(packet,"reserve_authority.json",authority)
    receipt=m.apply_reserve_v1(authority_ref,expected_source_sha256=CORE,expected_launcher_sha256=BASE)
    save(packet,"reserve_receipt.json",receipt)
    storage=dict(kind="project_budget_v9_storage_record",active=True,consumer_uses_v9_guard=True,
        accounting_mode=m.MODE,job_id=job,output_directory=output,reserved_new_bytes=CAP,
        reservation=receipt["value"],expires_at_utc=expiry,initial_ledger_pins=receipt["initial_ledger_pins"],
        completed_overlay_policy_ref=POLICY_REF,budget_components=dict(implementation=ref(m.budget.SOURCE_REL),
        launcher=ref(m.budget.LAUNCHER_REL),consumer_adapter=GUARD_REF))
    spec=dict(schema="C3_focal_plan_root_admission_v1",root_authorizer="root_agent",execution_authorized=True,
        GPU_authorized=False,parent_id=parent,job_id=job,output_directory=output,source_ref=ref(SOURCE),
        source_manifest_ref=manifest_ref,config_ref=cfg_ref,runtime_ref=runtime_ref,storage=storage,
        storage_adapter_ref=GUARD_REF,completed_overlay_policy_ref=POLICY_REF,marker_repair_authority_ref=REPAIR_REF,
        created_at_utc=now.isoformat(),expires_at_utc=expiry)
    result=save(packet,"admission.json",spec)
    print(json.dumps(dict(admission_ref=result,reserved_bytes=CAP,scientific_execution_started=False)),flush=True)

def enumerate_native_records(ts,get_path_key,*,max_path_nodes,resource_check):
    """Same sample/tree first-appearance ID order; sets are membership-only."""
    rows=[];offset=0
    for column,sample in enumerate(ts.samples()):
        seen=set()
        for number,tree in enumerate(ts.trees()):
            if number%256==0:resource_check()
            key=get_path_key(tree,int(sample))
            require(key and key[0]==int(sample) and len(key)<=max_path_nodes,"Complete author path, no truncation")
            if key not in seen:seen.add(key)
        rows.append(dict(sample_index=column,sample_node=int(sample),start_record_id=offset,unique_paths=len(seen)))
        offset+=len(seen)
        del seen
        if (column+1)%200==0:print(json.dumps(dict(stage="C3_native_focal_count",samples_completed=column+1,unique_paths=offset)),flush=True)
    return rows,offset

def make_focals(ts,rows,seed,get_path_key,*,max_path_nodes,count,resource_check):
    import numpy as np
    counts,total=enumerate_native_records(ts,get_path_key,max_path_nodes=max_path_nodes,resource_check=resource_check)
    reference_paths=sum(r["unique_paths"] for r in counts[:600])
    require(len(counts)==1600 and total-reference_paths>=count,"All native query path IDs, no fallback")
    # Deliberately identical ordered int64 array and Generator.choice call.
    query_ids=np.arange(reference_paths,total,dtype=np.int64)
    chosen=np.random.default_rng(seed).choice(query_ids,size=count,replace=False);chosen.sort();del query_ids
    columns=np.empty(count,dtype=np.int64);samples=np.empty(count,dtype=np.int64)
    intervals=np.empty((count,2),dtype=np.float64);filled=np.zeros(count,dtype=np.bool_)
    for item in counts[600:]:
        first=item["start_record_id"];last=first+item["unique_paths"]
        lo,hi=np.searchsorted(chosen,[first,last])
        if lo==hi:continue
        wanted={int(chosen[j]):j for j in range(lo,hi)}
        seen=set();next_id=first
        for number,tree in enumerate(ts.trees()):
            if number%256==0:resource_check()
            key=get_path_key(tree,item["sample_node"])
            require(key and key[0]==item["sample_node"] and len(key)<=max_path_nodes,"Exact untruncated path")
            if key in seen:continue
            seen.add(key);record=next_id;next_id+=1
            if record in wanted:
                j=wanted[record];columns[j]=item["sample_index"];samples[j]=item["sample_node"]
                intervals[j]=[tree.interval.left,tree.interval.right];filled[j]=True
        require(next_id==last,"Second complete per-sample enumeration count changed")
        del seen
    require(bool(filled.all()),"Every original selected record has its first-occurrence coordinate")
    return dict(record_ids=chosen,sample_nodes=samples,sample_indices=columns,
        positions_bp0=(intervals[:,0]+intervals[:,1])/2,first_occurrence_intervals_bp=intervals),counts,total,reference_paths

def execute(record):
    spec=read(record);cfg=read(spec["config_ref"])
    require(Path.cwd().resolve()==ROOT and os.environ.get("CUDA_VISIBLE_DEVICES")==""
        and spec["schema"]=="C3_focal_plan_root_admission_v1" and spec["root_authorizer"]=="root_agent"
        and spec["execution_authorized"] is True and spec["GPU_authorized"] is False
        and spec["source_ref"]==ref(SOURCE) and cfg["execution_enabled"] is True,
        "Exact current root CPU capability")
    require(datetime.now(timezone.utc)<datetime.fromisoformat(spec["expires_at_utc"]),"Unexpired actual root scope")
    require(cfg["metadata"]==parent_metadata(spec["parent_id"]),"Same completed observations and frozen focal seed")
    manifest=read(spec["source_manifest_ref"])
    for r in manifest["source_refs"]:bound(r)
    require(all(r in spec["storage"]["reservation"]["consumer_bindings"] for r in
        [*manifest["source_refs"],spec["source_manifest_ref"],spec["config_ref"],spec["runtime_ref"]]),
        "Actual full helper/producer/runtime closure reserved")
    require(read(spec["runtime_ref"])==runtime_report(),"Exact CPU interpreter/package metadata")
    m=guard_module(consumer=True);storage=spec["storage"]
    m.validate_storage_v9(spec,cfg,storage,source_ref=spec["source_ref"],config_ref=spec["config_ref"],runtime_ref=spec["runtime_ref"])
    w=Writer(spec,m);started=time.monotonic()
    def check():
        require(time.monotonic()-started<cfg["deadline_seconds"],"CPU parent deadline; no fallback")
        require(datetime.now(timezone.utc)<datetime.fromisoformat(spec["expires_at_utc"]),"Unexpired writer")
        rss=int(Path("/proc/self/statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
        require(rss<=cfg["max_owned_RSS_bytes"],"Current single-process CPU RSS cap")
    try:
        check()
        import numpy as np,tskit
        from arggnn import american_c3_paths_v1 as native
        meta=cfg["metadata"];observation=read(meta["observation_ref"]);roster=read(observation["roster_ref"]);rows=roster["samples"]
        require(len(rows)==1600 and [r["sample_index"] for r in rows]==list(range(1600))
            and [r["role"] for r in rows]==["pure_source_reference"]*600+["query"]*1000
            and [r["observed_reference_class"] for r in rows]==[2]*200+[0]*200+[1]*200+[None]*1000,
            "Exact R600/Q1000 AFR/EUR/EAS column roles, never query labels")
        ts=tskit.load(str(bound(observation["tree_ref"])))
        require(ts.num_samples==1600 and ts.sequence_length==2000000 and 0<ts.num_trees<=cfg["max_parent_trees"]
            and ts.num_nodes<2**31 and list(map(int,ts.samples()))==[r["observed_node_id"] for r in rows],
            "Original complete bounded inferred sampled tree, no subsampling")
        require(ts.num_migrations==ts.num_populations==0 and (ts.tables.nodes.population==-1).all(),
            "No internal source/population annotations")
        arrays,counts,total,reference_paths=make_focals(ts,rows,meta["path_sampling_seed"],native.author_functions()["get_path_key"],
            max_path_nodes=cfg["max_raw_path_nodes"],count=10000,resource_check=check)
        arrays.update(parent_id=np.asarray(meta["parent_id"]),class_order=np.asarray(CLASSES))
        focal_ref=w.npz("focals.npz",**arrays)
        count_ref=w.json("native_full_path_order_counts.json",dict(parent_id=meta["parent_id"],sample_rows=counts,
            complete_unique_paths=total,reference_unique_paths=reference_paths,record_order="sample_then_tree_first_appearance",
            sets_iterated=False,query_ancestry_truth_opened=False))
        prep_ref=w.json("focal_completed.json",dict(status="completed_C3_exact_native_focal_registry_only_v1",
            parent_id=meta["parent_id"],observation_ref=meta["observation_ref"],focals_ref=focal_ref,
            genealogy_kind=GENEALOGY,class_order=list(CLASSES),query_ancestry_truth_opened=False,
            query_focal_count=10000,path_sampling_seed=meta["path_sampling_seed"],
            full_native_MLM_corpus_prepared=False,native_model_training_performed=False,
            full_path_order_count_ref=count_ref,source_author_function_ref=ref(native.AUTHOR),
            original_native_focal_rule_source_ref=ref(native.SOURCE),operational_bridge_ref=ref(SOURCE),
            focal_coordinate_rule="unchanged_label_blind_first_occurrence_midpoint"))
        del ts,arrays
        check()
        from arggnn.american_c3_gnn_dataset_v1 import DEFAULTS,build_plan
        from arggnn.american_c3_gnn_stream_train_v1 import validate_plan_v1
        plan_cfg=dict(DEFAULTS,execution_enabled=True,observation_ref=meta["observation_ref"],
            focal_registry_ref=focal_ref,native_preparation_ref=prep_ref,expected_haplotypes=1600,reference_counts=[200]*3)
        plan=build_plan(plan_cfg);validate_plan_v1(plan,parent_id=meta["parent_id"],length_bp=2000000)
        check();plan_ref=w.json("block_index.json",plan)
        result=dict(status="completed_C3_short_TRAIN_native_focal_and_PyG_plan_v1",parent_id=meta["parent_id"],
            observation_ref=meta["observation_ref"],focal_registry_ref=focal_ref,native_preparation_ref=prep_ref,
            block_index_ref=plan_ref,trees=plan["trees"],sites=plan["sites"],blocks=len(plan["blocks"]),
            focal_records=10000,seconds=time.monotonic()-started,query_ancestry_truth_opened=False,
            native_MLM_or_training_dependency=False,all_trees_and_samples_retained=True,source_ref=ref(SOURCE))
        w.json("completed.json",result);print(json.dumps(result),flush=True)
    except Exception as error:
        w.json("failed.json",dict(status="failed_C3_focal_plan_preserved",error_type=type(error).__name__,error=str(error),
            parent_id=spec["parent_id"],automatic_replay=False,query_ancestry_truth_opened=False))
        raise

def collect_plans():
    """Read completed metadata only; never open focal/tree/genotype payloads."""
    refs={};completed={}
    for parent in TRAIN:
        path="experiments/results/C3_short_TRAIN_focal_plans_v1/"+parent+"/completed.json"
        completion_ref=ref(path);value=read(completion_ref);meta=parent_metadata(parent)
        require(value["status"]=="completed_C3_short_TRAIN_native_focal_and_PyG_plan_v1"
            and value["parent_id"]==parent and value["observation_ref"]==meta["observation_ref"]
            and value["focal_records"]==10000 and value["query_ancestry_truth_opened"] is False
            and value["all_trees_and_samples_retained"] is True and value["source_ref"]==ref(SOURCE),
            "Every original short TRAIN parent has an actual exact-focal/plan completion")
        refs[parent]=value["block_index_ref"];completed[parent]=completion_ref
    print(json.dumps(dict(status="all20_completed_C3_TRAIN_plan_metadata",parent_plan_refs=refs,
        completion_refs=completed,scientific_payload_opened=False),indent=2))

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group()
    g.add_argument("--admit",choices=TRAIN);g.add_argument("--execute",action="store_true");g.add_argument("--collect-plans",action="store_true")
    p.add_argument("--admission");p.add_argument("--expected-admission-sha256");args=p.parse_args(argv)
    if args.collect_plans:collect_plans();return 0
    if args.admit:admit(args.admit);return 0
    if args.execute:
        require(args.admission and args.expected_admission_sha256,"Exact active admission path/SHA")
        execute(dict(path=args.admission,sha256=args.expected_admission_sha256));return 0
    print(json.dumps(dict(status="source_only_no_guard_or_payload",TRAIN_parents=list(TRAIN),
        output_cap_per_parent_bytes=CAP,CPU_only=True,native_MLM_dependency=False,
        source_ref=ref(SOURCE)),indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
