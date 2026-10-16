"""TRAIN-only native-hold representation diagnostic, inert on import.
Root owns admission and invokes run_audit_v1 only under its original CPU monitor.
No learned model, graph decoding, optimizer, CUDA, heldout read or direct writer.
"""
from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,io,json,os

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
FIXED={"formula_contract": {"path": "research/reviews/post_H050_marker_tree_feedback_design_20261016_v1/TRAIN_audit_source_contract.v1.json", "sha256": "cb9b797176fc076f756b84a374b5f6b5303fe881bf6653c4f0f1eecd27e5f0d7"}, "payload_manifest": {"path": "research/reviews/native_hold_resolution_train_audit_source_20261016_v1/TRAIN_payload_manifest.source_frozen.v1.json", "sha256": "59c5149b0a5be6e3a1d8d48f4f43c4ef7de2d99cec8df1045237d0e3be4890ad"}, "registration": {"path": "research/reviews/native_hold_resolution_train_registration_root_20261016_v1/protocol.root_frozen_v2.json", "sha256": "e646153485aac3bdb88099982592d34e236069b71c88114990e78f82521ab4e9"}, "registration_v1": {"path": "research/reviews/native_hold_resolution_train_registration_root_20261016_v1/protocol.root_frozen_v1.json", "sha256": "44fbfaeb8c581bdababc4a76faff5f55b35fc47933bdad5738b7015a844dc8cd"}, "scientific_sources": {"intersections": {"path": "scripts/train_ancestralpaths_biological_v1.py", "sha256": "826a1e8c3053cf63b8374361f793903970dd75d44cbcd536755e9aaa3b985056"}, "native_mass": {"path": "src/arggnn/train_mutation_routed_full_panel_pyg_v1.py", "sha256": "c01c4d9506faae7bf537282e4d722b115b3c69cc5de1d4570758136de292dae3"}, "target": {"path": "src/arggnn/mutation_routed_full_panel_primary_v1.py", "sha256": "a94fd5af1cd10b1605f75621c36a130faa3f9d5ad71068f8b187d8e73c7e5374"}}}
TRAIN=tuple("ap_expanded_train_%03d"%i for i in range(32))
COHORT_COUNTS={"present_day":182,"Bronze_Age":162,"BAA":14,"Neo":302,"Yam":26}
_RAN=False

def require(ok,message):
    if not ok:raise ValueError(message)

def checked_bytes_v1(record,limit,resource_check):
    resource_check("before_exact_TRAIN_source_or_payload_read")
    require(isinstance(record,dict) and set(record)=={"path","sha256"},"Exact immutable record")
    name=record["path"];p=ROOT/name
    require(isinstance(name,str) and not Path(name).is_absolute() and ".." not in Path(name).parts
        and p.resolve()==p and p.is_file() and not p.is_symlink(),"Canonical project source/payload")
    size=p.stat().st_size
    require(0<size<=limit,"Declared bounded source/payload size")
    raw=p.read_bytes()
    require(len(raw)==size and hashlib.sha256(raw).hexdigest()==record["sha256"],"Original complete bytes/SHA")
    resource_check("after_exact_TRAIN_source_or_payload_read")
    return raw

def original_definition_v1(record,name,namespace,resource_check):
    source=checked_bytes_v1(record,2*2**20,resource_check).decode()
    nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name]
    require(len(nodes)==1,"One exact original function/class definition")
    tree=ast.Module(body=nodes,type_ignores=[])
    exec(compile(tree,record["path"],"exec"),namespace)
    return namespace[name]

class TRAINReaderV1:
    def __init__(self,records,resource_check):
        self.allowed={(r["path"],r["sha256"]) for r in records}
        self.root=ROOT;self.resource_check=resource_check
    def read(self,record,**kwargs):
        require((record["path"],record["sha256"]) in self.allowed,"Only exact32 TRAIN truth payloads")
        return checked_bytes_v1(record,256*2**20,self.resource_check)

def load_parent_v1(pid,metadata,np,torch,resource_check):
    require(pid in TRAIN and metadata["scoring"]["parent_id"]=={"json":pid}
        and metadata["scoring"]["split"]=={"json":"train"},"Exact TRAIN-only parent")
    raw=checked_bytes_v1(metadata["array_ref"],128*2**20,resource_check)
    require(len(raw)==metadata["compressed_bytes"],"Original compressed observation byte count")
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
        require(len(archive.files)==len(set(archive.files)),"No duplicate NPZ member names")
        required={*metadata["model_spec"].values(),*(v["array_key"] for v in metadata["scoring"].values() if "array_key" in v)}
        require(required.issubset(archive.files),"Exact selective original NPZ member contract")
        arrays={k:archive[v].copy() for k,v in metadata["model_spec"].items()}
        parent={k:archive[v["array_key"]].copy() if "array_key" in v else v["json"]
            for k,v in metadata["scoring"].items()}
    del raw
    hold,spans,columns=(arrays[k] for k in ("readout_hold","spans","query_column"))
    require(hold.dtype.kind in "iu" and hold.ndim==1 and spans.ndim==1
        and len(spans)>0 and np.isfinite(spans).all() and np.all(spans>0)
        and columns.dtype.kind in "iu" and columns.shape==(686,)
        and len(np.unique(columns))==686 and np.all((columns>=0)&(columns<952)),
        "Original complete CPU readout/hold/query members")
    edges=parent["edges"];positions=parent["positions"]
    require(parent["raw_query_nodes"].shape==(686,) and len(np.unique(parent["raw_query_nodes"]))==686
        and parent["cohorts"].shape==(686,) and edges.dtype==np.float64 and edges.shape==(len(positions)+1,)
        and len(positions)>0 and positions.ndim==1 and np.isfinite(edges).all()
        and np.isfinite(positions).all() and np.all(np.diff(edges)>0) and np.all(np.diff(positions)>0)
        and 0<=edges[0]<edges[-1]<=2000000 and hold.shape==positions.shape
        and np.all(np.diff(edges)<=100+1e-8) and np.allclose(np.diff(positions),100,atol=1e-8,rtol=0),
        "Original clipped physical scored-bin support")
    require({c:int((parent["cohorts"]==c).sum()) for c in COHORT_COUNTS}==COHORT_COUNTS,
        "Original full686 observed cohort identities")
    require(set(parent["cohorts"].tolist())==set(COHORT_COUNTS),"No unregistered query cohort")
    bank=SimpleNamespace(readout_hold=torch.from_numpy(hold),
        spans=torch.from_numpy(spans),query_column=torch.from_numpy(columns))
    resource_check("after_selective_TRAIN_parent_decode")
    return parent,bank

def cell_totals_v1(mass,np):
    total=mass.sum(axis=-1,dtype=np.float64)
    maximum=mass.max(axis=-1)
    active=total>0
    mixed=(mass>0).sum(axis=-1)>1
    return dict(support_mass_bp=float(total.sum(dtype=np.float64)),
        correct_oracle_mass_bp=float(maximum.sum(dtype=np.float64)),
        active_hold_query_cells=int(active.sum()),mixed_label_hold_query_cells=int(mixed.sum()),
        mixed_label_hold_query_mass_bp=float(total[mixed].sum(dtype=np.float64)))

def groups_v1(parent,np):
    c=parent["cohorts"]
    result={"all686":np.arange(686),"Present182":np.flatnonzero(c=="present_day"),
        "ancient504":np.flatnonzero(c!="present_day")}
    result.update({c+str(n):np.flatnonzero(parent["cohorts"]==c) for c,n in COHORT_COUNTS.items() if c!="present_day"})
    return result

def grid_oracle_v1(parent,target,helpers,groups,np,resource_check):
    # Stream256 original scored bins at a time, never materialize20k*686*4.
    bins,rows,spans=helpers.intersections(parent["edges"],target.intervals)
    n=len(parent["positions"]);result={g:dict(correct_oracle_mass_bp=0.,support_mass_bp=0.) for g in groups}
    for start in range(0,n,256):
        resource_check("before_original_grid_mass_chunk")
        stop=min(start+256,n)
        lo,hi=np.searchsorted(bins,[start,stop],side="left")
        mass=np.zeros((stop-start,686,4),dtype=np.float64)
        np.add.at(mass,(bins[lo:hi,None]-start,np.arange(686)[None,:],target.c4[rows[lo:hi]]),spans[lo:hi,None])
        require(np.allclose(mass.sum(-1),np.diff(parent["edges"])[start:stop,None],atol=1e-8,rtol=0),
            "Every original fixed-grid bin physical span")
        for group,columns in groups.items():
            selected=mass[:,columns,:]
            result[group]["correct_oracle_mass_bp"]+=float(selected.max(-1).sum(dtype=np.float64))
            result[group]["support_mass_bp"]+=float(selected.sum(dtype=np.float64))
    return result

def finalize_v1(hold,grid,physical_span,queries):
    require(hold["support_mass_bp"]>0 and hold["active_hold_query_cells"]>0,"Positive original scored support")
    return dict(hold,query_haplotypes=queries,original_scored_bp_per_query=physical_span,
        geometric_query_bp=physical_span*queries,
        hold_oracle_span_accuracy=hold["correct_oracle_mass_bp"]/hold["support_mass_bp"],
        mixed_label_hold_query_fraction=hold["mixed_label_hold_query_cells"]/hold["active_hold_query_cells"],
        mixed_label_hold_query_mass_fraction=hold["mixed_label_hold_query_mass_bp"]/hold["support_mass_bp"],
        original_grid_oracle_span_accuracy=grid["correct_oracle_mass_bp"]/grid["support_mass_bp"],
        original_grid_oracle_correct_mass_bp=grid["correct_oracle_mass_bp"],
        original_grid_support_mass_bp=grid["support_mass_bp"],
        grid_minus_hold_oracle_pp=100*(grid["correct_oracle_mass_bp"]/grid["support_mass_bp"]
            -hold["correct_oracle_mass_bp"]/hold["support_mass_bp"]))

def aggregate_v1(parents,np):
    result={}
    metrics=("hold_oracle_span_accuracy","mixed_label_hold_query_fraction",
        "mixed_label_hold_query_mass_fraction","original_grid_oracle_span_accuracy","grid_minus_hold_oracle_pp")
    for group in parents[0]["cohorts"]:
        rows=[p["cohorts"][group] for p in parents]
        summary={metric:dict(equal_parent_mean=float(np.mean([r[metric] for r in rows])),
            parent_sample_SD=float(np.std([r[metric] for r in rows],ddof=1)),
            unit="percentage_points" if metric.endswith("_pp") else "fraction") for metric in metrics}
        sums={k:sum(r[k] for r in rows) for k in ("support_mass_bp","correct_oracle_mass_bp",
            "active_hold_query_cells","mixed_label_hold_query_cells","mixed_label_hold_query_mass_bp",
            "original_grid_support_mass_bp","original_grid_oracle_correct_mass_bp")}
        summary.update(parent_count=len(rows),query_haplotypes_per_parent=rows[0]["query_haplotypes"],pooled_masses_and_counts=sums,
            pooled_hold_oracle_span_accuracy=sums["correct_oracle_mass_bp"]/sums["support_mass_bp"],
            pooled_grid_oracle_span_accuracy=sums["original_grid_oracle_correct_mass_bp"]/sums["original_grid_support_mass_bp"],
            pooled_mixed_label_hold_query_fraction=sums["mixed_label_hold_query_cells"]/sums["active_hold_query_cells"],
            pooled_mixed_label_hold_query_mass_fraction=sums["mixed_label_hold_query_mass_bp"]/sums["support_mass_bp"])
        result[group]=summary
    return result

def run_audit_v1(*,resource_check):
    global _RAN
    require(not _RAN and callable(resource_check),"One explicit root-monitored audit invocation")
    _RAN=True
    require(Path.cwd().resolve()==ROOT and os.environ.get("CUDA_VISIBLE_DEVICES")=="","Original project CPU-only context")
    resource_check("before_TRAIN_audit")
    protocol=json.loads(checked_bytes_v1(FIXED["registration"],65536,resource_check))
    require(protocol["schema"]=="native_hold_TRAIN_only_representability_protocol_root_v2"
        and protocol["version1_ref"]==FIXED["registration_v1"]
        and protocol["hypothesis_id"]=="H057" and protocol["status"]=="registered_before_numerical_audit"
        and protocol["train_parents"]==list(TRAIN) and protocol["fixed_grid_bp"]==100
        and protocol["VAL_or_outer_data_authorized"] is False and protocol["optimization_authorized"] is False
        and protocol["GPU_authorized"] is False and protocol["all_original_cohorts_required"] is True
        and protocol["prospective_utility_gate"]["minimum_percentage_points"]==.1,
        "Root-registered H057 TRAIN-only diagnostic and prospective0.10pp gate")
    doc=json.loads(checked_bytes_v1(FIXED["payload_manifest"],2*2**20,resource_check))
    require(doc["schema"]=="native_hold_resolution_TRAIN_payload_manifest_v1"
        and doc["execution_authorized"] is False and doc["TRAIN_parent_ids"]==list(TRAIN)
        and set(doc["parents"])==set(TRAIN) and doc["VAL_payloads_included"] is False
        and doc["outer_payloads_included"] is False,"Exact frozen TRAIN-only transitive manifest")
    import numpy as np
    import torch
    torch.set_num_threads(4)
    sources=FIXED["scientific_sources"]
    native_mass=original_definition_v1(sources["native_mass"],"native_hold_train_mass_v1",{"np":np},resource_check)
    Target=original_definition_v1(sources["target"],"FullPanelExternalTargetV1",{"np":np,"io":io},resource_check)
    intersections=original_definition_v1(sources["intersections"],"intersections",{"np":np},resource_check)
    helpers=SimpleNamespace(intersections=intersections)
    reader=TRAINReaderV1([doc["parents"][pid]["truth_binding"]["truth_npz"] for pid in TRAIN],resource_check)
    results=[]
    for pid in TRAIN:
        resource_check("before_TRAIN_parent")
        metadata=doc["parents"][pid]
        require(metadata["truth_binding"]["split"]=="train","Original TRAIN target binding")
        parent,bank=load_parent_v1(pid,metadata,np,torch,resource_check)
        target=Target(parent,metadata["truth_binding"],"TRAIN_loss",reader)
        mass=native_mass(parent,bank,target,helpers)
        require(mass.dtype==np.float32 and mass.shape==(bank.spans.numel(),686,4)
            and np.isfinite(mass).all() and np.all(mass>=0),"Unchanged original FP32 TRAIN class-span mass")
        groups=groups_v1(parent,np);grid=grid_oracle_v1(parent,target,helpers,groups,np,resource_check)
        physical=float(parent["edges"][-1]-parent["edges"][0]);cohorts={}
        for name,columns in groups.items():
            hold=cell_totals_v1(mass[:,columns,:],np)
            require(np.isclose(hold["support_mass_bp"],physical*len(columns),atol=1e-4,rtol=1e-6),
                "Original native FP32 target mass preserves scored physical support within cast tolerance")
            row=finalize_v1(hold,grid[name],physical,len(columns))
            require(-1e-6<=row["original_grid_oracle_span_accuracy"]-row["hold_oracle_span_accuracy"]<=1+1e-6,
                "Finer original-grid representation cannot be less expressive beyond FP32 mass rounding")
            cohorts[name]=row
        readout=bank.readout_hold.numpy()
        results.append(dict(parent_id=pid,split="train",native_holds=int(bank.spans.numel()),
            native_holds_with_scoring_bins=int(np.unique(readout).size),original_scored_bins=len(parent["positions"]),
            original_observation_payload_ref=metadata["array_ref"],
            original_TRAIN_truth_ref=metadata["truth_binding"]["truth_npz"],cohorts=cohorts))
        print(json.dumps(dict(stage="TRAIN_native_hold_resolution_audit",parent_id=pid,
            Present182_hold_oracle_accuracy=cohorts["Present182"]["hold_oracle_span_accuracy"])),flush=True)
        del mass,target,parent,bank
        resource_check("after_TRAIN_parent")
    result=dict(schema="native_hold_resolution_TRAIN_audit_v1",audit_completed=True,
        TRAIN_parent_ids=list(TRAIN),parent_count=32,cohort_counts=COHORT_COUNTS,
        per_parent=results,summary=aggregate_v1(results,np),sources=FIXED,
        optimizer_steps=0,model_fitted=False,GPU_used=False,VAL_data_read=False,outer_data_read=False,
        is_TRAIN_representation_oracle=True,learned_accuracy_claimed=False,
        aggregation="equal_parent_mean_and_sample_SD_plus_separately_labelled_pooled_bp",
        native_mass_precision="unchanged_original_FP32_targets_with_float64_scalar_reductions",
        grid_mass_precision="original_float64_coordinate_intersections_in256bin_chunks",
        support="same_original100bp_center_bins_clipped_to_observed_coverage_no_tail_extrapolation",
        truth="C4_ancestry_state_at350_generation_horizon_not_one_to_one_recombination_events",
        interpretation="Oracle representation diagnostic using TRAIN truth, not achievable model accuracy, novelty or heldout performance",
        boundary_metrics_computed=False)
    primary=result["summary"]["Present182"]["grid_minus_hold_oracle_pp"]
    result["hypothesis_id"]="H057"
    result["prospective_utility_gate"]=dict(registration_ref=FIXED["registration"],
        primary="Present182",effect_pp=primary["equal_parent_mean"],
        minimum_pp=.1,met=primary["equal_parent_mean"]>=.1,
        interpretation="TRAIN representation-gap gate only, never a model accuracy or achievable gain")
    require(len(json.dumps(result,allow_nan=False).encode())<=2**20,"Count-derived1MiB result-content cap")
    resource_check("after_complete_TRAIN_audit")
    return result
