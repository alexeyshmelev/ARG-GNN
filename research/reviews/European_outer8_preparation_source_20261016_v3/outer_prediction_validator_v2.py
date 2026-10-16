"""Observed-only native outer-product validation. No work occurs at import."""
from pathlib import Path
import hashlib, io, json, math, os, re, stat, zipfile
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/European_outer8_preparation_source_20261016_v3"
C4=["Ana","CHG","WHG","EHG"]
PARENTS=tuple("ap_expanded_outer_%03d"%i for i in range(8))
METHODS=("H050_stable_leaf_memory","H050_h045_continued_training","AncestralPaths_native_Dense5","ARGMix_current_native","ARGformer_native_within_ARG")
def need(ok,msg):
    if not ok:raise ValueError(msg)
def record(r,expected=None):
    need(isinstance(r,dict) and set(r)=={"path","sha256"},"Exact path/SHA record")
    name=r["path"];p=Path(name)
    need(isinstance(name,str) and not p.is_absolute() and p.as_posix()==name and name!="."
         and ".." not in p.parts and not any(x.lower() in ("truth","oracle") for x in p.parts),"Canonical nontruth record")
    need(isinstance(r["sha256"],str) and re.fullmatch("[0-9a-f]{64}",r["sha256"]) is not None,"SHA256")
    need(expected is None or name==expected,"Exact method/parent product path")
    return p
def ref(name):
    p=ROOT/name;return dict(path=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
class ObservedReaderV2:
    """Hash-checked no-follow reads; model/checkpoint/truth payloads excluded."""
    def __init__(self,guard):
        need(callable(guard),"Actual admitted caller supplies resource guard");self.guard=guard;self.reads={};self.root=ROOT
    def read(self,r,*,maximum=512*2**20):
        relative=record(r);need(relative.suffix not in (".pt",".keras",".pkl",".trees"),"Validator never reads model/corpus/tree payload")
        p=ROOT/relative;need(p.resolve()==p,"No symlinked read path");self.guard()
        fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
        fields=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        with os.fdopen(fd,"rb") as f:
            before=os.fstat(f.fileno())
            need(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and 0<before.st_size<=maximum,"Bounded singly linked product/metadata")
            need(fields(p.lstat())==fields(before),"Exact opened object")
            data=f.read(maximum+1);need(len(data)==before.st_size,"Exact complete read")
            need(fields(os.fstat(f.fileno()))==fields(before) and fields(p.lstat())==fields(before),"Immutable read")
        need(hashlib.sha256(data).hexdigest()==r["sha256"],"Actual bytes match sealed digest")
        self.reads[r["path"]]=r["sha256"];self.guard();return data
    def decode_json(self,r):
        return json.loads(self.read(r,maximum=32*2**20),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def archive(r,reader):
    import numpy as np
    data=reader.read(r)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        info=z.infolist();need(len(info)<=32 and len({i.filename for i in info})==len(info)
            and all(i.filename.endswith(".npy") and "/" not in i.filename for i in info)
            and sum(i.file_size for i in info)<=512*2**20,"Bounded unique native NPZ members")
        for item in info:
            with z.open(item) as member:
                version=np.lib.format.read_magic(member)
                need(version in ((1,0),(2,0)),"Native NPY header version")
                shape,order,dtype=(np.lib.format.read_array_header_1_0(member) if version==(1,0)
                    else np.lib.format.read_array_header_2_0(member))
                need(not dtype.hasobject and dtype.kind not in "OV" and len(shape)<=3
                    and all(type(x) is int and x>=0 for x in shape)
                    and math.prod(shape)*dtype.itemsize==item.file_size-member.tell(),"Exact bounded native array header/byte size")
    with np.load(io.BytesIO(data),allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in z.files}
    need(all(v.dtype.kind not in "OV" for v in arrays.values()),"No pickled/object fields")
    return arrays
def probability(value,shape):
    import numpy as np
    need(value.dtype==np.float32 and value.shape==shape and np.isfinite(value).all()
        and np.all(value>=0)
        and np.allclose(value.sum(-1),1,atol=1e-6,rtol=1e-5),"Complete finite nonnegative normalized raw FP32 probabilities")
    return value
def cell_directory(run,method,parent):
    need(isinstance(run,str) and re.fullmatch("[A-Za-z0-9][A-Za-z0-9_-]{0,63}",run) is not None,"Single canonical run identity")
    need(method in METHODS and parent in PARENTS,"Exact frozen method and registered parent")
    return "experiments/results/European_outer8_predictions_"+run+"/cells/"+method+"/"+parent
def product_path(run,method,parent,secondary=False):
    base=cell_directory(run,method,parent)
    leaf=("predictions/common_C4_original100bp.npz" if method=="AncestralPaths_native_Dense5" else
          "native/"+parent+"/sealed_C4_grid_probabilities.npz" if method=="ARGMix_current_native" else
          "prediction/C4_k5_full_support.npz" if method=="ARGformer_native_within_ARG" and secondary else
          "prediction/C4_k1_full_support.npz" if method=="ARGformer_native_within_ARG" else "probabilities.npz")
    return base+"/"+leaf
def observed_support(parent,entry_ref,reader):
    import numpy as np
    from arggnn.mutation_routed_full_panel_observations_v1 import validate_full_roster_v1
    from arggnn.full_panel_native_baseline_contracts_v1 import validate_observed_roster,validate_physical_support
    from arggnn.mutation_routed_parent_loader_v1 import _scoring_grid_v1
    record(entry_ref,"data/expanded_full_panel_v1/"+parent+"/full_panel_observed_entry.json")
    entry=reader.decode_json(entry_ref)
    need(entry["schema"]=="mutation_routed_full_panel_observed_entry_v1" and entry["parent_id"]==parent
        and entry["split"]=="outer" and entry["observations_only"] is True and entry["truth_or_oracle_read"] is False
        and (entry["samples"],entry["query_haplotypes"],entry["pure_reference_haplotypes"])==(952,686,266),"Full original observed outer identity")
    base="data/expanded_full_panel_v1/"+parent
    expected={"source_roster":base+"/observed/sample_manifest.json","derived_roster":base+"/relate/observed_sample_manifest.json",
        "scoring_coordinates":base+"/observed/scoring_coordinates.npz","inference_receipt":base+"/full_panel_observed_inference_receipt.json"}
    for key,path in expected.items():record(entry["refs"][key],path)
    record(entry["original_native_inference_receipt"],base+"/relate/inference.json")
    source=reader.decode_json(entry["refs"]["source_roster"]);derived=reader.decode_json(entry["refs"]["derived_roster"])
    validate_full_roster_v1(source);validate_observed_roster(derived)
    wrapper=reader.decode_json(entry["refs"]["inference_receipt"]);native=reader.decode_json(entry["original_native_inference_receipt"])
    need(wrapper["parent_id"]==native["parent_id"]==parent and native["split"]=="outer"
        and native["status"]=="post_five_iterations_inferred" and native["outer_command_returncodes"]==[0]*5
        and wrapper["truth_or_oracle_read"] is False and native["truth_or_oracle_read"] is False
        and wrapper["original_observed_sample_manifest"]==entry["refs"]["source_roster"]
        and wrapper["observed_sample_manifest"]==entry["refs"]["derived_roster"]
        and wrapper["inferred_tree"]==native["inferred_tree"]==entry["refs"]["native_tree"]
        and wrapper["native_inference_receipt"]==entry["original_native_inference_receipt"]
        and wrapper["coordinate_contract"]==entry["coordinate_contract"],"Actual original post-size observed provenance")
    for a,b in zip(source,derived):
        need(all(a[k]==b[k] for k in ("sample_index","sample_id","haplotype","observed_cohort","age_generations","role","observed_reference_class"))
            and a["observed_node_id"]==b["source_observed_node_id"],"Exact source/query/node/date join")
    need(wrapper["sample_index_to_inferred_tree_node_id"]==native["post_five_iterations"]["sample_index_to_inferred_tree_node_id"]
        ==[dict(sample_index=r["sample_index"],inferred_tree_node_id=r["inferred_tree_node_id"]) for r in derived],"Original observed node mapping")
    support=validate_physical_support(native["post_five_iterations"])
    need(all(entry["coordinate_contract"][k]==native["post_five_iterations"][k] for k in entry["coordinate_contract"]),"Entry/native support binding")
    positions,edges,indexes=_scoring_grid_v1(reader.read(entry["refs"]["scoring_coordinates"]),entry["coordinate_contract"])
    need(np.array_equal(positions,support["original_centers_bp"]) and np.array_equal(edges,support["clipped_bin_edges_bp"]),"Unchanged clipped physical support")
    q=[r for r in source if r["role"]=="query"];dq=[r for r in derived if r["role"]=="query"]
    need(len(q)==len(dq)==686 and len({(r["sample_id"],r["haplotype"]) for r in q})==686,"All unique ordered686 queries")
    return dict(parent_id=parent,split="outer",entry_ref=entry_ref,entry=entry,source=source,derived=derived,samples=source,coverage_original=list(support["coverage_original_bp0"]),
        positions=positions,edges=edges,retained_indexes=indexes,
        query_slots=np.asarray([r["sample_index"] for r in q]),raw_query_nodes=np.asarray([r["observed_node_id"] for r in q]),
        query_nodes=np.asarray([r["inferred_tree_node_id"] for r in dq]),cohorts=np.asarray([r["observed_cohort"] for r in q]),
        query_keys=[[r["sample_id"],r["haplotype"]] for r in q],coverage=np.asarray(support["coverage_original_bp0"]),
        source_roster_ref=entry["refs"]["source_roster"],scoring_coordinates_ref=entry["refs"]["scoring_coordinates"])
def same(a,b,msg):
    import numpy as np
    need(np.array_equal(a,b),msg)
def validate_product(r,parent,method,reader,*,argformer_roster_ref=None):
    import numpy as np
    z=archive(r,reader);shape=(len(parent["positions"]),686,4)
    checks={"positions_bp":parent["positions"],"bin_edges_bp":parent["edges"]}
    if method.startswith("H050_"):
        need(set(z)=={"C4_probabilities",*checks,"retained_fixed100bp_indexes","query_sample_indexes"},"Original C4 grid seal schema")
        checks.update(retained_fixed100bp_indexes=parent["retained_indexes"],query_sample_indexes=parent["query_slots"]);key="C4_probabilities"
    elif method=="AncestralPaths_native_Dense5":
        need(set(z)=={"source_probabilities",*checks,"query_node_ids","query_sample_indices","retained_fixed100bp_indexes",
            "coverage_original_bp0","local_to_original_offset_bp","raw_query_nodes"},"Native AP common-C4 schema")
        checks.update(query_node_ids=parent["query_nodes"],query_sample_indices=parent["query_slots"],
            retained_fixed100bp_indexes=parent["retained_indexes"],coverage_original_bp0=parent["coverage"],
            local_to_original_offset_bp=parent["coverage"][0],raw_query_nodes=parent["raw_query_nodes"]);key="source_probabilities"
    elif method=="ARGMix_current_native":
        need(set(z)=={"probabilities","supported","grid_indexes",*checks,"local_positions_bp","local_site_positions_bp",
            "original_site_positions_bp","query_nodes"},"Native ARGMix grid/support schema")
        checks.update(grid_indexes=parent["retained_indexes"],local_positions_bp=parent["positions"]-parent["coverage"][0],query_nodes=parent["query_nodes"])
        need(z["supported"].dtype==np.bool_ and z["supported"].shape==shape[:2] and z["supported"].all(),"No incomplete native ARGMix support")
        sites=reader.decode_json(parent["entry"]["original_native_inference_receipt"])["post_five_iterations"]["site_join"]
        checks.update(local_site_positions_bp=[r["local_position_0based"] for r in sites],
            original_site_positions_bp=[r["original_position_0based"] for r in sites]);key="probabilities"
    else:
        need(method=="ARGformer_native_within_ARG" and argformer_roster_ref is not None,"Actual native within-parent roster required")
        need(set(z)=={"probabilities","SNP_probabilities",*checks,"site_positions_bp","query_nodes","query_sample_indices",
            "raw_query_nodes","retained_fixed100bp_indexes","coverage_original_bp0"},"Native ARGformer raw-vote schema")
        rows=reader.decode_json(argformer_roster_ref)
        from arggnn.full_panel_native_baseline_contracts_v1 import validate_observed_roster
        validate_observed_roster(rows);queries=[r for r in rows if r["role"]=="query"]
        need([[r["sample_id"],r["haplotype"]] for r in queries]==parent["query_keys"],"Same query identities in native ARG namespace")
        same([r["source_observed_node_id"] for r in queries],parent["raw_query_nodes"],"Same original raw query IDs")
        checks.update(query_nodes=[r["inferred_tree_node_id"] for r in queries],query_sample_indices=parent["query_slots"],
            raw_query_nodes=parent["raw_query_nodes"],retained_fixed100bp_indexes=parent["retained_indexes"],coverage_original_bp0=parent["coverage"])
        sites=reader.decode_json(parent["entry"]["original_native_inference_receipt"])["post_five_iterations"]["site_join"]
        checks["site_positions_bp"]=[r["original_position_0based"] for r in sites]
        probability(z["SNP_probabilities"],(len(sites),686,4));key="probabilities"
    for name,expected in checks.items():
        if name in {"retained_fixed100bp_indexes","grid_indexes","query_sample_indexes","query_sample_indices","query_node_ids","query_nodes","raw_query_nodes","local_site_positions_bp"}:
            need(z[name].dtype.kind in "iu","Native integer ID/index array: "+name)
        elif name in {"positions_bp","bin_edges_bp","coverage_original_bp0","local_to_original_offset_bp","local_positions_bp","original_site_positions_bp","site_positions_bp"}:
            need(z[name].dtype==np.float64,"Native physical-coordinate float64 array: "+name)
        same(z[name],expected,"Actual native array/query/support equality: "+name)
    return probability(z[key],shape)
def reviewed_source(source_ref,review_ref,reader):
    record(source_ref);need(source_ref["path"].endswith(".py"),"Actual source path")
    reader.read(source_ref);review=reader.decode_json(review_ref)
    need(review.get("verdict") in ("PASS","SOURCE_PASS") and review.get("source_ref")==source_ref,"Actual independently reviewed adapter/driver/scorer source")
def completed_wait(cell,reader,completion_ref):
    """Bind separate actual OS wait, launch/input and ROOT-consumed outer terminal."""
    evidence=cell.get("evidence_directory",cell_directory(cell["run_id"],cell["method_id"],cell["parent_id"])+"/evidence")
    expected=cell_directory(cell["run_id"],cell["method_id"],cell["parent_id"])
    need(evidence in (expected+"/evidence",expected+"/fit/evidence",expected+"/observation/evidence",expected+"/cuts/evidence"),"Canonical stage evidence namespace")
    for name,leaf in (("actual_wait_ref","actual_wait.json"),("launch_ref","launch.json"),("input_ref","input.json"),("ROOT_terminal_receipt_ref","ROOT_terminal_receipt.json")):
        record(cell[name],evidence+"/"+leaf)
    wait=reader.decode_json(cell["actual_wait_ref"]);launch=reader.decode_json(cell["launch_ref"])
    admission=reader.decode_json(cell["input_ref"]);terminal=reader.decode_json(cell["ROOT_terminal_receipt_ref"])
    reviewed_source(cell["wait_driver_source_ref"],cell["wait_driver_review_ref"],reader)
    need(wait["schema"]=="European_outer_native_actual_wait_v2" and wait["returncode"]==0 and wait["actual_OS_wait_status"]==0
        and wait["actual_direct_worker_wait"] is True and wait["all_owned_processes_and_groups_drained"] is True
        and wait["operational_failure"] is None and wait["completion_ref"]==completion_ref
        and wait["input_ref"]==cell["input_ref"] and wait["launch_ref"]==cell["launch_ref"]
        and wait["driver_source_ref"]==cell["wait_driver_source_ref"]
        and wait["pid"]==launch["pid"] and wait["start_ticks"]==launch["start_ticks"]
        and type(wait["pid"]) is int and wait["pid"]>0 and type(wait["start_ticks"]) is int and wait["start_ticks"]>0
        and wait["group_observations"] and all(x["absent"] is True for x in wait["group_observations"])
        and all(x["returncode"]==0 for x in wait["adopted_waits"]),"Genuine source-bound wait0 and complete owned drain receipts")
    need(admission["schema"]=="European_outer_native_adapter_ROOT_request_v2" and admission["status"]=="active"
        and admission["scientific_execution_authorized"] is True and admission["outer_truth_access_authorized"] is False
        and admission["source_manifest_ref"]==ref(HERE+"/source_manifest.v1.json")
        and admission["root_authorizer"]=="root_agent" and admission["parent_id"]==cell["parent_id"]
        and admission["method_id"]==cell["method_id"] and admission["adapter_ref"]==cell["actual_inference_adapter_ref"]
        and terminal["root_authorizer"]=="root_agent" and terminal["actual_tool_terminal_consumed"] is True
        and terminal["actual_tool_exit_code"]==0 and terminal["actual_wait_ref"]==cell["actual_wait_ref"]
        and terminal["completion_ref"]==completion_ref and terminal["input_ref"]==cell["input_ref"],"Independent ROOT consumption binds actual native terminal")
def argmix_axis_order_v3(cell,native,identity,reader):
    base=cell_directory(cell["run_id"],cell["method_id"],cell["parent_id"])
    record(cell["adapter_axis_receipt_ref"],base+"/adapter_axis_receipt.json")
    receipt=reader.decode_json(cell["adapter_axis_receipt_ref"])
    config=reader.decode_json(receipt["config_ref"]);fit=reader.decode_json(identity["fit_receipt_ref"])
    need(receipt["schema"]=="European_outer_ARGMix_axis_receipt_v3"
        and receipt["parent_id"]==cell["parent_id"] and receipt["method_id"]==cell["method_id"]
        and receipt["adapter_source_ref"]==cell["actual_inference_adapter_ref"]
        and receipt["native_completion_ref"]==cell["native_completion_ref"]
        and receipt["fit_receipt_ref"]==identity["fit_receipt_ref"]
        and receipt["checkpoint_ref"]==fit["checkpoint"]==identity["checkpoint_ref"]
        and receipt["config_ref"]["sha256"]==fit["config_sha256"]
        and config["schema"]=="expanded_native_argmix_v1" and config["hypothesis_id"]=="H038"
        and config["owned_code_sha256"]==fit["owned_code_sha256"]
        and receipt["authored_source_ref"]["path"]=="src/arggnn/expanded_native_argmix_v1.py"
        and receipt["authored_source_ref"]["sha256"]==config["owned_code_sha256"][receipt["authored_source_ref"]["path"]]
        and receipt["class_order"]==config["class_order"]==["ANA","CHG","WHG","EHG"],
        "Separate actual authored-config/source/checkpoint axis receipt; author seal remains unchanged")
    reader.read(receipt["authored_source_ref"])
    return receipt["class_order"]
def metric_parent_view_v3(parent):
    import numpy as np
    result=dict(parent)
    need("source" in parent and "coverage" in parent,"Actual observed source rows/coverage required")
    result["samples"]=parent["source"];result["coverage_original"]=np.asarray(parent["coverage"],dtype=np.float64).tolist()
    return result
def preflight_metric_contract_v3(parent,metrics,columns=None):
    import numpy as np
    required={"samples","coverage_original","edges","query_slots","cohorts","retained_indexes","positions","raw_query_nodes"}
    need(required<=set(parent),"Complete inherited metric parent view before any target")
    need(len(parent["samples"])==952 and len(parent["query_slots"])==686
        and len(parent["raw_query_nodes"])==686 and parent["cohorts"].shape==(686,),"Full952 ordered686 metric identity")
    positions=np.asarray(parent["positions"]);edges=np.asarray(parent["edges"]);coverage=np.asarray(parent["coverage_original"])
    need(positions.dtype==edges.dtype==np.float64 and positions.ndim==1 and len(positions)>0
        and edges.shape==(len(positions)+1,) and np.isfinite(edges).all() and np.all(np.diff(edges)>0)
        and coverage.shape==(2,) and np.isfinite(coverage).all() and 0<=coverage[0]<=edges[0]<edges[-1]<=coverage[1]<=2000000
        and np.asarray(parent["retained_indexes"]).shape==positions.shape,"Complete physical metric support")
    query=np.asarray(parent["query_slots"])
    need(query.dtype.kind in "iu" and query.shape==(686,) and len(set(map(int,query)))==686 and np.all((query>=0)&(query<952)),"Metric query slot identity")
    for i,row in enumerate(parent["samples"]):need(row["sample_index"]==i,"Original source sample ordering")
    for i,slot in enumerate(query):
        row=parent["samples"][int(slot)]
        need(row["role"]=="query" and row["observed_cohort"]==parent["cohorts"][i]
            and row["observed_node_id"]==parent["raw_query_nodes"][i],"Metric source/query/cohort join")
    wanted={"present_day":182,"Bronze_Age":162,"BAA":14,"Neo":302,"Yam":26}
    need({c:int((parent["cohorts"]==c).sum()) for c in wanted}==wanted,"Complete frozen cohort counts")
    need(set(metrics)>={"calibration_bins","boundary_tolerance_bp","short_tract_cutoff_bp"}
        and metrics["calibration_bins"]==15 and metrics["boundary_tolerance_bp"]==1000.
        and metrics["short_tract_cutoff_bp"]==10000.,"Unchanged inherited diagnostic parameters")
    strata=[np.flatnonzero(parent["cohorts"]==c) for c in wanted]+[np.arange(686),np.flatnonzero(parent["cohorts"]!="present_day")]
    if columns is not None:strata.append(np.asarray(columns))
    for cols in strata:
        need(cols.ndim==1 and cols.dtype.kind in "iu" and len(cols)>0 and len(cols)%2==0 and np.all((cols>=0)&(cols<686)),"Complete ordered diploid metric stratum")
        for a,b in zip(cols[0::2],cols[1::2]):
            first,second=parent["samples"][int(query[a])],parent["samples"][int(query[b])]
            need(first["sample_id"]==second["sample_id"] and (first["haplotype"],second["haplotype"])==(0,1),"Actual ordered diploid partners before target access")
    return True
def validate_h050_bank_geometry_v3(query_column,readout_hold,spans,sample_shape,intervals,parent):
    import numpy as np
    same(query_column,parent["query_slots"],"Actual bank.query_column equals original ordered686 slots")
    need(np.asarray(query_column).dtype.kind in "iu" and np.asarray(query_column).shape==(686,),"Actual bank query index dtype/shape")
    intervals=np.asarray(intervals);readout=np.asarray(readout_hold);spans=np.asarray(spans)
    need(intervals.dtype==np.float64 and intervals.ndim==2 and intervals.shape[1]==2 and len(intervals)>0
        and np.isfinite(intervals).all() and np.all(intervals[:,1]>intervals[:,0])
        and np.array_equal(intervals[:-1,1],intervals[1:,0])
        and np.array_equal(intervals[[0,-1],[0,1]],parent["coverage"])
        and tuple(sample_shape)==(len(intervals),952),"Genuine complete original native-hold/full952 geometry")
    same(spans,(intervals[:,1]-intervals[:,0]).astype(np.float32),"Actual bank native physical spans")
    need(readout.dtype.kind in "iu" and readout.shape==parent["positions"].shape,"Actual bank readout shape/dtype")
    same(readout,np.searchsorted(intervals[:,1],parent["positions"],side="right"),"Actual bank readout_hold from original float64 intervals")
    need(np.all((readout>=0)&(readout<len(intervals))),"Actual bank readout support")
    return True
def h050_preparation_wait_v3(execution,cell,completion,stage,reader):
    need(all(execution[k]==cell[k] for k in ("run_id","method_id","parent_id","actual_inference_adapter_ref")),"Same H050 cell/source preparation execution")
    admission=reader.decode_json(execution["input_ref"])
    need(admission["stage"]==stage,"Genuine original bank/cut preparation stage")
    return completed_wait(execution,reader,completion)
def h050_provenance_v3(cell,native,parent,reader):
    p=native["bank_provenance"];base=cell_directory(cell["run_id"],cell["method_id"],cell["parent_id"])
    record(p["bank_preparation_ref"],base+"/observation/bank_preparation.json")
    prep=reader.decode_json(p["bank_preparation_ref"])
    need(prep["schema"]=="European_outer_H050_bank_preparation_v3" and prep["parent_id"]==parent["parent_id"]
        and prep["original_observed_entry_ref"]==parent["entry_ref"] and prep["producer_source_ref"]==ref(HERE+"/h050_observed_bridge_v3.py")
        and prep["cache_ref"]==p["cache_ref"] and prep["native_hold_geometry_ref"]==p["native_hold_geometry_ref"]
        and prep["targets_read"] is False,"Native predictor binds genuine observed bank preparation")
    cache=reader.decode_json(p["cache_ref"]);meta=cache["parents"][parent["parent_id"]]
    need(cache["entry_refs"][parent["parent_id"]]==parent["entry_ref"] and meta["array_ref"]==p["cache_array_ref"]
        and cache["targets_cached"] is False and cache["observations_only"] is True,"Exact original cache archive lineage")
    geometry=archive(p["native_hold_geometry_ref"],reader)
    need(set(geometry)=={"intervals_bp0","positions_bp","query_sample_indexes","readout_hold"},"Genuine native hold snapshot schema")
    same(geometry["positions_bp"],parent["positions"],"Exact snapshot grid");same(geometry["query_sample_indexes"],parent["query_slots"],"Exact snapshot query order")
    import numpy as np
    intervals=geometry["intervals_bp0"]
    validate_h050_bank_geometry_v3(geometry["query_sample_indexes"],geometry["readout_hold"],
        (intervals[:,1]-intervals[:,0]).astype(np.float32),(len(intervals),952),intervals,parent)
    cuts=reader.decode_json(p["cut_preparation_completion_ref"])
    need(cuts["observation_cache_ref"]==p["cache_ref"] and cuts["entry_refs"][parent["parent_id"]]==parent["entry_ref"]
        and cuts["targets_read"] is False and cuts["observations_only"] is True,"Same-bank original compact-cut completion")
    receipt=reader.decode_json(cuts["parents"][parent["parent_id"]]["receipt_ref"])
    need(receipt["original_cache_array_ref"]==p["cache_array_ref"] and receipt["original_observed_entry_ref"]==parent["entry_ref"]
        and receipt["lossless_roundtrip_verified"] is True,"Actual lossless original bank/cut recipe lineage")
    h050_preparation_wait_v3(p["bank_execution"],cell,p["bank_preparation_ref"],"prepare_bank",reader)
    h050_preparation_wait_v3(p["cut_execution"],cell,p["cut_preparation_completion_ref"],"prepare_cuts",reader)
    return True
def native_lineage(cell,parent,identity,reader):
    import numpy as np
    method=cell["method_id"];base=cell_directory(cell["run_id"],method,parent["parent_id"])
    completion_path=base+("/predictions/prediction_seal.json" if method=="AncestralPaths_native_Dense5" else
        "/native/"+parent["parent_id"]+"/prediction_seal.json" if method=="ARGMix_current_native" else
        "/prediction/prediction_seal.json" if method=="ARGformer_native_within_ARG" else "/prediction_seal.json")
    record(cell["native_completion_ref"],completion_path)
    n=reader.decode_json(cell["native_completion_ref"])
    native_order=(argmix_axis_order_v3(cell,n,identity,reader) if method=="ARGMix_current_native"
        else n.get("C4_class_order",n.get("class_order")))
    need(isinstance(native_order,list) and [("Ana" if x=="ANA" else x) for x in native_order]==C4,"Actual native probability class order")
    need(n["parent_id"]==parent["parent_id"],"Same native completion parent")
    if method.startswith("H050_"):
        fit=reader.decode_json(identity["training_completion_ref"])
        selected=reader.decode_json(identity["selection_completion_ref"])
        need(n["schema"]=="European_outer_H050_fixed_checkpoint_prediction_v2" and n["arm"]==identity["arm"]
            and n["pass_index"]==identity["selected_new_pass"] and n["checkpoint_ref"]==identity["checkpoint_ref"]
            and n["training_completion_ref"]==identity["training_completion_ref"] and n["selection_completion_ref"]==identity["selection_completion_ref"]
            and fit["checkpoint_groups"][str(identity["selected_new_pass"])][identity["arm"]]==identity["checkpoint_ref"]
            and selected["selected"][identity["arm"]]["checkpoint"]==identity["checkpoint_ref"]
            and n["query_keys"]==parent["query_keys"] and n["optimization_performed"] is False,"Exact preselected H050 lineage; no outer selection")
        h050_provenance_v3(cell,n,parent,reader)
    elif method=="AncestralPaths_native_Dense5":
        fit=reader.decode_json(identity["fit_receipt_ref"])
        need(n["schema"]=="expanded_native_ap_prediction_seal_v1" and n["split"]=="outer"
            and n["checkpoint_ref"]==fit["checkpoint_ref"]==identity["checkpoint_ref"] and n["fit_ref"]==identity["fit_receipt_ref"]
            and fit["epochs"]==5 and fit["VAL_selection"] is False and n["query_keys"]==parent["query_keys"]
            and n["targets_read"] is False and n["C4_collapse_before_argmax"]=="p1+p5,p2+p6,p3,p4","Native final Dense5 AP lineage")
        focal=archive(n["native6_focal_ref"],reader)
        need(set(focal)=={"native_probabilities","site_row","query_column","cohort","positions_bp","query_sample_indices","query_node_ids"},"Separate native6 focal schema")
        count=len(focal["site_row"]);need(count>0,"Nonempty original native6 focal support");probability(focal["native_probabilities"],(count,6))
        for k in ("query_column","site_row"):need(focal[k].dtype.kind in "iu" and focal[k].shape==(count,) and (focal[k]>=0).all(),"Native6 integer focal axes")
        need((focal["query_column"]<686).all(),"Native6 query columns")
        sites=reader.decode_json(parent["entry"]["original_native_inference_receipt"])["post_five_iterations"]["site_join"]
        need((focal["site_row"]<len(sites)).all(),"Native6 actual observed SNP rows")
        same(focal["positions_bp"],np.asarray([r["original_position_0based"] for r in sites])[focal["site_row"]],"Native6 exact original SNP coordinate join")
        for k,expected in (("query_sample_indices",parent["query_slots"]),("query_node_ids",parent["query_nodes"]),("cohort",parent["cohorts"])):
            same(focal[k],expected[focal["query_column"]],"Native6 exact focal query/cohort identity")
    elif method=="ARGMix_current_native":
        fit=reader.decode_json(identity["fit_receipt_ref"])
        need(n["checkpoint"]==fit["checkpoint"]==identity["checkpoint_ref"]
            and n["frame"]["source_observed_entry_ref"]==parent["entry_ref"]
            and n["full_support_complete"] is True and n["missing_grid_query_cells"]==0
            and n["native_replay_max_probability_error"]<=1e-6
            and n["original_raw_logits_are_per_chunk_not_log_of_averaged_probability"] is True
            and n["native_final_estimand"]=="source arithmetic average of overlapping-chunk softmax probabilities","Frozen native ARGMix checkpoint/overlap lineage")
        record(n["original_chunk_logits"]);record(n["stock_base_HDF5"]);record(n["stock_refined_HDF5"])
    else:
        record(n["fit_ref"],base+"/fit/fit_manifest.json");record(n["prepared_ref"],base+"/prepared/prepared_manifest.json")
        fit=reader.decode_json(n["fit_ref"]);prepared=reader.decode_json(n["prepared_ref"])
        need(n["schema"]=="expanded_native_argformer_tsinfer_prediction_seal_v1" and n["prepared_ref"]==fit["prepared_ref"]
            and n["namespace"]==fit["namespace"]==prepared["namespace"]
            and prepared["parent_id"]==fit["parent_id"]==parent["parent_id"] and prepared["split"]=="outer"
            and prepared["namespace"]==parent["parent_id"]+":"+prepared["inferred_tree_ref"]["sha256"]
            and fit["fresh_same_ARG_fit"] is True and fit["old_checkpoint_reused"] is False
            and fit["query_truth_used"] is False and prepared["true_query_ancestry_read"] is False
            and prepared["input_samples"]==952 and prepared["query_haplotypes"]==686 and prepared["reference_haplotypes"]==266
            and prepared["query_keys"]==parent["query_keys"] and n["primary_k"]==1 and n["sensitivity_k"]==5
            and n["raw_vote_zero_probabilities_retained"] is True and n["no_query_truth_read"] is True,"Actual same-parent full952 native vocabulary/fit lineage")
        for key,epochs,batch,seed,precision in (("MLM",5,512,17,"amp_bf16"),("contrastive",1,4096,36,"fp32")):
            phase=fit[key];expected=prepared["MLM_optimizer_updates" if key=="MLM" else "contrastive_optimizer_updates"]
            need(phase["epochs"]==epochs and phase["logical_batch"]==batch and phase["seed"]==seed and phase["precision"]==precision
                and type(phase["optimizer_updates"]) is int and phase["optimizer_updates"]==expected>0
                and phase["query_truth_used"] is False and phase["same_ARG_only"] is True
                and phase["source_batch_reduced"] is False and phase["gradient_cache_used"] is False,"Complete genuine native epochs/counters; no reduced fit")
            record(phase["checkpoint_ref"],base+"/fit/"+("mlm" if key=="MLM" else "contrastive")+"/composer_final.pt")
            need(parent["parent_id"] in Path(phase["checkpoint_ref"]["path"]).parts,"Distinct same-parent checkpoint path")
        need(fit["final_checkpoint_ref"]==fit["contrastive"]["checkpoint_ref"],"Native final checkpoint identity")
        need(cell["actual_checkpoint_refs"]==dict(mlm=fit["MLM"]["checkpoint_ref"],contrastive=fit["contrastive"]["checkpoint_ref"]),"Actual two checkpoint pins")
        need(cell["fit_execution"]["parent_id"]==cell["parent_id"] and cell["fit_execution"]["method_id"]==cell["method_id"],"Same-parent native fit OS lineage")
        completed_wait(cell["fit_execution"],reader,n["fit_ref"])
        need(cell["native_query_roster_ref"]==prepared["files"]["observed_roster.json"],"Actual native prepared roster")
    if method!="ARGformer_native_within_ARG":
        need(cell["actual_checkpoint_refs"]==dict(selected=identity["checkpoint_ref"]),"Frozen saved checkpoint equality")
    completed_wait(cell,reader,cell["native_completion_ref"]);return n
def validate_matrix(bundle_ref,reader):
    """Validate all40 actual products; target access remains a separate capability."""
    need(type(reader) is ObservedReaderV2,"Concrete observed-only hash reader required")
    bundle=reader.decode_json(bundle_ref)
    need(bundle["source_manifest_ref"]==ref(HERE+"/source_manifest.v1.json"),"Exact submitted v2 manifest")
    manifest=reader.decode_json(bundle["source_manifest_ref"])
    for source in manifest["refs"]:reader.read(source,maximum=32*2**20)
    inventory=reader.decode_json(ref(HERE+"/frozen_method_ids.v1.json"))
    freeze=reader.decode_json(bundle["method_freeze_ref"])
    from datetime import datetime,timezone
    frozen_at=datetime.fromisoformat(freeze["frozen_at_utc"])
    need(frozen_at.tzinfo and frozen_at<=datetime.now(timezone.utc),"Actual nonfuture prospective freeze")
    need(freeze["active"] is True and freeze["root_authorizer"]=="root_agent" and freeze["method_ids"]==list(METHODS)
        and freeze["identity_inventory_ref"]==ref(HERE+"/frozen_method_ids.v1.json")
        and freeze["all_methods_fixed_before_first_outer_generation"] is True and freeze["outer_data_or_truth_used_for_selection"] is False
        and freeze["H058_automatically_included"] is False,"Actual prospective immutable method commitment")
    need(bundle["schema"]=="European_outer8_native_prediction_bundle_v2" and bundle["method_ids"]==list(METHODS)
        and bundle["parent_ids"]==list(PARENTS) and len(bundle["cells"])==40,"Complete fixed five by eight matrix")
    need(bundle["scorer_source_ref"]==ref(HERE+"/score_outer_v2.py"),"Actual outer scorer path, not a protocol-only placeholder")
    reviewed_source(bundle["scorer_source_ref"],bundle["scorer_review_ref"],reader)
    reviewed_source(ref(HERE+"/outer_prediction_validator_v2.py"),bundle["validator_review_ref"],reader)
    keys=set();paths=set();parents={};products={m:{} for m in METHODS}
    products["ARGformer_top5_prespecified_secondary"]={}
    for cell in bundle["cells"]:
        m,p=cell["method_id"],cell["parent_id"];need((m,p) not in keys and m in METHODS and p in PARENTS,"Unique exact matrix cell");keys.add((m,p))
        need(cell["run_id"]==bundle["run_id"],"One fixed matrix run")
        base=cell_directory(bundle["run_id"],m,p);record(cell["seal_ref"],base+"/cell_seal.json")
        seal=reader.decode_json(cell["seal_ref"])
        need(all(seal[k]==cell[k] for k in ("method_id","parent_id","native_completion_ref","predictions_ref","actual_checkpoint_refs"))
            and seal["method_freeze_ref"]==bundle["method_freeze_ref"] and seal["class_order"]==C4
            and seal["frozen_identity"]==inventory["methods"][m],"Actual cell-bound frozen identity/class axis")
        need(cell["actual_inference_adapter_ref"]==ref(HERE+"/outer_adapters_v2.py"),"Exact actual submitted adapter source")
        reviewed_source(cell["actual_inference_adapter_ref"],cell["actual_adapter_review_ref"],reader)
        parent=observed_support(p,cell["observation_entry_ref"],reader)
        if p in parents:
            need(parents[p]["entry_ref"]==parent["entry_ref"],"One original observed entry per parent across methods")
        else:parents[p]=parent
        record(cell["predictions_ref"],product_path(bundle["run_id"],m,p))
        need(cell["predictions_ref"]["path"] not in paths,"No product reuse across cells");paths.add(cell["predictions_ref"]["path"])
        native=native_lineage(cell,parent,inventory["methods"][m],reader)
        expected_product=(native["common_C4_ref"] if m=="AncestralPaths_native_Dense5" else native["grid_prediction"] if m=="ARGMix_current_native"
            else native["prediction_refs"]["1"] if m=="ARGformer_native_within_ARG" else native["predictions_ref"])
        need(expected_product==cell["predictions_ref"],"Native completion binds actual validated probability bytes")
        products[m][p]=validate_product(cell["predictions_ref"],parent,m,reader,argformer_roster_ref=cell.get("native_query_roster_ref"))
        if m=="ARGformer_native_within_ARG":
            secondary=cell["top5_predictions_ref"];record(secondary,product_path(bundle["run_id"],m,p,True))
            need(secondary==native["prediction_refs"]["5"] and secondary["path"] not in paths,"Independent prespecified raw k5 product");paths.add(secondary["path"])
            top5=validate_product(secondary,parent,m,reader,argformer_roster_ref=cell["native_query_roster_ref"])
            import numpy as np
            products["ARGformer_top5_prespecified_secondary"][p]=top5
            need(np.isin(products[m][p],[0.,1.]).all() and np.allclose(top5*5,np.rint(top5*5),atol=1e-6,rtol=0),"Native raw k1/k5 class votes; no smoothing")
    need(keys=={(m,p) for m in METHODS for p in PARENTS},"No missing/failed method or parent")
    return bundle,parents,products
