"""Outer8 scorer integration source. Actual ROOT scoring admission remains mandatory."""
import io, json
from datetime import datetime,timezone
from .outer_prediction_validator_v2 import (
    ROOT,HERE,C4,PARENTS,METHODS,ObservedReaderV2,need,record,ref,reviewed_source,validate_matrix,archive,metric_parent_view_v3,preflight_metric_contract_v3)
class _TargetV2:
    """Original complete interval target schema and exact class-span integration."""
    def __init__(self,parent,binding,data):
        import numpy as np
        with np.load(io.BytesIO(data),allow_pickle=False) as archive:
            need(set(archive.files)=={"intervals_bp","query_raw_node_ids","native_paths_1based","source_classes_0based"},"Original complete interval truth schema")
            self.intervals=archive["intervals_bp"].copy();self.c4=archive["source_classes_0based"].copy();ids=archive["query_raw_node_ids"].copy()
            self.native6=archive["native_paths_1based"].copy()
        need(self.intervals.dtype==np.float64 and self.intervals.ndim==2 and self.intervals.shape[1]==2 and len(self.intervals)
            and self.intervals[0,0]==0 and self.intervals[-1,1]==2000000
            and np.array_equal(self.intervals[:-1,1],self.intervals[1:,0])
            and np.isfinite(self.intervals).all() and np.all(np.diff(self.intervals,axis=1)>0)
            and self.c4.shape==(len(self.intervals),686) and self.c4.dtype.kind in "iu"
            and np.all((self.c4>=0)&(self.c4<=3)) and np.array_equal(ids,parent["raw_query_nodes"]),"Original full686 C4 interval/raw-query identity")
        need(self.native6.shape==self.c4.shape and self.native6.dtype.kind in "iu" and np.all((self.native6>=1)&(self.native6<=6)),"Original native6 interval path axis")
        self.provenance=dict(record=binding,purpose="OUTER_after_all40_validated_products")
    def target_mass(self,parent,helpers):
        import numpy as np
        bins,rows,spans=helpers.intersections(parent["edges"],self.intervals)
        mass=np.zeros((len(parent["positions"]),686,4),dtype=np.float64)
        np.add.at(mass,(bins[:,None],np.arange(686)[None,:],self.c4[rows]),spans[:,None])
        need(np.allclose(mass.sum(-1),np.diff(parent["edges"])[:,None],atol=1e-8,rtol=0),"Exact clipped physical class-span mass")
        return mass
def _target_bytes_after_matrix(binding,parent,reader,request_ref):
    """Only this scorer can open an exact original target after full revalidation."""
    import hashlib,os,stat
    q=_scoring_authority(request_ref,reader)
    bundle,parents,products=validate_matrix(q["bundle_ref"],reader)
    need(bundle["method_freeze_ref"]==q["method_freeze_ref"] and len(parents)==8 and parent in parents
        and binding==q["target_bindings"][parent]["truth_npz"],"Repeat complete actual40 validation at every target-read boundary")
    generation=reader.decode_json(q["target_bindings"][parent]["generation_ref"])
    need(generation["parent_id"]==parent and generation["files"]["truth/full_truth_intervals.npz"]["sha256"]==binding["sha256"],"Same-parent original target provenance")
    metrics=dict(boundary_tolerance_bp=1000.,calibration_bins=15,short_tract_cutoff_bp=10000.)
    for value in parents.values():preflight_metric_contract_v3(metric_parent_view_v3(value),metrics)
    del products
    expected="data/expanded_full_panel_v1/"+parent+"/truth/full_truth_intervals.npz"
    need(isinstance(binding,dict) and set(binding)=={"path","sha256"} and binding["path"]==expected
        and isinstance(binding["sha256"],str) and len(binding["sha256"])==64,"Exact separately ROOT-bound original target")
    p=ROOT/expected;need(p.resolve()==p,"No symlink target path");reader.guard()
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW);fields=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    with os.fdopen(fd,"rb") as f:
        st=os.fstat(f.fileno());need(stat.S_ISREG(st.st_mode) and st.st_nlink==1 and 0<st.st_size<=512*2**20,"Bounded original target")
        need(fields(p.lstat())==fields(st),"Exact opened target")
        data=f.read(512*2**20+1)
        need(len(data)==st.st_size and fields(os.fstat(f.fileno()))==fields(st) and fields(p.lstat())==fields(st),"Immutable original target")
    need(hashlib.sha256(data).hexdigest()==binding["sha256"],"ROOT-pinned original target hash")
    reader.guard();return data
def _scoring_authority(request_ref,reader):
    q=reader.decode_json(request_ref)
    need(q["schema"]=="European_outer8_actual_scoring_ROOT_request_v2" and q["root_authorizer"]=="root_agent"
        and q["status"]=="active" and q["outer_scoring_authorized"] is True
        and q["scorer_source_ref"]==ref(HERE+"/score_outer_v2.py")
        and q["validator_source_ref"]==ref(HERE+"/outer_prediction_validator_v2.py")
        and q["source_manifest_ref"]==ref(HERE+"/source_manifest.v1.json"),"Separate exact ROOT scoring source/matrix authority")
    now=datetime.now(timezone.utc);start=datetime.fromisoformat(q["created_at_utc"]);end=datetime.fromisoformat(q["expires_at_utc"])
    need(start.tzinfo and end.tzinfo and start<=now<end,"Current actual ROOT scoring authority")
    reviewed_source(q["scorer_source_ref"],q["scorer_review_ref"],reader)
    integration=reader.decode_json(q["actual_integration_qualification_ref"])
    need(integration["schema"]=="European_outer8_actual_integration_qualification_v2" and integration["root_authorizer"]=="root_agent"
        and integration["status"]=="qualified" and integration["unresolved_items"]==[]
        and integration["source_manifest_ref"]==q["source_manifest_ref"],"Missing adapters/runtime/wait driver cannot be papered over")
    need(q["method_ids"]==list(METHODS) and q["parent_ids"]==list(PARENTS) and set(q["target_bindings"])==set(PARENTS),"Complete exact ROOT matrix")
    return q
def score_outer_v2(request_ref,*,guard):
    """Real evaluator; no launcher. Caller must hold separate actual scoring admission."""
    import numpy as np
    from arggnn.mutation_routed_fast_primary_v1 import mass_cache_v1,score_primary_v1
    from arggnn.census_tsinfer_fit_v1 import metric_helpers_v1
    reader=ObservedReaderV2(guard);q=reader.decode_json(request_ref)
    q=_scoring_authority(request_ref,reader)
    bundle,parents,products=validate_matrix(q["bundle_ref"],reader)
    need(bundle["method_freeze_ref"]==q["method_freeze_ref"] and q["method_ids"]==list(METHODS) and q["parent_ids"]==list(PARENTS)
        and set(q["target_bindings"])==set(PARENTS),"Exact same matrix/freeze/all original target bindings")
    helpers=metric_helpers_v1()
    metrics=dict(boundary_tolerance_bp=1000.,calibration_bins=15,normalization_length_bp=2000000,
        posthoc_renormalization=False,probability_floor=False,short_tract_cutoff_bp=10000.)
    parents={p:metric_parent_view_v3(parents[p]) for p in PARENTS}
    for parent in parents.values():preflight_metric_contract_v3(parent,metrics)
    cells={m:[] for m in METHODS};top5_cells=[];ap6_cells=[]
    # All40 revalidated probability/support arrays precede the first target read.
    for p in PARENTS:
        parent=parents[p];binding=q["target_bindings"][p]
        need(set(binding)=={"truth_npz","generation_ref"},"Explicit original target provenance binding")
        record(binding["generation_ref"],"data/expanded_full_panel_v1/"+p+"/generation.json")
        generation=reader.decode_json(binding["generation_ref"])
        need(generation["parent_id"]==p and generation["files"]["truth/full_truth_intervals.npz"]["sha256"]==binding["truth_npz"]["sha256"],"Same-parent original generation target hash")
        target=_TargetV2(parent,binding["truth_npz"],_target_bytes_after_matrix(binding["truth_npz"],p,reader,request_ref))
        mass=target.target_mass(parent,helpers);cache=mass_cache_v1(parent,mass,target_provenance=target.provenance,expected_present_queries=182)
        strata={"primary_present_day_182_full_diagnostics":np.flatnonzero(parent["cohorts"]=="present_day"),"all_queries_686":np.arange(686),"secondary_ancient_504":np.flatnonzero(parent["cohorts"]!="present_day")}
        strata.update({"age_stratum_"+c:np.flatnonzero(parent["cohorts"]==c) for c in ("Bronze_Age","BAA","Neo","Yam")})
        for m in METHODS:
            primary=score_primary_v1(products[m][p],cache)
            groups={"primary_present_day_182":primary}
            for name,columns in strata.items():
                value=helpers.full_span_score(products[m][p],parent,target,columns,metrics)
                value.pop("per_query",None);value.pop("retained_original_grid_indexes",None);groups[name]=value
            cells[m].append(dict(parent_id=p,groups=groups))
        secondary=products["ARGformer_top5_prespecified_secondary"][p]
        top5_cells.append(dict(parent_id=p,primary_present_day_182=score_primary_v1(secondary,cache)))
        from arggnn.expanded_native_ancestralpaths_v1 import native_focal_metrics_v1
        ap_cell=next(c for c in bundle["cells"] if c["method_id"]=="AncestralPaths_native_Dense5" and c["parent_id"]==p)
        native=reader.decode_json(ap_cell["native_completion_ref"]);focal=archive(native["native6_focal_ref"],reader)
        indexes=np.searchsorted(target.intervals[:,1],focal["positions_bp"],side="right")
        need(np.all(indexes<len(target.intervals)),"Native6 SNP positions inside original target")
        ap6_cells.append(dict(parent_id=p,native6_focal_ref=native["native6_focal_ref"],
            metrics=native_focal_metrics_v1(focal["native_probabilities"],target.native6[indexes,focal["query_column"]],cohorts=focal["cohort"])))

    summary={}
    for m in METHODS:
        accuracy=np.asarray([x["groups"]["primary_present_day_182"]["accuracy"] for x in cells[m]])
        nll=[x["groups"]["primary_present_day_182"] for x in cells[m]]
        summary[m]=dict(parent_accuracy=accuracy.tolist(),equal_parent_accuracy_mean=float(accuracy.mean()),
            accuracy_sample_SD=float(accuracy.std(ddof=1)),biological_parents=8,optimizer_seeds_do_not_equal_biological_replicates=True,
            NLL_is_infinite=any(x["NLL_is_infinite"] for x in nll),
            equal_parent_NLL_mean=None if any(x["NLL_is_infinite"] for x in nll) else float(np.mean([x["nll"] for x in nll])))
    a,b=METHODS[:2]
    nll_delta=[]
    for first,second in zip(cells[a],cells[b]):
        x,y=first["groups"]["primary_present_day_182"],second["groups"]["primary_present_day_182"]
        nll_delta.append(dict(parent_id=first["parent_id"],delta=None if x["NLL_is_infinite"] or y["NLL_is_infinite"] else x["nll"]-y["nll"],
            defined=not(x["NLL_is_infinite"] or y["NLL_is_infinite"]),stable_infinite=x["NLL_is_infinite"],continuation_infinite=y["NLL_is_infinite"]))
    delta=np.asarray(summary[a]["parent_accuracy"])-np.asarray(summary[b]["parent_accuracy"])
    result=dict(schema="European_outer8_fixed_method_score_v2",cells=cells,summary=summary,
        ARGformer_raw_top5_prespecified_secondary=top5_cells,AP_native6_SNP_focal_separate=ap6_cells,
        paired_stable_minus_continuation_NLL=nll_delta,paired_stable_minus_continuation_accuracy=dict(parent_deltas=delta.tolist(),mean=float(delta.mean()),sample_SD=float(delta.std(ddof=1))),
        no_success_only_subset=True,missing_cells=0,raw_zero_probability_NLL_retained=True,
        posthoc_calibration_or_floors=False,six_path_AP_metrics_separate=True,scoring_request_ref=request_ref)
    return helpers.json_value(result)
