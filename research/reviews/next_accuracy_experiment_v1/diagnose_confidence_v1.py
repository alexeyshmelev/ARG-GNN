"""Post-selection exact-span diagnostic, never a model input."""
from pathlib import Path
import hashlib,json,sys,traceback
import numpy as np
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=ROOT/"research/reviews/next_accuracy_experiment_v1"
def reference(p):
 return {"path":str(p.relative_to(ROOT)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
def main():
 completed=ROOT/"experiments/results/reference_dependence_full_storage_v1_seed101_GPU2/completed.json"
 d=json.loads(completed.read_text())
 tree=d["models"]["tree_gram"]["selected"]
 base=json.loads((completed.parent/"frozen_genomic_smoothing.metrics.json").read_text())
 by={x["parent_id"]:x for x in base["cells"]}
 bounds=np.array([0,.5,.6,.7,.8,.9,.95,.99,1.00001])
 rows=[]
 for cell in tree["cells"]:
  parent=cell["parent_id"];baseline=by[parent]
  ro=ROOT/f"data/official_argformer_tsinfer_parent_v3/{parent}/prepared/observed_roster.json"
  roster=json.loads(ro.read_text())
  present={x["sample_index"] for x in roster if x["role"]=="query" and x["observed_cohort"]=="present_day"}
  assert len(present)==64
  tp=ROOT/cell["prediction_seal"]["path"];bp=ROOT/baseline["prediction_seal"]["path"]
  assert reference(tp)==cell["prediction_seal"] and reference(bp)==baseline["prediction_seal"]
  with np.load(tp,allow_pickle=False) as z:
   ids=z["query_sample_indexes"].copy();edges=z["bin_edges_bp"].copy()
   take=np.array([i for i,v in enumerate(ids) if int(v) in present])
   pt=z["C4_probabilities"][:,take].copy()
  with np.load(bp,allow_pickle=False) as z:
   assert np.array_equal(ids,z["query_sample_indexes"]) and np.array_equal(edges,z["bin_edges_bp"])
   pb=z["C4_probabilities"][:,take].copy()
  truth=ROOT/f"data/biological_paths_execution_pilot_v1/{parent}/truth/full_truth_intervals.npz"
  with np.load(truth,allow_pickle=False) as z:
   intervals=z["intervals_bp"].copy();classes=z["source_classes_0based"][:,take].copy()
  # Existing target binding certifies the raw query-to-observed sample join.
  spec=json.loads((ROOT/d["input_ref"]["path"]).read_text())
  bindings=json.loads((ROOT/spec["refs"]["target_bindings"]["path"]).read_text())
  assert list(ids)==bindings["parents"][parent]["query_sample_indexes"]
  with np.load(truth,allow_pickle=False) as z:
   assert list(z["query_raw_node_ids"])==bindings["parents"][parent]["query_raw_node_ids"]
  ht=pt.argmax(-1);hb=pb.argmax(-1);confidence=pb.max(-1)
  cats=np.minimum(np.searchsorted(bounds,confidence,side="right")-1,len(bounds)-2)
  tally=np.zeros((len(bounds)-1,4),dtype=np.float64)
  i=j=0
  while i<len(intervals) and j<len(edges)-1:
   left=max(intervals[i,0],edges[j]);right=min(intervals[i,1],edges[j+1])
   if right>left:
    bc=hb[j]==classes[i];tc=ht[j]==classes[i]
    status=np.where(bc,np.where(tc,0,2),np.where(tc,1,3))
    np.add.at(tally,(cats[j],status),right-left)
   if intervals[i,1]<=edges[j+1]:i+=1
   else:j+=1
  total=tally.sum()
  ba=(tally[:,0].sum()+tally[:,2].sum())/total
  ta=(tally[:,0].sum()+tally[:,1].sum())/total
  assert abs(ba-baseline["groups"]["primary_present_day_64"]["accuracy"])<1e-10
  assert abs(ta-cell["groups"]["primary_present_day_64"]["accuracy"])<1e-10
  rows.append({"parent":parent,"prediction_refs":{"smoothing":reference(bp),"tree_gram":reference(tp)},
   "truth_ref":reference(truth),"query_bp":total,"baseline_accuracy":ba,"tree_accuracy":ta,
   "fixed_baseline_error_bp":tally[:,1].sum(),"introduced_error_bp":tally[:,2].sum(),
   "confidence_bins":[{"lower":float(bounds[k]),"upper":min(1.,float(bounds[k+1])),
    "both_correct_bp":tally[k,0],"fixed_baseline_error_bp":tally[k,1],
    "introduced_error_bp":tally[k,2],"both_wrong_bp":tally[k,3]} for k in range(len(bounds)-1)]})
 result={"status":"post_selection_development_diagnostic","source":reference(Path(__file__)),
  "completed_ref":reference(completed),"query_scope":"Present64","prediction_selection":"unchanged epoch10 minimumNLL",
  "truth_used_only_after_selected_sealed_predictions":True,"aggregation":"exact physical intersections, separately per parent",
  "rows":rows,"interpretation":"These counts locate error changes. They do not prove that confidence-aware feedback will help."}
 with (OUT/"confidence_diagnostic.json").open("x") as f:json.dump(result,f,indent=2,allow_nan=False);f.write("\n")
 for x in rows:print(x["parent"],"fixed_bp",x["fixed_baseline_error_bp"],"introduced_bp",x["introduced_error_bp"])
if __name__=="__main__":
 try:main()
 except Exception:traceback.print_exc(file=sys.stdout);raise SystemExit(1)
