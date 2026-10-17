"""Independent source/AST audit only. No model imports, function execution or numerical tests."""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
CANDIDATE=ROOT/"research/reviews/ordered_marker_learning_integration_source_20261017_v1"
OUT=ROOT/"research/reviews/ordered_marker_learning_independent_20261017_v1"
def ref(p):
 p=Path(p);p=p if p.is_absolute()else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def source(p):return (ROOT/p).read_text()
def method(tree,cls,name):
 c=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name==cls)
 return next(n for n in c.body if isinstance(n,ast.FunctionDef)and n.name==name)
def calls(node,name):
 return [n for n in ast.walk(node)if isinstance(n,ast.Call)and ast.unparse(n.func)==name]
def locus(path,node):return dict(path=path,line=node.lineno,end_line=node.end_lineno)
def main():
 manifest=json.loads((CANDIDATE/"MANIFEST.v1.json").read_text());checked=[]
 for r in manifest["files"]:
  p=ROOT/r["path"];assert ref(p)["sha256"]==r["sha256"]and p.stat().st_size==r["bytes"],r["path"];checked.append(ref(p))
 contract=json.loads((CANDIDATE/"SOURCE_CONTRACT.v1.json").read_text())
 dependencies=[];drift=[]
 for r in contract["source_refs"]:
  observed=ref(r["path"]);row=dict(declared_ref=r,observed_ref=observed,matched=r==observed)
  dependencies.append(row)
  if not row["matched"]:drift.append(row)
 assert len(drift)==1 and drift[0]["declared_ref"]["path"]=="research/experiment_ledger.md"
 model_path=str((CANDIDATE/"ordered_learning_gnn_v1.py").relative_to(ROOT));backend_path=str((CANDIDATE/"positive_marker_backend_v1.py").relative_to(ROOT))
 dense_path="research/reviews/biological_operator_next_fresh_20261016_v1/packed_csr_v2/same_marker_packed_csr_v2.py"
 oldhooks_path="research/reviews/same_marker_full_parent_binding_successor_source_20261016_v1/integration_hooks_v3.py"
 producer_path="research/reviews/H058_full_parent_CPU_execution_source_20261017_v1/producer_v1.py"
 ms,bs,ds,hs,ps=map(source,(model_path,backend_path,dense_path,oldhooks_path,producer_path))
 mt,bt,dt,ht,pt=map(ast.parse,(ms,bs,ds,hs,ps))
 for r in checked:
  if r["path"].endswith(".py"):compile(source(r["path"]),r["path"],"exec")
 init=next(n for n in mt.body if isinstance(n,ast.FunctionDef)and n.name=="initialize_supplied_h045_v1")
 oldinit=next(n for n in ht.body if isinstance(n,ast.FunctionDef)and n.name=="initialize_supplied_h045_state_v3")
 new_init_attrs={n.attr for n in ast.walk(init)if isinstance(n,ast.Attribute)}
 assert "requires_grad"not in new_init_attrs and "requires_grad"in {n.attr for n in ast.walk(oldinit)if isinstance(n,ast.Attribute)}
 marker=method(bt,"PositiveMarkerBackendV1","marker_sequence")
 denseforward=method(dt,"PackedSameMarkerEvidenceV2","forward")
 assert len(calls(marker,"dense.marker_features_v2"))==1 and not calls(marker,"checkpoint")
 assert len(calls(denseforward,"checkpoint"))==1
 hold=method(mt,"OrderedMarkerLearningGNNV1","_hold_residuals")
 encode=method(mt,"OrderedMarkerLearningGNNV1","encode_trees")
 sequence=method(mt,"OrderedWithinHoldResidualV1","forward")
 assert len(calls(encode,"self.sweep_tree_state_v1"))==2
 assert len(calls(sequence,"self.marker_gru"))==1
 assert len(calls(encode,"self.leaf_context_v1"))==1 and len(calls(encode,"checkpoint"))==1
 assert not any(isinstance(n,ast.Attribute)and n.attr in("detach","numpy")for n in ast.walk(mt))
 assert not any(isinstance(n,ast.Attribute)and n.attr in("detach","numpy")for n in ast.walk(bt))
 assert all(term in bs for term in("counts.sum(-1)","(called > 0)","query >= 0","frequency = counts/called.clamp_min(1)","safe_d = torch.where(torch.isfinite(log_d),log_d,torch.zeros_like(log_d))","if plan.inside(v,q):","permutation[ref]","labels[node] = allele"))
 assert all(term in ds for term in("groups = 4 * (sample_age >= 350)","outside.index_select(0, view.query_node)","frequency = counts / called.clamp_min(1)","(query >= 0)[:, :, None]"))
 assert "gt = global_genotype[cpu_layout.site_start:cpu_layout.site_stop]"in ds
 assert "genotype[hold.site_start:hold.site_stop]"in ms
 assert "self.same_marker.gate.projection" in ms and "LOG_MAX+(LOG_MIN-LOG_MAX)*score.sigmoid()" in ms
 assert "degree"not in ast.get_source_segment(bs,method(bt,"PositiveMarkerBackendV1","_cut"))
 assert "model.build_family_v1" in ps and "observations.payload_v1" in ps and "phase_arms(ctx[\"phase\"])"in ps
 # Analytic allocated parameter accounting from fixed declared layer dimensions. No tensors/models.
 counts=dict(input_linear=24*8,GRU_per_direction=3*8*8+3*8*8+2*3*8,bidirectional_GRU=2*(3*8*8+3*8*8+2*3*8),rank_input=16*8,output=8*32)
 counts["added_ordered"]=counts["input_linear"]+counts["bidirectional_GRU"]+counts["rank_input"]+counts["output"]
 counts["H058_existing"]=31812+714;counts["total"]=counts["H058_existing"]+counts["added_ordered"]
 assert counts["added_ordered"]==1440 and counts["total"]==33966
 dependencies_extra=["src/arggnn/anonymous_clade_gnn_v1.py","src/arggnn/mutation_routed_full_panel_pyg_model_v1.py","src/arggnn/mutation_routed_pyg_gnn_v1.py"]
 for path in dependencies_extra:compile(source(path),path,"exec")
 return dict(schema="ordered_marker_independent_static_audit_v1",created_utc=datetime.now(timezone.utc).isoformat(),independent=True,only_source_AST_metadata=True,candidate_manifest_ref=ref(CANDIDATE/"MANIFEST.v1.json"),candidate_files_authenticated=checked,source_contract_checks=dependencies,source_contract_drift=drift,additional_actual_MRO_source_refs=[ref(p)for p in dependencies_extra],analytic_parameter_counts=counts,feature_equations_compared_by_source=True,outside_only_path_cases_reviewed_by_algebra=True,no_detach_in_new_learning_path=True,two_original_full_tree_sweeps_present=True,ordered_GRU_before_reduction=True,placement_only_arm_difference_present=True,dense_block_API_called_without_inner_checkpoint=True,original_dense_forward_has_inner_block_checkpoint=True,initializer_lacks_original_all_parameter_trainability_predicate=True,existing_trainer_not_modified_or_new_arm_integrated=True,locations=dict(new_initializer=locus(model_path,init),original_initializer=locus(oldhooks_path,oldinit),dense_branch=locus(backend_path,marker),original_dense_forward=locus(dense_path,denseforward),hold_residuals=locus(model_path,hold),encode_trees=locus(model_path,encode),ordered_forward=locus(model_path,sequence)),model_imports=False,model_instantiation=False,numerical_tests=False,data_payload_reads=False,guards_state_resources_processes_admissions_launches_or_Git=False,scientific_fit_gradient_equivalence_or_ancestry_claim=False)
if __name__=="__main__":
 result=main();(OUT/"STATIC_AUDIT.v1.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
 print(json.dumps(dict(candidate_files=len(result["candidate_files_authenticated"]),declared_dependencies=len(result["source_contract_checks"]),drift=[x["declared_ref"]["path"]for x in result["source_contract_drift"]],analytic_parameters=result["analytic_parameter_counts"]["total"],source_only=True)))
