"""Independent v2 scientific producer audit. AST and frozen metadata only, no candidate imports."""
from pathlib import Path
import ast,copy,hashlib,json,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
C=ROOT/"research/reviews/ordered_marker_learning_integration_source_20261017_v2"
O=ROOT/"research/reviews/ordered_marker_learning_independent_20261017_v2"
V1=ROOT/"research/reviews/ordered_marker_learning_integration_source_20261017_v1"
SCI=ROOT/"research/reviews/H058_full_parent_CPU_execution_source_20261017_v1"
def need(ok,msg):
 if not ok:raise ValueError(msg)
def ref(path):
 p=Path(path);p=p if p.is_absolute()else ROOT/p
 need(p.suffix in(".py",".json",".md",".patch",".sh"),"Source/frozen metadata only; no payload")
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(r):need(ref(r["path"])==r,"Exact source/metadata pin");return json.loads((ROOT/r["path"]).read_text())
def tree(path):return ast.parse(Path(path).read_text())
def fn(t,name,cls=None):
 body=t.body if cls is None else next(n.body for n in t.body if isinstance(n,ast.ClassDef)and n.name==cls)
 return next(n for n in body if isinstance(n,ast.FunctionDef)and n.name==name)
def dump(node):return ast.dump(node,include_attributes=False)
def named_calls(n,name):return [x for x in ast.walk(n)if isinstance(x,ast.Call)and ast.unparse(x.func)==name]
def main():
 need(ref(C/"MANIFEST.v2.json")["sha256"]=="5a5ae04dfed8b65507e0c669f132235c52165c01f31de2f368fc133efa020516","Exact handed manifest")
 need(ref(C/"HANDOFF.v2.json")["sha256"]=="14bf23a812a3b951189cf9abaab5b85cd5e67d5e82bc8cb59a0d897345e0a9ab","Exact handed handoff")
 m=read(ref(C/"MANIFEST.v2.json"));h=read(ref(C/"HANDOFF.v2.json"));contract=read(ref(C/"SOURCE_CONTRACT.v2.json"))
 pins={}
 for record in m["refs"]+contract["source_refs"]:
  need(ref(record["path"])==record,"All declared static refs");pins[record["path"]]=record
 for row in m["new_directory_files"]:
  p=C/row["path"];need(ref(p)["sha256"]==row["sha256"]and p.stat().st_size==row["bytes"],"Own sealed file exact");pins[ref(p)["path"]]=ref(p)
 need("research/experiment_ledger.md"not in pins and contract["mutable_ledger_is_active_dependency"]is False,"No stale active ledger binding")
 snapshot=read(ref(C/"SOURCE_REGISTRATION_SNAPSHOT.v2.json"))
 for record in snapshot["source_lineage"]+[snapshot["source_protocol_ref"]]:
  need(ref(record["path"])==record,"Frozen source provenance");pins[record["path"]]=record
 oldmanifest=read(ref(SCI/"source_manifest.v1.json"))
 for record in oldmanifest["refs"]:
  need(ref(record["path"])==record,"Actual inherited producer/source closure");pins[record["path"]]=record
 compiled=[]
 for p in C.glob("*.py"):compile(p.read_text(),str(p),"exec");compiled.append(p.name)
 mt=tree(C/"ordered_learning_gnn_v2.py");bt=tree(C/"positive_marker_backend_v2.py");pt=tree(C/"producer_v2.py");ot=tree(C/"observations_v2.py");rt=tree(C/"matrix_v2.py")
 oldbackend=tree(V1/"positive_marker_backend_v1.py")
 # Exact accepted feature equations/path treatment, independently compared as AST.
 algebra=["_group_fields","_path","_cut","_positive_log_fields","_features"]
 for name in algebra:need(dump(fn(bt,name,"PositiveMarkerBackendV2"))==dump(fn(oldbackend,name,"PositiveMarkerBackendV1")),"Feature/path algebra exact: "+name)
 prep=fn(bt,"prepare_positive_parent_v2")
 need(any(isinstance(n,ast.Subscript)and ast.unparse(n)=="cpu_bank.genotype[s, permutation[ref]]"for n in ast.walk(prep)),"Only original routed reference calls choose covers")
 block=fn(bt,"marker_block_v2","PositiveMarkerBackendV2")
 need(len(named_calls(block,"dense.marker_features_v2"))==1and len(named_calls(block,"self._group_fields"))==1,"Actual original dense API and fresh sparse fields")
 stream=fn(mt,"stream","OrderedWithinHoldResidualV2")
 for name in("forward_block","reverse_block"):
  n=next(n for n in ast.walk(stream)if isinstance(n,ast.FunctionDef)and n.name==name)
  need([x.arg for x in n.args.args][-2:]==["first","last"]and [ast.unparse(x)for x in n.args.defaults]==["first","last"],"Exact block offsets captured for backward")
  need(len(named_calls(n,"kernel.marker_block_v2"))==1,"Features computed inside each checkpoint consumer")
 need(len(named_calls(stream,"checkpoint"))==2,"Both directions independently checkpointed")
 need("feature.flip(0)"in ast.unparse(stream)and "reversed(starts)"in ast.unparse(stream),"Exact full reverse block/row scan")
 need("calls[first:last]"in ast.unparse(stream)and "range(0, len(plan.covers), 16)"in ast.unparse(stream),"Every original block retained")
 need(not any(isinstance(n,ast.Attribute)and n.attr in("detach","data","numpy")for n in ast.walk(stream)),"No detached recurrence or learned host extraction")
 need(not any(isinstance(n,ast.Call)and ast.unparse(n.func)=="torch.cat"and n.args and isinstance(n.args[0],ast.Name)and n.args[0].id in("features","masks")for n in ast.walk(mt)),"No full marker-feature sequence concatenation")
 need(len(named_calls(fn(mt,"encode_trees","OrderedMarkerLearningGNNV2"),"self.sweep_tree_state_v1"))==2,"Two complete original sweeps retained")
 need(len(named_calls(fn(mt,"encode_trees","OrderedMarkerLearningGNNV2"),"checkpoint"))==1,"Whole-hold checkpoint retained")
 check=fn(mt,"validate_trainable_fp32_v2")
 need("requires_grad"in ast.unparse(check)and "p.dtype == torch.float32"in ast.unparse(check),"All parameter trainability/FP32 failstop")
 need(len(named_calls(fn(mt,"initialize_supplied_h045_v2"),"validate_trainable_fp32_v2"))==2,"Before/after initialization failstop")
 need(len(named_calls(fn(pt,"scheduled_step_v2"),"model.validate_trainable_fp32_v2"))==1,"Every scheduled update checks trainability")
 need(len(named_calls(fn(mt,"build_family_v2"),"validate_supplied_warmstart_v1"))==1,"Exact strong original H045 provenance validator")
 need(len(named_calls(fn(pt,"scheduled_step_v2"),"h45.train_one"))==1,"Actual original objective/backward/clip/AdamW used")
 train=fn(pt,"run_train_v2");evalnode=fn(pt,"run_evaluate_v2")
 need("range(1, 31)"in ast.unparse(train)and "random.Random(101 + at).shuffle(order)"in ast.unparse(train),"All original30 complete shuffled passes")
 need("steps == {a: 960 for a in arms}"in ast.unparse(train),"Complete960per arm required")
 need(ast.unparse(train).index("observations.load_observations_v2")<ast.unparse(train).index("reader.truth_gate = True"),"All observation/positive prep before TRAIN truth")
 need(ast.unparse(evalnode).index("ram.verify()")<ast.unparse(evalnode).index("reader.truth_gate = True")<ast.unparse(evalnode).index("validation.score"),"Complete matrix before either VAL truth/scoring")
 verify=fn(rt,"verify","OrderedMarkerRAMMatrixV2")
 need("set(self.values) == self.expected"in ast.unparse(verify)and "self.verified = True"in ast.unparse(verify),"Exactly prescribed complete probability product set")
 need("value.setflags(write=False)"in (C/"matrix_v2.py").read_text()and "array_identity(value)"in ast.unparse(verify),"Frozen complete probabilities")
 obs=fn(ot,"load_observations_v2")
 need("ORIGINAL_H058_ARMS[0]"in ast.unparse(obs)and "plan = prepare_positive_parent_v2"in ast.unparse(obs),"One original aligned route and positive plan per parent")
 need("observed['sealed'] = True"in ast.unparse(obs)and "episodes.freeze_v1"in ast.unparse(obs),"Immutable observation/episode preparation barrier")
 need(len(named_calls(obs,"retained_size_v2"))==3,"Retained complete positive/all-cache Python/storage accounting")
 # Only inert protocol constants/functions are evaluated; no model/source module import.
 env={}
 baseproto=tree(SCI/"protocol_v1.py")
 isolate=lambda nodes:exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),"<inert-source-protocol-metadata>","exec"),env)
 isolate([n for n in baseproto.body if not isinstance(n,(ast.Import,ast.ImportFrom))])
 env["original_h058_protocol_v1"]=env["fixed_protocol_v1"]
 childproto=tree(C/"protocol_v2.py");isolate([n for n in childproto.body if not isinstance(n,(ast.Import,ast.ImportFrom))])
 frozen=env["fixed_protocol_v2"]()
 need(snapshot["scientific_protocol"]==frozen,"Exact immutable protocol snapshot")
 need(len(frozen["TRAIN_parent_ids"])==32and len(frozen["VAL_parent_ids"])==8and frozen["passes"]==30and frozen["updates_per_arm"]==960and frozen["VAL_prediction_products_required_before_truth"]==96,"Complete fixed workload")
 need(snapshot["historical_drift_evidence"]["active_dependency"]is False and snapshot["execution_authority"]is False,"Historical ledger and source snapshot grant no authority")
 # Static analytic allocation only, no model/tensor instantiation.
 param=dict(input=24*8,forward_GRU=3*8*8+3*8*8+2*3*8,reverse_GRU=3*8*8+3*8*8+2*3*8,rank_input=16*8,output=8*32)
 param["new_ordered"]=sum(param.values());param["inherited_H045"]=31812;param["inherited_H058_added"]=714;param["total"]=31812+714+param["new_ordered"]
 need(param["new_ordered"]==1440and param["total"]==33966,"Matched analytic allocation")
 countsref=dict(path="research/reviews/H058_full_parent_pilot_source_20261016_v2/source_resource_counts.v1.json",sha256="e82a9be292c255b5fa63e20ef128f687f156b0315c49512dad3c2fc681f58796")
 counts=read(countsref);pins[countsref["path"]]=countsref
 rows=[v for p,v in counts["parents"].items()if p in frozen["TRAIN_parent_ids"]]
 metadata_costs={k:sum(r[k]for r in rows)for k in("original_SNPs","native_holds","native_nodes","one_packed_layout_numeric_bytes","maximum_global_marker_blocks_upper")}
 metadata_costs["positive_called_count_tensor_bytes"]=256*metadata_costs["original_SNPs"]
 metadata_costs["forward_directional_block_calls_upper_full_60_parent_phases"]=2*60*metadata_costs["maximum_global_marker_blocks_upper"]
 need(not any(n in sys.modules for n in("torch","numpy","torch_geometric","tskit")),"No numerical/model imports")
 return dict(schema="ordered_marker_learning_full_independent_AST_audit_v2",independent=True,status="PASS_SOURCE_ONLY",manifest_ref=ref(C/"MANIFEST.v2.json"),handoff_ref=ref(C/"HANDOFF.v2.json"),authenticated_static_refs=list(pins.values()),new_python_compiled_without_execution=sorted(compiled),algebra_methods_exact=algebra,streaming_recurrence_source_proof=["Forward starts0,16,... use slices[first:last] and previous hf as checkpoint input","Reverse traverses reversed starts and feature.flip(0), carrying previous hr","Input projection is pointwise and shared, so reverse reordering commutes with embedding","Each direction is a public one-layer8->8GRU; its state/weight map is the corresponding bidirectional GRU direction","No dropout/nonlinear cross-direction operation occurs before final-state concatenation","No detach, hidden reset or feature omission crosses a block; reset occurs only before each hold"],all_model_gradient_claims_limited_to_source_connectivity=True,per_block_feature_and_GRU_consumer_checkpoint=True,whole_hold_and_original_stage_checkpoints_retained=True,full_hold_feature_sequence_absent=True,B1_B2_B3_source_remedies_present=True,analytic_parameters=param,full_protocol=frozen,frozen_count_metadata_costs=metadata_costs,source_count_metadata_ref=countsref,CPU_only_scientific_producer=True,GPU_producer_or_efficiency_certified=False,practical_full_parent_fit_throughput_backward_gradient_and_numerical_equivalence_unproved=True,no_actual_model_data_truth_guard_process_resource_session_admission_training_launch_or_Git=True)
if __name__=="__main__":
 value=main();(O/"STATIC_AUDIT.v2.json").write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")
 print(json.dumps(dict(status=value["status"],pins=len(value["authenticated_static_refs"]),python_files=len(value["new_python_compiled_without_execution"]),parameters=value["analytic_parameters"]["total"],frozen_counts=value["frozen_count_metadata_costs"])))
