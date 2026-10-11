"""Focused fresh H024 producer/assembly/gradient checks on handmade records."""
from pathlib import Path
from types import SimpleNamespace as NS
from copy import deepcopy
import ast,dataclasses,hashlib,json,os,sys
import torch
from arggnn import anchor_parent_producer_v1 as p,anchor_parent_context_v3 as core,anchor_kernel as legacy,biological_candidate as frozen
torch.set_num_threads(1);torch.set_num_interop_threads(1)
R=p.ROOT;O=R/"research/reviews/anchor_parent_producer_v1_independent";A=R/"research/reviews/anchor_parent_producer_v1_author"
checks=[];details={};events=[];cpu_maps=[];tolerances=json.loads(core.CONFIG.read_text())["qualification"]["float32"]
assert os.environ["CUDA_VISIBLE_DEVICES"]==""
def ck(name,value=True):
 if not value:raise AssertionError(name)
 checks.append(name)
def audit(event,args):
 if event=="open" and not isinstance(args[0],int):
  q=Path(os.path.abspath(os.fsdecode(args[0])));mode,flags=args[1:3];writing=(isinstance(mode,str) and any(c in mode for c in "wax+")) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  own=q==O or O in q.parents
  if writing and not own:events.append("outside write "+str(q));raise RuntimeError(events[-1])
  if not writing and q==Path("/proc/self/maps"):cpu_maps.append(True);return
  if not writing and not own:
   for base in (R/"data",R/"experiments/results",R/".runtime/conda/envs",R/".runtime/gpup",R/".runtime/locks",R/".runtime/storage",Path("/proc")):
    if q==base or base in q.parents:events.append("actual state read "+str(q));raise RuntimeError(events[-1])
 if event in ("subprocess.Popen","os.system","os.posix_spawn","socket.connect","socket.bind"):events.append(event);raise RuntimeError(event)
sys.addaudithook(audit)
def sh(q):return hashlib.sha256(Path(q).read_bytes()).hexdigest()
pins={"src/arggnn/anchor_parent_producer_v1.py":"d11c9bb63048e343b96203059b5807e53db3cff38223f9f84a024606d0c838ee","configs/anchor_parent_producer_v1.json":"1a5a2e5d3d1080a285828959b013c648ae27ee28afd7229a0a42f78cd4884dcb","research/reviews/anchor_parent_producer_v1_author/author_receipt.json":"18b47cc0985e5ab61d431dfe5329934d9a398fb2c300795b13f486a22ebccc73","src/arggnn/anchor_parent_context_v3.py":"1b9f8f8148ed494373f9f76de677b847dff8ed256db15f6ff9b83ee0755f66e7"}
for q,h in pins.items():ck("pin_before_"+q,sh(R/q)==h)
def reject(name,fn):
 try:fn()
 except (ValueError,AssertionError):ck(name)
 else:raise AssertionError("accepted "+name)
# Borrow only fixture-construction definitions; fresh cases and numerical oracles below are independent.
s=(A/"metadata_checks.py").read_text().split("checks=[]")[0]
namespace={"__file__":str(O/"fresh_integration_checks.py")};exec(compile(s,str(A/"metadata_checks.py"),"exec"),namespace)
b,pr,ob,cfg,rows,derived,completion=[deepcopy(namespace[n]) for n in ("binding","prepared","observed","cfg","rows","derived","completion")]
def metadata(**changes):
 vals={"binding":b,"prepared":pr,"observed":ob,"cfg":cfg,"source_rows":rows,"derived_rows":derived,"completion":completion};vals.update(changes)
 return p.validate_metadata_records_v1(**vals)
ck("separate_metadata_positive_fixture",len(metadata()[2]["nodes"])==280)
for name,change in (
 ("completion_pointer_must_not_be_in_saved_record",lambda z:z.update(completion_provenance=ob["replicates"][0]["completion_provenance"])),
 ("completion_missing_original_field",lambda z:z.pop("commands_provenance")),
 ("completion_extra_unknown_field",lambda z:z.update(unregistered_value=17)),
):
 z=deepcopy(completion);change(z);reject(name,lambda z=z:metadata(completion=z))
z=deepcopy(derived);z[9]["age_generations"]=1.;reject("derived_observed_age_cannot_change",lambda:metadata(derived_rows=z))
z=deepcopy(pr);z["parents"][0]["actual_query_output_indexes"][3]=120;reject("actual120_output_join_cannot_change",lambda:metadata(prepared=z))
z=deepcopy(pr);z["parents"][0]["tiles"][4]["original_grid_indexes"].reverse();reject("readout_original_grid_order_cannot_change",lambda:metadata(prepared=z))
z=deepcopy(pr);z["parents"][0]["tiles"][2]["original_tree_ids"]=[2,2];reject("duplicate_serialized_tree_ID_cannot_pass",lambda:metadata(prepared=z))
z=deepcopy(pr);z["parents"][0]["tiles"][5]["all280_leaves_retained"]=False;reject("complete_native_leaf_retention_required",lambda:metadata(prepared=z))
z=deepcopy(pr);z["parents"][0]["reference_sample_indexes"].reverse();reject("pure96_reference_order_cannot_change",lambda:metadata(prepared=z))
# Native indexing is checked directly, with a nonzero physical offset and crossing trees.
nodes=[31+3*i for i in range(280)];cover=[500.,1999500.];cuts=[0.,390000.,810000.,1610000.,1999000.]
trees=[NS(index=i,num_roots=1,leaves=lambda:iter(nodes),interval=NS(left=cuts[i],right=cuts[i+1])) for i in range(4)]
ts=NS(num_trees=4,sequence_length=1999000.,samples=lambda:iter(nodes),trees=lambda:iter(trees))
record={"original_trees_preserved":4,"tiles":[]}
for i in range(8):
 left=max(cover[0],250000*i);right=min(cover[1],250000*(i+1))
 record["tiles"].append({"physical_interval_bp0":[left,right],"original_tree_ids":[t.index for t in trees if max(left,t.interval.left+cover[0])<min(right,t.interval.right+cover[0])]})
inventory=p.verify_native_inventory_v1(ts,record,{"nodes":nodes,"coverage":cover})
ck("native_inventory_exact_nonzero_offset",inventory[0]==(0,500.,390500.) and inventory[-1][-1]==1999500.)
bad=deepcopy(record);bad["tiles"][1]["original_tree_ids"].reverse();reject("native_crossing_tree_order_rejected",lambda:p.verify_native_inventory_v1(ts,bad,{"nodes":nodes,"coverage":cover}))
bad=deepcopy(trees);bad[1].index=9
badts=NS(num_trees=4,sequence_length=1999000.,samples=lambda:iter(nodes),trees=lambda:iter(bad))
reject("native_indices_never_inferred_by_equivalence",lambda:p.verify_native_inventory_v1(badts,record,{"nodes":nodes,"coverage":cover}))
# Adapt fixture setup to six leaves, five unique original IDs and target events at cuts.
tree=ast.parse((A/"handmade_checks.py").read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="inputs")
text=ast.unparse(node).replace("(4, 2, 2)","(6, 4, 2)").replace("max(0, i - 1)","(0, 0, 1, 1, 2, 3, 4, 4)[i]").replace("101 + tree_id","509 + 17 * tree_id").replace("* 3 + 11","* 7 + 13").replace("left + 0.03125","left")
ns={"torch":torch,"producer":p,"legacy":legacy,"NS":NS};exec(compile(text,str(O/"fresh_fixture_builder"),"exec"),ns);v=ns["inputs"]()
assemblies={g:p.assemble_parent_v1(v,g) for g in ("age","plain")}
for g,a in assemblies.items():
 ck("exact256_anchor_and_weight_union_"+g,len(a.parent.layout.anchors)==256 and torch.equal(a.parent.layout.anchors,torch.cat([t.layout.anchors for t in v.loaded_tiles])) and torch.equal(a.parent.layout.weights,torch.cat([t.layout.weights for t in v.loaded_tiles])))
 ck("all_original_occurrences_andIDs_retained_"+g,a.parent.original_tree_ids_by_cache_graph==(0,0,1,1,2,3,4,4))
 ck("fixed47_features_and_leaf_order_"+g,a.leaf.shape==(8,6,47) and not a.leaf.requires_grad and not a.response.requires_grad and torch.equal(a.leaf_ids,v.loaded_tiles[0].bank.sample_ids))
ck("age_plain_same_fixed_initial_inputs",torch.equal(assemblies["age"].leaf,assemblies["plain"].leaf) and torch.equal(assemblies["age"].response,assemblies["plain"].response))
bad=deepcopy(v);bad.loaded_tiles[7].bank.leaf_x[0,0,1]+=.01;reject("late_duplicated_original_tree_input_drift",lambda:p.assemble_parent_v1(bad,"age"))
bad=deepcopy(v);bad.loaded_tiles[0].bank.sample_ids=bad.loaded_tiles[0].bank.sample_ids.flip(0);reject("serialized_leaf_ID_order_drift",lambda:p.assemble_parent_v1(bad,"plain"))
reject("small_fixture_cannot_be_production",lambda:p.assemble_parent_v1(dataclasses.replace(v,fixture=False),"age"))
def close(name,x,y,atol=None,rtol=None):
 torch.testing.assert_close(x,y,atol=tolerances["vjp_atol"] if atol is None else atol,rtol=tolerances["vjp_rtol"] if rtol is None else rtol);ck(name)
for g,a in assemblies.items():
 model,opt=p.original_model_optimizer_v1(g,202)
 hook_rows=[];hook=model.encoder.register_forward_pre_hook(lambda mod,args:hook_rows.append(args[0].shape[0]))
 reset,context=p.forward_parent_v1(a,model,mode="reset",checkpoint_blocks=False);hook.remove()
 ck("encoder_once_on_five_unique_originalIDs_"+g,hook_rows==[5])
 reference=torch.cat([model(t) for t in v.loaded_tiles])
 close("reset_original_tile_output_at_cut_events_"+g,reset,reference,2e-5,2e-4)
 coefficient=torch.sin(torch.arange(reset.numel(),dtype=reset.dtype).reshape_as(reset)*.31)
 actual=torch.autograd.grad((reset*coefficient).sum(),tuple(model.parameters()))
 expected=torch.autograd.grad((reference*coefficient).sum(),tuple(model.parameters()))
 for i,(x,y) in enumerate(zip(actual,expected)):close("reset_complete_original_parameterVJP_"+g+str(i),x,y)
 carry,_=p.forward_parent_v1(a,model,mode="carry",checkpoint_blocks=False)
 ck("carry_retains_cross_cut_context_"+g,not torch.allclose(carry,reset,atol=1e-8,rtol=1e-8))
 leaf_before=a.leaf.clone();response_before=a.response.clone()
 for mode in ("reset","carry"):
  model,opt=p.original_model_optimizer_v1(g,202);rm=deepcopy(model)
  ro=torch.optim.AdamW(frozen.optimizer_groups(rm,json.loads((R/"configs/biological_candidate_pilot_v1.json").read_text())),lr=.001)
  ck("original_optimizer_groups_"+g+mode,[z["lr"] for z in opt.param_groups]==[.001,.001] and [z["weight_decay"] for z in opt.param_groups]==[.0001,0.])
  hooks=[];list(model.parameters())[0].register_hook(lambda grad:hooks.append(grad.detach().clone()))
  for iteration in range(2):
   def cell(which):
    def loss():
     x,_=p.forward_parent_v1(a,model,mode=mode,checkpoint_blocks=True)
     selected=x[:8] if which==0 else x[8:];coef=coefficient[:8] if which==0 else coefficient[8:]
     return 50*(selected*coef).sum()+.2*selected.square().sum(),2. if which==0 else 3.
    return loss
   result=core.matched_parent_step_v3(opt,tuple(model.parameters()),[cell(0),cell(1)])
   ro.zero_grad(set_to_none=True)
   rx=(torch.cat([rm(t) for t in v.loaded_tiles]) if mode=="reset" else p.forward_parent_v1(a,rm,mode="carry",checkpoint_blocks=False)[0])
   rl=(50*(rx*coefficient).sum()+.2*rx.square().sum())/5.;rl.backward();norm=torch.nn.utils.clip_grad_norm_(rm.parameters(),1.,error_if_nonfinite=True);ro.step()
   tag=g+mode+str(iteration)
   ck("one_actual_backward_hook_"+tag,len(hooks)==iteration+1)
   close("aggregate_loss_"+tag,torch.tensor(result["loss"]),rl.detach())
   close("aggregate_unclipped_norm_"+tag,torch.tensor(result["gradient_norm_before_single_clip"]),norm)
   ck("original_single_clip_step_"+tag,result["backward_calls"]==result["clip_calls"]==result["optimizer_steps"]==1 and result["gradient_norm_before_single_clip"]>1.)
   for i,(param,reference_param) in enumerate(zip(model.parameters(),rm.parameters())):
    ck("connected_gradient_"+tag+str(i),param.grad is not None and torch.isfinite(param.grad).all())
    close("complete_clipped_parameter_gradient_"+tag+str(i),param.grad,reference_param.grad)
    close("complete_parameter_update_"+tag+str(i),param,reference_param,1e-7,1e-5)
    ck("optimizer_state_keys_"+tag+str(i),opt.state[param].keys()==ro.state[reference_param].keys())
    for key,value in opt.state[param].items():close("complete_AdamW_state_"+tag+str(i)+key,value,ro.state[reference_param][key],1e-7,1e-5)
  details[g+"_"+mode]={"actual_helper_backward_hooks":len(hooks),"original_AdamW_steps":2,"synthetic_coefficients_no_targets":True}
 ck("fixed_original_inputs_unchanged_after_all_steps_"+g,torch.equal(a.leaf,leaf_before) and torch.equal(a.response,response_before))
# Integration does not bypass the previously reviewed forward-only callback guard.
model,opt=p.original_model_optimizer_v1("plain",202);before=[q.detach().clone() for q in model.parameters()]
def contaminated():
 x,_=p.forward_parent_v1(assemblies["plain"],model,mode="carry",checkpoint_blocks=False)
 list(model.parameters())[0].grad=torch.zeros_like(list(model.parameters())[0]);return x.sum(),1.
reject("producer_callbacks_keep_v3_gradient_purity",lambda:core.matched_parent_step_v3(opt,tuple(model.parameters()),[contaminated]))
ck("rejected_producer_callback_no_parameter_or_state_update",all(torch.equal(x,y) for x,y in zip(model.parameters(),before)) and not opt.state)
for q,h in pins.items():ck("pin_after_"+q,sh(R/q)==h)
ck("no_forbidden_access_events",not events)
report={"kind":"anchor_parent_producer_v1_fresh_independent_integration_checks","verdict":"PASS","count":len(checks),"checks":checks,"details":details,"audit_events":events,"allowed_own_CPU_library_map_reads":len(cpu_maps),"reviewed_pins":pins,"fixture_setup_disclosure":"Borrowed only author metadata/assembly fixture setup definitions; fresh offset/tree/join/unique-ID/cut-event/two-step optimizer and full-state cases and original-tile numerical oracles are independent.","actual_dataset_truth_TEST_GPU_profile_fit_budget_scheduling_or_Git":False,"handmade_model_and_disposable_optimizer_checks":True}
(O/"fresh_integration_checks.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
print(json.dumps({"verdict":"PASS","count":len(checks),"audit_events":events}))
