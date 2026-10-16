"""Narrow source-only checkpoint primitive and producer namespace joins; no payload imports."""
from pathlib import Path,PurePosixPath
import ast,copy,difflib,hashlib,json,random,sys,types
R=Path("/workspace-SR003.nfs2/ARG-GNN")
B="research/reviews/C3_CPU_checkpoint_schema_repair_independent_review_20261017_v1"
C=sys.argv[1]if len(sys.argv)>1 else"research/reviews/C3_CPU_checkpoint_schema_repair_source_20261017_v1"
VERSION=C.rsplit("_",1)[-1]
OLD="research/reviews/C3_CPU_actual_plan_lineage_integration_source_20261016_v1"
SCAN="research/reviews/C3_CPU_six_candidate_interleave_source_20261016_v1"
def text(path):return(R/path).read_text()
def defs(path):return {n.name:n for n in ast.parse(text(path)).body if isinstance(n,ast.FunctionDef)}
def execute(nodes,ns):exec(compile(ast.Module(body=copy.deepcopy(nodes),type_ignores=[]),"<exact_source_statements>","exec"),ns)
def require(ok,msg):
 if not ok:raise ValueError(msg)
def rejected(fn):
 try:fn()
 except (ValueError,KeyError,TypeError):return True
 return False
def fake(label):return dict(path="SYNTHETIC/"+label,sha256=hashlib.sha256(label.encode()).hexdigest())
# Exact five published diffs and all callable bodies outside explicit metadata delta.
V1="research/reviews/C3_CPU_checkpoint_schema_repair_source_20261017_v1"
BRIDGE=json.loads(text(C+"/source_handoff.v1.json"))["checkpoint_schema_bridge_ref"]["path"]
pairs=(("prediction_bridge_v1.py",SCAN),("acceptance_contract_v1.py",OLD),("prepare_evaluation_root_v1.py",OLD),("root_driver_v1.py",OLD),("worker_entry_v1.py",OLD))if VERSION=="v1"else tuple((n,V1)for n in ("acceptance_contract_v1.py","prepare_evaluation_root_v1.py","root_driver_v1.py","worker_entry_v1.py"))
for name,base in pairs:
 delta="".join(difflib.unified_diff(text(base+"/"+name).splitlines(True),text(C+"/"+name).splitlines(True),fromfile=base+"/"+name,tofile=C+"/"+name))
 assert delta==text(C+"/"+name+".patch"),name
old=defs(SCAN+"/prediction_bridge_v1.py");new=defs(BRIDGE)
assert ast.dump(old["new_candidate_models_v1"])==ast.dump(new["new_candidate_models_v1"])
a=copy.deepcopy(old["predict_v1"]);b=copy.deepcopy(new["predict_v1"])
la=next(n for n in a.body if isinstance(n,ast.For));lb=next(n for n in b.body if isinstance(n,ast.For))
ia=next(i for i,n in enumerate(la.body)if isinstance(n,ast.Expr)and any(isinstance(x,ast.Name)and x.id=="loaded"for x in ast.walk(n)))
ib=next(i for i,n in enumerate(lb.body)if isinstance(n,ast.Assign)and any(isinstance(x,ast.Name)and x.id=="loaded_metadata"for x in n.targets))
metadata_block=copy.deepcopy(lb.body[ib:ib+2]);oldassert=copy.deepcopy(la.body[ia])
del la.body[ia];del lb.body[ib:ib+2]
assert ast.dump(a)==ast.dump(b)
# The original scheduled metadata contract executes only on a synthetic epoch30 nested state.
trainpath="src/arggnn/american_c3_gnn_train_v1.py";train=defs(trainpath)
ns=dict(require=require,random=random,hashlib=hashlib,json=json,TRAIN=tuple("american_c3_short_train_%03d"%i for i in range(20)),
 CONDITION="fresh_C3_supervised_short_independent_complete_parents",SCHEMA="american_c3_checkpointed_complete_parent_training_v1",CHECKPOINT_EPOCHS=(5,10,15,20,25,30))
execute([defs("src/arggnn/american_c3_gnn_stream_train_v1.py")["json_sha"],train["epoch_order_v1"],train["_validate_progress"],train["scheduled_checkpoint_metadata_v1"]],ns)
history=[dict(global_step=(e-1)*20+i+1,epoch=e,parent_index=i,parent_id=p,records=10000,whole_parent_optimizer_updates=1,block_reset_or_truncated_BPTT=False)
 for e in range(1,31)for i,p in enumerate(ns["epoch_order_v1"](e))]
state=dict(schema=ns["SCHEMA"],global_step=600,history=history,focal_exposures=6000000,next_epoch=31,next_parent_index=0,next_parent_order=[],
 scheduler=dict(name="constant_learning_rate",completed_parent_updates=600,lr=.001),
 scheduled_checkpoints=[dict(epoch=e,global_step=e*20,checkpoint_ref=fake(str(e)))for e in (5,10,15,20,25)],
 protocol=dict(SYNTHETIC=True),protocol_sha256=ns["json_sha"](dict(SYNTHETIC=True)),observation_seal_sha256="a"*64,target_join_sha256="b"*64)
loaded=dict(schema="american_c3_fresh_whole_tree_GNN_checkpoint_v1",C4_initialization_or_head_slicing=False,model_seed=70001,training_condition=ns["CONDITION"],
 trainer_state=state,candidate=dict(eligible_for_VAL_selection=True,kind="scheduled_epoch",completed_epoch=30,global_step=600),
 trained_parent_ids=list(ns["TRAIN"]),train_sequence_lengths_bp=[2000000]*20)
checked=ns["scheduled_checkpoint_metadata_v1"](loaded,protocol_sha256=state["protocol_sha256"])
env=dict(ns,loaded=loaded,checked=checked,candidate=dict(epoch=30,global_step=600),training=dict(protocol_sha256=state["protocol_sha256"]))
execute(metadata_block,env);assert env["loaded_metadata"]==checked
assert rejected(lambda:execute([oldassert],env))
wrong=dict(env,checked=dict(checked,observation_seal_sha256="c"*64));assert rejected(lambda:execute(metadata_block,wrong))
wrong=dict(env,training=dict(protocol_sha256="d"*64));assert rejected(lambda:execute(metadata_block,wrong))
# Namespace/ROOT directive joins use exact source gate statements, no real producer or guard invocation.
contract=defs(C+"/acceptance_contract_v1.py");buildertree=ast.parse(text(C+"/prepare_evaluation_root_v1.py"))
builder_directive=next(ast.literal_eval(n.value)for n in buildertree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="DIRECTIVE"for t in n.targets))
ctree=ast.parse(text(C+"/acceptance_contract_v1.py"))
contract_directive=next(ast.literal_eval(n.value)for n in ctree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="DIRECTIVE"for t in n.targets))
tagassign=next(n for n in ast.walk(defs(C+"/prepare_evaluation_root_v1.py")["prepare_root_v1"])if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="tag"for t in n.targets))
prefixgate=next(n for n in contract["validate_request"].body if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)
 and any(isinstance(x,ast.Constant)and x.value=="Fresh separated future output and administrative scopes"for x in ast.walk(n)))
sourcegate=next(n for n in ast.walk(contract["stage_source_v1"])if isinstance(n,ast.Expr)and isinstance(n.value,ast.Call)
 and any(isinstance(x,ast.Constant)and x.value=="Exact new guarded future producer and separately owned scopes"for x in ast.walk(n)))
joins=[]
for phase in ("predict_val_003","predict_val_004","matrix","score"):
 tag_env=dict(args=types.SimpleNamespace(phase=phase,instance="SYNTHETIC"));execute([tagassign],tag_env);tag=tag_env["tag"]
 packet="research/reviews/C3_full30Mb_eval_ROOT_"+tag;output="experiments/results/C3_full30Mb_eval_"+tag
 gateenv=dict(require=require,req=dict(packet_directory=packet,output_directory=output),local=PurePosixPath,HERE=C)
 packet_pass=not rejected(lambda:execute([prefixgate],gateenv))
 sourceenv=dict(require=require,env=dict(root_driver_ref=fake("driver")),expected_driver=fake("driver"),spec=dict(packet_directory=packet))
 source_pass=not rejected(lambda:execute([sourcegate],sourceenv))
 joins.append(dict(phase=phase,packet=packet,output=output,request_namespace_accepted=packet_pass,completed_producer_namespace_accepted=source_pass))
v=dict(schema="C3_checkpoint_schema_narrow_independent_compatibility_checks_v1",source=C,
 metadata_primitive_nested_accepts=True,old_flat_assertion_rejects=True,metadata_or_protocol_drift_rejects=True,
 complete_prediction_body_outside_metadata_unchanged=True,published_patch_count=len(pairs),published_diffs_exact=True,
 builder_directive=builder_directive,contract_directive=contract_directive,directive_join=builder_directive==contract_directive,joins=joins,
 all_operational_joins_pass=builder_directive==contract_directive and all(x["request_namespace_accepted"]and x["completed_producer_namespace_accepted"]for x in joins),
 all_metadata_states_and_gate_inputs_synthetic=True,no_guard_models_payload_truth_jobs_or_Git=True)
with(R/B/("narrow_checks."+VERSION+".json")).open("x")as f:json.dump(v,f,sort_keys=True,indent=2);f.write("\n")
print(json.dumps(v,sort_keys=True))
