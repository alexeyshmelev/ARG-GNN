"""Synthetic metadata/fake tensor source fixtures only. Never imports Torch or opens scientific artifacts."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,os,subprocess,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
import saved_state_core_v1 as core
import acceptance_contract_v1 as c
def record(path):return dict(path=path,sha256=hashlib.sha256(path.encode()).hexdigest())
def tensor(n=1,shape=None,scalar=None,finite=True):
    return dict(shape=[n] if shape is None else shape,numel=n,dtype="torch.float32",CPU=True,finite=finite,
        sha256="a"*64,scalar=scalar)
class FakeReader:
    def __init__(self,values):self.values=values;self.reads=[];self.finishes=[]
    def load(self,r):self.reads.append(r["path"]);return self.values[r["path"]]
    def tensor(self,t):return {k:v for k,v in t.items() if k!="scalar"}
    def scalar_int(self,t):return t["scalar"]
    def finish(self,r):self.finishes.append(r["path"])
def fixture():
    meta={};authenticate=[]
    def store(path,value):
        r=record(path);meta[path]=value;return r
    registry=record("synthetic_registry.json");sources=[record("synthetic_source.py")]
    parents={};joins={};plans={}
    for pid in core.TRAIN:
        root="synthetic/"+pid+"/"
        focal=record(root+"focals.npz");sourceobs=record(root+"source_observed.json")
        obs=store(root+"observed.json",dict(parent_id=pid,split="train",sequence_length_bp=2000000,class_order=core.CLASSES,
            counts_haploid=dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000),query_ancestry_truth_in_inputs=False,
            sites_and_sample_identities_verified=True,migrations_and_internal_population_labels_removed=True,
            genealogy_kind="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG",source_observation_ref=sourceobs))
        prep=store(root+"prep.json",dict(parent_id=pid,status="completed_C3_exact_native_focal_registry_only_v1",
            observation_ref=obs,focals_ref=focal,query_focal_count=10000,query_ancestry_truth_opened=False,
            operational_bridge_ref=sources[0],original_native_focal_rule_source_ref=sources[0],source_author_function_ref=sources[0]))
        plan=store(root+"plan.json",dict(schema="american_c3_whole_tree_block_index_v1",parent_id=pid,
            sequence_length_bp=2000000,haplotypes=1600,focal_records=10000,targets_opened=False,all_trees_and_samples_retained=True,
            genomic_context="two_pass_full_parent_state_carry",class_order=core.CLASSES,observation_ref=obs,focal_registry_ref=focal,
            blocks=[dict(start_tree=0,stop_tree=1)],trees=1,config=dict(native_preparation_ref=prep)))
        plans[pid]=plan
        parents[pid]=dict(schema="american_c3_observation_blocks_sealed_v1",parent_id=pid,plan_ref=plan,observation_ref=obs,
            focal_registry_ref=focal,focal_identity_sha256="b"*64,observation_only=True,all_blocks_sealed_before_truth=True,
            all_trees_and_samples_retained=True,focal_records=10000,blocks=1,block_feature_sha256=["c"*64])
        truth=record(root+"NEVER_OPEN_private_truth.npz")
        sim=store(root+"sim.json",dict(status="completed_C3_simulation_reconstruction",parent_id=pid,class_order=core.CLASSES,
            observation_ref=sourceobs,private_truth_ref=truth))
        cfg=store(root+"cfg.json",dict(parent_id=pid,split="train",sequence_length_bp=2000000,class_order=core.CLASSES,
            sample_counts=dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000)))
        joins[pid]=dict(schema="C3_external_TRAIN_truth_focal_join_v1",parent_id=pid,split="train",records=10000,
            class_order=core.CLASSES,labels_in_observation_bank=False,purity_filter=False,target_equations_unchanged=True,
            original_scientific_registry_unchanged=True,nearest_or_majority_fallback=False,
            interval_rule="half_open_searchsorted_right_original_sample_index",focal_registry_ref=focal,
            focal_identity_sha256="b"*64,actual_simulation_completion_ref=sim,actual_simulation_config_ref=cfg,truth_ref=truth)
    obs=dict(schema="american_c3_all20_TRAIN_observation_barrier_v1",all20_ready=True,observation_only=True,
        truth_resolver_calls_before_barrier=0,parent_ids=core.TRAIN,parents=parents,registry_ref=registry)
    tar=dict(schema="american_c3_all20_external_TRAIN_target_joins_v1",all20_targets_ready=True,labels_in_observation_bank=False,
        purity_filter=False,parents=joins,registry_ref=registry,observation_seal_sha256=core.digest(obs))
    protocol=dict(source_refs=sources,parent_plan_refs=plans)
    candidates=[dict(epoch=e,global_step=e*20,checkpoint_ref=record(
        "synthetic/migrated_checkpoint_epoch_005.pt" if e==5 else
        "experiments/results/C3_GNN_efficient_resume_GPU0_train_v1/checkpoint_epoch_%03d.pt"%e)) for e in core.EPOCHS]
    history=[dict(global_step=i+1,epoch=e,parent_index=j,parent_id=p,records=10000,
        whole_parent_optimizer_updates=1,block_reset_or_truncated_BPTT=False) for i,(e,j,p) in enumerate(core.original_history_v1())]
    training=dict(status="completed_C3_checkpointed_complete_parent_training",schema="american_c3_checkpointed_complete_parent_training_v1",
        complete_parent_updates=600,epochs=30,focal_exposures=6000000,trained_parent_ids=core.TRAIN,
        train_sequence_lengths_bp=[2000000]*20,VAL_used_for_optimization=False,TEST_deferred=True,H043_extension=None,purity_filter=False,
        resume_ref=candidates[0]["checkpoint_ref"],protocol_sha256=core.digest(protocol),observation_seal_sha256=core.digest(obs),
        target_join_sha256=core.digest(tar),checkpoint_candidates=candidates,history=history,registry_ref=registry)
    migration=dict(migrated_boundary_ref=candidates[0]["checkpoint_ref"],actual_global_step=100,actual_AdamW_step=100,
        remaining_updates=500,passed=True,successor_protocol_sha256=training["protocol_sha256"],
        observation_seal_sha256=training["observation_seal_sha256"],target_join_sha256=training["target_join_sha256"],
        migrated_checkpoint_receipts=[dict(migrated_ref=candidates[0]["checkpoint_ref"])])
    contract=dict(schema="american_c3_fresh_whole_tree_GNN_checkpoint_v1",class_order=core.CLASSES,hidden_dim=32,
        donor_cohorts=2,donor_channels=6,expected_haplotypes=1600,initialization="fresh_no_C4_loading",
        site_feature_width=15,observed_leaf_width=5,recurrent_contract="one_layer_bidirectional_equations_as_two_directional_GRUs",
        tree_backbone="unchanged_PyG_gated_upward_recipient_excluded_downward_cavity",whole_parent_training_required=True)
    values={}
    for i,row in enumerate(candidates):
        step=row["global_step"];shapes=[29534]+[1]*37;model={str(j):tensor(n) for j,n in enumerate(shapes)}
        states={j:dict(step=tensor(shape=[],scalar=step),exp_avg=tensor(n),exp_avg_sq=tensor(n)) for j,n in enumerate(shapes)}
        state=dict(schema=training["schema"],global_step=step,protocol=protocol,protocol_sha256=core.digest(protocol),
            observation_seal_sha256=training["observation_seal_sha256"],target_join_sha256=training["target_join_sha256"],
            history=history[:step],scheduled_checkpoints=candidates[:i],focal_exposures=step*10000,next_epoch=step//20+1,
            next_parent_index=0,next_parent_order=core.order(step//20+1) if step<600 else [],
            scheduler=dict(name="constant_learning_rate",completed_parent_updates=step,lr=.001),
            rng_state=dict(python=[1],numpy=dict(algorithm="MT19937",keys=[0]*624),torch_CPU=tensor(8),torch_device=tensor(8),device_type="cuda"))
        values[row["checkpoint_ref"]["path"]]=dict(schema=contract["schema"],contract=contract,model_seed=70001,
            training_condition=core.CONDITION,C4_initialization_or_head_slicing=False,trained_parent_ids=core.TRAIN,
            train_sequence_lengths_bp=[2000000]*20,trainer_state=state,
            candidate=dict(eligible_for_VAL_selection=True,kind="scheduled_epoch",completed_epoch=row["epoch"],global_step=step),
            model_state=model,optimizer_state=dict(state=states,param_groups=[dict(params=list(range(38)),lr=.001,weight_decay=.0001,betas=[.9,.999],eps=1e-8)]))
    return training,migration,obs,tar,FakeReader(values),meta,authenticate
checks=[]
def yes(name,test):
    assert test,name;checks.append(name)
def reject(name,fn):
    try:fn()
    except (ValueError,KeyError,TypeError,AssertionError):checks.append(name);return
    raise AssertionError("Expected rejection: "+name)
def run(f):
    t,m,o,j,r,meta,auth=f
    def read(q):
        assert q["path"].endswith(".json") and "private_truth" not in q["path"]
        auth.append(q["path"]);return meta[q["path"]]
    return core.audit_saved_v1(t,m,o,j,r,read,lambda q:auth.append(q["path"]))
f=fixture();result=run(f)
yes("fake_interface_all6_states_and_all20_provenance",result["all_six_verified"] and result["provenance"]["parents"]==20 and len(f[4].reads)==6 and len(f[4].finishes)==6)
yes("no_truth_payload_opened",all("private_truth" not in p for p in f[6]))
def mutate(name,fn):
    q=copy.deepcopy(fixture());fn(q);reject(name,lambda:run(q))
mutate("reject_final_only",lambda q:q[0].update(checkpoint_candidates=q[0]["checkpoint_candidates"][-1:]))
mutate("reject_wrong_actual_optimizer_count",lambda q:q[4].values[q[0]["checkpoint_candidates"][-1]["checkpoint_ref"]["path"]]["optimizer_state"]["state"][0]["step"].update(scalar=599))
mutate("reject_nonfinite_model",lambda q:next(iter(q[4].values.values()))["model_state"]["0"].update(finite=False))
mutate("reject_wrong_moment_shape",lambda q:next(iter(q[4].values.values()))["optimizer_state"]["state"][0]["exp_avg"].update(shape=[1]))
mutate("reject_history_order",lambda q:next(iter(q[4].values.values()))["trainer_state"]["history"][0].update(parent_id="wrong"))
mutate("reject_protocol_self_hash",lambda q:next(iter(q[4].values.values()))["trainer_state"]["protocol"].update(extra=True))
mutate("reject_missing_saved_RNG",lambda q:next(iter(q[4].values.values()))["trainer_state"]["rng_state"].pop("torch_device"))
mutate("reject_missing_TRAINseal",lambda q:q[2]["parents"].pop(core.TRAIN[-1]))
mutate("reject_cropped_TRAINplan",lambda q:q[5][q[2]["parents"][core.TRAIN[0]]["plan_ref"]["path"]].update(haplotypes=1599))
mutate("reject_query_truth_in_input",lambda q:q[5][q[2]["parents"][core.TRAIN[0]]["observation_ref"]["path"]].update(query_ancestry_truth_in_inputs=True))
mutate("reject_wrong_simulation_join",lambda q:q[5][q[3]["parents"][core.TRAIN[0]]["actual_simulation_completion_ref"]["path"]].update(parent_id="wrong"))
# Real adapter interface uses six tiny synthetic bytes and a fake Torch API, never real checkpoint/model data.
class FakeArray:
    def tobytes(self):return b"synthetic_tensor"
class FakeTensor:
    shape=(1,);device=type("Device",(),dict(type="cpu"))();dtype="torch.float32"
    def numel(self):return 1
    def contiguous(self):return self
    def numpy(self):return FakeArray()
    def item(self):return 100
class Finite:
    def all(self):return True
class FakeTorch:
    cuda=type("CUDA",(),dict(is_initialized=staticmethod(lambda:False)))()
    def __init__(self):self.loads=[]
    def load(self,p,**kw):self.loads.append(kw);return {"synthetic":True}
    def is_tensor(self,t):return isinstance(t,FakeTensor)
    def isfinite(self,t):return Finite()
synthetic=HERE/"synthetic_interface";synthetic.mkdir(exist_ok=True);expected=[]
for i in range(6):
 p=synthetic/("checkpoint%d.pt"%i)
 if not p.exists():p.write_bytes(("synthetic%d"%i).encode())
 expected.append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
ft=FakeTorch();real=core.TorchCPUReader(ft,ROOT,expected)
for r in expected:yes("real_adapter_mock_load_"+r["path"][-4:],real.load(r)=={"synthetic":True});real.finish(r)
yes("real_adapter_CPU_weights_only_mmap_flags",all(x==dict(map_location="cpu",weights_only=True,mmap=True) for x in ft.loads))
yes("real_adapter_tensor_and_scalar_interfaces",real.tensor(FakeTensor())["CPU"] and real.scalar_int(FakeTensor())==100)
reject("real_adapter_rejects_unpinned_checkpoint",lambda:real.load(dict(expected[0],sha256="0"*64)))
# Acceptance evidence fixture uses only synthetic in-memory JSON objects.
training_ref=record("synthetic_training.json");source_ref=record("synthetic_contract.py");driver_ref=record("synthetic_driver.py");manifest_ref=record("synthetic_manifest.json")
inp=record("synthetic_input.json");complete=record("synthetic_completed.json");vr=record("synthetic_result.json");waitref=record("synthetic_wait.json")
bundle=dict(schema="C3_saved600_ROOT_artifact_acceptance_v1",root_authorizer="root_agent",accepted_saved_artifacts_only=True,
 original_invocation_success_claimed=False,original_outerUNKNOWN_explicitly_acknowledged=True,
 future_original_C3_training_writes_cooperatively_revoked=True,storage_close_or_credit_authorized=False,
 original_history=copy.deepcopy(c.ORIGINAL_HISTORY),training_completion_ref=training_ref,source_manifest_ref=manifest_ref,
 validator_input_ref=inp,validation_result_ref=vr,validator_completion_ref=complete,actual_wait_ref=waitref)
env=dict(status="completed_C3_saved600_artifact_validation",stage=c.ACCEPT_PHASE,root_driver_ref=driver_ref,
 training_completion_ref=training_ref,input_ref=inp,result_ref=vr)
result.update(original_history=copy.deepcopy(c.ORIGINAL_HISTORY),dedicated_validator_CPU_runtime_checked=True,CUDA_initialized=False,source_manifest_ref=manifest_ref)
wait=dict(returncode=0,actual_direct_launcher_wait=True,all_owned_processes_and_groups_drained=True,operational_failure=None,
 driver_ref=driver_ref,input_ref=inp,completed_ref=complete,pid=123)
terminal=dict(actual_terminal_consumed=True,actual_exit_code=0,completed_ref=complete,actual_wait_ref=waitref,input_ref=inp,driver_ref=driver_ref)
def acceptance(args):return c.acceptance_values_v1(*args,source_ref,driver_ref,manifest_ref,training_ref,f[0]["checkpoint_candidates"])
aa=[bundle,env,result,wait,terminal];acceptance(aa);checks.append("synthetic_acceptance_own_wait0_drain_and_original1_UNKNOWN")
for name,index,key,value in [
 ("reject_fake_original_success",0,"original_history",dict(c.ORIGINAL_HISTORY,producer_cli_exit_code=0)),
 ("reject_fabricated_original_outer0",0,"original_history",dict(c.ORIGINAL_HISTORY,original_outer_74136_exit_code=0)),
 ("reject_missing_ROOT_failure_ack",0,"original_outerUNKNOWN_explicitly_acknowledged",False),
 ("reject_boolean_wait0",3,"returncode",False),
 ("reject_boolean_original_exit1",0,"original_history",dict(c.ORIGINAL_HISTORY,producer_cli_exit_code=True)),
 ("reject_boolean_own_outer0",4,"actual_exit_code",False),
 ("reject_boolean_optimizer_update0",2,"optimizer_updates_performed",False),
 ("reject_validator_wait1",3,"returncode",1),("reject_unknown_validator_wait",3,"actual_direct_launcher_wait",False),
 ("reject_incomplete_own_drain",3,"all_owned_processes_and_groups_drained",False),
 ("reject_unconsumed_own_outer",4,"actual_terminal_consumed",False),
 ("reject_wrong_own_outer_join",4,"actual_wait_ref",record("wrong.json")),
 ("reject_final_only_result",2,"checkpoints",result["checkpoints"][-1:]),
 ("reject_unverified_CPU_runtime",2,"dedicated_validator_CPU_runtime_checked",False)]:
 q=copy.deepcopy(aa);q[index][key]=value;reject(name,lambda q=q:acceptance(q))
# Exact original30cell barrier executed against synthetic metadata, never Torch/arrays/truth.
p=ROOT/"src/arggnn/american_c3_gnn_predict_isolated_v2.py";source=p.read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="seal_validation_matrix_v1")
VAL=tuple("american_c3_val_%03d"%i for i in range(5));seals={};refs=[]
for row in f[0]["checkpoint_candidates"]:
 for parent in VAL:
  refx=record("synthetic_seal_%s_%s.json"%(parent,row["epoch"]));refs.append(refx)
  seals[refx["path"]]=dict(status="sealed_C3_original_heldout_predictions",split="val",epoch=row["epoch"],parent_id=parent,
    training_completion_ref=training_ref,protocol_sha256=f[0]["protocol_sha256"],private_truth_opened=False,records=10000,
    purity_filter=False,checkpoint_ref=row["checkpoint_ref"],prediction_ref=record("synthetic_prediction.npz"),
    focal_registry_ref=record("synthetic_focals.npz"),focal_identity_sha256="a"*64)
class Writer:
 def save(self,name,value):return value
bound_calls=[]
ns=dict(require=core.require,read=lambda r:seals[r["path"]],CHECKPOINT_EPOCHS=core.EPOCHS,VAL=VAL,
 completed_training_v1=lambda r:f[0],bound=lambda r:bound_calls.append(r))
exec(compile(ast.Module(body=[node],type_ignores=[]),"<original30cellbarrier>","exec"),ns)
matrix=lambda rows:ns["seal_validation_matrix_v1"](rows,training_completion_ref=training_ref,writer=Writer())
yes("original_matrix_all30_pretruth",len(matrix(refs)["cells"])==30 and len(bound_calls)==60)
reject("original_matrix_rejects29",lambda:matrix(refs[:-1]))
reject("original_matrix_rejects_oneVAL",lambda:matrix([r for r in refs if "_000_" in r["path"]]))
reject("original_matrix_rejects_duplicate_cell",lambda:matrix(refs[:-1]+refs[:1]))
# AST bodies of numerical entrypoints and supervisor functions remain exact.
old=ROOT/"research/reviews/C3_full30Mb_evaluation_readiness_source_20261016_v1"
def functions(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}
a=functions(old/"worker_entry_v1.py");b=functions(HERE/"worker_entry_v1.py")
for name in ("predict_v1","matrix_v1","score_v1"):yes("unchanged_numerical_body_"+name,a[name]==b[name])
driver=(HERE/"root_driver_v1.py").read_text()
yes("supervisor_exact_AST_reuse",all(name in driver for name in ("_stat","_tree","_absent_group","_harvest_owned_roster","_signal_owned","launch")))
yes("current_native14_mode_and_policy",c.GUARD_REF["sha256"]=="60d010e393bda25f662381051b3fd3d4e4decfa034b10c3baad3c3f30a6a196c"
 and c.POLICY_REF["sha256"]=="72ed727feb1a824fa8f9c2b31ce374dca6728ca4e655e9562b4dedc84122f417"
 and 'storage["accounting_mode"]=="native14_stopped_retention_v3"' in (HERE/"acceptance_contract_v1.py").read_text())
for name in ("worker_entry_v1.py","root_driver_v1.py"):
 r=subprocess.run([sys.executable,"-B",str(HERE/name)],cwd=ROOT,capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1"),timeout=10)
 yes("inert_CLI_"+name,r.returncode==0 and "SOURCE_ONLY_INACTIVE" in r.stdout and r.stderr=="")
 r=subprocess.run([sys.executable,"-B",str(HERE/name),"--execute" if name.startswith("worker") else "--launch"],cwd=ROOT,
    capture_output=True,text=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1"),timeout=10)
 yes("missing_authority_blocks_"+name,r.returncode!=0 and ("Separate reviewed ROOT admission required" in r.stderr or "Exact separately admitted ROOT input reference" in r.stderr))
yes("fixture_never_imported_Torch_numpy_PyG",not any(k in sys.modules for k in ("torch","numpy","torch_geometric")))
out=dict(schema="C3_saved600_source_only_synthetic_fixture_v1",status="PASS",checks=len(checks),check_names=checks,
 actual_checkpoint_or_training_payload_read=False,real_Torch_import=False,real_guard_or_runtime_or_process_probe=False,
 actual_acceptance_admission_execution_or_accounting=False,synthetic_real_adapter_uses_fake_Torch_API=True)
(HERE/"source_fixture.v1.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps(dict(status="PASS",checks=len(checks),source_only=True,real_Torch_import=False)))
