"""Synthetic source-contract checks only. No real pipeline/data/process imports.
Execute explicitly with --run. All prerequisite receipts and writers are fake.
"""
from pathlib import Path
import ast,copy,hashlib,json,sys,types
from datetime import timedelta
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/native_VAL000_downstream_execution_source_20261016_v1"

def run():
    text=(ROOT/HERE/"worker_context_v1.py").read_text();tree=ast.parse(text)
    science=(ROOT/"research/reviews/native_VAL000_downstream_scientific_stages_source_20261016_v1/scientific_stages_v1.py").read_text();stree=ast.parse(science)
    refs=ast.literal_eval(next(n.value for n in stree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="REFS"))
    def require(ok,message):
        if not ok:raise ValueError(message)
    sns=dict(require=require,REFS=refs)
    funcs=[n for n in stree.body if isinstance(n,ast.FunctionDef) and n.name in ("full_mlm_metadata_prerequisite_v1","full_contrastive_metadata_prerequisite_v1","canonical_actual_wait_metadata_v1")]
    exec(compile(ast.Module(body=funcs,type_ignores=[]),"<sealed-metadata-prerequisite-functions-only>","exec"),sns)
    basemodule=types.SimpleNamespace(**{n.name:sns[n.name] for n in funcs})
    def r(name):return dict(path="SYNTHETIC_ONLY/"+name,sha256=hashlib.sha256(name.encode()).hexdigest())
    data={}
    def put(name,value):
        record=r(name);data[(record["path"],record["sha256"])]=value;return record
    def read(record):return copy.deepcopy(data[(record["path"],record["sha256"])])
    manifest=r("actual_source_manifest");policy=r("4096_policy");oldmanifest=r("native_MLM_manifest");oldchild=r("unchanged_scientific_child")
    prep=read0=None
    actualprep=json.loads((ROOT/refs["preparation"]["path"]).read_text())
    data[(refs["preparation"]["path"],refs["preparation"]["sha256"])]=actualprep
    c=types.SimpleNamespace(require=require,read=read,raw=lambda record:None,OWNED_ROSTER_POLICY_REF=policy,
        MANIFEST="manifest",ref=lambda p:manifest,PREP_REF=refs["preparation"],PARENT="american_c3_val_000",
        original=types.SimpleNamespace(ref=lambda p:oldchild,CONTEXT="original_scientific_child"))
    put("native_MLM_manifest",dict(owned_roster_policy_ref=policy,original_scientific_child_ref=oldchild))
    auth=put("native_MLM_actual_ROOT_authority",dict(root_authorizer="root_agent",independent_source_review_passed=True,
        adapter_manifest_ref=oldmanifest,independent_source_review_ref=r("independent_review")))
    ns=dict(c=c,source_stages_v1=lambda:basemodule)
    definitions=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in
        ("_bundle_v1","prerequisite_metadata_v1","PendingPrerequisites","check_external_prerequisites_v1")]
    exec(compile(ast.Module(body=definitions,type_ignores=[]),"<actual-prerequisite-contract-with-fake-receipts>","exec"),ns)
    def bundle(name,completion,MLM=False):
        inp=r(name+"_input");completion=dict(completion,parent_id="american_c3_val_000",query_ancestry_truth_opened=False,child_input_ref=inp)
        cr=put(name+"_completion",completion)
        wait=dict(returncode=0,actual_direct_worker_wait=True,actual_OS_wait_status=0,all_owned_processes_and_groups_drained=True,
            complete_owned_roster_known=True,completed_ref=cr,operational_failure=None,group_observations=[dict(absent=True)],
            input_ref=inp,adapter_manifest_ref=oldmanifest if MLM else manifest)
        if MLM:wait.update(owned_roster_policy_ref=policy,actual_execution_amendment_ref=auth)
        wr=put(name+"_wait",wait)
        outer=dict(actual_exit_code=0,actual_terminal_consumed=True,actual_direct_worker_returncode=0,
            all_owned_processes_and_groups_drained=True,actual_wait_ref=wr,actual_tool_session_id=1234)
        out=put(name+"_outer",outer)
        return dict(completion_ref=cr,wait_ref=wr,outer_terminal_ref=out)
    mlm_checkpoint=r("MLM_checkpoint");contrastive_checkpoint=r("contrastive_checkpoint")
    mlm=bundle("MLM",dict(stage="mlm",completed_epochs=5,actual_optimizer_updates=227990,observed_counters_verified=True,
        trainer_optimizer_counters=dict(unique_step_values=[227990]),checkpoint_optimizer_counters=dict(unique_step_values=[227990]),
        initializer_checkpoint_ref=None,checkpoint_ref=mlm_checkpoint,preparation_ref=refs["preparation"],author_source_manifest_ref=refs["author_manifest"]),True)
    contrastive=bundle("contrastive",dict(stage="contrastive",completed_epochs=1,actual_optimizer_updates=2135,
        checkpoint_optimizer_counters=dict(unique_step_values=[2135]),initializer_checkpoint_ref=mlm_checkpoint,
        checkpoint_ref=contrastive_checkpoint,author_source_manifest_ref=refs["author_manifest"]))
    embeddings={}
    for key in ("contrastive/train","contrastive/val","query_paths/train","query_paths/val"):
        view,split=key.split("/")
        embeddings[key]=bundle(key.replace("/","_"),dict(stage="embedding",view=view,split=split,checkpoint_ref=contrastive_checkpoint,
            raw_dataset_ref=actualprep["native_dataset_refs"][key],row_order_ref=actualprep["full_path_id_order_refs"][key]))
    RAM=bundle("RAM",dict(status="completed_native_RAM_retrieval_and_truth_free_seal",full_reference_rows=9721236,full_query_rows=10000,
        checkpoint_ref=contrastive_checkpoint,native_top1_arithmetic_or_ties_modified=False))
    q=dict(phase="finalize",prerequisites=dict(MLM=mlm,contrastive=contrastive,embeddings=embeddings,RAM_retrieve_and_seal=RAM))
    ns["check_external_prerequisites_v1"](q)
    negatives=[]
    for target,changes in [(mlm["wait_ref"],dict(returncode=-15)),(mlm["wait_ref"],dict(actual_OS_wait_status=None)),
        (mlm["wait_ref"],dict(complete_owned_roster_known=False)),(mlm["wait_ref"],dict(group_observations=[dict(absent=None)])),
        (mlm["outer_terminal_ref"],dict(actual_exit_code=1)),(mlm["outer_terminal_ref"],dict(actual_terminal_consumed=False)),
        (mlm["completion_ref"],dict(trainer_optimizer_counters=dict(unique_step_values=[2]))),
        (contrastive["completion_ref"],dict(checkpoint_optimizer_counters=dict(unique_step_values=[2]))),
        (RAM["completion_ref"],dict(full_reference_rows=100)),
        (embeddings["contrastive/train"]["completion_ref"],dict(checkpoint_ref=r("wrong_checkpoint")))]:
        k=(target["path"],target["sha256"]);original=copy.deepcopy(data[k]);data[k].update(changes)
        rejected=False
        try:ns["check_external_prerequisites_v1"](q)
        except ValueError:rejected=True
        finally:data[k]=original
        assert rejected;negatives.append(changes)
    pending=False
    try:ns["check_external_prerequisites_v1"](dict(phase="contrastive",prerequisites=dict(MLM=None)))
    except ns["PendingPrerequisites"]:pending=True
    assert pending
    # Exact actual GuardedWriter JSON branch with fake path/guard/save only.
    class FakePath:
        def __truediv__(self,name):return self
        def exists(self):return False
    events=[];saved={}
    def save0(path,name,value):
        encoded=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
        assert ("guard",len(encoded)+8192) in events
        saved[name]=encoded;events.append(("saved",name));return r(name)
    wc=types.SimpleNamespace(require=require,local=lambda path:FakePath(),active_amendment_v1=lambda:events.append(("active",)),
        charged_census_v1=lambda s,required_write_bytes:events.append(("census",required_write_bytes)),
        save=save0,ref=lambda p:manifest,MANIFEST="manifest",current_amendment_ref_v1=lambda:r("current_input"))
    writerclass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="GuardedWriter")
    wns=dict(c=wc,Path=Path,json=json)
    exec(compile(ast.Module(body=[writerclass],type_ignores=[]),"<actual-guarded-JSON-writer-with-fakes>","exec"),wns)
    guard=types.SimpleNamespace(budget_check_v9=lambda storage,**kw:events.append(("guard",kw.get("required_write_bytes",0))))
    writer=wns["GuardedWriter"](dict(output_directory="fake"),{},guard,{})
    writer.save("receipt.json",dict(status="SYNTHETIC_ONLY_NOT_REAL_COMPLETION"))
    assert "receipt.json" in saved
    writer.save("completed.json",dict(status="SYNTHETIC_ONLY_NOT_REAL_COMPLETION",actual_input_ref=r("current_input")))
    assert json.loads(saved["completed.json"])["child_input_ref"]==r("current_input")
    rejected=False
    try:writer.save("completed.json",dict(child_input_ref=r("wrong_input")))
    except ValueError:rejected=True
    assert rejected
    # Genuine compact progress metadata schema only. All model/state values and
    # torch/file access below are fake, not checkpoint or scientific payloads.
    native_path=ROOT/"research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/native_child_v1.py"
    native_text=native_path.read_text();ntree=ast.parse(native_text)
    observed_path=ROOT/"research/reviews/native_VAL000_GPU0_borrowed32_ROOT_actual_20261016_v2/worker_aux/ROOT_owned_roster_v4_saved_optimizer_progress.root_v1.json"
    observed=json.loads(observed_path.read_text());timestamp=copy.deepcopy(observed["actual_Composer_timestamp_raw"])
    assert set(timestamp)=={"Timestamp"} and timestamp["Timestamp"]["batch"]==5500
    for key,value in timestamp["Timestamp"].items():
        if key.endswith("_wct"):
            hours,minutes,seconds=value.split(":")
            timestamp["Timestamp"][key]=timedelta(hours=int(hours),minutes=int(minutes),seconds=float(seconds))
    class FakeCheckpoint:
        def lstat(self):return types.SimpleNamespace(st_size=100)
    fakepath=FakeCheckpoint()
    nns=dict(Path=Path,_checkpoint_entries=lambda directory:([fakepath],None))
    funcs=[n for n in ntree.body if isinstance(n,ast.FunctionDef) and n.name in ("_int_time","optimizer_counter_evidence","_final_checkpoint")]
    exec(compile(ast.Module(body=funcs,type_ignores=[]),"<original-final-checkpoint-with-entirely-fake-payload-file-and-torch>","exec"),nns)
    native_ref=dict(path=str(native_path.relative_to(ROOT)),sha256=hashlib.sha256(native_text.encode()).hexdigest())
    bc=types.SimpleNamespace(require=require,original=types.SimpleNamespace(CONTEXT="native",ref=lambda p:native_ref),
        raw=lambda record:None,local=lambda p:native_path if p==native_ref["path"] else ROOT/HERE/"worker_context_v1.py",
        CONTEXT="current_context",ref=lambda p:r("current_context"))
    bns=dict(c=bc,ast=ast,copy=copy,timedelta=timedelta)
    defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ("validated_composer_timestamp_v1","checkpoint_metadata_bridge_v1")]
    exec(compile(ast.Module(body=defs,type_ignores=[]),"<actual-nested-timestamp-bridge-with-fakes-only>","exec"),bns)
    assert bns["validated_composer_timestamp_v1"](dict(timestamp=timestamp))["batch"]==5500
    fake_state=dict(timestamp=timestamp,model=dict(fake_parameter=0),optimizers=dict(DecoupledAdamW=dict(state={0:dict(step=5500,exp_avg=0,exp_avg_sq=0)})))
    fake_torch=types.SimpleNamespace(load=lambda *a,**k:dict(state=copy.deepcopy(fake_state)))
    args=dict(epochs=5,updates=227990,max_files=2,max_bytes=1000,torch=fake_torch,reference=lambda p:r("fake_checkpoint"))
    adapted=bns["checkpoint_metadata_bridge_v1"](types.SimpleNamespace(_final_checkpoint=nns["_final_checkpoint"]))
    rejected=False
    try:adapted(None,**args)
    except ValueError:rejected=True
    assert rejected
    fake_state["timestamp"]["Timestamp"].update(epoch=5,batch=227990,batch_in_epoch=0)
    fake_state["optimizers"]["DecoupledAdamW"]["state"][0]["step"]=227990
    result=adapted(None,**args)
    assert result["actual_optimizer_updates"]==227990 and result["Composer_timestamp"]==dict(epoch=5,batch=227990,batch_in_epoch=0)
    parser_negatives=0
    for bad in [fake_state["timestamp"]["Timestamp"],dict(Timestamp=dict(fake_state["timestamp"]["Timestamp"],epoch=True)),
        dict(Timestamp={k:v for k,v in fake_state["timestamp"]["Timestamp"].items() if k!="total_wct"})]:
        rejected=False
        try:bns["validated_composer_timestamp_v1"](dict(timestamp=bad))
        except ValueError:rejected=True
        assert rejected;parser_negatives+=1
    return dict(schema="native_VAL000_concrete_execution_synthetic_contract_fixture_v1",status="PASS_SOFTWARE_ONLY",
        fake_complete_prerequisite_chain_accepted=True,invalid_or_partial_prerequisite_cases_rejected=len(negatives),
        missing_genuine_MLM_returns_pending=True,actual_JSON_writer_prechecks_exact_bytes_before_fake_save=True,actual_completion_input_bridge_and_wrong_input_rejection=True,
        genuine_compact_metadata_nested_Timestamp_schema_covered=True,fake_final_checkpoint_full_step_and_epoch_checks_preserved=True,
        genuine5500_metadata_rejected_as227990_completion=True,invalid_nested_Timestamp_schema_cases_rejected=parser_negatives,
        real_payloads_models_truth_processes_devices_or_project_imports=False,
        actual_training_numerical_equivalence_or_memory_fit_claimed=False)

if __name__=="__main__":
    if sys.argv[1:]!=["--run"]:raise SystemExit("Explicit --run required, fake contract checks only")
    print(json.dumps(run(),sort_keys=True,indent=2))
