"""Actual guarded native context, prerequisite readers and bounded output writer.
No project/scientific imports or payload reads occur until an authorized worker.
"""
from pathlib import Path
import ast,copy,hashlib,importlib,importlib.util,json,os,resource,stat,sys,types
from datetime import timedelta
from research.reviews.native_saved_MLM_contrastive8_phase_source_20261017_v1 import contract_v1 as c
ROOT=c.ROOT
from . import science_binding_v1 as saved_science
IPC_NAMESPACE_BY_PHASE_V2={'contrastive':'val000_c8_cl'}

def phase_namespace_v2(phase):
    c.require(phase in IPC_NAMESPACE_BY_PHASE_V2,"Only one explicit original downstream phase namespace")
    return IPC_NAMESPACE_BY_PHASE_V2[phase]

def validated_composer_timestamp_v1(state):
    outer=state.get("timestamp")
    c.require(type(outer) is dict and set(outer)=={"Timestamp"},"Exact installed Composer nested Timestamp serialization")
    value=outer["Timestamp"]
    counters={"iteration","epoch","batch","sample","token","epoch_in_iteration","token_in_iteration","batch_in_epoch","sample_in_epoch","token_in_epoch"}
    walls={"total_wct","iteration_wct","epoch_wct","batch_wct"}
    c.require(type(value) is dict and set(value)==counters|walls
        and all(type(value[k]) is int and value[k]>=0 for k in counters)
        and all(isinstance(value[k],timedelta) and value[k].total_seconds()>=0 for k in walls),
        "Exact actual native timestamp keys, nonnegative integer counters and timedeltas")
    return value

def checkpoint_metadata_bridge_v1(helper):
    # One metadata-load expression only. Original full model/optimizer counters,
    # epoch/update budget, file/alias selection and ambiguity checks stay intact.
    record=c.original.ref(c.original.CONTEXT);c.raw(record)
    source=c.local(record["path"]).read_text();tree=ast.parse(source)
    function=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_final_checkpoint"))
    matches=[n for n in ast.walk(function) if isinstance(n,ast.Assign) and len(n.targets)==1
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=="timestamp"]
    expected=ast.parse("timestamp=state.get("+chr(34)+"timestamp"+chr(34)+",{})").body[0]
    c.require(len(matches)==1 and ast.dump(matches[0],include_attributes=False)==ast.dump(expected,include_attributes=False),
        "Only original direct timestamp lookup is adapted for installed Composer schema")
    matches[0].value=ast.Call(func=ast.Name(id="_actual_native_nested_timestamp_v1",ctx=ast.Load()),args=[ast.Name(id="state",ctx=ast.Load())],keywords=[])
    ns=dict(helper._final_checkpoint.__globals__,_actual_native_nested_timestamp_v1=validated_composer_timestamp_v1)
    ast.fix_missing_locations(function)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(c.local(c.CONTEXT))+":nested_timestamp_metadata_bridge","exec"),ns)
    adapted=ns["_final_checkpoint"]
    def checked(*args,**kwargs):
        result=adapted(*args,**kwargs)
        return dict(result,actual_checkpoint_timestamp_schema="state.timestamp.Timestamp",
            checkpoint_metadata_parser_bridge_source_ref=c.ref(c.CONTEXT),original_checkpoint_model_optimizer_and_selection_checks_preserved=True)
    return checked

def _bind_private_scientific_helper_v1(module):
    native=module._native_v1
    def prepare(ctx):
        prepared,helper=native(ctx)
        helper._final_checkpoint=checkpoint_metadata_bridge_v1(helper)
        return prepared,helper
    module._native_v1=prepare
    return module

def source_stages_v1():
    return _bind_private_scientific_helper_v1(c.load(c.SCIENTIFIC_STAGES_REF,"_downstream_sealed_original_scientific_stages"))
def source_IO_v1():
    module=c.load(c.IO_SOURCE_REF,"_downstream_sealed_reviewed_IO_option");base=module._base_v1
    def stages():return _bind_private_scientific_helper_v1(base())
    module._base_v1=stages
    return module

def _bundle_v1(bundle,stage):
    c.require(type(bundle) is dict and set(bundle)=={"completion_ref","wait_ref","outer_terminal_ref"},"Exact genuine prerequisite triple")
    complete=c.read(bundle["completion_ref"]);wait=c.read(bundle["wait_ref"]);outer=c.read(bundle["outer_terminal_ref"])
    c.require(complete.get("parent_id")==c.PARENT and complete.get("query_ancestry_truth_opened") is False
        and wait.get("returncode")==0 and wait.get("actual_direct_worker_wait") is True
        and wait.get("actual_OS_wait_status")==0 and wait.get("all_owned_processes_and_groups_drained") is True
        and wait.get("complete_owned_roster_known") is True and wait.get("completed_ref")==bundle["completion_ref"]
        and wait.get("operational_failure") is None and wait.get("group_observations")
        and all(row.get("absent") is True for row in wait["group_observations"])
        and complete.get("child_input_ref",complete.get("active_input_ref"))==wait["input_ref"]
        and outer.get("actual_exit_code")==0 and outer.get("actual_terminal_consumed") is True
        and outer.get("actual_direct_worker_returncode")==0
        and outer.get("all_owned_processes_and_groups_drained") is True
        and outer.get("actual_wait_ref")==bundle["wait_ref"]
        and type(outer.get("actual_tool_session_id")) is int and outer["actual_tool_session_id"]>0,
        "Actual successful source-bound worker/OS wait, full current roster/group drain and consumed genuine outer0")
    if stage=="contrastive":
        c.require(wait.get("actual_direct_child_wait")is True and wait.get("child_completion_ref")==bundle["completion_ref"]
            and wait.get("process_group_ESRCH")is True
            and outer.get("schema")=="native_saved_MLM_downstream_actual_outer_terminal_v1"
            and outer.get("terminal_consumed")is True and outer.get("actual_tool_terminal_exit_code")==0
            and type(outer.get("tool_session_id"))is int and outer["tool_session_id"]==outer["actual_tool_session_id"]
            and outer.get("child_completion_ref")==bundle["completion_ref"],
            "Same actual contrastive wait/completion/consumed outer0/session in both original and saved qualification fields")
    c.require(stage=="contrastive"and wait["adapter_manifest_ref"]==c.ref(c.MANIFEST)
        and complete.get("actual_execution_source_manifest_ref")==c.ref(c.MANIFEST),
        "Genuine new contrastive producer manifest, no relabelled historical waits")
    return complete,wait,outer

def prerequisite_metadata_v1(bundle,stage):
    complete,wait,outer=_bundle_v1(bundle,stage)
    if stage in ("mlm","contrastive"):
        c.require(complete.get("stage")==stage,"Exact prerequisite training phase")
        result=dict(completion=complete,wait=wait,outer=outer,completion_ref=bundle["completion_ref"],
            wait_ref=bundle["wait_ref"],checkpoint_ref=complete["checkpoint_ref"])
        return result
    return complete,wait,outer

class PendingPrerequisites(ValueError):pass

def check_external_prerequisites_v1(q):
 c.require(q["phase"]=="contrastive","Only saved-MLM initialized contrastive; no later-stage route")
 if q.get("ROOT_saved_MLM_artifact_acceptance_ref")is None:
  raise PendingPrerequisites("Genuine ROOT saved-MLM artifact acceptance absent")
 integration=saved_science.integration_v1(q["storage_policy_selector_ref"])
 binding=dict(storage_policy_selector_ref=q["storage_policy_selector_ref"],saved_MLM_integration_source_ref=c.SAVED_INTEGRATION_REF,
  ROOT_saved_MLM_artifact_acceptance_ref=q["ROOT_saved_MLM_artifact_acceptance_ref"])
 integration.bind_stage_v1(binding,binding)
 bridge,eligible,checkpoint=integration.saved_acceptance_v1(binding,binding)
 return dict(checkpoint_ref=checkpoint,saved_artifact_only=True,saved_MLM_eligibility=eligible.record(),historical_MLM_outer_exit_code=1,
  ROOT_saved_MLM_artifact_acceptance_ref=q["ROOT_saved_MLM_artifact_acceptance_ref"])

def stable_reference_v1(path):
    path=c.local(path);before=path.lstat()
    c.require(stat.S_ISREG(before.st_mode) and before.st_nlink==1,"Own single-link regular scientific product")
    h=hashlib.sha256();fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
    def stamp(s):return (s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
    try:
        c.require(stamp(os.fstat(fd))==stamp(before),"Stable scientific output open identity")
        total=0
        while True:
            block=os.read(fd,8*2**20)
            if not block:break
            total+=len(block);h.update(block)
        c.require(total==before.st_size and stamp(os.fstat(fd))==stamp(before)==stamp(path.lstat()),"Scientific output stable during streaming authentication")
    finally:os.close(fd)
    return dict(path=str(path.relative_to(ROOT)),sha256=h.hexdigest())

class GuardedWriter:
    def __init__(self,s,q,g,storage):self.path=c.local(s["output_directory"]);self.s=s;self.q=q;self.g=g;self.storage=storage
    def boundary(self,event,*,required_write_bytes=0):
        c.active_amendment_v1();c.charged_census_v1(self.s,required_write_bytes=required_write_bytes)
        self.g.budget_check_v9(self.storage,boundary=True,required_write_bytes=required_write_bytes);c.active_amendment_v1()
    def save(self,name,value,kind=None):
        c.require(kind is None,"Only contrastive JSON metadata; later-stage arrays are outside this admission")
        c.require(type(name) is str and Path(name).name==name,"Flat own output only")
        path=self.path/name;c.require(not path.exists(),"Exclusive new scientific output, no replay")
        if kind is None:
            if name=="completed.json":
                c.require(value.get("child_input_ref",c.current_amendment_ref_v1())==c.current_amendment_ref_v1(),
                    "Actual completion belongs to this exact phase input")
                value=dict(value,child_input_ref=c.current_amendment_ref_v1())
            value=dict(value,actual_execution_source_manifest_ref=c.ref(c.MANIFEST),actual_ROOT_phase_input_ref=c.current_amendment_ref_v1())
            encoded=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
            c.require(len(encoded)<=8*2**20,"Bounded true scientific metadata")
            self.boundary("before_JSON_write",required_write_bytes=len(encoded)+8192)
            result=c.save(self.path,name,value)
        elif kind=="npz":
            import numpy as np
            c.require(all(not a.dtype.hasobject for a in value.values()),"Numerical truth-free output arrays")
            size=sum(a.nbytes for a in value.values());bound=size+(size+99)//100+2**20
            self.boundary("before_NPZ_write",required_write_bytes=bound+8192)
            with path.open("xb") as stream:np.savez_compressed(stream,**value)
            c.require(path.stat().st_size<=bound,"Actual serializer output bound")
            result=stable_reference_v1(path)
        else:raise ValueError("Only actual small JSON/NPZ writes through guarded writer. Author tensor/checkpoint writes stay original.")
        self.boundary("after_scientific_write")
        return result

class NativeContext:
    def __init__(self,input_ref,s,q,cfg,storage,ticket,native_guard):
     self.input_ref=input_ref;self.s=s;self.q=q;self.cfg=cfg;self.storage=storage
     self.ticket=ticket;self.native_guard=native_guard;self.g=c.guard_v1(q);self.writer=GuardedWriter(s,q,self.g,storage)
     self.base=source_stages_v1();self.prepared=None;self.verify_imports=None;self.helper=None;self.authenticated_inputs=None
     self.prep,self.pcfg,self.counts=c.frozen_science_metadata_v1()
     self.MLM=check_external_prerequisites_v1(q)
     c.require(s["phase"]=="contrastive","Only contrastive context");self.contrastive=None
     self.fit_view_descriptor=None
     self.embedding_view_descriptor=None
    def fast_guard(self,event):
        c.active_amendment_v1();self.native_guard.apply_and_validate_title()
        lease=self.native_guard.validate_lease()
        c.require(lease["physical_gpu"]==0 and lease["visible_gpu_uuid"]==c.GPU0_UUID
            and lease["borrowed_physical_inode"] is True,"Actual phase GPU0 borrowed lease/title")
    def prepare_native(self):
        if self.prepared is not None:return self.prepared
        self.fast_guard("before_native_imports")
        runtime=c.read(self.q["runtime_ref"])
        c.require(Path(sys.executable).resolve()==c.local(c.NATIVE_PYTHON_REF["path"])
            and Path(sys.prefix).resolve()==c.local(runtime["native_prefix"])
            and sys.dont_write_bytecode and sys.flags.no_user_site and os.environ.get("PYTHONPATH","")==""
            and os.environ.get("PYTHONNOUSERSITE")=="1","Exact isolated native interpreter and no user site")
        sys.path[:0]=[str(ROOT/"src"),str(ROOT/"research/reviews/American_C3_native_MLM_amendment_v12"),
            str(ROOT/"research/reviews/American_C3_native_MLM_full_fit_v8"),str(ROOT/"research/reviews/American_C3_native_MLM_full_source_v5")]
        from research.reviews.native_VAL000_GPU0_borrowed32_source_20261016_v2 import native_runtime_v1,gpu0_abstract_ipc_adapter_v3,native_child_v1
        inventory,verify=native_runtime_v1.validate_native_runtime_v1(self.q["runtime_ref"],runtime)
        c.require(inventory["status"]=="passed","Full actual native installed inventory before science")
        gpu0_abstract_ipc_adapter_v3.install_private_module_before_native_child_native32_v1()
        from abstract_unix_bootstrap_v1 import bootstrap_native_execution_v1
        ipc=bootstrap_native_execution_v1(phase_namespace_v2(self.s["phase"]),physical_gpu=0,gpu_uuid=c.GPU0_UUID)
        self.writer.save("operational_IPC_bootstrap.json",ipc)
        sys.path.insert(0,str(ROOT/c.AUTHOR_ROOT))
        import torch
        from omegaconf import OmegaConf
        c.require(torch.cuda.device_count()==1,"Exactly one actual visible physicalGPU0")
        device=torch.device("cuda",0);props=torch.cuda.get_device_properties(device)
        torch.cuda.set_per_process_memory_fraction(c.CAPS["max_cuda_reserved_bytes"]/props.total_memory,device)
        lease=self.native_guard.validate_lease()
        identity=native_child_v1._actual_context_uuid(c.original,lease,torch)
        verify();self.fast_guard("after_first_actual_CUDA_UUID_context")
        checkpoint_validation={}
        for stage,metadata in (("contrastive",self.contrastive),):
            if metadata is None:continue
            checkpoint=metadata["checkpoint_ref"]
            c.require(stable_reference_v1(ROOT/checkpoint["path"])==checkpoint,"Exact actual prerequisite checkpoint bytes")
            result=checkpoint_metadata_bridge_v1(native_child_v1)((ROOT/checkpoint["path"]).parent,
                epochs=5 if stage=="mlm" else 1,updates=227990 if stage=="mlm" else 2135,
                max_files=2,max_bytes=3*2**30,torch=torch,reference=stable_reference_v1)
            c.require(result["checkpoint_ref"]==checkpoint,"Actually opened saved model/optimizer at exact full epoch/update budget")
            checkpoint_validation[stage]=result
        self.writer.save("actual_saved_optimizer_prerequisite_verification.json",dict(schema="native_VAL000_actual_checkpoint_optimizer_validation_v1",
            stages=checkpoint_validation,runtime_inventory=inventory,actual_GPU_UUID_identity=identity,
            no_optimizer_job_or_accuracy_claim=True,all_prerequisite_workers_waited_and_drained=False,
            saved_MLM_historical_outer_exit_code_preserved=1,saved_MLM_payload_not_hashed_or_loaded_before_original_inline_initialization=True))
        self.confirm_authenticated_inputs()
        self.helper=native_child_v1;self.verify_imports=verify
        self.prepared=dict(torch=torch,OmegaConf=OmegaConf,physical_gpu=0,runtime_inventory_passed=True,
            one_actual_visible_CUDA_UUID=True,actual_device_identity=identity)
        os.chdir(self.writer.path)
        return self.prepared
    def authenticate_inputs(self,preparation_ref,counts_ref):
        c.require(preparation_ref==c.PREP_REF and counts_ref==c.COUNTS_REF,"Only exact frozen native input metadata")
        auth=c.load({"path":"research/reviews/American_C3_native_MLM_amendment_v12/corpus_auth_v2.py","sha256":"2f02c99d49502493abab82e5b6c08c8abc782538d7d821a71cd5116f62b0cd98"},"_downstream_exact_streaming_corpus_auth")
        views=["contrastive"]
        verified={}
        for view in views:
            for split in ("train","val"):
                key=view+"/"+split;item=self.counts["datasets"][key];record=item["dataset_ref"]
                self.fast_guard("before_actual_raw_auth")
                verified[key]=auth.authenticate_file(record,c.local(c.PREP_OUTPUT)/view/split/("raw_"+split+"_sequences_and_vocab.pkl"),item["raw_pickle_bytes"])
                if view=="contrastive":
                    label=self.counts["reference_label_refs"][split]
                    verified[key+"/labels"]=auth.authenticate_file(label,c.local(c.PREP_OUTPUT)/view/split/"labels.pkl",5*item["rows"]+16777216)
        verified["vocab"]=auth.authenticate_file(self.prep["vocab_ref"],c.local(c.PREP_OUTPUT)/"vocab.pkl",64*self.prep["vocabulary_nodes"]+16777216)
        self.authenticated_inputs=verified
        self.confirm_authenticated_inputs()
        self.writer.save("native_consumed_input_authentication.json",dict(schema="native_VAL000_downstream_quiescent_input_auth_v1",inputs=verified,
            exact_existing_bytes_and_identity_verified=True,query_ancestry_truth_opened=False,corpus_decoded_or_copied=False))
    def confirm_authenticated_inputs(self):
        if self.authenticated_inputs is None:return
        for row in self.authenticated_inputs.values():
            actual=c.local(row["ref"]["path"]).lstat()
            identity=[actual.st_dev,actual.st_ino,actual.st_mode,actual.st_size,actual.st_mtime_ns,actual.st_ctime_ns,actual.st_nlink]
            c.require(identity==row["identity"],"Authenticated exact corpus identity remains unchanged before native loading")
        for key in self.prep["native_dataset_refs"]:
            view,split=key.split("/")
            c.require(not (c.local(c.PREP_OUTPUT)/view/split/("tokenized_"+split+"_sequences_and_vocab.pkl")).exists(),"No shadow tokenized corpus before native loading")
    def preparation_callback(self,spec):
        c.require(spec["refs"]["preparation"]==c.PREP_REF and spec["refs"]["preparation_config"]==c.PREP_CONFIG_REF,"Exact VAL000 preparation callback refs")
        c.frozen_science_metadata_v1()
        return self.prep,self.pcfg,c.local(c.PREP_OUTPUT)
    def counts_callback(self,spec,prep):
        c.require(spec["refs"]["counts"]==c.COUNTS_REF and prep==self.prep,"Exact actual counts callback")
        return self.counts
    def fit_callback(self,record,spec):
        raise ValueError("Embedding/retrieval fit dispatch is outside this contrastive-only admission")
    def metadata_callback(self,record):
        raise ValueError("Later-stage product metadata dispatch is outside this contrastive-only admission")
    def context_and_request(self):
        phase="contrastive"
        request=dict(schema="native_VAL000_downstream_scientific_phase_v1",execution_authorized=True,phase=phase,parent_id=c.PARENT,
            separate_future_storage_admission_ref=self.storage["reservation"]["root_admission"],original_MLM_7GiB_scope_extended=False,
            ROOT_context_source_ref=c.ref(c.CONTEXT),operational_IO_source_ref=c.IO_SOURCE_REF,operational_IO_policy_ref=c.IO_POLICY_REF,
            keep1_rotation_of_new_scope_authorized=True,original_author_recipe_claimed_exact=False)
        request.update(storage_policy_selector_ref=self.q["storage_policy_selector_ref"],saved_MLM_integration_source_ref=c.SAVED_INTEGRATION_REF,
            scientific_selector_binding_bridge_ref=c.ref(c.SCIENCE_BINDING),
            ROOT_saved_MLM_artifact_acceptance_ref=self.q["ROOT_saved_MLM_artifact_acceptance_ref"],
            saved_MLM_qualification_ref=self.q.get("saved_MLM_qualification_ref"))
        def validate(actual):c.require(actual==request,"Exact ROOT-authorized actual scientific request mapping");c.active_amendment_v1()
        ctx=dict(root_authorizer="root_agent",admitted=True,validate_phase_authority_v1=validate,fast_guard=self.fast_guard,
            reference=stable_reference_v1,writer=self.writer,output_directory=str(self.writer.path),physical_gpu=0,
            context_source_ref=c.ref(c.CONTEXT),operational_IO_source_ref=c.IO_SOURCE_REF,operational_IO_policy_ref=c.IO_POLICY_REF,
            input_ref=self.input_ref,MLM_prerequisite_metadata=self.MLM,prepare_native_runtime_v1=self.prepare_native,
            verify_native_imports_v1=lambda:self.verify_imports(),max_checkpoint_files=2,max_checkpoint_bytes=3*2**30,
            authenticate_native_inputs_once_v1=self.authenticate_inputs,prepared_metadata=self.prep,boundary_guard=self.writer.boundary,
            validated_VAL000_preparation_v1=self.preparation_callback,validated_VAL000_counts_v1=self.counts_callback,
            validated_VAL000_fit_completion_v1=self.fit_callback,validated_VAL000_metadata=self.metadata_callback)
        ctx.update(storage_policy_selector_ref=self.q["storage_policy_selector_ref"],saved_MLM_integration_source_ref=c.SAVED_INTEGRATION_REF,
            scientific_selector_binding_bridge_ref=c.ref(c.SCIENCE_BINDING),
            ROOT_saved_MLM_artifact_acceptance_ref=self.q["ROOT_saved_MLM_artifact_acceptance_ref"],
            saved_MLM_qualification_spec=self.q.get("saved_MLM_qualification_spec"))
        if self.contrastive is not None:ctx["contrastive_prerequisite_metadata"]=self.contrastive
        return ctx,request
    def run(self):
     c.require(self.s["phase"]=="contrastive","Only original saved-MLM contrastive science")
     self.fast_guard("before_own_native_output")
     c.require(not self.writer.path.exists(),"Genuinely fresh native phase output")
     self.writer.boundary("before_native_output_mkdir",required_write_bytes=8192);self.writer.path.mkdir()
     limit=3*2**30
     resource.setrlimit(resource.RLIMIT_FSIZE,(limit,limit))
     ctx,request=self.context_and_request()
     return saved_science.entry_v1(self.q["storage_policy_selector_ref"]).contrastive_keep1_saved_MLM_v1(ctx,request)



