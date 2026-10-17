"""Actual guarded native context, prerequisite readers and bounded output writer.
No project/scientific imports or payload reads occur until an authorized worker.
"""
from pathlib import Path
import ast,copy,hashlib,importlib,importlib.util,json,os,resource,stat,sys,types
from datetime import timedelta
from research.reviews.native_ARGformer_postdrain_pool_source_20261017_v2 import downstream_contract_v1 as c
ROOT=c.ROOT
IPC_NAMESPACE_BY_PHASE_V2={'contrastive': 'val000_ds2_cl', 'embedding_contrastive_train': 'val000_ds2_ect', 'embedding_contrastive_val': 'val000_ds2_ecv', 'embedding_query_paths_train': 'val000_ds2_eqt', 'embedding_query_paths_val': 'val000_ds2_eqv', 'RAM_retrieve_and_seal': 'val000_ds2_ret', 'finalize': 'val000_ds2_fin'}

def phase_namespace_v2(phase):
    c.require(phase in IPC_NAMESPACE_BY_PHASE_V2,"Only one explicit original downstream phase namespace")
    return "v"+c.PARENT[-3:]+"_ds3_"+{"contrastive":"cl","embedding_contrastive_train":"ect","embedding_contrastive_val":"ecv","embedding_query_paths_train":"eqt","embedding_query_paths_val":"eqv","RAM_retrieve_and_seal":"ret","finalize":"fin"}[phase]

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
    record={"path":"research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/native_child_v1.py","sha256":"f2770222f02979b6064a7f1ffea80953cd234114c96ddde0845372479292379e"};c.raw(record)
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
    return _bind_private_scientific_helper_v1(contextual_module_v1(c.SCIENTIFIC_STAGES_REF,"_downstream_sealed_original_scientific_stages"))
def source_IO_v1():
    module=contextual_module_v1(c.IO_SOURCE_REF,"_downstream_sealed_reviewed_IO_option");base=module._base_v1
    def stages():return source_stages_v1()
    module._base_v1=stages
    return module

def _bundle_v1(bundle,stage):
    c.require(type(bundle) is dict and set(bundle)=={"completion_ref","wait_ref","outer_terminal_ref"},"Exact genuine prerequisite triple")
    outer=c.read(bundle["outer_terminal_ref"])
    if outer.get("actual_pool_logical_outer_OS_wait_consumed") is True:return pool_prerequisite_bundle_v1(bundle,stage)
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
    if stage=="mlm":
        legacy_manifest={"path":"research/reviews/native_VAL000_full_state_resume_source_20261016_v12/source_manifest.v1.json","sha256":"734c8f55fc6a06c20fba9b63b9d72bf733ee9e6b49261f8da26c8bb6b27ae664"}
        c.require(c.PARENT=="american_c3_val_000" and wait.get("adapter_manifest_ref")==legacy_manifest
            and complete.get("adapter_manifest_ref")==legacy_manifest
            and complete.get("child_input_ref")==wait["input_ref"]
            and complete.get("actual_execution_amendment_ref")==wait.get("actual_execution_amendment_ref"),
            "Only authentic current VAL000 v12 full-state lineage in legacy MLM branch; no fresh-pool manifest fallback")
        c.require(wait.get("owned_roster_policy_ref")==c.OWNED_ROSTER_POLICY_REF,"Reviewed4096 native MLM lifecycle lineage, no truncated64 completion")
        actual_authority=c.read(wait["actual_execution_amendment_ref"])
        c.require(actual_authority["root_authorizer"]=="root_agent"
            and actual_authority["independent_source_review_passed"] is True
            and actual_authority["adapter_manifest_ref"]==wait["adapter_manifest_ref"],"Actual reviewed native MLM execution amendment")
        c.require(complete.get("actual_original_full_state_resume") is True
            and complete.get("original_fresh_run_initializer_checkpoint_ref") is None
            and complete.get("resume_checkpoint_ref")==actual_authority["resume_checkpoint_ref"],
            "Genuine original v12 full-state resume retained, never fresh-optimizer/no-resume relabelling")
        c.raw(actual_authority["independent_source_review_ref"])
        manifest=c.read(wait["adapter_manifest_ref"])
        c.require(manifest["owned_roster_policy_ref"]==c.OWNED_ROSTER_POLICY_REF
            and manifest["original_scientific_child_ref"]=={"path":"research/reviews/native_VAL000_GPU0_borrowed32_source_20261016_v2/native_child_v1.py","sha256":"f2770222f02979b6064a7f1ffea80953cd234114c96ddde0845372479292379e"},"Unchanged full native MLM scientific child and reviewed roster policy")
    else:
        c.require(wait["adapter_manifest_ref"]==c.ref(c.MANIFEST),"Actual completed stage used this reviewed downstream source")
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
    prerequisites=q["prerequisites"]
    needed=["MLM"]+([] if q["phase"]=="contrastive" else ["contrastive"])
    if q["phase"] in ("RAM_retrieve_and_seal","finalize"):needed.append("embeddings")
    if q["phase"]=="finalize":needed.append("RAM_retrieve_and_seal")
    missing=[name for name in needed if prerequisites.get(name) is None]
    if missing:raise PendingPrerequisites("Genuine external prerequisite refs absent: "+",".join(missing))
    base=source_stages_v1();mlm=prerequisite_metadata_v1(prerequisites["MLM"],"mlm")
    base.full_mlm_metadata_prerequisite_v1(**mlm)
    if q["phase"]!="contrastive":
        contrastive=prerequisite_metadata_v1(prerequisites["contrastive"],"contrastive")
        base.full_contrastive_metadata_prerequisite_v1(initializer_checkpoint_ref=mlm["checkpoint_ref"],**contrastive)
    if q["phase"] in ("RAM_retrieve_and_seal","finalize"):
        c.require(set(prerequisites["embeddings"])=={"contrastive/train","contrastive/val","query_paths/train","query_paths/val"},"All four genuinely completed embedding prerequisites")
        for key,bundle in prerequisites["embeddings"].items():
            child,wait,outer=prerequisite_metadata_v1(bundle,"embedding")
            view,split=key.split("/")
            c.require(child["stage"]=="embedding" and child["view"]==view and child["split"]==split
                and child["checkpoint_ref"]==contrastive["checkpoint_ref"]
                and child["raw_dataset_ref"]==c.read(c.PREP_REF)["native_dataset_refs"][key]
                and child["row_order_ref"]==c.read(c.PREP_REF)["full_path_id_order_refs"][key],"Actual same-checkpoint full native embedding lineage")
    if q["phase"]=="finalize":
        final,wait,outer=prerequisite_metadata_v1(prerequisites["RAM_retrieve_and_seal"],"RAM_retrieve_and_seal")
        c.require(final["status"]=="completed_native_RAM_retrieval_and_truth_free_seal"
            and final["full_reference_rows"]==c.EXPECTED_CONFIG["scientific_protocol"]["retrieval"]["reference_rows"] and final["full_query_rows"]==10000
            and final["checkpoint_ref"]==contrastive["checkpoint_ref"]
            and final["native_top1_arithmetic_or_ties_modified"] is False,"Genuine completed original native top1/seal, no invented score")
    return mlm

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
        check_external_prerequisites_v1(q)
        self.MLM=prerequisite_metadata_v1(q["prerequisites"]["MLM"],"mlm")
        self.contrastive=None if s["phase"]=="contrastive" else prerequisite_metadata_v1(q["prerequisites"]["contrastive"],"contrastive")
        self.fit_view_descriptor=dict(schema="native_VAL000_authenticated_fit_virtual_metadata_v1",MLM=q["prerequisites"]["MLM"],contrastive=q["prerequisites"].get("contrastive"))
        self.embedding_view_descriptor=dict(schema="native_VAL000_authenticated_four_embeddings_virtual_metadata_v1",items=q["prerequisites"].get("embeddings"))
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
        from research.reviews.native_ARGformer_postdrain_pool_source_20261017_v2 import native_runtime_v1,gpu0_abstract_ipc_adapter_v3
        from research.reviews.native_VAL000_GPU0_borrowed32_source_20261016_v2 import native_child_v1
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
        for stage,metadata in (("mlm",self.MLM),("contrastive",self.contrastive)):
            if metadata is None:continue
            checkpoint=metadata["checkpoint_ref"]
            c.require(stable_reference_v1(ROOT/checkpoint["path"])==checkpoint,"Exact actual prerequisite checkpoint bytes")
            result=checkpoint_metadata_bridge_v1(native_child_v1)((ROOT/checkpoint["path"]).parent,
                epochs=5 if stage=="mlm" else 1,updates=c.read(c.COUNTS_REF)["MLM_optimizer_updates"] if stage=="mlm" else c.EXPECTED_CONFIG["scientific_protocol"]["contrastive"]["updates"],
                max_files=2,max_bytes=3*2**30,torch=torch,reference=stable_reference_v1)
            c.require(result["checkpoint_ref"]==checkpoint,"Actually opened saved model/optimizer at exact full epoch/update budget")
            checkpoint_validation[stage]=result
        self.writer.save("actual_saved_optimizer_prerequisite_verification.json",dict(schema="native_VAL000_actual_checkpoint_optimizer_validation_v1",
            stages=checkpoint_validation,runtime_inventory=inventory,actual_GPU_UUID_identity=identity,
            no_optimizer_job_or_accuracy_claim=True,all_prerequisite_workers_waited_and_drained=True))
        self.confirm_authenticated_inputs()
        self.helper=native_child_v1;self.verify_imports=verify
        self.prepared=dict(torch=torch,OmegaConf=OmegaConf,physical_gpu=0,runtime_inventory_passed=True,
            one_actual_visible_CUDA_UUID=True,actual_device_identity=identity)
        os.chdir(self.writer.path)
        return self.prepared
    def authenticate_inputs(self,preparation_ref,counts_ref):
        c.require(preparation_ref==c.PREP_REF and counts_ref==c.COUNTS_REF,"Only exact frozen native input metadata")
        auth=c.load({"path":"research/reviews/American_C3_native_MLM_amendment_v12/corpus_auth_v2.py","sha256":"2f02c99d49502493abab82e5b6c08c8abc782538d7d821a71cd5116f62b0cd98"},"_downstream_exact_streaming_corpus_auth")
        views=["contrastive"] if self.s["phase"]=="contrastive" else ([self.q["view"]] if self.s["phase"].startswith("embedding_") else [])
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
        c.require(record==self.fit_view_descriptor and self.contrastive is not None,"Explicit genuine fit virtual metadata, never a fake file/SHA receipt")
        self.base.full_mlm_metadata_prerequisite_v1(**self.MLM)
        self.base.full_contrastive_metadata_prerequisite_v1(initializer_checkpoint_ref=self.MLM["checkpoint_ref"],**self.contrastive)
        return dict(schema="native_VAL000_fit_metadata_compatibility_view_v1",phases=dict(mlm=dict(checkpoint_ref=self.MLM["checkpoint_ref"]),contrastive=dict(checkpoint_ref=self.contrastive["checkpoint_ref"])))
    def metadata_callback(self,record):
        if record==self.embedding_view_descriptor:
            check_external_prerequisites_v1(self.q);outputs={}
            for key,bundle in self.q["prerequisites"]["embeddings"].items():
                child,wait,outer=prerequisite_metadata_v1(bundle,"embedding")
                outputs[key]=dict(child_completion_ref=bundle["completion_ref"],actual_wait_ref=bundle["wait_ref"],embedding_ref=child["embedding_ref"])
            return dict(status="completed_C3_native_four_single_split_embeddings",parent_id=c.PARENT,
                fit_completion_ref=self.fit_view_descriptor,preparation_ref=c.PREP_REF,counts_ref=c.COUNTS_REF,
                checkpoint_ref=self.contrastive["checkpoint_ref"],outputs=outputs,
                actual_four_waits_and_consumed_outer_terminals_verified=True,operational_virtual_view_not_a_forged_file=True)
        for bundle in self.q["prerequisites"].get("embeddings",{}).values():
            child,wait,outer=prerequisite_metadata_v1(bundle,"embedding")
            if record==bundle["completion_ref"]:return child
            if record==bundle["wait_ref"]:return self.base.canonical_actual_wait_metadata_v1(wait,bundle["wait_ref"])
        raise ValueError("Unknown metadata callback reference, no private truth or arbitrary payload access")
    def context_and_request(self):
        phase="embedding" if self.s["phase"].startswith("embedding_") else self.s["phase"]
        request=dict(schema="native_VAL000_downstream_scientific_phase_v1",execution_authorized=True,phase=phase,parent_id=c.PARENT,
            separate_future_storage_admission_ref=self.storage["reservation"]["root_admission"],original_MLM_7GiB_scope_extended=False,
            ROOT_context_source_ref=c.ref(c.CONTEXT),operational_IO_source_ref=c.IO_SOURCE_REF,operational_IO_policy_ref=c.IO_POLICY_REF,
            keep1_rotation_of_new_scope_authorized=True,original_author_recipe_claimed_exact=False)
        if phase=="embedding":request.update(view=self.q["view"],split=self.q["split"])
        def validate(actual):c.require(actual==request,"Exact ROOT-authorized actual scientific request mapping");c.active_amendment_v1()
        ctx=dict(root_authorizer="root_agent",admitted=True,validate_phase_authority_v1=validate,fast_guard=self.fast_guard,
            reference=stable_reference_v1,writer=self.writer,output_directory=str(self.writer.path),physical_gpu=0,
            context_source_ref=c.ref(c.CONTEXT),operational_IO_source_ref=c.IO_SOURCE_REF,operational_IO_policy_ref=c.IO_POLICY_REF,
            input_ref=self.input_ref,MLM_prerequisite_metadata=self.MLM,prepare_native_runtime_v1=self.prepare_native,
            verify_native_imports_v1=lambda:self.verify_imports(),max_checkpoint_files=2,max_checkpoint_bytes=3*2**30,
            authenticate_native_inputs_once_v1=self.authenticate_inputs,prepared_metadata=self.prep,boundary_guard=self.writer.boundary,
            validated_VAL000_preparation_v1=self.preparation_callback,validated_VAL000_counts_v1=self.counts_callback,
            validated_VAL000_fit_completion_v1=self.fit_callback,validated_VAL000_metadata=self.metadata_callback)
        if self.contrastive is not None:ctx["contrastive_prerequisite_metadata"]=self.contrastive
        return ctx,request
    def run(self):
        self.fast_guard("before_own_native_output")
        c.require(not self.writer.path.exists(),"Genuinely fresh native phase output")
        self.writer.boundary("before_native_output_mkdir",required_write_bytes=8192);self.writer.path.mkdir()
        limit=3*2**30 if self.s["phase"]=="contrastive" else (48*2**30 if self.s["phase"].startswith("embedding_") else 256*2**20)
        resource.setrlimit(resource.RLIMIT_FSIZE,(limit,limit))
        ctx,request=self.context_and_request()
        if self.s["phase"]=="contrastive":return source_IO_v1().contrastive_keep1_v1(ctx,request)
        if self.s["phase"].startswith("embedding_"):
            self.authenticate_inputs(c.PREP_REF,c.COUNTS_REF)
            return self.base.embedding_v1(ctx,request)
        spec=dict(parent_id=c.PARENT,refs=dict(preparation=c.PREP_REF,preparation_config=c.PREP_CONFIG_REF,counts=c.COUNTS_REF,
            fit_completion=self.fit_view_descriptor,embedding_completion=self.embedding_view_descriptor))
        return source_IO_v1().RAM_retrieval_and_seal_v1(ctx,request,spec,dict(max_embedding_input_bytes=48*2**30))


def finalize_worker_v1(input_ref,s,q,cfg,storage):
    c.require(s["phase"]=="finalize" and os.environ.get("CUDA_VISIBLE_DEVICES")==""
        and "torch" not in sys.modules and "torch_geometric" not in sys.modules,"CPU-only metadata finalizer, no model/import/probe")
    check_external_prerequisites_v1(q)
    g=c.guard_v1(q);writer=GuardedWriter(s,q,g,storage)
    c.require(not writer.path.exists() and not (c.local(c.STATE[0]["output_directory"])/"completed.json").exists(),"Fresh finalizer and immutable master completion")
    writer.boundary("before_finalizer_mkdir",required_write_bytes=8192);writer.path.mkdir()
    resource.setrlimit(resource.RLIMIT_FSIZE,(8*2**20,8*2**20))
    master_writer=GuardedWriter(s,q,g,storage);master_writer.path=c.local(c.STATE[0]["output_directory"])
    final,wait,outer=prerequisite_metadata_v1(q["prerequisites"]["RAM_retrieve_and_seal"],"RAM_retrieve_and_seal")
    master_completion=master_writer.save("completed.json",dict(schema="native_VAL000_actual_complete_downstream_pipeline_v1",
        status="completed_native_VAL000_downstream_scientific_pipeline",parent_id=c.PARENT,
        original_preparation_ref=c.PREP_REF,actual_counts_ref=c.COUNTS_REF,all_prior_prerequisite_bundles=q["prerequisites"],
        final_RAM_native_completion_ref=q["prerequisites"]["RAM_retrieve_and_seal"]["completion_ref"],
        prediction_seal_ref=final["prediction_seal_ref"],all_six_scientific_stages_actual_wait0_and_consumed_outer0=True,
        this_finalizer_worker_wait_and_outer_terminal_still_pending=True,
        no_storage_close_or_credit_applied=True,query_ancestry_truth_opened=False,
        native_Table1_PCA_reproduction_claimed=False,ancestry_accuracy_claimed=False))
    return writer.save("completed.json",dict(schema="native_VAL000_pipeline_CPU_finalizer_child_v1",
        status="completed_native_VAL000_pipeline_finalizer_child",parent_id=c.PARENT,child_input_ref=input_ref,master_completion_ref=master_completion,
        all_six_scientific_stages_actual_wait0_and_consumed_outer0=True,query_ancestry_truth_opened=False,
        current_worker_wait_and_outer_terminal_still_pending=True))

def contextual_module_v1(record,name):
    source=c.raw(record).decode();tree=ast.parse(source);counts=c.read(c.COUNTS_REF)
    changes={"experiments/results/American_C3_val000_full30mb_shared_v1/native_paths/contrastive":c.PREP_OUTPUT+"/contrastive",227990:counts["MLM_optimizer_updates"],2135:c.EXPECTED_CONFIG["scientific_protocol"]["contrastive"]["updates"],9721236:c.EXPECTED_CONFIG["scientific_protocol"]["retrieval"]["reference_rows"],"american_c3_val_000":c.PARENT,"experiments/results/American_C3_val000_full30mb_shared_v1/native_paths":c.PREP_OUTPUT}
    class BindMetadata(ast.NodeTransformer):
        def visit_Constant(self,node):
            if type(node.value)in(int,str)and node.value in changes:return ast.copy_location(ast.Constant(changes[node.value]),node)
            return node
    tree=BindMetadata().visit(tree);ast.fix_missing_locations(tree);module=types.ModuleType(name);module.__file__=str(c.local(record["path"]))
    exec(compile(tree,module.__file__+":original_parent_metadata_only","exec"),module.__dict__)
    if hasattr(module,"REFS"):module.REFS=dict(module.REFS,preparation=c.PREP_REF,preparation_config=c.PREP_CONFIG_REF,counts=c.COUNTS_REF)
    if hasattr(module,"ROW_COUNTS"):module.ROW_COUNTS={key:item["rows"]for key,item in counts["datasets"].items()}
    return module

def pool_prerequisite_bundle_v1(bundle,stage):
    complete=c.read(bundle["completion_ref"]);wait=c.read(bundle["wait_ref"]);outer=c.read(bundle["outer_terminal_ref"])
    registry=c.read(outer["immutable_pool_registry_snapshot_ref"]);jobs=[j for j in registry["jobs"]if j["logical_job"]==outer["job_id"]]
    c.require(len(jobs)==1,"Actual pool-owned outer native leader wait")
    job=jobs[0];spec=c.read(wait["input_ref"]);authority=c.read(spec["root_stage_request_ref"])
    capability=c.read(outer["pool_capability_ref"]);declaration=capability["jobs"][outer["job_id"]]
    expected_phase="embedding_"+complete.get("view","")+"_"+complete.get("split","") if stage=="embedding" else stage
    c.require(spec["job_id"]==outer["job_id"] and spec["parent_id"]==c.PARENT
        and declaration["worker_kind"]=="native" and declaration["parent_id"]==c.PARENT
        and declaration["native_phase"]==expected_phase
        and spec["phase"]==authority["phase"]==expected_phase
        and authority["postdrain_pool_capability_ref"]==outer["pool_capability_ref"]
        and complete.get("child_input_ref",complete.get("active_input_ref"))==wait["input_ref"]
        and wait.get("group_observations") and all(row.get("absent") is True for row in wait["group_observations"]),
        "Same declared original parent/phase/source input and complete groups, no cross-stage bundle substitution")
    c.require(complete.get("parent_id")==c.PARENT and complete.get("query_ancestry_truth_opened") is False
        and wait.get("returncode")==0 and wait.get("actual_OS_wait_status")==0 and wait.get("actual_direct_worker_wait") is True
        and wait.get("complete_owned_roster_known") is True and wait.get("all_owned_processes_and_groups_drained") is True
        and wait.get("operational_failure") is None and wait.get("completed_ref")==bundle["completion_ref"] and wait.get("adapter_manifest_ref")==c.ref(c.MANIFEST)
        and job["state"]=="succeeded" and job["exit_code"]==0 and job["leader_wait_status_known"] is True and job["owned_drain_observed"] is True
        and outer.get("root_authorizer")=="root_agent" and outer.get("actual_exit_code")==0 and outer.get("actual_terminal_consumed") is True
        and outer.get("actual_wait_ref")==bundle["wait_ref"] and outer.get("original_standalone_or_historical_wait_relabelled") is False
        and outer["pool_capability_ref"]==registry["capability_ref"]==c.STATE[2]["postdrain_pool_capability_ref"],"New phase genuine inner0/drain and pool outer0/drain, no historical wait substitution")
    if stage=="mlm":
        c.require(complete.get("completed_epochs")==5 and complete.get("actual_optimizer_updates")==c.read(c.COUNTS_REF)["MLM_optimizer_updates"]
            and complete.get("preparation_ref")==c.PREP_REF and complete.get("initializer_checkpoint_ref") is None
            and complete.get("resume_checkpoint_ref") is None,"Complete same-parent fresh MLM")
    return complete,wait,outer
