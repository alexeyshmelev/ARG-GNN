"""Independent fake-only review of later C3 evaluation preparation; no active endpoints."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import ast,copy,hashlib,importlib,json,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");sys.path.insert(0,str(ROOT))
REVIEW="research/reviews/C3_full30Mb_eval_admission_readiness_independent_review_20261016_v1"
BUILDER="research/reviews/C3_full30Mb_eval_admission_readiness_source_20261016_v1"
MAIN="research/reviews/C3_saved600_artifact_acceptance_source_20261016_v1"
PREP="research/reviews/C3_saved600_validator_ROOT_preparation_source_20261016_v1"
MAIN_REVIEW=dict(path="research/reviews/C3_saved600_artifact_acceptance_independent_review_20261016_v1/review.v1.json",sha256="b89ce51b5fc08420bb7b275a21817146a54791438c4d8fa74ccdf810b4f1fe51")
b=importlib.import_module(BUILDER.replace("/",".")+".prepare_evaluation_root_v1")
c=importlib.import_module(MAIN.replace("/",".")+".acceptance_contract_v1")
helper=importlib.import_module(PREP.replace("/",".")+".prepare_root_v1")
manifest=json.loads((ROOT/MAIN/"source_manifest.v1.json").read_text())
checks=[]
def check(name,detail=None):checks.append(dict(name=name,status="PASS_SOFTWARE_ONLY",**({"detail":detail} if detail is not None else {})))
def reject(fn):
    try:fn()
    except (ValueError,KeyError,TypeError,AssertionError,AttributeError):return
    raise AssertionError("Expected rejection")
def extracted(path,name,ns):
    fn=next(n for n in ast.parse((ROOT/path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),"<exact-source:"+path+":"+name+">","exec"),ns)
    return ns[name]
opns=dict(require=b.require,copy=copy,MODE="native14_stopped_retention_v3",RETIRE_OPERATION="SYNTHETIC_UNUSED_RETIRE",datetime=datetime,timezone=timezone)
for name in ("operation_draft_v1","_pins","_expire","_authority"):
    extracted(c.GUARD_REF["path"],name,opns)
bindns=dict(require=b.require,copy=copy,_raw=lambda r:b.require(isinstance(r,dict) and set(r)=={"path","sha256"},"Fake exact reference"))
bindings=extracted(c.GUARD_REF["path"],"consumer_bindings_v1",bindns)
storens=dict(require=b.require,_raw=lambda r:None,budget_check_v9=lambda storage,**kw:kw)
storage_check=extracted(c.GUARD_REF["path"],"validate_storage_v9",storens)

class FakeFlow:
    def __init__(self,phase="predict_val_000",mode="normal"):
        self.phase=phase;self.mode=mode;self.mem={};self.reads=[];self.events=[];self.sizes={};self.dirs=set();self.applies=0
        self.counter=0;self.expiry=(datetime.now(timezone.utc)+timedelta(days=3)).isoformat()
        outer=self
        class VirtualPath(type(Path())):
            def exists(self):
                key=str(self.relative_to(ROOT))
                if key.startswith("research/reviews/C3_full30Mb_eval_ROOT_") or key.startswith("experiments/results/C3_full30Mb_eval_"):
                    return str(self) in outer.dirs or key in outer.mem
                return Path(self).exists()
            def is_symlink(self):return False
            def mkdir(self,*a,**kw):
                assert str(self).startswith(str(ROOT/"research/reviews/C3_full30Mb_eval_ROOT_"))
                assert str(self) not in outer.dirs;outer.dirs.add(str(self));outer.events.append("fake_mkdir")
        def local(path):
            p=Path(str(path));p=p if p.is_absolute() else ROOT/p
            assert p.is_relative_to(ROOT) and ".." not in p.parts
            return VirtualPath(p)
        def ref(path):
            p=local(path);key=str(p.relative_to(ROOT))
            assert p.suffix in (".json",".py",".sh",".md"),"No scientific payload hashing"
            if key in self.mem:return dict(path=key,sha256=self.mem[key][1])
            return c.ref(key)
        def read(record,cap=33554432):
            assert isinstance(record,dict) and set(record)=={"path","sha256"},"Exact fake JSON ref"
            assert Path(record["path"]).suffix==".json","No scientific payload read"
            self.reads.append(record["path"])
            if record["path"] in self.mem:
                value,sha=self.mem[record["path"]];assert sha==record["sha256"]
                return copy.deepcopy(value)
            return c.read(record,cap=cap)
        def save(directory,name,value):
            key=str((local(directory)/name).relative_to(ROOT));assert key not in self.mem and Path(name).name==name
            raw=(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
            assert len(raw)<=65536
            charge=sum(size+8192 for path,size in self.sizes.items() if str(Path(path).parent)==str(Path(key).parent))+len(raw)+8192
            assert charge<=524288
            self.charge=charge;self.sizes[key]=len(raw);self.mem[key]=(copy.deepcopy(value),hashlib.sha256(raw).hexdigest())
            self.events.append("save:"+name);return ref(key)
        self.local=local;self.ref=ref;self.read=read
        ns=dict(c.__dict__,local=local,ref=ref,read=read,save=save,manifest_v1=lambda:copy.deepcopy(manifest))
        for name in ("require","active","review_values_v1","history_values_v1","acceptance_values_v1","caps_v1",
            "root_validation_request_v1","training_ready_v1","terminal0_v1","plan_ready_v1","_evaluation_request_v1",
            "validate_request","validate_input_v1","make_config","required_direct_refs_v1"):
            ns[name]=types.FunctionType(getattr(c,name).__code__,ns,name,getattr(c,name).__defaults__)
        self.ns=ns
        self.training_ref=self.put(dict(status="SYNTHETIC_TRAIN_COMPLETION_NOT_REAL"))
        self.candidates={epoch:dict(path=REVIEW+"/SYNTHETIC_checkpoint_"+str(epoch)+".pt",sha256=str(i+1)*64) for i,epoch in enumerate(c.EPOCHS)}
        self.candidate_rows=[dict(epoch=e,global_step=100*(i+1),checkpoint_ref=self.candidates[e]) for i,e in enumerate(c.EPOCHS)]
        self.training=dict(checkpoint_candidates=copy.deepcopy(self.candidate_rows),protocol_sha256="a"*64,observation_seal_sha256="b"*64,target_join_sha256="c"*64)
        ns["frozen_v1"]=lambda:dict(training_completion_ref=self.training_ref,checkpoint_candidates=copy.deepcopy(self.candidate_rows))
        def acceptance(req):
            ns["history_values_v1"](req["original_history"])
            assert req["training_completion_ref"]==self.training_ref
            return dict(training_completion_ref=self.training_ref),copy.deepcopy(self.training)
        ns["acceptance_inputs_v1"]=acceptance
        request=json.loads((ROOT/MAIN/"ROOT_validation_request.INACTIVE.template.v1.json").read_text())
        request.update(active=True,root_authorizer="root_agent",execution_authorized=True,expires_at_utc=self.expiry,
            training_completion_ref=self.training_ref,source_manifest_ref=b.MAIN_MANIFEST_REF,source_review_ref=MAIN_REVIEW,
            caps_ref=c.ref(MAIN+"/caps_validate_saved600.INACTIVE.v1.json"),
            ROOT_accepts_original_producer1_and_outerUNKNOWN_for_artifact_validation_only=True,
            future_original_C3_training_writes_cooperatively_revoked=True,controller_preserved_for_borrowed_native_work=True)
        validation_req=self.put(request)
        validation_input=self.put(dict(root_stage_request_ref=validation_req,phase=c.ACCEPT_PHASE,main_source_ref=c.ref(c.ENTRY),
            root_driver_ref=c.ref(c.DRIVER),source_sha256=copy.deepcopy(manifest["source_sha256"])))
        result=dict(status="validated_C3_saved600_artifacts_after_failed_producer",all_six_verified=True,saved_effective_logical_updates=600,
            optimizer_updates_performed=0,VAL_or_TEST_truth_opened=False,root_acceptance_still_required=True,
            original_history=copy.deepcopy(c.ORIGINAL_HISTORY),dedicated_validator_CPU_runtime_checked=True,CUDA_initialized=False,
            source_manifest_ref=b.MAIN_MANIFEST_REF,checkpoints=[dict(row,all_checks_passed=True,actual_AdamW_step_min=row["global_step"],
                actual_AdamW_step_max=row["global_step"],optimizer_parameter_states=38,model_parameters=29571) for row in self.candidate_rows],
            provenance=dict(parents=20,all_original_block_seals=True),original_producer_success_claimed=False,
            original_worker_drain_manifest_reconstructed=False,storage_close_or_credit_authorized=False,
            training_completion_ref=self.training_ref,protocol_sha256="a"*64,observation_seal_sha256="b"*64,target_join_sha256="c"*64)
        validation_result=self.put(result)
        valenv=self.put(dict(status="completed_C3_saved600_artifact_validation",stage=c.ACCEPT_PHASE,root_driver_ref=c.ref(c.DRIVER),
            training_completion_ref=self.training_ref,input_ref=validation_input,result_ref=validation_result))
        valwait=self.put(dict(returncode=0,actual_direct_launcher_wait=True,all_owned_processes_and_groups_drained=True,
            operational_failure=None,driver_ref=c.ref(c.DRIVER),input_ref=validation_input,completed_ref=valenv,pid=1234))
        valterm=self.put(dict(actual_terminal_consumed=True,actual_exit_code=0,completed_ref=valenv,actual_wait_ref=valwait,
            input_ref=validation_input,driver_ref=c.ref(c.DRIVER)))
        self.acceptance=dict(schema="C3_saved600_ROOT_artifact_acceptance_v1",root_authorizer="root_agent",active=True,
            execution_authorized=True,expires_at_utc=self.expiry,accepted_saved_artifacts_only=True,original_invocation_success_claimed=False,
            original_outerUNKNOWN_explicitly_acknowledged=True,future_original_C3_training_writes_cooperatively_revoked=True,
            storage_close_or_credit_authorized=False,training_completion_ref=self.training_ref,source_manifest_ref=b.MAIN_MANIFEST_REF,
            source_review_ref=MAIN_REVIEW,original_history=copy.deepcopy(c.ORIGINAL_HISTORY),validator_input_ref=validation_input,
            validator_request_ref=validation_req,validator_completion_ref=valenv,validation_result_ref=validation_result,
            actual_wait_ref=valwait,outer_terminal_ref=valterm)
        self.acceptance_ref=self.put(self.acceptance)
        self.plans={};self.plan_evidence={};self.simulations={};self.seals={};self.sixes={};self.predictions={}
        for parent in c.VAL:
            source_obs=self.put(dict(SYNTHETIC_SOURCE_OBSERVATION=parent))
            sim=self.put(dict(status="completed_C3_simulation_reconstruction",parent_id=parent,class_order=["EUR","EAS","AFR"],
                observation_ref=source_obs,private_truth_ref=dict(path=REVIEW+"/SYNTHETIC_NEVER_OPENED_"+parent+".npz",sha256="e"*64)))
            self.simulations[parent]=sim
            obs=self.put(dict(parent_id=parent,split="val",sequence_length_bp=30000000,
                counts_haploid=dict(AFR=200,EUR=200,ASIA=200,ADMIX=1000),query_ancestry_truth_in_inputs=False,source_observation_ref=source_obs))
            focal=dict(path=REVIEW+"/SYNTHETIC_focal_"+parent+".json",sha256="f"*64)
            prepref=self.put(dict(status="completed_native_C3_path_preparation",parent_id=parent,focals_ref=focal,
                observation_ref=obs,query_focal_count=10000,query_ancestry_truth_opened=False))
            plan=dict(schema="american_c3_whole_tree_block_index_v1",parent_id=parent,sequence_length_bp=30000000,haplotypes=1600,
                focal_records=10000,all_trees_and_samples_retained=True,targets_opened=False,class_order=["EUR","EAS","AFR"],
                config=dict(native_preparation_ref=prepref),observation_ref=obs,focal_registry_ref=focal)
            self.plans[parent]=plan;planref=self.put(plan)
            planresult=self.put(dict(status="completed_C3_observation_only_block_plan",parent_id=parent,targets_opened=False,block_index_ref=planref))
            plane=self.stage("SYNTHETIC_plan",planresult)
            close=self.put(dict(kind="concurrent_completed_scope_close_receipt_v1",value=dict(future_writes_and_relaunch_revoked=True,
                request=dict(completed_ref=plane["completion_ref"],direct_wait_ref=plane["actual_wait_ref"]))))
            self.plan_evidence[parent]=dict(plane,parent_id=parent,plan_ref=planref,close_receipt_ref=close)
            seals=[]
            for epoch in c.EPOCHS:
                seal=dict(status="sealed_C3_original_heldout_predictions",split="val",epoch=epoch,parent_id=parent,records=10000,
                    private_truth_opened=False,probabilities_modified=False,training_completion_ref=self.training_ref,
                    checkpoint_ref=self.candidates[epoch],prediction_ref=dict(path=REVIEW+"/SYNTHETIC_probability_"+parent+"_"+str(epoch)+".npy",sha256="9"*64),
                    focal_registry_ref=focal)
                self.seals[(epoch,parent)]=seal;seals.append(self.put(seal))
            six=dict(status="sealed_C3_one_original_VAL_all6_candidates",parent_id=parent,training_completion_ref=self.training_ref,
                private_truth_opened=False,prediction_seal_refs=seals)
            self.sixes[parent]=six;self.predictions[parent]=self.stage("predict_val_"+parent[-3:],self.put(six))
        cells=[]
        for epoch in c.EPOCHS:
            for parent in c.VAL:
                seal=self.seals[(epoch,parent)]
                cells.append(dict(epoch=epoch,parent_id=parent,checkpoint_ref=self.candidates[epoch],prediction_seal_ref=self.sixes[parent]["prediction_seal_refs"][list(c.EPOCHS).index(epoch)],
                    prediction_ref=seal["prediction_ref"],focal_registry_ref=seal["focal_registry_ref"]))
        self.matrix=dict(status="sealed_C3_all30_original_VAL_predictions",training_completion_ref=self.training_ref,
            all_predictions_sealed_before_VAL_truth=True,TEST_deferred=True,checkpoint_epochs=list(c.EPOCHS),
            validation_parent_ids=list(c.VAL),cells=cells)
        self.matrix_evidence=self.stage("matrix",self.put(self.matrix))
        self.evidence=dict(schema="C3_full30Mb_eval_actual_stage_evidence_v1",phase=phase,ROOT_artifact_acceptance_ref=self.acceptance_ref,
            closed_plan_evidence_by_parent=copy.deepcopy(self.plan_evidence))
        if phase=="matrix":self.evidence["VAL_predictions"]=copy.deepcopy(self.predictions)
        if phase=="score":self.evidence.update(matrix=copy.deepcopy(self.matrix_evidence),VAL_simulation_completion_refs=copy.deepcopy(self.simulations))
        self.builder_review=dict(schema="C3_full30Mb_eval_preparation_independent_review_v1",independent=True,verdict="SOURCE_PASS",
            source_ref=b.reference(b.SOURCE),source_handoff_ref=b.reference(BUILDER+"/source_handoff.v1.json"),
            main_manifest_ref=b.MAIN_MANIFEST_REF,validator_preparation_source_ref=b.PREP_HELPER_REF)
        self.builder_review_ref=self.put(self.builder_review)
        guard=types.SimpleNamespace(MODE="native14_stopped_retention_v3")
        v4=types.SimpleNamespace(__name__="_synthetic_original_v4")
        def original():raise AssertionError("Never invoked source-only original mutation probe")
        original.__module__=v4.__name__;v4.canonical=v4.file_bytes=v4.mutex=original
        if mode=="wrapped":v4.canonical=lambda:None
        guard.budget=types.SimpleNamespace(v4=v4,SOURCE_REL="scripts/project_budget_v9.py",LAUNCHER_REL="scripts/launch_reserved_gpu_v6.py")
        def inspect(**kw):
            assert kw==dict(expected_source_sha256=c.CORE,expected_launcher_sha256=c.BASE)
            roles=copy.deepcopy(manifest["current_guard_roles"])
            roles["SYNTHETIC_UNEXPECTED_CURRENT_ROLE"]=dict(path=REVIEW+"/SYNTHETIC_extra_current.json",sha256="8"*64)
            if mode=="overflow":
                for i in range(8):roles["SYNTHETIC_extra_"+str(i)]=dict(path=REVIEW+"/SYNTHETIC_extra_"+str(i)+".json",sha256="7"*64)
            self.events.append("fake_inspect")
            return dict(bindings=roles,base_pins=dict(SYNTHETIC=1),completed_overlay_prefix=dict(events=321,tail_sha256="6"*64))
        def apply(record,**kw):
            self.applies+=1;self.events.append("fake_apply_once");self.precharge=self.charge
            opns.update(_json=read,budget=guard.budget)
            auth=opns["_authority"](record,"reserve",read(record)["bindings"],fresh=True)
            if mode=="race":raise ValueError("SYNTHETIC prefix race")
            return dict(kind="concurrent_completed_scope_reserve_receipt_v1",value=dict(auth["request"],root_admission=record),
                initial_ledger_pins=dict(base_pins=dict(SYNTHETIC=1),completed_overlay_prefix=dict(events=322,tail_sha256="5"*64)))
        guard.inspect_v1=inspect;guard._pins=opns["_pins"];guard.operation_draft_v1=opns["operation_draft_v1"];guard.apply_reserve_v1=apply
        guard.consumer_bindings_v1=bindings
        guard.validate_storage_v9=lambda spec,cfg,storage,**kw:storage_check(spec,cfg,storage,**kw)
        ns["original_guard"]=lambda:guard;self.guard=guard;self.fakec=types.SimpleNamespace(**ns)
        bn=dict(b.__dict__)
        def bound(record):
            if record["path"] in self.mem:
                return json.dumps(read(record),sort_keys=True,separators=(",",":")).encode()
            return b.bound(record)
        def reference(path):
            return ref(path) if path in self.mem else b.reference(path)
        def load(record,label):
            assert record in (b.PREP_HELPER_REF,dict(path=MAIN+"/acceptance_contract_v1.py",sha256=manifest["source_sha256"][MAIN+"/acceptance_contract_v1.py"]))
            return helper if record==b.PREP_HELPER_REF else self.fakec
        bn.update(bound=bound,reference=reference,load=load)
        for name in ("builder_review_v1","actual_stage_v1","prediction_metadata_v1","matrix_metadata_v1","evidence_request_fields_v1","prepare_root_v1"):
            bn[name]=types.FunctionType(getattr(b,name).__code__,bn,name,getattr(b,name).__defaults__)
        self.bn=bn
        if mode=="late":
            end=datetime.fromisoformat(self.expiry)
            class Clock:
                count=0
                @classmethod
                def fromisoformat(cls,value):return datetime.fromisoformat(value)
                @classmethod
                def now(cls,tz):
                    cls.count+=1
                    return end-timedelta(seconds=86700) if cls.count>=3 else end-timedelta(days=3)
            bn["datetime"]=Clock
    def put(self,value):
        self.counter+=1;path=REVIEW+"/SYNTHETIC_"+str(self.counter)+".json"
        raw=json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
        sha=hashlib.sha256(raw).hexdigest();self.mem[path]=(copy.deepcopy(value),sha)
        return dict(path=path,sha256=sha)
    def stage(self,phase,result):
        inp=dict(path=REVIEW+"/SYNTHETIC_stage_input_"+str(self.counter)+".json",sha256="4"*64)
        env=self.put(dict(stage=phase,status=c.STATUSES.get(phase,"SYNTHETIC_PLAN"),root_driver_ref=c.ref(c.DRIVER),
            training_completion_ref=self.training_ref,input_ref=inp,result_ref=result))
        wait=self.put(dict(driver_ref=c.ref(c.DRIVER),input_ref=inp,completed_ref=env,returncode=0,
            actual_direct_launcher_wait=True,all_owned_processes_and_groups_drained=True,operational_failure=None))
        term=self.put(dict(actual_terminal_consumed=True,actual_exit_code=0,completed_ref=env,actual_wait_ref=wait))
        return dict(completion_ref=env,actual_wait_ref=wait,outer_terminal_ref=term)
    def value(self,name):
        values=[v[0] for k,v in self.mem.items() if k.endswith("/"+name)]
        assert len(values)==1
        return copy.deepcopy(values[0])
    def fields(self):
        return self.bn["evidence_request_fields_v1"](self.fakec,self.evidence,self.phase)
    def run(self):
        evidence_ref=self.put(self.evidence)
        args=types.SimpleNamespace(phase=self.phase,root_directive=b.DIRECTIVE,instance="SYNTHETIC_"+self.phase+"_"+self.mode,
            main_review_path=MAIN_REVIEW["path"],main_review_sha256=MAIN_REVIEW["sha256"],
            builder_review_path=self.builder_review_ref["path"],builder_review_sha256=self.builder_review_ref["sha256"],
            evidence_path=evidence_ref["path"],evidence_sha256=evidence_ref["sha256"],expires_at_utc=self.expiry)
        return self.bn["prepare_root_v1"](args)
    def rejected_before_writes(self):
        reject(self.run);assert not self.dirs and not self.sizes and self.applies==0

summaries=[]
for phase in b.PHASES:
    f=FakeFlow(phase);result=f.run();spec=f.value("input.json");req=f.value("request.root_active_v1.json");storage=f.value("storage.json")
    assert result["status"]=="ADMITTED_C3_CPU_EVALUATION_STAGE_NOT_EXECUTED" and f.applies==1
    assert result["consumer_binding_count"]==58
    assert f.value("config.json")["output_cap_bytes"]==storage["reserved_new_bytes"]==33554432
    assert storage["reservation"]["auxiliary_scopes"]==[str(ROOT/req["packet_directory"]/"worker_aux"),str(ROOT/req["packet_directory"]/"worker_tmp")]
    assert storage["reservation"]==f.value("reserve_receipt.json")["value"]
    assert storage["initial_ledger_pins"]==f.value("reserve_receipt.json")["initial_ledger_pins"]
    assert req["expires_at_utc"]==spec["expires_at_utc"]==storage["expires_at_utc"]==f.expiry
    assert ("VAL_truth_capability" in req)==(phase=="score")
    if phase=="score":assert req["VAL_truth_capability"]==f.simulations
    assert f.value("admitted.json")["execution_started"] is False
    assert not any((ROOT/key).exists() for key in f.mem)
    assert not any(Path(p).suffix in (".pt",".npy",".npz") for p in f.reads)
    argv=result["exact_one_ROOT_stage_execution_argv"]
    assert argv[:5]==["setsid",str(ROOT/".runtime/env/bin/python"),"-B",str(ROOT/c.DRIVER),"--launch"]
    assert argv[5:]==["--input",result["input_ref"]["path"],"--expected-input-sha256",result["input_ref"]["sha256"]]
    f.fakec.set_active_spec_v1=lambda record:f.ns["validate_input_v1"](record)
    dn=dict(argparse=importlib.import_module("argparse"),json=json,c=f.fakec,SOURCE=c.DRIVER,__doc__="fake source-only")
    dn["bind_reviewed_supervisor_v1"]=lambda:lambda phase:f.events.append("fake_supervisor:"+phase) or 0
    assert extracted(MAIN+"/root_driver_v1.py","main",dn)(argv[4:])==0
    summaries.append(dict(phase=phase,roles=result["consumer_binding_count"],pre_reserve_metadata_charge=f.precharge,total_packet_metadata_charge=f.charge,
        truth_capability_present="VAL_truth_capability" in req,all_writes_in_memory=True))
check("all7_full_fake_flows_actual_acceptance_plan_request_input_native14_authority_and_driver_bodies",summaries)
for mode in ("wrapped","overflow","race","late"):
    f=FakeFlow("score",mode);reject(f.run)
    assert f.applies==(1 if mode in ("race","late") else 0)
    if mode=="race":assert not any(k.endswith(("/storage.json","/input.json","/admitted.json")) for k in f.mem)
    check("full_flow_"+mode+"_rejects_without_retry_or_execution")
for parent in c.VAL:
    f=FakeFlow();f.evidence["closed_plan_evidence_by_parent"].pop(parent);f.rejected_before_writes()
check("every_one_of5_missing_plan_keys_rejected_before_writes")
for parent in c.VAL:
    f=FakeFlow();f.evidence["closed_plan_evidence_by_parent"][parent]["close_receipt_ref"]=None;f.rejected_before_writes()
check("every_one_of5_missing_actual_close_receipts_rejected_before_writes")
f=FakeFlow();parent=c.VAL[3];plan=copy.deepcopy(f.plans[parent]);plan["sequence_length_bp"]=2000000
f.evidence["closed_plan_evidence_by_parent"][parent]["plan_ref"]=f.put(plan);f.rejected_before_writes()
check("incomplete_full30Mb_plan_rejected")
for patch in (dict(accepted_saved_artifacts_only=False),dict(outer_terminal_ref=None),dict(original_outerUNKNOWN_explicitly_acknowledged=False)):
    f=FakeFlow();accepted=copy.deepcopy(f.acceptance);accepted.update(patch);f.evidence["ROOT_artifact_acceptance_ref"]=f.put(accepted);f.rejected_before_writes()
check("genuine_ROOT_acceptance_saved_only_unknown_and_own_terminal_prerequisites")
for field,patch in (("actual_wait_ref",dict(returncode=False)),("actual_wait_ref",dict(all_owned_processes_and_groups_drained=False)),
    ("outer_terminal_ref",dict(actual_terminal_consumed=False)),("outer_terminal_ref",dict(actual_exit_code=1)),
    ("validation_result_ref",dict(all_six_verified=False)),("validation_result_ref",dict(optimizer_updates_performed=1))):
    f=FakeFlow();accepted=copy.deepcopy(f.acceptance);value=f.read(accepted[field]);value.update(patch);accepted[field]=f.put(value)
    f.evidence["ROOT_artifact_acceptance_ref"]=f.put(accepted);f.rejected_before_writes()
check("acceptance_real_wait_type_drain_outer_consumed_all6_and_no_updates_negatives")
for phase in ("predict_val_000","matrix"):
    for key in ("VAL_truth_capability","external_truth_ref","VAL_simulation_completion_refs","TEST_truth"):
        f=FakeFlow(phase);f.evidence[key]={};f.rejected_before_writes()
check("truth_and_extra_capability_keys_rejected_before_prediction_and_matrix")
for field,patch in (("actual_wait_ref",dict(returncode=False)),("actual_wait_ref",dict(all_owned_processes_and_groups_drained=False)),
    ("outer_terminal_ref",dict(actual_exit_code=False)),("outer_terminal_ref",dict(actual_terminal_consumed=False)),
    ("completion_ref",dict(root_driver_ref=dict(path="SYNTHETIC_wrong.py",sha256="0"*64))),("completion_ref",dict(stage="predict_val_004")),
    ("completion_ref",dict(training_completion_ref=dict(path="SYNTHETIC_wrong.json",sha256="0"*64)))):
    f=FakeFlow("matrix");p=c.VAL[0];e=f.evidence["VAL_predictions"][p];value=f.read(e[field]);value.update(patch);e[field]=f.put(value)
    f.rejected_before_writes()
f=FakeFlow("matrix");f.evidence["VAL_predictions"][c.VAL[0]]["outer_terminal_ref"]=None;f.rejected_before_writes()
check("predecessor_wait_integer0_drain_outer0_consumed_exact_driver_phase_training_required")
for change in ("missing","duplicate","wrong_checkpoint","truth_opened","probabilities_modified","wrong_parent"):
    f=FakeFlow("matrix");p=c.VAL[0];six=copy.deepcopy(f.sixes[p])
    if change=="missing":six["prediction_seal_refs"].pop()
    elif change=="duplicate":six["prediction_seal_refs"][-1]=six["prediction_seal_refs"][0]
    else:
        seal=f.read(six["prediction_seal_refs"][-1])
        patch={"wrong_checkpoint":dict(checkpoint_ref=f.candidates[c.EPOCHS[0]]),"truth_opened":dict(private_truth_opened=True),
            "probabilities_modified":dict(probabilities_modified=True),"wrong_parent":dict(parent_id=c.VAL[1])}[change]
        seal.update(patch);six["prediction_seal_refs"][-1]=f.put(seal)
    f.evidence["VAL_predictions"][p]=f.stage("predict_val_000",f.put(six));f.rejected_before_writes()
check("all6_unique_prediction_metadata_checkpoint_parent_and_raw_pretruth_negatives")
for change in ("missing","duplicate","wrong_checkpoint","wrong_focal","wrong_prediction","truth_opened"):
    f=FakeFlow("score");matrix=copy.deepcopy(f.matrix)
    if change=="missing":matrix["cells"].pop()
    elif change=="duplicate":matrix["cells"][-1]=copy.deepcopy(matrix["cells"][0])
    elif change=="wrong_checkpoint":matrix["cells"][-1]["checkpoint_ref"]=f.candidates[c.EPOCHS[0]]
    elif change=="wrong_focal":matrix["cells"][-1]["focal_registry_ref"]=dict(path="SYNTHETIC_wrong.json",sha256="0"*64)
    elif change=="wrong_prediction":matrix["cells"][-1]["prediction_ref"]=dict(path="SYNTHETIC_wrong.npy",sha256="0"*64)
    else:
        seal=f.read(matrix["cells"][-1]["prediction_seal_ref"]);seal["private_truth_opened"]=True
        matrix["cells"][-1]["prediction_seal_ref"]=f.put(seal)
    f.evidence["matrix"]=f.stage("matrix",f.put(matrix));f.reads=[];f.rejected_before_writes()
    assert not any(r["path"] in f.reads for r in f.simulations.values())
check("complete30cell_metadata_before_any_score_simulation_capability_reads")
for change in ("missing","TEST","wrong_parent","wrong_observation"):
    f=FakeFlow("score")
    if change=="missing":f.evidence["VAL_simulation_completion_refs"].pop(c.VAL[-1])
    elif change=="TEST":f.evidence["VAL_simulation_completion_refs"]["american_c3_test_000"]=f.simulations[c.VAL[0]]
    else:
        p=c.VAL[-1];sim=f.read(f.simulations[p])
        sim.update(dict(parent_id=c.VAL[0]) if change=="wrong_parent" else dict(observation_ref=dict(path="SYNTHETIC_wrong.json",sha256="0"*64)))
        f.evidence["VAL_simulation_completion_refs"][p]=f.put(sim)
    f.rejected_before_writes()
check("score_same5parent_simulation_metadata_no_TEST_or_unrelated_observation")
for patch in (dict(independent=False),dict(source_ref={"path":"SYNTHETIC_wrong.py","sha256":"0"*64}),dict(main_manifest_ref={"path":"SYNTHETIC_wrong.json","sha256":"0"*64}),
    dict(validator_preparation_source_ref={"path":"SYNTHETIC_wrong.py","sha256":"0"*64})):
    f=FakeFlow();review=copy.deepcopy(f.builder_review);review.update(patch);f.builder_review_ref=f.put(review);f.rejected_before_writes()
check("actual_builder_review_schema_and_exact_source_handoff_main_manifest_helper_gates")
assert not any(name in sys.modules for name in ("torch","numpy","torch_geometric"))
print(json.dumps(dict(schema="C3_full30Mb_eval_builder_independent_fake_checks_v1",status="PASS_SOFTWARE_ONLY",check_count=len(checks),checks=checks,
    exact_builder_function_code=True,actual_guard_resource_process_probe_accounting_authority_admission_acceptance_execution_evaluation_or_Git=False,
    actual_scientific_checkpoint_truth_prediction_or_focal_payloads_opened=False,
    fake_only_endpoints=["source_loader","current_guard_inspection","reserve_apply","budget_boundary","in_memory_JSON_packets","frozen_checkpoint_metadata","acceptance_input_metadata","supervisor"],
    real_pure_schema_bodies=["builder_all_functions","training_ready_v1","acceptance_values_v1","root_validation_request_v1","plan_ready_v1","terminal0_v1","evaluation_request_v1","validate_input_v1","required_direct_refs_v1","native14_operation_draft_pins_authority_expire_consumer_bindings_validate_storage","root_driver_main"],
    all_synthetic_evidence_is_in_memory=True,source_ref=b.reference(b.SOURCE)),sort_keys=True,indent=2))
