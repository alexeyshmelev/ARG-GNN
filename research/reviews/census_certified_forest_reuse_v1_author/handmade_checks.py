"""Bounded handmade CPU checks; no actual data, CUDA, fit loop or targets."""
import ast,copy,gc,hashlib,importlib.util,json,os,random,sys,time,traceback,weakref
from dataclasses import fields,replace
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
import torch
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
torch.set_num_threads(1);torch.set_num_interop_threads(1)
ROOT=Path.cwd();OUT=Path(__file__).resolve().parent
(OUT/"_tmp").mkdir(exist_ok=True)
os.environ["TMPDIR"]=str(OUT/"_tmp")
import tempfile
tempfile.tempdir=str(OUT/"_tmp")
from arggnn import census_certified_forest_reuse_v1 as reuse
from arggnn import census_biological_fit_v2 as fit
from arggnn import census_biological_bank_v1 as bank
from arggnn import census_tree_encoder_v1 as tree
spec=importlib.util.spec_from_file_location("handmade_join_fixture",ROOT/"tests/test_census_biological_bank_v1.py")
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
checks=[];rows=[];forbidden=[];started=time.monotonic()
baseline=json.loads((OUT/"baseline_pins.json").read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(event,args):
    if event=="open":
        raw=args[0]
        if not isinstance(raw,(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(raw)).absolute()
        mode=args[1];flags=args[2]
        write=(isinstance(mode,str) and any(c in mode for c in "wax+")) or (isinstance(flags,int) and bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if str(p).startswith(str(ROOT)):
            rel=p.relative_to(ROOT)
            if rel.parts and rel.parts[0] in ("data","results"):
                forbidden.append({"event":event,"path":str(p)});raise PermissionError("actual scientific read/write forbidden")
            if str(rel).startswith("experiments/results/"):
                forbidden.append({"event":event,"path":str(p)});raise PermissionError("actual experiment output read/write forbidden")
        if write and not str(p).startswith(str(OUT)+"/"):
            forbidden.append({"event":event,"path":str(p)});raise PermissionError("author writes outside own evidence")
    if event in ("subprocess.Popen","os.system"):
        forbidden.append({"event":event});raise PermissionError("no subprocess or external job")
sys.addaudithook(audit)
def ck(name,ok=True):
    if not bool(ok):raise AssertionError(name)
    checks.append(name)
def eq(name,a,b):
    ck(name,torch.equal(a,b) if isinstance(a,torch.Tensor) else a==b)
def rejects(name,fn,match=None):
    try:fn()
    except (ValueError,TypeError,RuntimeError,KeyError) as e:
        if match is not None:ck(name+"_reason",match in str(e))
        ck(name);return
    raise AssertionError("did not reject "+name)
def certify(c):
    return replace(c,certificate_sha256=bank.fingerprint_sha256_v1(bank._content(c)))
def call(adapter,model,c,control="proposed"):
    kw={"group_permutation":torch.tensor([2,3,1,0])} if control=="permuted_groups" else {}
    return adapter(model,c,expected_certificate_sha256=c.certificate_sha256,control=control,readout_chunk_size=3,**kw)
def model(dtype=torch.float64):
    torch.manual_seed(519)
    return tree.CensusTreeGNNV1(7,3,4,2,3,initial_exchange=(5.,20.,80.),initial_damping=1.,initial_eta=1.).to(dtype=dtype)
def compare_outputs(label,a,b):
    for f in fields(a):eq(label+"_"+f.name,getattr(a,f.name),getattr(b,f.name))
def compare_optimizer(label,a,b):
    aa=a.state_dict();bb=b.state_dict();ck(label+"_groups",aa["param_groups"]==bb["param_groups"])
    ck(label+"_state_keys",aa["state"].keys()==bb["state"].keys())
    for k,av in aa["state"].items():
        ck(label+"_state_fields_"+str(k),av.keys()==bb["state"][k].keys())
        for field,x in av.items():eq(label+"_state_"+str(k)+"_"+field,x,bb["state"][k][field])
report={"scope":"handmade TS and synthesized targets only; no actual payloads/targets/GPU or original fit loop","status":"RUNNING"}
try:
    cfg=reuse.load_config_v1();ck("inactive_source_protocol",cfg["actual_profile_or_fit_authorized"] is False)
    for p,h in baseline["original_sha256"].items():ck("original_pin_before_"+p,digest(ROOT/p)==h)
    cert=fixture.prepare();ck("handmade_toy_two_whole_trees",len(cert.inputs)==2)
    eq("handmade_scored_two",int(cert.bank.partitions[0].scored_query_mask.sum()),2)
    original_certificate_hash=cert.certificate_sha256
    fit_cfg=fit.load_config_v1()
    for dtype in (torch.float32,torch.float64):
        adapter=reuse.CertifiedForestReuseV1()
        for control in reuse.CONTROLS:
            a=model(dtype);b=copy.deepcopy(a);parameter_ids=tuple(id(x) for x in b.parameters())
            rng=torch.get_rng_state().clone()
            kw={"group_permutation":torch.tensor([2,3,1,0])} if control=="permuted_groups" else {}
            ref=bank.forward_certified_census_biological_v1(a,cert,expected_certificate_sha256=cert.certificate_sha256,control=control,readout_chunk_size=3,**kw)
            got=call(adapter,b,cert,control)
            label=str(dtype)+"_"+control
            compare_outputs(label+"_forward",ref,got)
            eq(label+"_RNG",rng,torch.get_rng_state())
            ck(label+"_same_parameter_objects",parameter_ids==tuple(id(x) for x in b.parameters()))
            ck(label+"_encoder_restored","forward" not in b.encoder.__dict__)
            gen=torch.Generator().manual_seed(751)
            mass=torch.rand(ref.logits.shape,generator=gen,dtype=torch.float64)+.25
            states=[]
            for m in (a,b):
                opt=torch.optim.AdamW(fit.optimizer_groups_v1(m,fit_cfg),lr=fit_cfg["training"]["learning_rate"],betas=tuple(fit_cfg["training"]["betas"]),eps=fit_cfg["training"]["epsilon"])
                states.append({"model":m,"optimizer":opt,"steps":0})
            for step in (1,2):
                row_a=fit.train_step_v1(states[0],cert,cert.certificate_sha256,mass,control,torch.tensor([2,3,1,0]),fit_cfg,"cpu")
                with reuse.installed_fit_forward_v1(adapter):
                    row_b=fit.train_step_v1(states[1],cert,cert.certificate_sha256,mass,control,torch.tensor([2,3,1,0]),fit_cfg,"cpu")
                ck(label+"_step_"+str(step)+"_loss_norm_steps",row_a==row_b)
                for (name,x),(other,y) in zip(a.named_parameters(),b.named_parameters()):
                    ck(label+"_name_"+name,name==other)
                    eq(label+"_step_"+str(step)+"_parameter_"+name,x,y)
                    ck(label+"_step_"+str(step)+"_gradient_presence_"+name,(x.grad is None)==(y.grad is None))
                    if x.grad is not None:eq(label+"_step_"+str(step)+"_gradient_"+name,x.grad,y.grad)
                compare_optimizer(label+"_optimizer_step_"+str(step),states[0]["optimizer"],states[1]["optimizer"])
                ck(label+"_original_fit_helper_restored",fit.forward_certified_census_biological_v1 is reuse.ORIGINAL_FORWARD)
            with torch.no_grad():
                reference=fit.raw_prediction_v1(states[0],cert,cert.certificate_sha256,control,torch.tensor([2,3,1,0]),fit_cfg)
                with reuse.installed_fit_forward_v1(adapter):
                    prediction=fit.raw_prediction_v1(states[1],cert,cert.certificate_sha256,control,torch.tensor([2,3,1,0]),fit_cfg)
                for index,(x,y) in enumerate(zip(reference,prediction)):ck(label+"_VAL_raw_prediction_"+str(index),np.array_equal(x,y))
            rows.append({"dtype":str(dtype),"control":control,"two_AdamW_steps":True,"all_output_loss_gradient_optimizer_and_VAL_arrays_bitwise_equal":True})
        ck(str(dtype)+"_one_forest_binding",adapter.bindings==1)
        ck(str(dtype)+"_remaining_forward_calls_reuse",adapter.reuses==adapter.calls-1)
    eq("certificate_unchanged_after_handmade_steps",bank.fingerprint_sha256_v1(bank._content(cert)),original_certificate_hash)
    full=fixture.prepare(full=True);helper=reuse.CertifiedForestReuseV1()
    ck("handmade_full280_roster",len(full.native_trees[0].sample_ids)==280)
    ck("handmade_full559_nodes",len(full.inputs[0].parent)==559)
    ck("handmade_full_actual120_scored",int(full.bank.partitions[0].scored_query_mask.sum())==120)
    for control in reuse.CONTROLS:
        m=model(torch.float32)
        kw={"group_permutation":torch.randperm(len(full.bank.query_ids),generator=torch.Generator().manual_seed(48))} if control=="permuted_groups" else {}
        with torch.no_grad():
            a=bank.forward_certified_census_biological_v1(m,full,expected_certificate_sha256=full.certificate_sha256,control=control,readout_chunk_size=3,**kw)
            b=helper(m,full,expected_certificate_sha256=full.certificate_sha256,control=control,readout_chunk_size=3,**kw)
        compare_outputs("full280_"+control,a,b)
    ck("full280_one_binding_four_reuses",helper.bindings==1 and helper.reuses==4)

    # Count retained input validators and forest builds, rather than assuming
    # reuse silently skips original representation checks.
    builds=[0];validations=[0]
    original_builder=tree.prepare_retained_observed_forest_v1;original_check=tree._checked_inputs
    def counted_builder(*a,**k):builds[0]+=1;return original_builder(*a,**k)
    def counted_check(*a,**k):validations[0]+=1;return original_check(*a,**k)
    tree.prepare_retained_observed_forest_v1=counted_builder;tree._checked_inputs=counted_check
    try:
        h=reuse.CertifiedForestReuseV1();m=model()
        for c in reuse.CONTROLS:
            with torch.no_grad():call(h,m,cert,c)
        ck("one_original_forest_build_for_five_calls",builds[0]==1)
        ck("retained_input_checks_every_graph_every_call",validations[0]==len(cert.inputs)*5)
    finally:
        tree.prepare_retained_observed_forest_v1=original_builder;tree._checked_inputs=original_check

    mutation_cases={
        "native_R_date":lambda c:c.native_trees[0].node_times.__setitem__(c.native_trees[0].sample_nodes[-1],c.native_trees[0].node_times[c.native_trees[0].sample_nodes[-1]]+1),
        "donor_weights":lambda c:c.bank.donor_weights.__setitem__((0,0,0,0),c.bank.donor_weights[0,0,0,0]+.1),
        "F47":lambda c:c.inputs[0].features.__setitem__((0,8),c.inputs[0].features[0,8]+.1),
        "forest_node":lambda c:c.forest.node.__setitem__((0,8),c.forest.node[0,8]+.1),
        "forest_edge":lambda c:c.forest.edge_time.__setitem__((0,0),c.forest.edge_time[0,0]+.1),
        "forest_parent":lambda c:c.forest.parent.__setitem__(0,0),
        "forest_child_count":lambda c:c.forest.child_count.__setitem__(0,c.forest.child_count[0]+1),
        "forest_up_members":lambda c:c.forest.upward_levels[0].__setitem__(0,1),
        "forest_down_members":lambda c:c.forest.downward_levels[0].__setitem__(0,1),
        "source_SHA":lambda c:c.provenance["source_bindings"]["native_observed_ts"].__setitem__("sha256","f"*64),
        "physical_grid":lambda c:c.provenance["original_grid_indexes"].__setitem__(0,1),
        "cut_membership":lambda c:c.bank.group_index.__setitem__((0,0),3),
    }
    for name,mutate in mutation_cases.items():
        c=copy.deepcopy(cert);h=reuse.CertifiedForestReuseV1();m=model()
        with torch.no_grad():call(h,m,c)
        mutate(c)
        rejects("cached_stale_"+name,lambda:call(h,m,c),"Changed/stale")
        ck("mutation_"+name+"_no_forward_override","forward" not in m.encoder.__dict__)
    for name in ("forest_node","forest_edge","forest_parent","forest_child_count","forest_up_members","forest_down_members"):
        c=copy.deepcopy(cert);mutation_cases[name](c);c=certify(c)
        rejects("fresh_recertified_incoherent_"+name,lambda:call(reuse.CertifiedForestReuseV1(),model(),c),"differs from original")
    c=copy.deepcopy(cert);h=reuse.CertifiedForestReuseV1();m=model()
    with torch.no_grad():call(h,m,c)
    c.provenance["tile_id"]=1;object.__setattr__(c,"certificate_sha256",bank.fingerprint_sha256_v1(bank._content(c)))
    rejects("same_object_rebound_new_hash",lambda:call(h,m,c),"hash changed")
    c=copy.deepcopy(cert);c=replace(c,forest=replace(c.forest,node=c.forest.node.double()));c=certify(c)
    rejects("recertified_forest_dtype_mismatch",lambda:call(reuse.CertifiedForestReuseV1(),model(),c),"differs from original")
    c=copy.deepcopy(cert);inp=replace(c.inputs[0],upward_order=tuple(reversed(c.inputs[0].upward_order)));c=certify(replace(c,inputs=(inp,*c.inputs[1:])))
    rejects("recertified_invalid_sweep_keeps_original_check",lambda:call(reuse.CertifiedForestReuseV1(),model(),c),"Tree sweeps")
    c=copy.deepcopy(cert);c.bank.partitions[0].query_ages[0]+=1.;c=certify(c)
    rejects("recertified_query_role_age_keeps_GNN_check",lambda:call(reuse.CertifiedForestReuseV1(),model(),c),"queries/native ages")
    m=model();m.encoder.time_scale=9999.
    rejects("changed_time_scale",lambda:call(reuse.CertifiedForestReuseV1(),m,cert),"time_scale")
    m=model();m.encoder.forward=lambda x:None
    rejects("preexisting_encoder_override",lambda:call(reuse.CertifiedForestReuseV1(),m,cert),"overridden")
    ck("preexisting_override_untouched","forward" in m.encoder.__dict__)
    m=model();m.forward=lambda *a,**k:None
    rejects("preexisting_model_override",lambda:call(reuse.CertifiedForestReuseV1(),m,cert),"Original Census model")
    for control,permutation in (("proposed",torch.tensor([2,3,1,0])),("permuted_groups",None)):
        rejects("permutation_gate_"+control,lambda control=control,permutation=permutation:reuse.CertifiedForestReuseV1()(model(),cert,expected_certificate_sha256=cert.certificate_sha256,control=control,group_permutation=permutation))
    rejects("unknown_control_original_reject",lambda:call(reuse.CertifiedForestReuseV1(),model(),cert,"unknown"),"Unknown H021")
    for target in ("model","encoder"):
        m=model();h=reuse.CertifiedForestReuseV1()
        def fail(*args):raise RuntimeError("handmade hook failure")
        handle=getattr(m,target,None).register_forward_pre_hook(fail) if target=="encoder" else m.register_forward_pre_hook(fail)
        try:rejects(target+"_exception",lambda:call(h,m,cert),"handmade hook failure")
        finally:handle.remove()
        ck(target+"_exception_method_restored","forward" not in m.encoder.__dict__ and not h._routing)
        with torch.no_grad():call(h,m,cert)
        ck(target+"_success_after_exception")
    m=model();h=reuse.CertifiedForestReuseV1()
    handle=m.register_forward_pre_hook(lambda module,args:(list(args[0]),args[1]))
    try:rejects("route_exact_tuple_identity",lambda:call(h,m,cert),"exact certified original")
    finally:handle.remove()
    ck("tuple_identity_failure_restores_encoder","forward" not in m.encoder.__dict__)
    h=reuse.CertifiedForestReuseV1()
    with reuse.installed_fit_forward_v1(h):
        rejects("nested_fit_install",lambda:reuse.installed_fit_forward_v1(h).__enter__(),"already replaced")
    ck("fit_helper_after_nested_rejection",fit.forward_certified_census_biological_v1 is reuse.ORIGINAL_FORWARD)
    try:
        with reuse.installed_fit_forward_v1(h):raise RuntimeError("handmade scope failure")
    except RuntimeError:pass
    ck("fit_helper_exception_restoration",fit.forward_certified_census_biological_v1 is reuse.ORIGINAL_FORWARD)
    temporary=copy.deepcopy(cert);r=weakref.ref(temporary);h=reuse.CertifiedForestReuseV1()
    with torch.no_grad():call(h,model(),temporary)
    del temporary;gc.collect()
    ck("weak_binding_does_not_retain_certificate",r() is None and h._certificate() is None)

    # Fake metadata reader and fit delegate prove gates/entry/restoration only.
    # No original run_fit body or actual capability is executed.
    cfg=reuse.load_config_v1();source_sha=digest(ROOT/reuse.SOURCE_PATH)
    base_ref={"path":"handmade/base","sha256":"b"*64}
    review_ref={"path":"handmade/review","sha256":"c"*64}
    source_ref={"path":reuse.SOURCE_PATH,"sha256":source_sha}
    config_ref={"path":reuse.CONFIG_PATH,"sha256":reuse.CONFIG_SHA256}
    cap_ref={"path":"handmade/reuse","sha256":"d"*64}
    now=datetime.now(timezone.utc)
    admission={"schema":"census_certified_forest_reuse_admission_v1","status":"active","fit_authorized":True,"observed_profile_authorized":True,"not_before_utc":(now-timedelta(minutes=1)).isoformat(),"expires_at_utc":(now+timedelta(minutes=10)).isoformat(),"base_admission_ref":base_ref,"source_ref":source_ref,"config_ref":config_ref,"review_ref":review_ref}
    review={"status":"PASS","actual_admission":False,"artifact_sha256":{reuse.SOURCE_PATH:source_sha,reuse.CONFIG_PATH:reuse.CONFIG_SHA256}}
    request_log=[]
    class FakeReader:
        def decode_json(self,ref):
            request_log.append(("decode",dict(ref)))
            if ref==cap_ref:return copy.deepcopy(admission)
            if ref==review_ref:return copy.deepcopy(review)
            if ref==config_ref:return copy.deepcopy(cfg)
            raise ValueError("unknown handmade JSON reference")
        def read(self,ref):
            request_log.append(("read",dict(ref)))
            if ref==source_ref:return (ROOT/reuse.SOURCE_PATH).read_bytes()
            if ref==base_ref:return b"handmade inactive original gate sentinel"
            raise ValueError("unknown handmade byte reference")
    real_reader=fit.VerifiedReaderV1;real_runner=fit.run_fit_v1;delegations=[]
    fit.VerifiedReaderV1=FakeReader
    def fake_run(ref):
        ck("delegate_uses_adapter",type(fit.forward_certified_census_biological_v1) is reuse.CertifiedForestReuseV1)
        ck("delegate_receives_unchanged_original_ref",ref==base_ref)
        delegations.append(dict(ref));return {"status":"handmade_stub_only","sentinel":519}
    fit.run_fit_v1=fake_run
    try:
        result=reuse.run_original_fit_v1(base_ref,cap_ref)
        ck("original_stub_receipt_unmodified",result["original_fit_receipt"]=={"status":"handmade_stub_only","sentinel":519})
        ck("fake_metadata_gate_reads_base_ref",("read",base_ref) in request_log)
        ck("no_original_fit_called",len(delegations)==1)
        ck("delegate_restore",fit.forward_certified_census_biological_v1 is reuse.ORIGINAL_FORWARD)
        mutations=[
            ("inactive","status","inactive"),("no_fit","fit_authorized",False),("expired","expires_at_utc",(now-timedelta(minutes=1)).isoformat()),("not_started","not_before_utc",(now+timedelta(minutes=1)).isoformat()),("naive_date","expires_at_utc","2099-01-01T00:00:00"),("wrong_base","base_admission_ref",{"path":"handmade/other","sha256":"e"*64}),("wrong_config","config_ref",{"path":reuse.CONFIG_PATH,"sha256":"e"*64}),("wrong_source","source_ref",{"path":"handmade/other_source","sha256":"e"*64})]
        for name,key,value in mutations:
            old=admission[key];admission[key]=value
            try:rejects("admission_"+name,lambda:reuse.run_original_fit_v1(base_ref,cap_ref))
            finally:admission[key]=old
        review["status"]="REJECT";rejects("fresh_exact_review_required",lambda:reuse.run_original_fit_v1(base_ref,cap_ref));review["status"]="PASS"
        review["artifact_sha256"][reuse.SOURCE_PATH]="e"*64;rejects("stale_source_review_reject",lambda:reuse.run_original_fit_v1(base_ref,cap_ref));review["artifact_sha256"][reuse.SOURCE_PATH]=source_sha
        ck("rejected_gates_never_delegate",len(delegations)==1)
        profile_model=model();state_before={n:p.detach().clone() for n,p in profile_model.named_parameters()}
        rng=torch.get_rng_state().clone()
        timing=reuse.observed_timing_v1(profile_model,lambda:cert,expected_certificate_sha256=cert.certificate_sha256,base_admission_ref=base_ref,reuse_admission_ref=cap_ref,readout_chunk_size=3)
        ck("observed_timing_all_phases",all(k in timing for k in ("preparation_seconds","phase_seconds","synthetic_backward_seconds")))
        ck("observed_timing_no_fit_or_speed_claim",timing["TRAIN_loss_or_fit"] is False and timing["speedup_claim"] is False)
        ck("observed_timing_synchronized_semantics",timing["timing_semantics"]=="synchronized future profile phases")
        ck("observed_profile_restores_all_None_gradients",all(p.grad is None for p in profile_model.parameters()))
        for n,p in profile_model.named_parameters():eq("observed_profile_no_optimizer_change_"+n,p,state_before[n])
        eq("observed_profile_no_RNG_change",rng,torch.get_rng_state())
        rejects("observed_profile_external_expected_SHA_required",lambda:reuse.observed_timing_v1(model(),lambda:cert,expected_certificate_sha256="e"*64,base_admission_ref=base_ref,reuse_admission_ref=cap_ref),"Changed/stale")
        profile_model.decoder.weight.grad=torch.ones_like(profile_model.decoder.weight)
        rejects("observed_profile_rejects_existing_gradients",lambda:reuse.observed_timing_v1(profile_model,lambda:cert,expected_certificate_sha256=cert.certificate_sha256,base_admission_ref=base_ref,reuse_admission_ref=cap_ref),"initially absent")
        admission["observed_profile_authorized"]=False
        rejects("profile_fresh_capability_required",lambda:reuse.observed_timing_v1(model(),lambda:cert,expected_certificate_sha256=cert.certificate_sha256,base_admission_ref=base_ref,reuse_admission_ref=cap_ref))
        ck("profile_never_delegated_to_fit",len(delegations)==1)
    finally:
        fit.VerifiedReaderV1=real_reader;fit.run_fit_v1=real_runner

    ck("all_original_module_sources_unchanged",all(digest(ROOT/p)==h for p,h in baseline["original_sha256"].items()))
    ck("original_handmade_fixture_source_unchanged",digest(ROOT/baseline["handmade_fixture_source_ref"]["path"])==baseline["handmade_fixture_source_ref"]["sha256"])
    ck("no_forbidden_actual_access_or_jobs",forbidden==[])
    report.update(status="PASS",checks=checks,checks_passed=len(checks),parity_rows=rows,forbidden_access_events=forbidden,source_sha256=digest(ROOT/reuse.SOURCE_PATH),config_sha256=digest(ROOT/reuse.CONFIG_PATH),elapsed_seconds=time.monotonic()-started,observed_timing_CPU_fixture_only=timing,actual_speedup_claim=False,original_fit_body_executed=False)
except Exception as e:
    report.update(status="FAIL",checks=checks,checks_passed=len(checks),error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),forbidden_access_events=forbidden,elapsed_seconds=time.monotonic()-started,source_sha256=digest(ROOT/reuse.SOURCE_PATH))
    raise
finally:
    (OUT/"handmade_checks_attempt04.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:report[k] for k in ("status","checks_passed","elapsed_seconds") if k in report}),flush=True)
