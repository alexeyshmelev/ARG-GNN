"""End-to-end ROOT preparation and dual-scope checks with virtual files and fake guards only."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import ast,copy,hashlib,importlib.util,json,sys,types,traceback,subprocess
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");HERE="research/reviews/native_VAL000_full_state_resume_source_20261016_v12"
checks=[]
def require(ok,message):
    if not ok:raise ValueError(message)
def good(name,ok=True):
    assert ok,name
    checks.append(dict(name=name,status="PASS"))
def rejects(name,fn):
    try:fn()
    except (ValueError,KeyError,FileNotFoundError):checks.append(dict(name=name,status="PASS_REJECTED"));return
    raise AssertionError("Invalid fixture accepted: "+name)
def import_source(path,name):
    sp=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
def run():
    sys.path.insert(0,str(ROOT))
    c=import_source(HERE+"/amendment_contract_v1.py","_native11_fake_integration_contract")
    pp=import_source(HERE+"/prepare_continuation_root_v12.py","_native11_fake_integration_ROOT_prep")
    good("inert_source_import_no_scientific_runtime",not any(n in sys.modules for n in ("torch","composer","numpy","tskit")))
    originals=copy.deepcopy(c.ORIGINAL_CAPS);fresh=copy.deepcopy(c.CAPS)
    good("only_fresh_reservation_and_retained_checkpoint_subtraction",fresh==dict(originals,reservation_bytes=5*2**30,scientific_output_bytes=originals["scientific_output_bytes"]-2*2**30))
    good("exact5GiB_analytic_full_envelope",4*2**30+134283264+fresh["auxiliary_bytes"]+fresh["directory_and_accounting_allowance_bytes"]==5*2**30 and fresh["auxiliary_bytes"]==768*2**20)
    for n in ("max_seconds","max_cpu_rss_bytes","max_cuda_reserved_bytes","cleanup_seconds","work_cutoff_margin_seconds","terminal_evidence_margin_seconds","log_bytes","max_checkpoint_files","max_checkpoint_bytes"):
        good("original_"+n+"_unchanged",fresh[n]==originals[n])
    component=json.loads((ROOT/c.LIVENESS_COMPONENT_MANIFEST_REF["path"]).read_text())
    current_direct=ast.parse((ROOT/c.DIRECT).read_text())
    frozen_direct=ast.parse((ROOT/component["direct_source_ref"]["path"]).read_text())
    imports=[n for n in current_direct.body if isinstance(n,ast.ImportFrom) and n.module==HERE.replace("/",".")+".amendment_contract_v1"]
    good("exactly_one_direct_contract_import",len(imports)==1)
    imports[0].module="research.reviews.native_VAL000_full_state_resume_source_20261016_v11.amendment_contract_v1"
    good("sealed_liveness_direct_AST_only_namespace",ast.dump(current_direct,include_attributes=False)==ast.dump(frozen_direct,include_attributes=False))
    good("real_liveness_component_validation",c.validate_liveness_component_v11(dict(liveness_component_manifest_ref=c.LIVENESS_COMPONENT_MANIFEST_REF)) is None)
    nt=ast.parse((ROOT/c.SOURCE).read_text())
    fn=next(n for n in nt.body if isinstance(n,ast.FunctionDef) and n.name=="terminal_snapshot_authority_v11")
    good("sealed_liveness_authority_AST_unchanged",hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest()==component["terminal_authority_AST_sha256"])
    for n in ("execution_only_v1.py","corpus_metadata_bridge_v1.py","resume_scientific_bridge_v5.py","alias_namespace_v7.py","fresh_entry_v9.py"):
        expected=(ROOT/"research/reviews/native_VAL000_full_state_resume_source_20261016_v10"/n).read_text().replace("native_VAL000_full_state_resume_source_20261016_v10","native_VAL000_full_state_resume_source_20261016_v12")
        good(n+"_only_namespace",expected==(ROOT/HERE/n).read_text())
    # VFS is in RAM; all actual source imports/readers above were inert metadata/source only.
    current=json.loads((ROOT/"research/reviews/C3_saved600_artifact_acceptance_source_20261016_v1/source_manifest.v1.json").read_text())["current_guard_roles"]
    known={r["path"]:r for r in current.values()};docs={};files={};dirs=set();events=[]
    def fref(path):
        path=str(path);path=path[len(str(ROOT))+1:] if path.startswith(str(ROOT)+"/") else path
        if path in known:return copy.deepcopy(known[path])
        return dict(path=path,sha256=hashlib.sha256(("VIRTUAL:"+path).encode()).hexdigest())
    def fread(record):return copy.deepcopy(docs[record["path"]])
    class VPath:
        def __init__(self,key):
            text=str(key);self.key=text[len(str(ROOT))+1:] if text.startswith(str(ROOT)+"/") else text
        def __str__(self):return str(ROOT/self.key)
        def __truediv__(self,name):return VPath(self.key+"/"+name)
        def exists(self):return self.key in dirs or self.key in files
        def is_dir(self):return self.key in dirs
        def is_file(self):return self.key in files
        def is_symlink(self):return False
        def mkdir(self,*a,**k):
            require(not self.exists(),"Fresh virtual directory")
            dirs.add(self.key)
        def is_relative_to(self,other):
            return PurePosixPath(self.key).is_relative_to(PurePosixPath(other.key))
        def iterdir(self):
            prefix=self.key+"/"
            return [VPath(k) for k in files if k.startswith(prefix) and "/" not in k[len(prefix):]]
        def lstat(self):
            if self.key not in files:raise FileNotFoundError(self.key)
            return types.SimpleNamespace(st_size=len(files[self.key]))
    def save(directory,name,value):
        key=directory.key+"/"+name;require(key not in docs,"Immutable virtual metadata")
        docs[key]=copy.deepcopy(value);files[key]=(json.dumps(value,sort_keys=True,indent=2)+"\n").encode()
        return fref(key)
    old=json.loads((ROOT/c.ORIGINAL_INPUT["path"]).read_text())
    oldq=json.loads((ROOT/old["root_stage_request_ref"]["path"]).read_text())
    oldstorage=json.loads((ROOT/old["refs"]["storage"]["path"]).read_text())
    audit=json.loads((ROOT/c.V11_SAVED22000_AUDIT_REF["path"]).read_text())
    outer=json.loads((ROOT/c.V10_OUTER_TERMINAL_REF["path"]).read_text())
    template=json.loads((ROOT/HERE/"ROOT_amendment.INACTIVE.template.v12.json").read_text())
    mref=fref(c.MANIFEST);review_ref=fref("VIRTUAL/review.json");component_review=fref("VIRTUAL/component_review.json")
    manifest=dict(source_only=True,new_executed_source_refs=[fref(HERE+"/"+n) for n in ("__init__.py","amendment_contract_v1.py","direct_gpu0_v1.py","execution_only_v1.py","corpus_metadata_bridge_v1.py","resume_scientific_bridge_v5.py","alias_namespace_v7.py","fresh_entry_v9.py")])
    docs.update({c.MANIFEST:manifest,review_ref["path"]:dict(verdict="PASS_SOURCE_ONLY",candidate_manifest_ref=mref),
        component_review["path"]:dict(verdict="PASS_SOURCE_ONLY",candidate_manifest_ref=c.LIVENESS_COMPONENT_MANIFEST_REF),
        c.ORIGINAL_INPUT["path"]:old,old["root_stage_request_ref"]["path"]:oldq,old["refs"]["storage"]["path"]:oldstorage,
        c.V11_SAVED22000_AUDIT_REF["path"]:audit,c.V10_OUTER_TERMINAL_REF["path"]:outer,
        HERE+"/ROOT_amendment.INACTIVE.template.v12.json":template})
    c.ref=fref;c.read=fread;c.raw=lambda r:None;c.local=lambda p:VPath(p);c.save=save;c.active=lambda a:None
    class OldGuard:
        def validate_storage_v9(self,s,cfg,storage,**kwargs):
            require(s==old and storage==oldstorage and cfg["output_cap_bytes"]==7*2**30,"Original exact7GiB retained guard")
            events.append(("old_validate",kwargs))
        def budget_check_v9(self,storage,**kwargs):
            require(storage==oldstorage and kwargs.get("required_write_bytes",0)==0,"No new writes charged to old scope")
            events.append(("old_check",kwargs));return {}
    oldguard=OldGuard()
    c.original.guard_v1=lambda *a,**k:oldguard
    class FreshGuard:
        MODE="native14_stopped_retention_v3"
        budget=types.SimpleNamespace(SOURCE_REL="scripts/project_budget_v9.py",LAUNCHER_REL="scripts/launch_reserved_gpu_v6.py")
        def inspect_v1(self,**kw):
            events.append(("fresh_inspect",kw));return dict(projected_bytes=970000000000,bindings=copy.deepcopy(current))
        def _pins(self,v):return dict(actual_test_prefix=True)
        def consumer_bindings_v1(self,*groups):
            out={}
            for group in groups:
                for r in group:
                    require(r["path"] not in out or out[r["path"]]==r,"Role conflict")
                    out[r["path"]]=r
            require(len(out)<=64,"All roles<=64");return list(out.values())
        def operation_draft_v1(self,op,value,**kw):
            require(op=="reserve" and value["reserved_bytes"] in c.ACCOUNTING_CHOICES_V11,"Fresh5GiB only")
            return dict(request=copy.deepcopy(value),expected_before=kw["expected_before"])
        def apply_reserve_v1(self,record,**kw):
            request=fread(record)["request"]
            self.own=copy.deepcopy(request)
            events.append(("FAKE_ONLY_RESERVE",request["reserved_bytes"]))
            return dict(value=request,initial_ledger_pins=dict(unchanged_returned_prefix="FAKE"))
        def validate_storage_v9(self,s,cfg,storage,*,source_ref,config_ref,runtime_ref):
            require(storage["reservation"]==self.own and storage["initial_ledger_pins"]==dict(unchanged_returned_prefix="FAKE")
                and s["job_id"]==storage["job_id"] and s["output_directory"]==storage["output_directory"]
                and 0<storage["reserved_new_bytes"]<=cfg["output_cap_bytes"]==5*2**30
                and all(r in self.own["consumer_bindings"] for r in (source_ref,config_ref,runtime_ref)),
                "Actual original fresh guard bindings/cap invariant on fake state")
            events.append(("fresh_validate",cfg["output_cap_bytes"]))
        def budget_check_v9(self,storage,**kw):
            require(storage["reservation"]==self.own,"Actual fresh reservation")
            events.append(("fresh_check",kw));return {}
    g=FreshGuard();c.fresh_guard_runtime_v11=lambda **kw:g
    c._consumer_guards_v12=lambda:(oldguard,g)
    def set_amend(record):c.AMENDMENT_REF=record
    c.set_amendment_v1=set_amend
    c.active_amendment_v1=lambda:(fread(c.AMENDMENT_REF),old,oldq)
    def validate_original(record):
        require(record==c.ORIGINAL_INPUT,"Original input remains immutable retained source")
        c.ORIGINAL_STATE=(copy.deepcopy(old),copy.deepcopy(oldq),copy.deepcopy(oldstorage))
        c.RESUME_STATE=dict(actual_saved_progress=dict(batch=22000),checkpoint_ref=c.V11_RESUME_CHECKPOINT_REF)
        return c.fresh_continuation_input_v11(fread(c.AMENDMENT_REF),old,oldq,dict(max_checkpoint_bytes=2*2**30),oldstorage,{})
    c.validate_input_v1=validate_original
    pp.load_contract=lambda:c;pp.ref=fref;pp.read=fread
    args=types.SimpleNamespace(root_directive=pp.DIRECTIVE,instance="IN_MEMORY_TEST_ONLY",review_path=review_ref["path"],review_sha256=review_ref["sha256"],component_review_path=component_review["path"],component_review_sha256=component_review["sha256"])
    out=pp.prepare_v11(args)
    good("full_ROOT_prepare_path_with_fake_storage_only",out["status"]=="FRESH_NATIVE_CONTINUATION_ADMITTED_NOT_EXECUTED" and out["execution_command_emitted_not_invoked"] is True)
    good("all_actual37_and_complete_needed_roles",1<=out["consumer_binding_count"]<=64 and all(r in g.own["consumer_bindings"] for r in current.values()))
    good("initial_ledger_pins_never_rebased",c.FRESH_STATE[2]["initial_ledger_pins"]==dict(unchanged_returned_prefix="FAKE"))
    good("old7GiB_guard_and_fresh5GiB_guard_both_validate",any(k=="old_validate" for k,v in events) and any(k=="fresh_validate" for k,v in events))
    spec,req,storage=copy.deepcopy(c.FRESH_STATE);amend=fread(c.AMENDMENT_REF)
    good("fresh_native_storage_job_and_original_attachment_identity_distinct",spec["job_id"]!=oldq["job_id"] and c.fresh_continuation_input_v11(amend,old,oldq,{},oldstorage,{})[1]["job_id"]==oldq["job_id"])
    baseline_docs=copy.deepcopy(docs)
    def validate_fresh():return c.fresh_continuation_input_v11(amend,old,oldq,{},oldstorage,{})
    for field,value in [("reserved_new_bytes",4*2**30),("completed_overlay_policy_ref",fref("wrongpolicy")),("output_directory","wrong")]:
        docs=copy.deepcopy(baseline_docs);docs[spec["refs"]["storage"]["path"]][field]=value
        rejects("fresh_storage_"+field,validate_fresh)
    for label,fn in [
        ("aux_scope_extra",lambda st:st["reservation"]["auxiliary_scopes"].append("EXTRA")),
        ("required_role_missing",lambda st:st["reservation"]["consumer_bindings"].remove(fref(c.ACTUAL_DRIVER))),
        ("65roles",lambda st:st["reservation"].update(consumer_bindings=[fref("X"+str(i)) for i in range(65)]))]:
        docs=copy.deepcopy(baseline_docs);fn(docs[spec["refs"]["storage"]["path"]])
        rejects(label,validate_fresh)
    for field,value in [("resume_checkpoint_ref",c.RESUME_CHECKPOINT_REF),("expires_at_utc","2099-01-01T00:00:00+00:00"),("independent_source_review_ref",fref("wrongreview")),("fresh_caps",originals)]:
        docs=copy.deepcopy(baseline_docs);docs[spec["root_stage_request_ref"]["path"]][field]=value
        rejects("fresh_request_"+field,validate_fresh)
    docs=copy.deepcopy(baseline_docs)
    c.fresh_continuation_input_v11(amend,old,oldq,{},oldstorage,{})
    # Complete scope fixture keeps both old checkpoint files, full old charge and new2-slot writer.
    oldout=old["output_directory"];oldaux=old["worker_auxiliary_scope"];newout=spec["output_directory"];newaux=spec["worker_auxiliary_scope"]
    def row(parts,size,kind="regular"):return (parts,size+4096,kind)
    rosters={oldout:[row(("same_charge_owned_roster_v4","checkpoints","ep0-ba6500-rank0.pt"),2017935360),row(("same_charge_full_state_resume_v10","checkpoints","ep0-ba22000-rank0.pt"),2017947392)],
        oldaux:[row(("meta.json",),100000)],newout:[],newaux:[]}
    baseline_rosters=copy.deepcopy(rosters)
    c.original._sizes=lambda p,**kw:copy.deepcopy(rosters[p.key])
    totals=c.charged_census_v1(spec)
    good("both_retained_checkpoints_full_old_scope_and_fresh_empty_scope",set(totals)=={oldout,oldaux,newout,newaux})
    rosters[newout]=[row(("checkpoints","ep0-ba22500-rank0.pt"),2*2**30),row(("checkpoints","ep0-ba23000-rank0.pt"),2*2**30)]
    rosters[newaux]=[row(("tmp",),fresh["auxiliary_bytes"]-4096)]
    good("two_full2GiB_products_and_full768MiB_AUX_fit5GiB",c.charged_census_v1(spec)[newaux]==fresh["auxiliary_bytes"])
    rosters[newout].append(row(("checkpoints","extra.pt"),1))
    rejects("third_new_checkpoint_rejected",lambda:c.charged_census_v1(spec))
    rosters=copy.deepcopy(baseline_rosters);rosters[newout]=[row(("checkpoints","oversize.pt"),2*2**30+1)]
    rejects("unchanged2GiB_file_hard_bound",lambda:c.charged_census_v1(spec))
    rosters=copy.deepcopy(baseline_rosters);rosters[newaux]=[row(("tmp",),fresh["auxiliary_bytes"]+1)]
    rejects("unchanged768MiB_AUX_limit",lambda:c.charged_census_v1(spec))
    rosters=copy.deepcopy(baseline_rosters);rosters[oldout].append(row(("extra",),7*2**30))
    rejects("old_scope_not_excluded_from_original7GiB",lambda:c.charged_census_v1(spec))
    rosters=copy.deepcopy(baseline_rosters);rosters[newout]=[row(("metadata",),134283264)]
    rejects("prospective_both2GiB_products_and_metadata_prewrite_guard",lambda:c.charged_census_v1(spec,required_write_bytes=1))
    rosters=copy.deepcopy(baseline_rosters)
    bridge=c.guard_v1({})
    bridge.budget_check_v9(storage,boundary=True,required_write_bytes=123)
    good("old_check_has_zero_growth_fresh_check_has_actual_growth",events[-2:]==[("old_check",dict(boundary=True,required_write_bytes=0)),("fresh_check",dict(boundary=True,required_write_bytes=123))])
    c.alias_module_v7=lambda:types.SimpleNamespace()
    c.alias_strict_boundary_v7=lambda *a,**k:dict(fake_strict_boundary=True)
    good("terminal_alias_route_accepts_only_actual_fresh_scope_after_drain",c.normalize_resumed_alias_v7(spec,{},bridge,storage,prove_drained=lambda:True,minimal_wait_facts={})["alias_present"] is False)
    rejects("terminal_alias_no_undrained_route",lambda:c.normalize_resumed_alias_v7(spec,{},bridge,storage,prove_drained=lambda:False,minimal_wait_facts={}))
    wrong=copy.deepcopy(spec);wrong["output_directory"]=old["output_directory"]
    rejects("terminal_alias_never_targets_old_scope",lambda:c.normalize_resumed_alias_v7(wrong,{},bridge,storage,prove_drained=lambda:True,minimal_wait_facts={}))
    good("all_fake_ROOT_packet_metadata_within512KiB",sum(len(v)+8192 for v in files.values())<=512*1024)
    args.instance="IN_MEMORY_DOWNWARD_ONLY";args.reservation_bytes=17*2**28
    down=pp.prepare_v11(args)
    downspec,downreq,downstorage=copy.deepcopy(c.FRESH_STATE)
    good("full_ROOT_explicit17quarterGiB_accounting_path",down["reserved_bytes"]==17*2**28 and downstorage["reserved_new_bytes"]==17*2**28 and downreq["fresh_caps"]==fresh and downreq["reservation_only_downward_failstop_authorized"] is True)
    downout=downspec["output_directory"];downaux=downspec["worker_auxiliary_scope"]
    rosters=copy.deepcopy(baseline_rosters);rosters[downout]=[row(("checkpoints","a.pt"),2*2**30),row(("checkpoints","b.pt"),2*2**30)];rosters[downaux]=[row(("cache",),128*2**20)]
    good("selected_lower_cap_preserves_two2GiB_bounds_and_actual128MiB_AUX",c.charged_census_v1(downspec)[downaux]>128*2**20 and c.CAPS["auxiliary_bytes"]==768*2**20)
    rosters[downaux]=[row(("cache",),300*2**20)]
    rejects("lower_reservation_rejects_growth_even_below_original768MiB_AUX",lambda:c.charged_census_v1(downspec))
    bad=copy.deepcopy(fread(c.AMENDMENT_REF));bad["fresh_accounting_reservation_bytes"]=5*2**30
    rejects("selected_accounting_label_mismatch",lambda:c.fresh_continuation_input_v11(bad,old,oldq,{},oldstorage,{}))
    bad=copy.deepcopy(fread(c.AMENDMENT_REF));bad["reservation_only_downward_failstop_authorized"]=False
    rejects("lower_reservation_requires_explicit_ROOT_failstop",lambda:c.fresh_continuation_input_v11(bad,old,oldq,{},oldstorage,{}))
    args.instance="IN_MEMORY_WRONG_CAP";args.reservation_bytes=4*2**30
    rejects("unreviewed_accounting_choice_rejected",lambda:pp.prepare_v11(args))
    good("two_separate_virtual_ROOT_packets_each512KiB",all(sum(len(v)+8192 for k,v in files.items() if k.startswith(packet+"/"))<=512*1024 for packet in dirs))

    for script in ("prepare_continuation_root_v12.py","execution_only_v1.py"):
        for args2,rc in [([],0),(["--help"],0)]:
            p=subprocess.run([sys.executable,"-B","-s","-m",HERE.replace("/",".")+"."+script[:-3],*args2],cwd=ROOT,capture_output=True,text=True,timeout=10)
            good(script+"_inert_"+str(args2),p.returncode==rc)
    good("no_actual_model_or_GPU_import",not any(n in sys.modules for n in ("torch","composer","numpy","tskit")))
    return dict(schema="native12_fresh5GiB_dual_scope_ROOT_inert_integration_v1",status="PASS_SOFTWARE_ONLY",checks=checks,check_count=len(checks),prospective_role_union=out["consumer_binding_count"],limit=64,actual_admission_or_mutation=False,actual_execution_or_probe_or_payload=False,all_file_mutations_in_RAM_virtual_FS=True,scientific_and_liveness_invariants_preserved=True,finished_utc=datetime.now(timezone.utc).isoformat())
def save(name,o):
    with (ROOT/HERE/name).open("x") as f:json.dump(o,f,sort_keys=True,indent=2);f.write("\n")
if __name__=="__main__":
    try:
        v=run();save("fresh_scope_integration_inert.v12.json",v);print(json.dumps(dict(status=v["status"],checks=v["check_count"],roles=v["prospective_role_union"]),indent=2))
    except BaseException as e:save("fresh_scope_integration_inert.attempt5.FAILURE.json",dict(error=repr(e),traceback=traceback.format_exc(),checks=checks));raise
