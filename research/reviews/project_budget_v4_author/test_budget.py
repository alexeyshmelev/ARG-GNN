"""Handmade metadata and tiny opaque byte files only."""
from pathlib import Path
import copy, fcntl, hashlib, importlib.util, json, os
import pytest
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
spec=importlib.util.spec_from_file_location("budget_handmade",ROOT/"src/arggnn/project_budget_v4.py")
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+"\n")
def ref(root,p):return {"path":str(p.relative_to(root)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
def setup(tmp_path,monkeypatch):
    root=tmp_path;c=copy.deepcopy(b.FIXED_POLICY);c["root"]=str(root)
    source=root/"src/budget.py";source.parent.mkdir(parents=True);source.write_text("handmade sealed budget source\n")
    human=root/"research/human.json";dump(human,{"kind":"human_project_storage_authorization","project_root":str(root),"authorized_hard_limit_bytes":1000000000000,"future_scheduling_ceiling_bytes":950000000000})
    pool=root/"src/legacy.py";pool.write_text("handmade legacy source\n")
    c["human_authorization"]=ref(root,human);c["legacy_pool_source"]=ref(root,pool)
    config=root/"configs/budget.json";dump(config,c)
    monkeypatch.setattr(b,"ROOT",root);monkeypatch.setattr(b,"SOURCE",source);monkeypatch.setattr(b,"CONFIG",config);monkeypatch.setattr(b,"FIXED_POLICY",c)
    lock=root/b.LOCK;lock.parent.mkdir(parents=True);lock.write_bytes(b"unchanged mutex")
    legacy={"schema":"gpu_pool_storage_ledger_v1","root":str(root),**c["legacy_invariants"],"controllers":{"active":{"pid":111,"reserved_bytes":67108864}},"jobs":{"active":{"output":str(root/"existing/active"),"reserved_bytes":60000000000}}}
    dump(root/b.LEGACY,legacy)
    consumer=root/"src/consumer.py";consumer.write_text("handmade consumer source\n")
    runtime=root/"research/runtime_authority.json";dump(runtime,{"kind":"handmade_runtime_authority"})
    request={"key":"native_argformer_tsinfer_fit_v1/parent","output":str(root/"experiments/results/fit/parent"),"auxiliary_scopes":[str(root/".runtime/tmp/short/parent")],"reserved_bytes":20000000000,"consumer_bindings":[ref(root,consumer),ref(root,runtime)]}
    bindings={"implementation":ref(root,source),"config":ref(root,config),"human_authorization":c["human_authorization"]}
    review=root/"research/independent.json";dump(review,{"kind":"project_budget_v4_independent_review","review_role":"independent_reviewer","verdict":"PASS",**bindings,"actual_apply_performed":False})
    admission={"kind":"project_budget_v4_root_reserve_admission","root_authorizer":"root_agent","active":True,**bindings,"all_budget_writers_use_scheduler_mutex":True,"no_legacy_mutation_authorized":True,"GPU_authorized":False,"scientific_execution_authorized":False,"global_rebase_authorized":False,"independent_review":ref(root,review),"expected_before_ledger_pins":{"legacy_ledger_sha256":ref(root,root/b.LEGACY)["sha256"],"supplemental_ledger_sha256":None},"reservations":[request]}
    authority=root/"research/root_reserve.json";dump(authority,admission)
    return root,c,request,admission,authority,bindings
def reserve(f):
    root,c,request,admission,authority,bindings=f
    return b.reserve(root,ref(root,authority),expected_source_sha256=bindings["implementation"]["sha256"],expected_config_sha256=bindings["config"]["sha256"])
def check(f,reservation,**kwargs):
    root,c,request,admission,authority,bindings=f
    return b.check_reservation(root,reservation,expected_source_sha256=bindings["implementation"]["sha256"],expected_config_sha256=bindings["config"]["sha256"],**kwargs)
def test_default_source_only_and_missing_optin_do_not_read_live_paths(tmp_path,monkeypatch,capsys):
    f=setup(tmp_path,monkeypatch)
    monkeypatch.setattr(b,"reserve",lambda *a,**k:(_ for _ in ()).throw(AssertionError("runtime reached")))
    assert b.main([])==0
    assert json.loads(capsys.readouterr().out)["live_ledger_or_scope_access"] is False
    with pytest.raises(ValueError,match="opt-in"):b.main(["--reserve"])
def test_four20GB_scopes_add_only_supplemental_and_retain_legacy_bytes(tmp_path,monkeypatch):
    f=setup(tmp_path,monkeypatch);root,c,r,a,path,bindings=f
    a["reservations"]=[]
    for i in range(4):
        q=copy.deepcopy(r);q["key"]+=str(i);q["output"]+=str(i);q["auxiliary_scopes"]=[r["auxiliary_scopes"][0]+str(i)];a["reservations"].append(q)
    dump(path,a);before=(root/b.LEGACY).read_bytes();lock=(root/b.LOCK).read_bytes();receipt=reserve(f)
    assert (root/b.LEGACY).read_bytes()==before and (root/b.LOCK).read_bytes()==lock
    assert len(receipt["admitted_reservations"])==4
    value=next(iter(receipt["admitted_reservations"].values()));view=check(f,value,boundary=True)
    assert view["supplemental_reserved_bytes"]==80000000000 and view["projected_bytes"]==437403749376
    assert view["limit_bytes"]==950000000000 and view["hard_limit_bytes"]==1000000000000
    assert view["scope_measurement"]["conservative_scope_bytes"]==0
def test_first_hashes_exact_later_unrelated_legacy_mutation_permitted(tmp_path,monkeypatch):
    f=setup(tmp_path,monkeypatch);root,*_=f;receipt=reserve(f);own=next(iter(receipt["admitted_reservations"].values()))
    first=check(f,own);pins={k:first[k] for k in ["legacy_ledger_sha256","supplemental_ledger_sha256"]}
    assert check(f,own,initial_ledger_pins=pins)["reservation"]==own
    legacy=json.loads((root/b.LEGACY).read_text());legacy["jobs"]["other"]={"output":str(root/"other"),"reserved_bytes":1234};dump(root/b.LEGACY,legacy)
    with pytest.raises(ValueError,match="first-read"):check(f,own,initial_ledger_pins=pins)
    later=check(f,own);assert later["projected_bytes"]==first["projected_bytes"]+1234 and later["reservation"]==own
def test_later_legitimate_supplemental_addition_permitted(tmp_path,monkeypatch):
    f=setup(tmp_path,monkeypatch);root,c,r,a,path,bindings=f;receipt=reserve(f);own=next(iter(receipt["admitted_reservations"].values()));before=check(f,own)
    second=copy.deepcopy(a);second["reservations"]=[copy.deepcopy(r)];q=second["reservations"][0];q["key"]="second";q["output"]=str(root/"second/output");q["auxiliary_scopes"]=[str(root/"second/tmp")];q["reserved_bytes"]=12345
    second["expected_before_ledger_pins"]={k:before[k] for k in ["legacy_ledger_sha256","supplemental_ledger_sha256"]}
    path2=root/"research/root_second.json";dump(path2,second)
    b.reserve(root,ref(root,path2),expected_source_sha256=bindings["implementation"]["sha256"],expected_config_sha256=bindings["config"]["sha256"])
    assert check(f,own)["projected_bytes"]==before["projected_bytes"]+12345
@pytest.mark.parametrize("change",["baseline","external","snapshot","limit","root","schema","float_charge","negative_charge"])
def test_legacy_invariant_or_charge_drift_rejected_without_new_ledger(tmp_path,monkeypatch,change):
    f=setup(tmp_path,monkeypatch);root,*_=f;p=root/b.LEGACY;l=json.loads(p.read_text())
    if change=="float_charge":l["jobs"]["active"]["reserved_bytes"]=1.0
    elif change=="negative_charge":l["jobs"]["active"]["reserved_bytes"]=-1
    else:
        field={"baseline":"baseline_bytes","external":"external_reserve_bytes","snapshot":"snapshot_sha256","limit":"limit_bytes","root":"root","schema":"schema"}[change]
        l[field]=l[field]+1 if type(l[field]) is int else str(l[field])+"changed"
    dump(p,l);before=p.read_bytes()
    with pytest.raises(ValueError):reserve(f)
    assert p.read_bytes()==before and not (root/b.SUPPLEMENTAL).exists()
@pytest.mark.parametrize("change",["inactive","source","config","human","review_verdict","review_role","oldhash"])
def test_exact_root_authority_review_and_first_hash_gates(tmp_path,monkeypatch,change):
    f=setup(tmp_path,monkeypatch);root,c,r,a,p,bindings=f
    if change=="inactive":a["active"]=False
    elif change in ["source","config","human"]:a[{"source":"implementation","config":"config","human":"human_authorization"}[change]]["sha256"]="0"*64
    elif change=="oldhash":a["expected_before_ledger_pins"]["legacy_ledger_sha256"]="0"*64
    else:
        reviewpath=root/a["independent_review"]["path"];review=json.loads(reviewpath.read_text());review[{"review_verdict":"verdict","review_role":"review_role"}[change]]="BAD";dump(reviewpath,review);a["independent_review"]=ref(root,reviewpath)
    dump(p,a)
    with pytest.raises(ValueError):reserve(f)
    assert not (root/b.SUPPLEMENTAL).exists()
@pytest.mark.parametrize("change",["same_legacy_output","nested_legacy_output","same_key","overlap_aux","existing_output","accounting_scope","same_batch_output"])
def test_double_count_or_occupied_scope_rejected(tmp_path,monkeypatch,change):
    f=setup(tmp_path,monkeypatch);root,c,r,a,p,bindings=f;q=a["reservations"][0]
    if change=="same_legacy_output":q["output"]=str(root/"existing/active")
    elif change=="nested_legacy_output":q["output"]=str(root/"existing/active/sub")
    elif change=="same_key":q["key"]="active"
    elif change=="overlap_aux":q["auxiliary_scopes"]=[q["output"]+"/tmp"]
    elif change=="existing_output":Path(q["output"]).mkdir(parents=True)
    elif change=="accounting_scope":q["output"]=str(root/".runtime/storage")
    else:
        q2=copy.deepcopy(q);q2["key"]="second";a["reservations"].append(q2)
    dump(p,a)
    with pytest.raises(ValueError):reserve(f)
    assert not (root/b.SUPPLEMENTAL).exists()
@pytest.mark.parametrize("delta",[-1,0,1])
def test_combined_strict950GB_scheduling_boundary(tmp_path,monkeypatch,delta):
    f=setup(tmp_path,monkeypatch);root,c,r,a,p,bindings=f
    legacy=json.loads((root/b.LEGACY).read_text());base=legacy["baseline_bytes"]+legacy["external_reserve_bytes"]+sum(x["reserved_bytes"] for coll in ["jobs","controllers"] for x in legacy[coll].values())
    a["reservations"][0]["reserved_bytes"]=950000000000-base+delta;dump(p,a)
    if delta<0:assert reserve(f)["projected_bytes"]==949999999999
    else:
        with pytest.raises(ValueError,match="ceiling"):reserve(f)
        assert not (root/b.SUPPLEMENTAL).exists()
def test_actual_output_plus_short_tmp_boundary_and_planned_write(tmp_path,monkeypatch):
    f=setup(tmp_path,monkeypatch);receipt=reserve(f);own=next(iter(receipt["admitted_reservations"].values()))
    out=Path(own["output"]);tmp=Path(own["auxiliary_scopes"][0]);out.mkdir(parents=True);tmp.mkdir(parents=True);(out/"opaque.pt").write_bytes(b"a"*100);(tmp/"opaque.tmp").write_bytes(b"b"*200)
    v=check(f,own,boundary=True);assert v["scope_measurement"]["entries"]==4 and v["scope_measurement"]["conservative_scope_bytes"]>=300
    check(f,own,boundary=True,required_write_bytes=v["remaining_scope_bytes"])
    with pytest.raises(ValueError,match="boundary"):check(f,own,boundary=True,required_write_bytes=v["remaining_scope_bytes"]+1)
    with pytest.raises(ValueError,match="requires scope"):check(f,own,required_write_bytes=1)
@pytest.mark.parametrize("change",["symlink","hardlink","fifo","traversal"])
def test_boundary_unsafe_namespace_rejected(tmp_path,monkeypatch,change):
    f=setup(tmp_path,monkeypatch);root,*_=f;receipt=reserve(f);own=next(iter(receipt["admitted_reservations"].values()));out=Path(own["output"]);out.mkdir(parents=True);p=out/"opaque";p.write_bytes(b"x")
    if change=="symlink":(out/"link").symlink_to(p)
    elif change=="hardlink":os.link(p,out/"alias")
    elif change=="fifo":os.mkfifo(out/"fifo")
    else:
        with pytest.raises(ValueError):b.canonical(root,root/"../escape")
        return
    with pytest.raises(ValueError):check(f,own,boundary=True)
def test_own_reservation_or_consumer_authority_drift_blocks(tmp_path,monkeypatch):
    f=setup(tmp_path,monkeypatch);root,*_=f;receipt=reserve(f);own=next(iter(receipt["admitted_reservations"].values()))
    changed=copy.deepcopy(own);changed["reserved_bytes"]+=1
    with pytest.raises(ValueError,match="own"):check(f,changed)
    consumer=root/own["consumer_bindings"][0]["path"];consumer.write_text("changed")
    with pytest.raises(ValueError,match="bound"):check(f,own)
def test_mutex_exclusive_existing_only_and_bytes_preserved(tmp_path,monkeypatch):
    f=setup(tmp_path,monkeypatch);root,*_=f;p=root/b.LOCK;before=p.read_bytes()
    with p.open("r+") as other:
        with b.mutex(root):
            with pytest.raises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(other,fcntl.LOCK_UN)
    assert p.read_bytes()==before;p.unlink()
    with pytest.raises(FileNotFoundError):
        with b.mutex(root):pass
    assert not p.exists()
def test_scandir_errors_fail_closed(tmp_path,monkeypatch):
    f=setup(tmp_path,monkeypatch);receipt=reserve(f);own=next(iter(receipt["admitted_reservations"].values()));Path(own["output"]).mkdir(parents=True)
    def broken(*args,**kwargs): raise OSError("handmade scan failure")
    monkeypatch.setattr(b.os,"scandir",broken)
    with pytest.raises(OSError,match="scan failure"):check(f,own,boundary=True)

def test_directory_membership_drift_after_listing_is_rejected(tmp_path,monkeypatch):
    import contextlib
    f=setup(tmp_path,monkeypatch);receipt=reserve(f);own=next(iter(receipt["admitted_reservations"].values()));out=Path(own["output"]);out.mkdir(parents=True);(out/"first").write_bytes(b"x")
    original=b.os.scandir;changed=False
    @contextlib.contextmanager
    def raced(path):
        nonlocal changed
        with original(path) as entries:captured=list(entries)
        if Path(path)==out and not changed:
            (out/"late").write_bytes(b"later bytes");changed=True
        yield iter(captured)
    monkeypatch.setattr(b.os,"scandir",raced)
    with pytest.raises(ValueError,match="membership changed"):check(f,own,boundary=True)

def test_atomic_write_failure_keeps_legacy_and_no_automatic_rollback(tmp_path,monkeypatch):
    f=setup(tmp_path,monkeypatch);root,*_=f;before=(root/b.LEGACY).read_bytes()
    def fail(*args):raise OSError("handmade atomic replacement failure")
    monkeypatch.setattr(b.os,"replace",fail)
    with pytest.raises(OSError,match="atomic replacement failure"):reserve(f)
    assert (root/b.LEGACY).read_bytes()==before and not (root/b.SUPPLEMENTAL).exists()
    assert not list((root/b.SUPPLEMENTAL).parent.glob(".*.tmp"))

def test_own_stored_root_authority_rewrite_and_header_tamper_fail(tmp_path,monkeypatch):
    f=setup(tmp_path,monkeypatch);root,c,r,a,p,bindings=f;receipt=reserve(f);own=next(iter(receipt["admitted_reservations"].values()))
    a["active"]=False;dump(p,a)
    with pytest.raises(ValueError,match="bound"):check(f,own)
    dump(p,{**a,"active":True})
    data=json.loads((root/b.SUPPLEMENTAL).read_text());data["hard_limit_bytes"]+=1;dump(root/b.SUPPLEMENTAL,data)
    with pytest.raises(ValueError,match="header"):check(f,own)

def test_metadata_hardlink_and_duplicate_JSON_rejected(tmp_path,monkeypatch):
    f=setup(tmp_path,monkeypatch);root,*_=f;p=root/b.LEGACY;os.link(p,root/"metadata_alias")
    with pytest.raises(ValueError,match="singly linked"):reserve(f)
    for raw in ['{"x":1,"x":2}','{"x":NaN}']:
        with pytest.raises(ValueError):b.parse(raw)

@pytest.mark.parametrize("change",["unknown_header_collection","extra_job_scope","extra_controller_output","missing_job_charge","missing_controller_pid","bool_controller_pid","zero_controller_pid","float_controller_pid","empty_job_key","oversized_job_key","invalid_controller_key","relative_output","noncanonical_absolute","legacy_ancestor","legacy_descendant","legacy_accounting_parent","legacy_accounting_child","legacy_mutex","legacy_ledger"])
def test_exact_legacy_schema_keys_paths_and_all_overlap_repair(tmp_path,monkeypatch,change):
    f=setup(tmp_path,monkeypatch);root,c,r,a,p,bindings=f
    ledger=root/b.LEGACY;l=json.loads(ledger.read_text())
    if change=="unknown_header_collection":l["unknown_reserved_collection"]={"x":{"reserved_bytes":700000000000}}
    elif change=="extra_job_scope":l["jobs"]["active"]["auxiliary_scopes"]=[r["output"]]
    elif change=="extra_controller_output":l["controllers"]["active"]["output"]=r["output"]
    elif change=="missing_job_charge":l["jobs"]["active"].pop("reserved_bytes")
    elif change=="missing_controller_pid":l["controllers"]["active"].pop("pid")
    elif change=="bool_controller_pid":l["controllers"]["active"]["pid"]=True
    elif change=="zero_controller_pid":l["controllers"]["active"]["pid"]=0
    elif change=="float_controller_pid":l["controllers"]["active"]["pid"]=111.0
    elif change=="empty_job_key":l["jobs"][""]=l["jobs"].pop("active")
    elif change=="oversized_job_key":l["jobs"]["x"*130]=l["jobs"].pop("active")
    elif change=="invalid_controller_key":l["controllers"]["bad/key"]=l["controllers"].pop("active")
    elif change=="relative_output":l["jobs"]["active"]["output"]="relative/output"
    elif change=="noncanonical_absolute":l["jobs"]["active"]["output"]=str(root)+"/existing//active"
    elif change in ["legacy_ancestor","legacy_descendant"]:
        parent=l["jobs"]["active"]["output"]
        other=parent+"/nested" if change=="legacy_descendant" else str(Path(parent).parent)
        l["jobs"]["other"]={"output":other,"reserved_bytes":1}
    elif change=="legacy_accounting_parent":l["jobs"]["active"]["output"]=str(root/".runtime/storage")
    elif change=="legacy_accounting_child":l["jobs"]["active"]["output"]=str(root/".runtime/storage/unrelated_scope")
    elif change=="legacy_mutex":l["jobs"]["active"]["output"]=str(root/b.LOCK)
    else:l["jobs"]["active"]["output"]=str(root/b.LEGACY)
    dump(ledger,l);before=ledger.read_bytes()
    with pytest.raises(ValueError):reserve(f)
    assert ledger.read_bytes()==before and not (root/b.SUPPLEMENTAL).exists()

def test_exact_controller_entry_has_pid_and_no_invented_output_and_settlement_remains_dynamic(tmp_path,monkeypatch):
    f=setup(tmp_path,monkeypatch);root,*_=f;receipt=reserve(f);own=next(iter(receipt["admitted_reservations"].values()))
    before=check(f,own);ledger=root/b.LEGACY;l=json.loads(ledger.read_text())
    l["controllers"]["new_controller"]={"reserved_bytes":1234,"pid":222}
    l["jobs"]["active"]["reserved_bytes"]-=10000
    dump(ledger,l);later=check(f,own)
    assert later["projected_bytes"]==before["projected_bytes"]-10000+1234
    assert later["reservation"]==own and later["legacy_ledger_sha256"]!=before["legacy_ledger_sha256"]

@pytest.mark.parametrize("existing_count",[4095,4096])
def test_combined_map_cap_before_proposal_construction_or_write(tmp_path,monkeypatch,existing_count):
    # Inject an already-validated latest ledger view; all authority/new-scope/write gates remain real.
    f=setup(tmp_path,monkeypatch);root,c,r,a,p,bindings=f
    _,actual_bindings=b.policy(root,bindings["implementation"]["sha256"],bindings["config"]["sha256"])
    legacy,empty,view=b.current(root,c,actual_bindings)
    big=b.empty_ledger(root,c,actual_bindings)
    for i in range(existing_count):
        item=copy.deepcopy(r);item["key"]="existing-"+str(i);item["output"]=str(root/"existing_supplemental"/str(i));item["auxiliary_scopes"]=[];item["reserved_bytes"]=1;item["root_admission"]=ref(root,p)
        big["reservations"][item["key"]]=item
    original_state=copy.deepcopy(big)
    view["supplemental_reserved_bytes"]=existing_count;view["projected_bytes"]+=existing_count
    monkeypatch.setattr(b,"current",lambda *args:(legacy,big,view))
    original_scope=b.scope_spec
    def existing_scope(root,value,stored=False):
        if value["key"].startswith("existing-"):
            return [Path(value["output"])]
        return original_scope(root,value,stored)
    monkeypatch.setattr(b,"scope_spec",existing_scope)
    before_legacy=(root/b.LEGACY).read_bytes()
    if existing_count==4096:
        original_copy=b.copy.deepcopy
        def forbidden_proposal(value,*args,**kwargs):
            if value is big:raise AssertionError("proposal constructed before map cap rejection")
            return original_copy(value,*args,**kwargs)
        monkeypatch.setattr(b.copy,"deepcopy",forbidden_proposal)
        with pytest.raises(ValueError,match="combined supplemental reservation map cap"):
            reserve(f)
        assert not (root/b.SUPPLEMENTAL).exists()
    else:
        receipt=reserve(f);written=json.loads((root/b.SUPPLEMENTAL).read_text())
        assert len(written["reservations"])==4096 and len(receipt["admitted_reservations"])==1
        assert written["reservations"][r["key"]]["reserved_bytes"]==20000000000
    assert big==original_state and (root/b.LEGACY).read_bytes()==before_legacy
