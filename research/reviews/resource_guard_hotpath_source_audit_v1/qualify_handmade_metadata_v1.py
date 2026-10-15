"""Handmade metadata fixtures only; never import/configure a live guard."""
from pathlib import Path
import ast,contextlib,fcntl,hashlib,importlib.util,json,os,stat,sys,threading,types

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=ROOT/"research/reviews/resource_guard_hotpath_source_audit_v1"
SOURCE=ROOT/"src/arggnn/project_budget_v4.py"
ADAPTER=OUT/"transaction_memo_adapter_v1.py"

def load_fixture_module(directory):
    source=SOURCE.read_text();tree=ast.parse(source)
    names={"canonical","file_bytes","reference","bound","parse","mutex","measure_scopes"}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==names
    m=types.ModuleType("_handmade_v4_"+directory.name);m.__file__=str(SOURCE)
    m.__dict__.update(Path=Path,contextlib=contextlib,fcntl=fcntl,hashlib=hashlib,
                      json=json,os=os,stat=stat,ROOT=directory,LOCK="fixture_scheduler.lock",MAX_JSON=32*1024*1024)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),"exec"),m.__dict__)
    return m

spec=importlib.util.spec_from_file_location("_handmade_transaction_memo",ADAPTER)
adapter_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter_module)
source_sha=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
fixtures=OUT/"fixtures_r002"
if fixtures.exists():raise RuntimeError("Fixture run exists; preserve it rather than overwrite")
fixtures.mkdir()
results=[]

def make_case(name):
    d=fixtures/name;d.mkdir();(d/"fixture_scheduler.lock").write_text("")
    (d/"metadata.json").write_text('{"allow":true,"value":17}\n')
    (d/"journal_fixture.json").write_text('{"revoked":false}\n')
    (d/"owned").mkdir();(d/"owned"/"payload.txt").write_text("fixture")
    return d,load_fixture_module(d)

def install(m,**kwargs):
    return adapter_module.install(m,expected_source_path=SOURCE,expected_source_sha256=source_sha,**kwargs)

def profile_count(code,body):
    count=0
    def profile(frame,event,arg):
        nonlocal count
        if event=="call" and frame.f_code is code:count+=1
    sys.setprofile(profile)
    try:body()
    finally:sys.setprofile(None)
    return count

def check(m,d,ref,repeat=8):
    for _ in range(repeat):
        observed=m.reference(d,"metadata.json")
        if observed!=ref:m.bound(d,ref,True)
        assert observed==ref
        assert m.bound(d,ref,True)["value"]==17
        state=m.parse(m.file_bytes(d,"journal_fixture.json")[0])
        if state["revoked"]:raise ValueError("Fixture reservation revoked")
    return m.measure_scopes(d,[d/"owned"])

def outcome(body):
    try:body();return "accepted"
    except (ValueError,RuntimeError) as e:return type(e).__name__+":"+str(e)

d,m=make_case("repeated");ref=m.reference(d,"metadata.json")
original_read=m.file_bytes;original_canonical=m.canonical
plain={}
def plain_body():
    with m.mutex(d):plain["measurement"]=check(m,d,ref)
plain_reads=profile_count(original_read.__code__,plain_body)
handle=install(m)
memo={}
def memo_body():
    with m.mutex(d):memo["measurement"]=check(m,d,ref)
memo_reads=profile_count(original_read.__code__,memo_body)
assert plain["measurement"]==memo["measurement"]
assert plain_reads==24 and memo_reads==2
assert handle.last_operation_counts()["raw_hits"]==22
results.append(dict(test="same_acceptance_measurement_unique_reads",plain_reads=plain_reads,memo_reads=memo_reads,memo_counts=handle.last_operation_counts(),passed=True))

# Every operation freshly rereads both files, even without a namespace change.
memo_again=profile_count(original_read.__code__,memo_body)
assert memo_again==2
results.append(dict(test="no_cross_operation_cache",second_operation_reads=memo_again,passed=True))
handle.uninstall()
assert m.file_bytes is original_read and m.canonical is original_canonical

# Equal rejections with and without adapter for bad SHA, invalid JSON and path.
for case in ("sha","json","outside","revoked"):
    entries=[]
    for mode in ("plain","memo"):
        d,m=make_case(case+"_"+mode);ref=m.reference(d,"metadata.json")
        if case=="sha":ref["sha256"]="0"*64
        if case=="json":(d/"metadata.json").write_text('{"value":17,"value":17}\n');ref=m.reference(d,"metadata.json")
        if case=="revoked":(d/"journal_fixture.json").write_text('{"revoked":true}\n')
        if mode=="memo":h=install(m)
        def body():
            with m.mutex(d):
                if case=="outside":m.canonical(d,d.parent/"outside")
                else:check(m,d,ref)
        entries.append(outcome(body))
        if mode=="memo":assert h._state() is None;h.uninstall()
    assert entries[0]==entries[1] and entries[0]!="accepted"
    results.append(dict(test="equivalent_rejection_"+case,outcomes=entries,passed=True))

# Revocation and bytes changes between operations cannot reuse prior evidence.
for case in ("between_revocation","between_tamper"):
    d,m=make_case(case);ref=m.reference(d,"metadata.json");h=install(m)
    with m.mutex(d):check(m,d,ref)
    if case=="between_revocation":(d/"journal_fixture.json").write_text('{"revoked":true}\n')
    else:(d/"metadata.json").write_text('{"allow":true,"value":18}\n')
    def body():
        with m.mutex(d):check(m,d,ref)
    rejected=outcome(body)
    assert rejected!="accepted" and h._state() is None
    results.append(dict(test=case,outcome=rejected,passed=True));h.uninstall()

# Tamper during one snapshot is caught before successful context exit.
d,m=make_case("during_tamper");ref=m.reference(d,"metadata.json");h=install(m)
def during():
    with m.mutex(d):
        m.bound(d,ref)
        (d/"metadata.json").write_text('{"allow":true,"value":18}\n')
        m.bound(d,ref)  # cached; final uncached identity check must reject.
rejected=outcome(during)
assert rejected!="accepted" and h._state() is None
results.append(dict(test="during_operation_tamper_final_identity",outcome=rejected,passed=True));h.uninstall()

# Direct stat/list/fingerprint/scope operations remain uncached.
d,m=make_case("fresh_scope");h=install(m)
with m.mutex(d):
    before=m.measure_scopes(d,[d/"owned"])
    (d/"owned"/"second.txt").write_text("new")
    after=m.measure_scopes(d,[d/"owned"])
    assert after["entries"]==before["entries"]+1
    assert m.file_bytes(d,"metadata.json")==m.file_bytes(d,"metadata.json")
assert h.last_operation_counts()["raw_misses"]==1
results.append(dict(test="uncached_scope_measurement",before_entries=before["entries"],after_entries=after["entries"],passed=True));h.uninstall()

# Bound overflow falls back to original reads without changing acceptance.
d,m=make_case("bounded");ref=m.reference(d,"metadata.json");h=install(m,max_bytes=512,max_entries=1)
with m.mutex(d):check(m,d,ref)
assert h.last_operation_counts()["charge"]<=512 and h.last_operation_counts()["capacity_bypasses"]>0
results.append(dict(test="bounded_capacity_fallback",counts=h.last_operation_counts(),passed=True));h.uninstall()

# Nested entry rejects before the original flock. Exception leaves no active cache.
d,m=make_case("cleanup");h=install(m)
def nested():
    with m.mutex(d):
        with m.mutex(d):raise AssertionError("Unreachable")
assert outcome(nested).startswith("RuntimeError:Nested")
assert h._state() is None and h.active_count==0
def raises():
    with m.mutex(d):
        m.file_bytes(d,"metadata.json")
        raise RuntimeError("handmade failure")
assert outcome(raises)=="RuntimeError:handmade failure"
assert h._state() is None and h.active_count==0
with m.mutex(d):m.file_bytes(d,"metadata.json")
assert h.last_operation_counts()["raw_misses"]==1
results.append(dict(test="nested_rejection_exception_cleanup",passed=True));h.uninstall()

# Separate fixture locks allow simultaneous threads to prove TLS isolation.
d,m=make_case("threads");other=d/"second_lock";other.mkdir()
(other/"fixture_scheduler.lock").write_text("");(other/"metadata.json").write_text('{"value":29}\n')
h=install(m);barrier=threading.Barrier(2);observed=[];errors=[]
def thread_body(base,expected):
    try:
        with m.mutex(base):
            assert m.parse(m.file_bytes(base,"metadata.json")[0])["value"]==expected
            barrier.wait(timeout=5)
            m.file_bytes(base,"metadata.json")
        observed.append(h.last_operation_counts())
    except BaseException as e:errors.append(str(e))
a=threading.Thread(target=thread_body,args=(d,17));b=threading.Thread(target=thread_body,args=(other,29))
a.start();b.start();a.join(10);b.join(10)
assert not a.is_alive() and not b.is_alive() and not errors
assert len(observed)==2 and all(row["raw_misses"]==1 and row["raw_hits"]==1 for row in observed)
assert h._state() is None and h.active_count==0
results.append(dict(test="thread_local_isolation",counts=observed,passed=True));h.uninstall()

receipt=dict(schema="resource_guard_handmade_memo_fixture_qualification_v1",status="passed",tests=results,
 original_source=dict(path=str(SOURCE.relative_to(ROOT)),sha256=source_sha),
 adapter_ref=dict(path=str(ADAPTER.relative_to(ROOT)),sha256=hashlib.sha256(ADAPTER.read_bytes()).hexdigest()),
 original_functions_identity_example=handle.receipt,
 live_guard_imported=False,live_guard_configured=False,live_journal_read=False,
 scientific_payloads_read=False,live_processes_or_GPU_accessed=False,
 limitations="Handmade AST-extracted I/O helpers, not live accounting replay, timing or production qualification.")
(OUT/"handmade_fixture_results.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(dict(status="passed",tests=len(results),plain_reads=plain_reads,memo_reads=memo_reads)))
