"""Pure mocked renewal fixtures. No guard/import hook/process/accounting action."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone,timedelta
import ast,copy,hashlib,json,types
HERE=Path(__file__).resolve().parent
SOURCE=HERE/"marker_repair_renewal_v1.py"
class FakePath(PurePosixPath):
    def resolve(self):return self
class FakeLoader:
    calls=[]
    def __init__(self,path):self.path=path
    def exec_module(self,module):
        self.calls.append(self.path)
        module._bound=lambda record:copy.deepcopy(data[record["path"]])
original_exec=FakeLoader.exec_module
fake_importlib=types.SimpleNamespace(machinery=types.SimpleNamespace(SourceFileLoader=FakeLoader))
fake_sys=types.SimpleNamespace(modules={})
ns=dict(Path=FakePath,datetime=datetime,timezone=timezone,timedelta=timedelta,copy=copy,
    importlib=fake_importlib,sys=fake_sys)
tree=ast.parse(SOURCE.read_text())
nodes=[n for n in tree.body if isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef)
    and n.name in ("require","validate_certificate_v1","_wrap_private_marker_v1","install_renewal_v1")]
module=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(module)
exec(compile(module,"<isolated_pure_renewal>","exec"),ns)
old_path=Path("/workspace-SR003.nfs2/ARG-GNN")/ns["OLD_AUTHORITY_REF"]["path"]
old=json.loads(old_path.read_text())
certificate_ref=dict(path="fixture/renewal_certificate.json",sha256="c"*64)
refs={ns["SOURCE"]:dict(path=ns["SOURCE"],sha256="1"*64),
    ns["BOOTSTRAP"]:dict(path=ns["BOOTSTRAP"],sha256="2"*64),
    ns["OLD_SOURCE_REF"]["path"]:ns["OLD_SOURCE_REF"]}
now=datetime.now(timezone.utc)
certificate=dict(schema="H039_marker_repair_expiry_renewal_root_certificate_v1",active=True,root_authorizer="root_agent",
    project_root=str(ns["ROOT"]),source_ref=refs[ns["SOURCE"]],bootstrap_ref=refs[ns["BOOTSTRAP"]],
    old_authority_ref=ns["OLD_AUTHORITY_REF"],old_repair_source_ref=ns["OLD_SOURCE_REF"],
    old_activation_barrier_ref=ns["OLD_BARRIER_REF"],reservation_sha256=ns["RESERVATION_SHA"],
    retained_allowance_bytes=15000000,accounting_credit_bytes=0,
    issued_at_utc=(now-timedelta(minutes=1)).isoformat(),renewed_expires_at_utc=(now+timedelta(days=1)).isoformat(),
    read_only_marker_authority_expiry_renewal=True,fresh_admissions_and_existing_read_close_authorized=True,
    all_allocation_expiries_unchanged=True,historical_journal_mutation_authorized=False,
    historical_source_or_binding_mutation_authorized=False,global_rebase_authorized=False,
    malformed_key_future_writes_or_relaunch_authorized=False,old_terminal_barrier_reused_without_new_absence_claim=True)
data={certificate_ref["path"]:certificate,ns["OLD_AUTHORITY_REF"]["path"]:old,
    ns["OLD_BARRIER_REF"]["path"]:{"fixture":"original barrier bytes delegated unchanged"},"other.json":{"expires_at_utc":"original"}}
reads=[]
def read(record,limit=None):
    reads.append(record["path"])
    return copy.deepcopy(data[record["path"]])
ns["_json"]=read;ns["reference"]=lambda name:copy.deepcopy(refs[name])
results=[]
def passed(name,ok):
    assert ok,name;results.append(dict(case=name,passed=True))
def rejects(name,call):
    try:call()
    except ValueError:passed(name,True)
    else:raise AssertionError(name)
ns["validate_certificate_v1"](certificate_ref);passed("exact_active_root_certificate_validated_without_guard",True)
target=str(ns["ROOT"]/ns["OLD_SOURCE_REF"]["path"])
private=types.SimpleNamespace(__file__=target,_bound=read)
ns["_wrap_private_marker_v1"](private,certificate_ref)
renewed=private._bound(ns["OLD_AUTHORITY_REF"])
passed("only_returned_old_authority_expiry_changes",
    {k:v for k,v in renewed.items() if k!="expires_at_utc"}=={k:v for k,v in old.items() if k!="expires_at_utc"}
    and renewed["expires_at_utc"]==certificate["renewed_expires_at_utc"]
    and old["expires_at_utc"]=="2026-10-15T21:36:04.459612+00:00")
passed("all_other_bound_refs_delegate_unchanged",private._bound(dict(path="other.json",sha256="d"*64))==data["other.json"])
saved=certificate["renewed_expires_at_utc"];certificate["renewed_expires_at_utc"]=(now-timedelta(seconds=1)).isoformat()
rejects("renewal_revalidated_on_every_old_authority_read",lambda:private._bound(ns["OLD_AUTHORITY_REF"]))
certificate["renewed_expires_at_utc"]=saved
for name,value in [("accounting_credit_bytes",1),("retained_allowance_bytes",14999999),
    ("malformed_key_future_writes_or_relaunch_authorized",True),("all_allocation_expiries_unchanged",False),
    ("global_rebase_authorized",True),("root_authorizer","not_root")]:
    before=certificate[name];certificate[name]=value
    rejects("reject_"+name,lambda:ns["validate_certificate_v1"](certificate_ref));certificate[name]=before
before=refs[ns["OLD_SOURCE_REF"]["path"]];refs[ns["OLD_SOURCE_REF"]["path"]]=dict(before,sha256="0"*64)
rejects("old_source_pin_rechecked_each_use",lambda:private._bound(ns["OLD_AUTHORITY_REF"]))
refs[ns["OLD_SOURCE_REF"]["path"]]=before
FakeLoader.exec_module=original_exec;fake_sys.modules.clear()
ns["install_renewal_v1"](certificate_ref)
external=types.SimpleNamespace(__file__="/fixture/unrelated.py")
FakeLoader("/fixture/unrelated.py").exec_module(external)
passed("unrelated_source_import_delegates_without_private_wrapper",not hasattr(external,"_root_expiry_renewal_ref"))
fresh=types.SimpleNamespace(__file__=target)
FakeLoader(target).exec_module(fresh)
passed("exact_future_marker_import_gets_narrow_private_wrapper",
    fresh._bound(ns["OLD_AUTHORITY_REF"])["expires_at_utc"]==saved)
same_hook=FakeLoader.exec_module;ns["install_renewal_v1"](certificate_ref)
passed("same_certificate_install_is_idempotent",FakeLoader.exec_module is same_hook)
other_ref=dict(certificate_ref,path="fixture/other_certificate.json");data[other_ref["path"]]=copy.deepcopy(certificate)
rejects("no_certificate_swap_in_one_interpreter",lambda:ns["install_renewal_v1"](other_ref))
FakeLoader.exec_module=original_exec;fake_sys.modules={"existing":types.SimpleNamespace(__file__=target)}
rejects("existing_loaded_marker_is_never_patched",lambda:ns["install_renewal_v1"](certificate_ref))
passed("historical_authority_file_bytes_unchanged",hashlib.sha256(old_path.read_bytes()).hexdigest()==ns["OLD_AUTHORITY_REF"]["sha256"])
print(json.dumps(dict(schema="H039_marker_repair_renewal_pure_fixtures_v1",source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    cases=results,cases_passed=len(results),actual_guard_calls_import_hooks_processes_signals_waits_admissions_jobs_or_Git=0),indent=2,sort_keys=True))
