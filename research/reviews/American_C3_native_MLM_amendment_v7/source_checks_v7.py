"""No guard/model imports, actual process probes, accounting or scope inventories."""
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,importlib.util,json,unittest
from unittest.mock import patch
from types import SimpleNamespace
import errno,os
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("amendment_v7_checks",HERE/"amendment_v7.py")
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
def node(path,name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
class SourceChecks(unittest.TestCase):
    def test_expiry_original_absolute(self):
        good=dict(created_at_utc=a.CREATED,expires_at_utc=a.EXPIRES)
        a.check_expiry(good,now=datetime(2026,10,15,12,tzinfo=timezone.utc))
        for changed in [dict(good,expires_at_utc="2026-10-17T10:01:33.907094+00:00"),dict(good,created_at_utc="2026-10-15T11:01:33.907094+00:00")]:
            with self.assertRaises(ValueError):a.check_expiry(changed,now=datetime(2026,10,15,12,tzinfo=timezone.utc))
        with self.assertRaises(ValueError):a.check_expiry(good,now=datetime(2026,10,16,11,tzinfo=timezone.utc))
    def test_inactive_or_wrong_allocation_rejected_without_guard(self):
        value=dict(a.fixed_authority_fields(),active=False,source_closure_ref={},input_ref={},root_directive_ref={},old_writer_absence_ref={})
        with patch.object(a,"read",return_value=value):
            with self.assertRaises(ValueError):a.validate_authority({"path":"fake","sha256":"fake"})
        for field,bad in [("key","other"),("reserved_bytes",1),("physical_gpu",5),("expires_at_utc","renewed")]:
            value=dict(a.fixed_authority_fields(),active=True,source_closure_ref={},input_ref={},root_directive_ref={},old_writer_absence_ref={})
            value[field]=bad
            with patch.object(a,"read",return_value=value):
                with self.assertRaises(ValueError):a.validate_authority({"path":"fake","sha256":"fake"})
    def test_actual_input_provenance_and_original_science_separated(self):
        old=dict(source_sha256={"old":"oldsha"},main_source_ref={"old":"producer"},
            actual_launcher_ref={"old":"launcher"},root_context_module_ref={"old":"context"},
            transitive_source_closure_ref={"old":"closure"},input_binding_mode="old",
            refs={"config":{"path":"frozen-config","sha256":"cfg"},"runtime":{"path":"frozen-runtime","sha256":"run"}},
            output_directory=a.OUTPUT,physical_gpu=4,job_id=a.JOB,created_at_utc=a.CREATED,expires_at_utc=a.EXPIRES)
        auth={"source_closure_ref":{"path":"closure","sha256":"new"}}
        closure={"source_sha256":{"old":"oldsha","actual":"actualsha"}}
        def read(record):return old if record==a.ORIGINAL_INPUT_REF else closure
        def ref(path):return {"path":path,"sha256":"sha:"+path}
        expected=copy.deepcopy(old)
        expected.update(main_source_ref=ref(a.ENTRY),actual_launcher_ref=ref(a.LAUNCHER),
            root_context_module_ref=ref(a.REL+"/context_factory_v1.py"),source_sha256=closure["source_sha256"],
            transitive_source_closure_ref=auth["source_closure_ref"],input_binding_mode="root_execution_amendment_v7")
        with patch.object(a,"read",side_effect=read),patch.object(a,"ref",side_effect=ref):
            a.validate_input_fields(auth,expected)
            for field,bad in [("physical_gpu",5),("output_directory","new-output"),("job_id","new-job"),("expires_at_utc","renewed"),("main_source_ref",old["main_source_ref"])]:
                changed=dict(expected);changed[field]=bad
                with self.assertRaises(ValueError):a.validate_input_fields(auth,changed)
            changed=copy.deepcopy(expected);changed["refs"]["runtime"]["sha256"]="replacement"
            with self.assertRaises(ValueError):a.validate_input_fields(auth,changed)
    def test_authenticated_authority_chain_and_tampering(self):
        old=a.read(a.ORIGINAL_INPUT_REF)
        sources=dict(old["source_sha256"])
        for path in a.required_execution_sources():sources[path]=a.ref(path)["sha256"]
        closure=dict(schema="native_MLM_execution_source_closure_v7",original_storage_ref=a.ORIGINAL_STORAGE_REF,
            actual_backend_ref=a.BACKEND_REF,source_sha256=sources,v5_provenance_source_refs=[a.BACKEND_REF])
        closure_ref={"path":"mock-closure","sha256":"mock-closure-sha"}
        input_ref={"path":"mock-input","sha256":"mock-input-sha"}
        directive_ref={"path":"mock-directive","sha256":"mock-directive-sha"}
        authority_ref={"path":"mock-authority","sha256":"mock-authority-sha"}
        auth=dict(a.fixed_authority_fields(),active=True,source_closure_ref=closure_ref,input_ref=input_ref,
            root_directive_ref=directive_ref,old_writer_absence_ref={"path":"root-cert","sha256":"pending"})
        spec=copy.deepcopy(old)
        spec.update(main_source_ref=a.ref(a.ENTRY),actual_launcher_ref=a.ref(a.LAUNCHER),
            root_context_module_ref=a.ref(a.REL+"/context_factory_v1.py"),source_sha256=sources,
            transitive_source_closure_ref=closure_ref,input_binding_mode="root_execution_amendment_v7")
        directive=dict(schema="native_MLM_execution_amendment_root_directive_v7",root_authorizer="root_agent",
            active=True,key=a.KEY,source_closure_ref=closure_ref,input_ref=input_ref,
            amendment_source_ref=a.ref(a.SOURCE),original_storage_ref=a.ORIGINAL_STORAGE_REF,
            all_original_v4_v5_v6_launch_resume_write_capabilities_revoked=True,
            preserved_v6_refs=a.PREVIOUS_V6_REFS,actual_NFS_flock_probe_ref=a.NFS_FLOCK_PROBE_REF,
            cooperative_root_writer_model_acknowledged=True,
            immutable_old_code_mechanical_rejection_claimed=False,no_accounting_or_original_source_mutation=True,
            reserved_bytes=a.CAP,expires_at_utc=a.EXPIRES)
        fixtures={"mock-authority":auth,"mock-closure":closure,"mock-input":spec,"mock-directive":directive}
        real_read=a.read
        def read(record,**kw):
            if record["path"] in fixtures:return fixtures[record["path"]]
            return real_read(record,**kw)
        with patch.object(a,"read",side_effect=read),patch.object(a,"check_expiry",return_value=None):
            a.validate_authority(authority_ref)
            directive["immutable_old_code_mechanical_rejection_claimed"]=True
            with self.assertRaises(ValueError):a.validate_authority(authority_ref)
            directive["immutable_old_code_mechanical_rejection_claimed"]=False
            path=next(iter(old["source_sha256"]));saved=sources.pop(path)
            with self.assertRaises(ValueError):a.validate_authority(authority_ref)
            sources[path]=saved
            spec["phase"]="mlm"
            with self.assertRaises(ValueError):a.validate_authority(authority_ref)
            spec["phase"]="qualify_mlm"
            sources[a.ENTRY]="tampered"
            with self.assertRaises(ValueError):a.validate_authority(authority_ref)
    def test_owned_ancestry_and_PID_reuse(self):
        rows={10:dict(pid=10,start=100,ppid=1,state="S"),11:dict(pid=11,start=101,ppid=10,state="S"),12:dict(pid=12,start=102,ppid=11,state="S")}
        owner=dict(pid=10,start=100)
        self.assertTrue(a.descends_from(12,owner,rows.get))
        rows[10]["start"]=999
        self.assertFalse(a.descends_from(12,owner,rows.get))
        rows[10]["start"]=100;rows[10]["state"]="Z"
        self.assertFalse(a.descends_from(12,owner,rows.get))
        rows[11]["ppid"]=11
        self.assertFalse(a.descends_from(12,owner,rows.get))
    def test_unknown_historical_wait_is_explicit(self):
        t=(HERE/"amendment_v7.py").read_text()
        self.assertIn('cert["original_outer_OS_wait"]=="unknown"',t)
        old_observer=ast.unparse(node(HERE/"amendment_v7.py","observe_old_absence"))
        self.assertNotIn('wait["returncode"]',old_observer)
        self.assertNotIn("actual_direct_outer_supervisor_wait",old_observer)
        self.assertIn('started["actual_wait_observed"] is False',t)
    def test_writable_independent_lock_probe_regression(self):
        descriptor=SimpleNamespace(st_mode=0o100600,st_nlink=1,st_dev=10,st_ino=20)
        lock=SimpleNamespace(lstat=lambda:descriptor)
        record={"path":"authorized","sha256":"bound"}
        token=dict(schema="native_MLM_execution_attempt_token_v7",authority_ref=record,
            attempt_directory=str(a.ATTEMPT.relative_to(a.ROOT)),owner=dict(pid=10,start=100))
        raw=json.dumps(token).encode()
        calls=[]
        def fake_open(path,flags):
            calls.append(flags)
            if not flags & os.O_RDWR:raise OSError(errno.EBADF,"NFS exclusive flock needs writable independent fd")
            return 44
        def fake_flock(fd,operation):
            self.assertEqual(fd,44)
            raise BlockingIOError(errno.EAGAIN,"root owner EX lock is held")
        with patch.object(a,"LOCK",lock),patch.object(a,"check_expiry",return_value=None),patch.object(a,"lifetime_fd",return_value=33),patch.object(a,"authority_ref",return_value=record),patch.object(a,"descends_from",return_value=True),patch.object(a.os,"fstat",return_value=descriptor),patch.object(a.os,"pread",return_value=raw),patch.object(a.os,"getpid",return_value=11),patch.object(a.os,"open",side_effect=fake_open),patch.object(a.os,"close") as closed,patch.object(a.fcntl,"flock",side_effect=fake_flock):
            self.assertEqual(a.check_lifetime(),token)
            closed.assert_called_once_with(44)
        self.assertEqual(calls,[os.O_RDWR|os.O_NOFOLLOW])
        self.assertEqual(a.ENV_REF,"ARGGNN_MLM_AMENDMENT_V7_REF")
        self.assertEqual(a.ENV_FD,"ARGGNN_MLM_AMENDMENT_V7_FD")
        self.assertNotEqual(a.LOCK,a.V6_LOCK)
        self.assertNotEqual(a.ATTEMPT,a.V6_ATTEMPT)
    def test_real_root_NFS_and_failed_v6_receipts_bound(self):
        # Exact actual ROOT receipts/source hashes, with no process/lock syscalls by this test.
        a.validate_previous_v6_and_real_lock_probe()
        real_read=a.read
        def bad_probe(record,**kw):
            value=real_read(record,**kw)
            if record==a.NFS_FLOCK_PROBE_REF:
                value=copy.deepcopy(value);value["result"]["read_write"]["errno"]=9
            return value
        with patch.object(a,"read",side_effect=bad_probe):
            with self.assertRaises(ValueError):a.validate_previous_v6_and_real_lock_probe()
    def test_v6_token_never_reset_and_exact_previous_entries(self):
        class FakeDirectory:
            def __init__(self,name,children):self.name=name;self.children=children
            def is_dir(self):return True
            def iterdir(self):return [SimpleNamespace(name=x) for x in self.children]
        aux=FakeDirectory("aux",["amendment_v6.activation","amended_v6_attempt1"])
        attempt=FakeDirectory("amended_v6_attempt1",["admitted.json"])
        by_path={r["path"]:r for r in a.PREVIOUS_V6_REFS.values()}
        with patch.object(a,"AUX",aux),patch.object(a,"V6_ATTEMPT",attempt),patch.object(a,"ref",side_effect=lambda path:by_path[path]):
            a.validate_preserved_auxiliary_lineage()
            attempt.children.append("unexpected_worker_output")
            with self.assertRaises(ValueError):a.validate_preserved_auxiliary_lineage()
        activation=ast.unparse(node(HERE/"amendment_v7.py","activate"))
        self.assertIn("validate_preserved_auxiliary_lineage()",activation)
        self.assertNotIn("unlink",activation)
        self.assertIn("not LOCK.exists()",activation)
    def test_attempt_lock_and_lifetime_on_all_supported_boundaries(self):
        t=(HERE/"amendment_v7.py").read_text()
        activation=ast.unparse(node(HERE/"amendment_v7.py","activate"))
        self.assertIn("guard.budget.v4.mutex(ROOT)",activation)
        self.assertIn("os.O_EXCL",activation);self.assertIn("fcntl.LOCK_EX",activation)
        self.assertIn("validate_authority(record)",activation)
        self.assertNotIn("unlink",activation)
        for filename in ["launcher_v7.py","root_native_driver_v1.py","supervisor_v1.py"]:
            tree=ast.parse((HERE/filename).read_text())
            calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=="subprocess.Popen"]
            self.assertEqual(len(calls),1,filename)
            self.assertTrue(any(k.arg=="pass_fds" and "amendment.lifetime_fd()" in ast.unparse(k.value) for k in calls[0].keywords),filename)
        self.assertIn("amendment.check_lifetime()",ast.unparse(node(HERE/"native_child_v1.py","_save_json")))
        self.assertIn("amendment.check_lifetime()",ast.unparse(node(HERE/"context_factory_v1.py","_fast_guard")))
        self.assertIn("amendment.check_lifetime()",ast.unparse(node(HERE/"context_factory_v1.py","_PrimitiveJSONWriter")))
    def test_no_new_accounting_or_positive_numeric_signals(self):
        for name in a.required_execution_sources():
            path=a.ROOT/name
            if path==a.ROOT/a.BACKEND_REF["path"]:continue
            text=path.read_text();tree=ast.parse(text)
            for call in [n for n in ast.walk(tree) if isinstance(n,ast.Call)]:
                fn=ast.unparse(call.func)
                self.assertNotIn(fn.split(".")[-1],["apply_reserve_v1","apply_retire_failed_H039_v1"])
                if fn in ["os.kill","os.killpg"]:
                    self.assertGreaterEqual(len(call.args),2)
                    self.assertEqual(ast.literal_eval(call.args[1]),0,name)
            if name.endswith("root_native_driver_v1.py"):self.assertNotIn("def admit(",text)
    def test_science_and_probed_backend_preserved(self):
        v5=a.ROOT/"research/reviews/American_C3_native_MLM_full_source_v5"
        for filename,names in [("qualifier_v1.py",["qualify"]),("native_child_v1.py",["_int_time","_final_checkpoint","_verify_author_import_origins","main"])]:
            for name in names:
                old=ast.dump(node(v5/filename,name),include_attributes=False)
                new=ast.dump(node(HERE/filename,name),include_attributes=False)
                self.assertEqual(old.replace(str(v5.relative_to(a.ROOT)),a.REL),new,(filename,name))
        self.assertEqual(hashlib.sha256((a.ROOT/a.BACKEND_REF["path"]).read_bytes()).hexdigest(),"61be5d89357d1b95611e0ba26dd39d381ececf3ce6a36c347507cdc062373eca")
        for name in a.required_execution_sources():ast.parse((a.ROOT/name).read_text(),filename=name)
if __name__=="__main__":unittest.main()
