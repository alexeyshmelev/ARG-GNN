"""Pure mocked source tests; no real socket, rename, unlink, GPU, guard or payload."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import ast,copy,hashlib,importlib.util,io,json,os,stat,sys,unittest
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/"research/reviews/American_C3_native_MLM_full_fit_v8"
def load(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+".py"))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def s(mode,ino=1,size=0,nlink=1):
    return NS(st_dev=1,st_ino=ino,st_mode=mode,st_nlink=nlink,st_size=size,st_mtime_ns=1,st_ctime_ns=1)
class Checks(unittest.TestCase):
    def test_fixed_terminal_capture_commit_checksum_and_no_growth(self):
        c=load("terminal_capture_v9");raw=bytearray(c.WAIT_BYTES);writes=[]
        obj=c.Capture.__new__(c.Capture);obj.role="native";obj.fds={"wait":10};obj.committed=set()
        obj.manifest={"files":{"native.wait":[1,2,c.WAIT_BYTES]}}
        def write(fd,data,off):raw[off:off+len(data)]=data;writes.append((off,len(data)));return len(data)
        with patch.object(c.os,"fstat",return_value=s(stat.S_IFREG|0o600,2,c.WAIT_BYTES)),patch.object(c.os,"pwrite",side_effect=write),patch.object(c.os,"fsync") as sync:
            obj.event(0,{"actual_direct_child_wait":True,"returncode":1,"drain_state":"not_yet_certified"})
            self.assertTrue(sync.call_count>=2)
            self.assertTrue(all(a+n<=c.WAIT_BYTES for a,n in writes))
            self.assertEqual(c.decode_slot(bytes(raw),0)["returncode"],1)
            with self.assertRaises(ValueError):obj.event(0,{"returncode":0})
            with self.assertRaises(ValueError):obj.event(1,{"huge":"x"*c.WAIT_BYTES})
        raw[c.HEADER]^=1
        with self.assertRaises(ValueError):c.decode_slot(bytes(raw),0)
    def test_raw_tail_retains_bytes_without_JSON_expansion(self):
        c=load("terminal_capture_v9");obj=c.Capture.__new__(c.Capture);obj.role="native";obj.fds={"tail":10}
        obj.manifest={"files":{"native.tail":[1,2,4194304]}}
        data=b"\\\xff\x00actual";written=[]
        with patch.object(c.os,"fstat",return_value=s(stat.S_IFREG|0o600,2,4194304)),patch.object(c.os,"pwrite",side_effect=lambda fd,b,o:(written.append(bytes(b)) or len(b))),patch.object(c.os,"fsync"):
            result=obj.tail(NS(parts=[data],total=100))
        self.assertEqual(b"".join(written),data);self.assertEqual(result["retained_raw_bytes"],len(data))
    def test_own_dead_sockets_preserved_before_unlink_and_live_rejected(self):
        m=load("terminal_namespace_v9");tmp=Path("/pure-test/tmp");directory=tmp/"pymp-own";node=directory/"listener-own"
        stats={tmp:s(stat.S_IFDIR|0o700),directory:s(stat.S_IFDIR|0o700),node:s(stat.S_IFSOCK|0o700,3)}
        events=[]
        def children(path):return iter([directory] if path==tmp else [node] if path==directory else [])
        with patch.object(Path,"exists",lambda path:path in stats),patch.object(Path,"lstat",lambda path:stats[path]),patch.object(Path,"iterdir",children),patch.object(Path,"read_text",return_value="header\n"),patch.object(Path,"unlink",lambda path:events.append(("unlink",str(path)))):
            rows=m.clean_dead_sockets(tmp,prove_drained=lambda:True,preserve=lambda value:events.append(("preserve",value)))
        self.assertEqual(len(rows),1);self.assertEqual([x[0] for x in events],["preserve","unlink"])
        live="Num RefCount Protocol Flags Type St Inode Path\n0 2 0 10000 1 01 1 "+str(node)+"\n"
        self.assertEqual(m.kernel_listeners([str(node)],live),[str(node)])
        with patch.object(m,"socket_candidates",return_value=rows),patch.object(Path,"read_text",return_value=live):
            with self.assertRaises(ValueError):m.clean_dead_sockets(tmp,prove_drained=lambda:True,preserve=lambda value:None)
        with self.assertRaises(ValueError):m.clean_dead_sockets(tmp,prove_drained=lambda:False,preserve=lambda value:None)
    def test_unexpected_symlink_or_hardlink_not_socket_cleanup(self):
        m=load("terminal_namespace_v9");tmp=Path("/pure-test/tmp");node=tmp/"unexpected"
        for mode,nlink in [(stat.S_IFLNK|0o700,1),(stat.S_IFREG|0o600,2)]:
            stats={tmp:s(stat.S_IFDIR|0o700),node:s(mode,2,nlink=nlink)}
            with patch.object(Path,"exists",return_value=True),patch.object(Path,"lstat",lambda path:stats[path]),patch.object(Path,"iterdir",lambda path:iter([node])):
                with self.assertRaises(ValueError):m.socket_candidates(tmp)
    def test_latest_alias_independent_same_bytes_with_no_follow(self):
        m=load("terminal_namespace_v9");directory=Path("/pure-test/checkpoints")
        alias=directory/"latest-rank0.pt";target=directory/"ep5-ba224775-rank0.pt";temp=directory/".latest-rank0.pt.regularizing-v9"
        data=b"actual-checkpoint-bytes";events=[];written=[];reads=[data,b"",data,b""]
        stats={directory:s(stat.S_IFDIR|0o700),alias:s(stat.S_IFLNK|0o777,3,25),target:s(stat.S_IFREG|0o600,2,len(data)),temp:s(stat.S_IFREG|0o600,4,len(data))}
        def openfd(path,flags,*args):
            events.append(("open",str(path),flags));return 10 if path==target else 11 if path==temp else 12
        def fstat(fd):return stats[target] if fd==10 else s(stat.S_IFREG|0o600,4,len(data))
        tree=ast.parse((HERE/"native_child_v1.py").read_text())
        names={"_int_time","optimizer_counter_evidence","_checkpoint_entries","_final_checkpoint"}
        ns={"Path":Path,"stat":stat,"os":os}
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),"<pure-semantic-checkpoint>","exec"),ns)
        with patch.object(Path,"lstat",lambda path:stats[path]),patch.object(Path,"iterdir",lambda path:iter([alias,target])),patch.object(Path,"open",lambda path,*a,**k:io.BytesIO(data)),patch.object(m.os,"readlink",return_value=target.name),patch.object(m.os,"open",side_effect=openfd),patch.object(m.os,"read",side_effect=lambda *a:reads.pop(0)),patch.object(m.os,"write",side_effect=lambda fd,b:(written.append(bytes(b)) or len(b))),patch.object(m.os,"fstat",side_effect=fstat),patch.object(m.os,"fsync"),patch.object(m.os,"close"),patch.object(m.os,"replace",side_effect=lambda a,b:events.append(("replace",str(a),str(b)))):
            loaded=[]
            values={"state":{"timestamp":{"epoch":5,"batch":224775,"batch_in_epoch":0},"model":{"actual":1},
                "optimizers":{"DecoupledAdamW":{"state":{0:{"step":224775,"exp_avg":1,"exp_avg_sq":2}}}}}}
            def read_checkpoint(path,**kwargs):loaded.append(path);return values
            semantic=ns["_final_checkpoint"](directory,epochs=5,updates=224775,max_files=2,max_bytes=1000,
                torch=NS(load=read_checkpoint),reference=lambda path:{"path":str(path),"sha256":hashlib.sha256(data).hexdigest()})
            self.assertEqual(loaded,[target]);self.assertFalse(semantic["checkpoint_latest_alias"]["alias_followed_or_opened"])
            self.assertFalse(semantic["checkpoint_latest_alias"]["normalized_future_bytes_claimed"])
            self.assertEqual(semantic["actual_optimizer_updates"],224775)
            self.assertFalse(any(x[0]=="replace" for x in events))
            result=m.normalize_latest(directory,prove_drained=lambda:True,preserve=lambda value:events.append(("preserve",value)))
        self.assertEqual(b"".join(written),data);self.assertEqual(result["sha256"],hashlib.sha256(data).hexdigest())
        self.assertEqual(events[0][0],"preserve")
        self.assertTrue(events[1][2]&os.O_NOFOLLOW)
        self.assertTrue(any(x[0]=="replace" for x in events))
    def test_latest_alias_escape_or_missing_drain_rejected(self):
        m=load("terminal_namespace_v9")
        with self.assertRaises(ValueError):m.normalize_latest(Path("/pure-test"),prove_drained=lambda:False,preserve=lambda v:None)
        for target in ["../outside.pt","/outside.pt","subdir/file.pt"]:
            with patch.object(Path,"lstat",return_value=s(stat.S_IFLNK|0o777,3,10)),patch.object(m.os,"readlink",return_value=target):
                with self.assertRaises(ValueError):m.normalize_latest(Path("/pure-test"),prove_drained=lambda:True,preserve=lambda v:None)
    def test_unchanged_science_optimizer_checkpoint_and_lifecycle(self):
        def fn(path,name):
            return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
        for name in ["_io_config","optimizer_counter_evidence"]:
            self.assertEqual(fn(OLD/"native_child_v1.py",name),fn(HERE/"native_child_v1.py",name))
        for name in ["_tree","_signal_owned"]:
            self.assertEqual(fn(OLD/"root_native_driver_v1.py",name),fn(HERE/"root_native_driver_v1.py",name))
        child=(HERE/"native_child_v1.py").read_text()
        self.assertIn("trainer=module.main(config,return_trainer=True)",child)
        self.assertIn('observed_optimizer["unique_step_values"]==[224775]',child)
    def test_wait_first_before_any_postprocess_in_both_layers(self):
        inner=(HERE/"supervisor_v1.py").read_text();outer=(HERE/"root_native_driver_v1.py").read_text()
        self.assertIn("returncode=proc.wait();capture_direct(returncode)",inner)
        self.assertIn("rc=proc.wait();capture_direct(rc)",outer)
        self.assertLess(inner.index('terminal.event(1'),inner.index('clean_dead_sockets(tmp'))
        self.assertLess(inner.index('terminal.event(1'),inner.index('ctx["boundary_guard"]("after_native'))
        self.assertLess(outer.index('terminal.event(1'),outer.index('complete_path=ROOT/output'))
        self.assertIn("amendment.lifetime_fd()",inner)
        self.assertIn("terminal.event(0",(HERE/"launcher_v9.py").read_text())
    def test_no_new_reservation_or_mutation_of_original_binding(self):
        driver=(HERE/"root_native_driver_v1.py").read_text()
        self.assertNotIn("apply_reserve_v1",driver)
        for node in ast.walk(ast.parse(driver)):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=="operation_draft_v1":
                self.assertNotEqual(ast.literal_eval(node.args[0]),"reserve")
        self.assertNotIn("--admit",driver)
        amendment=load("amendment_v9")
        self.assertEqual(amendment.CAP,14*2**30)
        self.assertEqual(amendment.EXPIRES,"2026-10-18T11:47:27.217781+00:00")
        self.assertEqual(amendment.ORIGINAL_DRIVER_REF["sha256"],"0af05152d32d58f5e1485b4d0ba33ebdc6aa099d92609cef5661407d756a7230")
        for row in amendment.FIXED_REFS.values():
            self.assertEqual(hashlib.sha256((ROOT/row["path"]).read_bytes()).hexdigest(),row["sha256"])
    def test_truthful_execution_authority_preserves_old_sources_and_science(self):
        m=load("amendment_v9")
        old=json.loads((ROOT/m.ORIGINAL_INPUT_REF["path"]).read_text())
        storage=json.loads((ROOT/m.ORIGINAL_STORAGE_REF["path"]).read_text())
        refs=lambda name:{"path":name,"sha256":"mock-only"}
        sources=dict(old["source_sha256"])
        for path in m.required_execution_sources():sources[path]=m.ref(path)["sha256"]
        closure={"schema":"native_MLM_fullfit_execution_source_closure_v9",
            "original_storage_ref":m.ORIGINAL_STORAGE_REF,"source_sha256":sources}
        spec=copy.deepcopy(old)
        spec.update(main_source_ref=m.ref(m.ENTRY),actual_launcher_ref=m.ref(m.LAUNCHER),
            root_context_module_ref=m.ref(m.REL+"/context_factory_v1.py"),source_sha256=sources,
            transitive_source_closure_ref=refs("closure"),input_binding_mode="actual_root_execution_amendment_v9")
        auth={"schema":"native_MLM_fullfit_execution_authority_v9","root_authorizer":"root_agent","active":True,
            "key":m.KEY,"job_id":m.JOB,"original_refs":m.FIXED_REFS,"reserved_bytes":m.CAP,
            "created_at_utc":m.CREATED,"expires_at_utc":m.EXPIRES,"physical_gpu":4,"GPU_UUID":m.GPU_UUID,
            "accounting_or_original_binding_mutation_authorized":False,"scientific_scope_changed":False,
            "all_v8_launch_resume_write_capabilities_cooperatively_revoked":True,
            "immutable_old_code_mechanical_rejection_claimed":False,
            "dead_owned_socket_cleanup_after_preserved_metadata_authorized":True,
            "checkpoint_latest_regular_copy_after_drain_authorized":True,"actual_old_outer_wait_ref":refs("outer"),
            "old_writer_absence_ref":refs("absence"),"archive_receipt_ref":refs("archive"),
            "source_closure_ref":refs("closure"),"input_ref":refs("input")}
        data={"auth":auth,m.ORIGINAL_INPUT_REF["path"]:old,m.ORIGINAL_STORAGE_REF["path"]:storage,
            "outer":{"actual_direct_outer_supervisor_wait":True,"pid":2817234,"returncode":1},
            "absence":{"root_authorizer":"root_agent","active":True,"key":m.KEY,
                "all_old_writers_and_capabilities_listed":True,"all_old_launch_resume_write_capabilities_revoked":True,
                "native_wait_exit":"UNKNOWN","actual_optimizer_updates":"UNKNOWN","exact_old_kernel_listener_matches":[]},
            "archive":{"root_authorizer":"root_agent","active":True,"all_scientific_and_regular_files_preserved":True,
                "all_socket_metadata_preserved_before_narrow_cleanup":True,"archive_regular_namespace_confirmed":True,
                "same_filesystem_rename":True,"retained_archive_and_metadata_bytes":1000000,"old_writer_absence_ref":refs("absence")},
            "closure":closure,"input":spec}
        with patch.object(m,"read",side_effect=lambda ref:copy.deepcopy(data[ref["path"]])),patch.object(m,"check_expiry"):
            m.validate_authority(refs("auth"))
            for kind in ["new_expiry","changed_science","missing_old_source","fake_native_success"]:
                saved=copy.deepcopy(data)
                if kind=="new_expiry":data["auth"]["expires_at_utc"]="2099-01-01T00:00:00+00:00"
                if kind=="changed_science":data["input"]["refs"]["config"]={"path":"shortened","sha256":"fake"}
                if kind=="missing_old_source":data["closure"]["source_sha256"].pop(next(iter(old["source_sha256"])))
                if kind=="fake_native_success":data["absence"]["native_wait_exit"]=0
                with self.assertRaises(ValueError):m.validate_authority(refs("auth"))
                data.clear();data.update(saved)
    def test_all_actual_sources_exist_and_original_caps_are_accepted(self):
        sys.path.insert(0,str(ROOT/"src"))
        c=load("root_native_contract_v1")
        for path in c.source_paths():self.assertTrue((ROOT/path).is_file(),path)
        m=load("amendment_v9")
        request=json.loads((ROOT/m.ORIGINAL_REQUEST_REF["path"]).read_text())
        caps=json.loads((ROOT/m.ORIGINAL_CAPS_REF["path"]).read_text())
        c.validate_caps(caps,request)
        self.assertEqual(c.make_config(request,caps),json.loads((ROOT/m.ORIGINAL_CONFIG_REF["path"]).read_text()))
        from datetime import datetime,timedelta
        m.check_expiry({"created_at_utc":m.CREATED,"expires_at_utc":m.EXPIRES},
            now=datetime.fromisoformat(m.CREATED)+timedelta(hours=1))
        with self.assertRaises(ValueError):m.check_expiry({"created_at_utc":m.CREATED,"expires_at_utc":"2099-01-01T00:00:00+00:00"})
    def test_archive_helper_requires_explicit_authority_and_absence_before_mutation(self):
        m=load("root_archive_v9");archive=ROOT/"pure-source-only-archive"
        authority=dict(root_authorizer="root_agent",active=True,archive_failed_v8_by_same_filesystem_rename=True,
            preserve_all_scientific_and_regular_files=True,narrow_dead_socket_cleanup_after_metadata=True,
            archive_allowance_bytes=5242880,old_output=str(m.OLD_OUTPUT),old_tmp=str(m.OLD_TMP),
            archive_root=str(archive),old_writer_absence_ref={"path":"mock","sha256":"mock"})
        with patch.object(Path,"resolve",lambda p:p),patch.object(Path,"exists",return_value=False),patch.object(Path,"mkdir") as mkdir,patch.object(m.os,"rename") as rename,patch.object(m,"snapshot") as snapshot:
            with self.assertRaises(ValueError):m.preserve_failed_v8(archive,authority=authority,prove_old_writers_absent=lambda:False)
            mkdir.assert_not_called();rename.assert_not_called();snapshot.assert_not_called()
    def test_evidence_failure_does_not_skip_owned_drain(self):
        for name in ["supervisor_v1.py","root_native_driver_v1.py"]:
            tree=ast.parse((HERE/name).read_text())
            nested=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="capture_direct")
            self.assertTrue(any(isinstance(n,ast.Try) and n.handlers for n in nested.body))
    def test_immediate_wait_stdout_precedes_capture_and_bounded_relay(self):
        for name in ["supervisor_v1.py","root_native_driver_v1.py"]:
            tree=ast.parse((HERE/name).read_text())
            fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="capture_direct")
            source=ast.unparse(fn)
            self.assertLess(source.index("print(json.dumps"),source.index("terminal.event"))
        c=load("terminal_capture_v9");notices=[];relay=c.WaitRelay(emit=notices.append)
        line=b'{"event":"actual_direct_native_wait_v9","pid":123,"returncode":1}\n'
        relay.feed(b"unrelated author console\n"+line[:12]);relay.feed(line[12:])
        self.assertEqual(len(notices),1);self.assertEqual(json.loads(notices[0])["returncode"],1)
        for _ in range(100):relay.feed(line)
        self.assertEqual(len(notices),8)
    def test_exclusive_lease_repeat_uses_same_original_job_and_honest_logical_identity(self):
        m=load("amendment_v9");launcher=(HERE/"launcher_v9.py").read_text()
        self.assertIn('"ARGGNN_LOGICAL_JOB_NAME": args.job',launcher)
        self.assertIn("fcntl.LOCK_EX | fcntl.LOCK_NB",launcher)
        self.assertIn('return record["index"]==4',launcher)
        self.assertNotIn("gpu_pool_v1",launcher)
        source=(ROOT/"src/arggnn/census_tsinfer_fit_v1.py").read_text()
        fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="require_logical_job_v1")
        ns={"os":os}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),"<pure-original-logical-job-check>","exec"),ns)
        lease={"launcher_metadata":{"job":m.JOB}};spec={"job_id":m.JOB}
        with patch.dict(os.environ,{"ARGGNN_JOB_NAME":m.JOB,"ARGGNN_LOGICAL_JOB_NAME":m.JOB}):
            ns["require_logical_job_v1"](lease,spec)
        with patch.dict(os.environ,{"ARGGNN_JOB_NAME":m.JOB,"ARGGNN_LOGICAL_JOB_NAME":"other-attempt"}):
            with self.assertRaises(ValueError):ns["require_logical_job_v1"](lease,spec)
    def test_syntax_and_sys_path_call_shapes(self):
        for path in HERE.glob("*.py"):
            tree=ast.parse(path.read_text(),filename=str(path))
            for n in ast.walk(tree):
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and ast.unparse(n.func)=="sys.path.insert":
                    self.assertEqual(len(n.args),2)
if __name__=="__main__":unittest.main()
