"""Focused pure mocks for honest lost-history/current-writer correction."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import copy,hashlib,importlib.util,inspect,json,os,sys,time,unittest
HERE=Path(__file__).resolve().parent
def load(name):
    sp=importlib.util.spec_from_file_location(name,HERE/(name+".py"))
    m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
def current_fixture():
    m=load("current_writer_proof_v9")
    rootrows=[{"pid":pid,"absent":True,"method":"targeted os.kill(pid,0) ESRCH","checked_utc":"2026-10-15T12:00:00+00:00"} for pid in m.KNOWN_ROOTS]
    groups=[{"group":g,"absent":True,"method":"targeted os.killpg(group,0) ESRCH","checked_utc":"2026-10-15T12:00:00+00:00"} for g in m.KNOWN_GROUPS]
    scan={"complete":True,"stable_same_uid_membership":True,"observer_effective_uid":os.geteuid(),
        "same_uid_processes_classified":12,"current_matching_processes":[],
        "unrelated_process_contents_recorded":False}
    proof={"schema":"native_MLM_honest_current_writer_observation_v9_v2",
        "checked_utc":"2026-10-15T12:00:00+00:00","historical_ids_complete":False,
        "historical_descendant_ids_lost":True,"native_wait_exit":"UNKNOWN","actual_optimizer_updates":"UNKNOWN",
        "known_old_roots":rootrows,"known_original_groups":groups,"current_same_uid_scan":scan,
        "external_NFS_writers_outside_observation":True,"other_UID_and_remote_host_writers_outside_observation":True,
        "future_relaunch_prevented_by_cooperative_root_revocation":True}
    cert={"schema":"native_MLM_fullfit_current_writer_absence_root_v9_v2","root_authorizer":"root_agent",
        "active":True,"key":m.KEY,"historical_ids_complete":False,"historical_descendant_ids_lost":True,
        "all_v8_future_launch_resume_write_capabilities_cooperatively_revoked":True,
        "native_wait_exit":"UNKNOWN","actual_optimizer_updates":"UNKNOWN","exact_old_kernel_listener_matches":[],
        "external_NFS_writers_outside_observation":True,"current_writer_observation":proof}
    return m,proof,cert
class Focused(unittest.TestCase):
    def test_exact_old_metadata_matches_and_unrelated_contents_discarded(self):
        m,_,_=current_fixture()
        metadata=(b"python\x00unrelated-secret-content\x00",
            b"PRIVATE_TOKEN=unrelated-secret-content\x00ARGGNN_JOB_NAME="+m.JOB.encode()+b"\x00",
            {"1":"/unrelated/secret"})
        with patch.object(m,"_process_identity",return_value=(111,"S")),patch.object(Path,"lstat",return_value=NS(st_uid=1000)),patch.object(m,"_metadata",return_value=metadata):
            reasons=m._classify(123,111,1000,time.monotonic()+10)
        self.assertEqual(reasons,["exact_old_job_environment"])
        self.assertNotIn("unrelated-secret",json.dumps(reasons))
        self.assertEqual(m._path_reasons(m.OUTPUT+"-other/file"),[])
        self.assertTrue(m._path_reasons(m.TMP+"/mlm/pymp-own"))
        job_argv=(b"python"+bytes([0])+b"--job"+bytes([0])+m.JOB.encode()+bytes([0]),b"",{})
        with patch.object(m,"_process_identity",return_value=(111,"S")),patch.object(Path,"lstat",return_value=NS(st_uid=1000)),patch.object(m,"_metadata",return_value=job_argv):
            self.assertEqual(m._classify(123,111,1000,time.monotonic()+10),["exact_old_job_argv"])
    def test_complete_stable_current_scan_and_nonempty_match_blocks(self):
        m,proof,_=current_fixture()
        with patch.object(m,"_snapshot",return_value={101:1,102:2}),patch.object(m,"_classify",return_value=[]):
            scan=m.scan_current_same_uid()
        self.assertTrue(scan["complete"]);self.assertEqual(scan["same_uid_processes_classified"],2)
        self.assertNotIn("101",json.dumps(scan))
        bad=copy.deepcopy(scan);bad["current_matching_processes"]=[{"pid":101,"start_ticks":1,"match_reasons":["exact_old_scope_path"]}]
        with patch.object(m,"observe_known_roots_and_groups",return_value=(proof["known_old_roots"],proof["known_original_groups"])),patch.object(m,"scan_current_same_uid",return_value=bad):
            with self.assertRaises(ValueError):m.prove_current_absence()
    def test_churn_limit_permission_and_metadata_limits_fail_closed(self):
        m,_,_=current_fixture()
        with patch.object(m,"_snapshot",side_effect=[{1:1},{2:2}]*3),patch.object(m,"_classify",return_value=[]):
            with self.assertRaises(ValueError):m.scan_current_same_uid()
        with patch.object(m,"_snapshot",return_value={1:1}),patch.object(m,"_classify",side_effect=PermissionError("unreadable metadata")):
            with self.assertRaises(PermissionError):m.scan_current_same_uid()
        with patch.object(m.os,"open",return_value=10),patch.object(m.os,"close"),patch.object(m.os,"read",side_effect=[b"abc",b"def"]):
            with self.assertRaises(ValueError):m._read(Path("/pure-test"),4)
        with patch.object(Path,"iterdir",return_value=iter([Path(str(i)) for i in range(m.MAX_PROCESSES+1)])):
            with self.assertRaises(ValueError):m._snapshot(1000,time.monotonic()+10)
    def test_self_fd_iterator_stays_open_and_missing_fds_still_fail(self):
        m,_,_=current_fixture();state={"open":False}
        class Entries:
            def __enter__(self):state["open"]=True;return self
            def __iter__(self):
                for name in ("0","1","99"):
                    yield NS(name=name,path="/mock-proc/123/fd/"+name)
            def __exit__(self,*args):state["open"]=False
        def target(path):
            name=path.rsplit("/",1)[-1]
            if name=="99" and not state["open"]:raise FileNotFoundError("closed self-inspection fd")
            return "/mock-proc/123/fd" if name=="99" else "/unrelated/"+name
        # The old listdir-shaped sequence materialized names before target reads.
        with Entries() as entries:stale=list(entries)
        with self.assertRaises(FileNotFoundError):target(stale[-1].path)
        with patch.object(m.os,"scandir",side_effect=lambda _:Entries()),patch.object(m.os,"readlink",side_effect=target):
            first=m._fd_targets(123,time.monotonic()+10)
            second=m._fd_targets(123,time.monotonic()+10)
        self.assertEqual(first,second);self.assertEqual(first["99"],"/mock-proc/123/fd")
        self.assertFalse(state["open"])
        for pid in (os.getpid(),456):
            with patch.object(m.os,"scandir",side_effect=lambda _:Entries()),patch.object(m.os,"readlink",side_effect=FileNotFoundError("gone fd")):
                with self.assertRaises(m.Changed):m._fd_targets(pid,time.monotonic()+10)
    def test_all_known_roots_and_groups_signal_zero_only_repeated_after_scan(self):
        m,proof,_=current_fixture();calls=[]
        def absent(pid,sig):calls.append(("pid",pid,sig));raise ProcessLookupError()
        def absent_group(pid,sig):calls.append(("group",pid,sig));raise ProcessLookupError()
        with patch.object(m.os,"kill",side_effect=absent),patch.object(m.os,"killpg",side_effect=absent_group),patch.object(m,"scan_current_same_uid",return_value=proof["current_same_uid_scan"]):
            observed=m.prove_current_absence()
        self.assertEqual(len(calls),16);self.assertTrue(all(row[2]==0 for row in calls))
        self.assertFalse(observed["historical_ids_complete"]);self.assertEqual(observed["native_wait_exit"],"UNKNOWN")
        with patch.object(m.os,"kill",return_value=None):
            with self.assertRaises(ValueError):m.observe_known_roots_and_groups()
    def test_historical_complete_fake_or_partial_proofs_rejected(self):
        m,proof,cert=current_fixture();m.validate_static_certificate(cert)
        for kind in ["complete_history","missing_group","partial_scan","native_success"]:
            bad=copy.deepcopy(cert)
            if kind=="complete_history":bad["historical_ids_complete"]=True
            if kind=="missing_group":bad["current_writer_observation"]["known_original_groups"].pop()
            if kind=="partial_scan":bad["current_writer_observation"]["current_same_uid_scan"]["complete"]=False
            if kind=="native_success":bad["native_wait_exit"]=0
            with self.assertRaises(ValueError):m.validate_static_certificate(bad)
    def test_archive_has_no_constant_proof_callback_and_live_path_has_no_rescan(self):
        archive=load("root_archive_v9")
        self.assertNotIn("prove_old_writers_absent",inspect.signature(archive.preserve_failed_v8).parameters)
        source=(HERE/"root_archive_v9.py").read_text()
        self.assertIn("initial_proof=prove_current_absence()",source)
        amendment=(HERE/"amendment_v9.py").read_text()
        import ast
        tree=ast.parse(amendment)
        for name in ["validate_authority","check_lifetime"]:
            fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertNotIn("prove_current_absence()",ast.get_source_segment(amendment,fn))
        activate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="activate")
        self.assertEqual(ast.get_source_segment(amendment,activate).count("preserve_prestart_proof("),2)
        driver=(HERE/"root_native_driver_v1.py").read_text()
        self.assertIn('all_current_v9_attempt_owned_PIDs_and_groups_listed',driver)
        self.assertIn('historical_v8_ids_complete=False',driver)
        self.assertNotIn('all_genuinely_owned_PIDs_and_groups_listed',driver)
    def test_pre_fix_packet_and_original31pins_preserved(self):
        snap=HERE/"pre_current_writer_fix_v1"
        manifest=json.loads((snap/"preservation_manifest.json").read_text())
        for row in manifest["files"]:
            self.assertEqual(hashlib.sha256((HERE.parents[2]/row["preserved_path"]).read_bytes()).hexdigest(),row["sha256"])
        old=json.loads((snap/"execution_source_closure.json").read_text())
        original=json.loads((HERE.parents[2]/"research/reviews/American_C3_native_MLM_full_fit_v8/admitted_mlm_v1/input.json").read_text())
        for path,sha in original["source_sha256"].items():
            self.assertEqual(old["source_sha256"][path],sha)
            self.assertEqual(hashlib.sha256((HERE.parents[2]/path).read_bytes()).hexdigest(),sha)
if __name__=="__main__":unittest.main()
