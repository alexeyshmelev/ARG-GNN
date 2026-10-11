"""Tiny handmade arithmetic/stat/proc cases; never inspect real products or /proc."""
from pathlib import Path
import contextlib
import io
import json
import os
import sys
import unittest
import uuid
from unittest import mock

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
sys.path.insert(0, str(ROOT / "scripts"))
import project_budget_v6 as b

FIXTURES = ROOT / "research/reviews/project_budget_v6_author" / ("handmade_" + uuid.uuid4().hex)
FIXTURES.mkdir()

class Cases(unittest.TestCase):
    def setUp(self):
        self.root = FIXTURES / self._testMethodName
        self.root.mkdir()
        self.scope = self.root / "output"
        self.scope.mkdir()
        (self.scope / "tiny.txt").write_text("x")
        self.r = {"key":"old", "reserved_bytes":100, "output":str(self.scope),
                  "auxiliary_scopes":[], "consumer_bindings":[], "root_admission":{}}
        self.view = {"projected_bytes":850, "limit_bytes":1000, "hard_limit_bytes":1100}
    def close(self, retained=20):
        return {"reservation_sha256":b.digest(self.r),
                "measurement":{"conservative_scope_bytes":retained}}
    def fakeproc(self, flags=None, mapping=False, state="S", cwd=None, directory=False):
        proc=self.root/"proc";proc.mkdir();p=proc/"11";p.mkdir()
        (p/"stat").write_text("11 (toy) "+" ".join([state]+["0"]*18+["123"]))
        (p/"cwd").symlink_to(cwd or self.root)
        (p/"fd").mkdir();(p/"fdinfo").mkdir()
        if flags is not None:
            (p/"fd"/"1").symlink_to(self.scope if directory else self.scope/"tiny.txt")
            (p/"fdinfo"/"1").write_text("flags:\t"+flags+"\n")
        (p/"maps").write_text("0-1 rw-s 0 00:00 0 "+str(self.scope/"tiny.txt")+"\n" if mapping else "")
        return proc
    def test_open_unchanged(self):
        self.assertEqual(b.reconcile(self.view,{"old":self.r},{},{})["projected_bytes"],850)
    def test_closed_replace_once(self):
        a=b.reconcile(self.view,{"old":self.r},{},{"old":self.close()})
        self.assertEqual(a["projected_bytes"],770);self.assertEqual(a["closed_allowance_credit_bytes"],80)
    def test_add_other_reservation_preserves_baseline(self):
        new={**self.r,"key":"new","reserved_bytes":50}
        self.assertEqual(b.reconcile(self.view,{"old":self.r},{"new":new},{"old":self.close()})["projected_bytes"],820)
    def test_zero_retained(self):
        self.assertEqual(b.reconcile(self.view,{"old":self.r},{},{"old":self.close(0)})["projected_bytes"],750)
    def test_full_retained(self):
        self.assertEqual(b.reconcile(self.view,{"old":self.r},{},{"old":self.close(100)})["projected_bytes"],850)
    def test_oversize_rejected(self):
        with self.assertRaises(ValueError):b.reconcile(self.view,{"old":self.r},{},{"old":self.close(101)})
    def test_unknown_closed_rejected(self):
        with self.assertRaises(ValueError):b.reconcile(self.view,{"old":self.r},{},{"missing":self.close()})
    def test_duplicate_key_rejected(self):
        with self.assertRaises(ValueError):b.reconcile(self.view,{"old":self.r},{"old":self.r},{})
    def test_drift_rejected(self):
        with self.assertRaises(ValueError):b.reconcile(self.view,{"old":{**self.r,"reserved_bytes":101}},{},{"old":self.close()})
    def test_strict_ceiling(self):
        with self.assertRaises(ValueError):b.reconcile({**self.view,"projected_bytes":1000},{},{},{})
    def test_handmade_measured_manifest(self):
        a=b.measure(self.root,[self.scope])
        self.assertGreater(a["conservative_scope_bytes"],0)
        self.assertEqual(a,b.measure(self.root,[self.scope]))
    def test_missing_scope(self):
        a=b.measure(self.root,[self.root/"absent"])
        self.assertEqual(a["conservative_scope_bytes"],0)
    def test_scope_drift(self):
        a=b.measure(self.root,[self.scope])
        (self.scope/"tiny.txt").write_text("changed")
        self.assertNotEqual(a,b.measure(self.root,[self.scope]))
    def test_hardlink_rejected(self):
        os.link(self.scope/"tiny.txt",self.scope/"link")
        with self.assertRaises(ValueError):b.measure(self.root,[self.scope])
    def test_symlink_rejected(self):
        (self.scope/"link").symlink_to(self.scope/"tiny.txt")
        with self.assertRaises(ValueError):b.measure(self.root,[self.scope])
    def test_absent_terminal_no_writer(self):
        proc=self.fakeproc()
        a=b.no_live_writers([self.scope],[99],proc_root=proc)
        self.assertFalse(a["terminal_pids_live"]);self.assertFalse(a["remote_writers_verified"])
    def test_live_recorded_pid_rejected(self):
        with self.assertRaises(ValueError):b.no_live_writers([self.scope],[11],proc_root=self.fakeproc())
    def test_zombie_terminal_rejected(self):
        with self.assertRaises(ValueError):
            b.no_live_writers([self.scope],[11],proc_root=self.fakeproc(state="Z"))
    def test_writable_fd_rejected(self):
        with self.assertRaises(ValueError):b.no_live_writers([self.scope],[99],proc_root=self.fakeproc(flags="0100001"))
    def test_readonly_fd_allowed(self):
        b.no_live_writers([self.scope],[99],proc_root=self.fakeproc(flags="0100000"))
    def test_directory_fd_rejected(self):
        with self.assertRaises(ValueError):b.no_live_writers([self.scope],[99],proc_root=self.fakeproc(flags="0100000",directory=True))
    def test_writable_mapping_rejected(self):
        with self.assertRaises(ValueError):b.no_live_writers([self.scope],[99],proc_root=self.fakeproc(mapping=True))
    def test_scope_cwd_rejected(self):
        with self.assertRaises(ValueError):b.no_live_writers([self.scope],[99],proc_root=self.fakeproc(cwd=self.scope))
    def test_own_user_unreadable_rejected(self):
        proc=self.fakeproc()
        with mock.patch.object(b,"proc_identity",side_effect=PermissionError):
            with self.assertRaises(ValueError):b.no_live_writers([self.scope],[99],proc_root=proc)
    def journal_event(self, sequence=1, previous=None):
        return {"schema":"project_budget_v6_event","root":str(self.root),"sequence":sequence,
          "previous_sha256":previous,"bindings":{},"operation":"close","authority":{},"value":{}}
    def write_event(self,event,name="000001.json"):
        d=self.root/b.JOURNAL_REL;d.mkdir(parents=True,exist_ok=True)
        p=d/name;p.write_bytes(b.v4.encoded(event)+b"\n")
        return p
    def test_empty_journal(self):
        self.assertEqual(b.load_journal(self.root,{}),([],None))
    def test_journal_chain(self):
        first=self.write_event(self.journal_event())
        sha=b.v4.file_bytes(self.root,first)[1]
        second=self.write_event(self.journal_event(2,sha),"000002.json")
        records,tail=b.load_journal(self.root,{})
        self.assertEqual(len(records),2);self.assertEqual(tail,b.v4.file_bytes(self.root,second)[1])
    def test_journal_gap_rejected(self):
        self.write_event(self.journal_event(2),"000002.json")
        with self.assertRaises(ValueError):b.load_journal(self.root,{})
    def test_journal_previous_digest_rejected(self):
        self.write_event(self.journal_event())
        self.write_event(self.journal_event(2,"0"*64),"000002.json")
        with self.assertRaises(ValueError):b.load_journal(self.root,{})
    def test_journal_foreign_file_rejected(self):
        self.write_event(self.journal_event(),"unknown.json")
        with self.assertRaises(ValueError):b.load_journal(self.root,{})
    def test_journal_byte_cap(self):
        self.write_event(self.journal_event())
        with mock.patch.object(b,"MAX_JOURNAL_BYTES",8):
            with self.assertRaises(ValueError):b.load_journal(self.root,{})
    def test_append_only_no_overwrite(self):
        lp=self.root/b.v4.LEGACY;lp.parent.mkdir(parents=True);lp.write_text("{}")
        sp=self.root/b.v4.SUPPLEMENTAL;sp.parent.mkdir(parents=True);sp.write_text("{}")
        before={"journal_events":0,"journal_tail_sha256":None,
           "legacy_ledger_sha256":b.v4.file_bytes(self.root,lp)[1],
           "supplemental_ledger_sha256":b.v4.file_bytes(self.root,sp)[1]}
        sha=b.append_event(self.root,self.journal_event(),before,{}, {})
        path=self.root/b.JOURNAL_REL/"000001.json";raw=path.read_bytes()
        with self.assertRaises(ValueError):b.append_event(self.root,self.journal_event(),before,{}, {})
        self.assertEqual(path.read_bytes(),raw);self.assertEqual(sha,b.v4.file_bytes(self.root,path)[1])
        self.assertEqual(lp.read_text(),"{}");self.assertEqual(sp.read_text(),"{}")
    def test_unknown_exit_valid_with_truthful_attestation(self):
        lease={"lease_path":str(self.root/".runtime/locks/gpu_2.lock"),"lease_device":0,
          "lease_inode":1,"launcher_metadata":{"launcher_pid":12}}
        cp=self.scope/"completed.json";cp.write_text(json.dumps({"status":"completed_toy","runtime":{"pid":11,"lease":lease}}))
        cr={"path":str(cp),"sha256":b.v4.file_bytes(self.root,cp)[1]}
        op=self.root/"observation.json";op.write_text("{}")
        observed={"path":str(op),"sha256":b.v4.file_bytes(self.root,op)[1]}
        terminal={"kind":"project_budget_v6_terminal_receipt","root_authorizer":"root_agent",
          "reservation_sha256":b.digest(self.r),"completed":cr,"pids":[11,12],
          "evidence_method":"root_observed_absence","status":"driver_completed_launcher_exit_unknown",
          "returncode":None,"exit_status_known":False,"observed_utc":"2026-10-11T00:00:00Z",
          "recorded_pids_absent":True,"original_lease_exclusive_acquire_release_verified":True,
          "no_recorded_gpu_context":True,"external_NFS_writers_outside_observation":True,
          "lease_path":lease["lease_path"],"original_lease_identity":{"device":0,"inode":1},
          "process_observation":observed,"lease_observation":observed,"gpu_observation":observed}
        tp=self.root/"terminal.json";tp.write_text(json.dumps(terminal))
        request={"key":"old","reservation_sha256":b.digest(self.r),"completed":cr,
          "completed_identity":{"status":"completed_toy"},"terminal_receipt":{"path":str(tp),"sha256":b.v4.file_bytes(self.root,tp)[1]},
          "scope_paths":[str(self.scope)],"expected_scope_measurement":{}}
        self.assertEqual(b.terminal_binding(self.root,request,self.r,[self.scope]),[11,12])
    def test_closed_state_and_drift(self):
        self.r["reserved_bytes"]=1000000
        view={"projected_bytes":3000000,"limit_bytes":10000000,"hard_limit_bytes":11000000,
          "legacy_ledger_sha256":"a"*64,"supplemental_ledger_sha256":"b"*64}
        measurement=b.measure(self.root,[self.scope])
        request={"key":"old","reservation_sha256":b.digest(self.r),"expected_scope_measurement":measurement}
        authority={"expected_before":{**b.pins({**view,"journal_tail_sha256":None})},
          "closure":request,"scope_write_authority_revoked_for_all_cooperative_project_jobs":True,
          "closed_scope_future_writes_and_relaunch_revoked":True,"external_NFS_writers_outside_observation":True}
        ap=self.root/"authority.json";ap.write_text(json.dumps(authority))
        ref={"path":str(ap),"sha256":b.v4.file_bytes(self.root,ap)[1]}
        bindings={"v4_implementation":{},"v4_config":{},"human_authorization":{}}
        event={**self.journal_event(),"bindings":bindings,"authority":ref,
          "value":{"request":request,"reservation_sha256":b.digest(self.r),"measurement":measurement,"writer_check":{}}}
        self.write_event(event)
        with mock.patch.object(b.v4,"current",return_value=({"jobs":{}},{"reservations":{"old":self.r}},view)), \
             mock.patch.object(b.v4,"scope_spec",return_value=[self.scope]), \
             mock.patch.object(b,"check_authority",return_value=authority), \
             mock.patch.object(b,"terminal_binding",return_value=[99]), \
             mock.patch.object(b,"no_live_writers",return_value={}):
            account=b.state(self.root,{},bindings)[4]
            self.assertEqual(account["projected_bytes"],2000000+measurement["conservative_scope_bytes"])
            (self.scope/"tiny.txt").write_text("changed")
            with self.assertRaises(ValueError):b.state(self.root,{},bindings)
            with mock.patch.object(b.v4,"current",return_value=({"jobs":{}},{"reservations":{"old":self.r}},
                              {**view,"supplemental_ledger_sha256":"c"*64})):
                with self.assertRaises(ValueError):b.state(self.root,{},bindings,verify_closed=False)
    def review_authority_fixture(self,review_kind="project_budget_v6_independent_review",verdict="PASS",wrong_source=False):
        bindings={"implementation":{"path":"scripts/project_budget_v6.py","sha256":"6"*64},
         "launcher":{"path":"scripts/launch_reserved_gpu_v3.py","sha256":"3"*64},
         "v4_implementation":{"path":"src/arggnn/project_budget_v4.py","sha256":"4"*64},
         "v4_config":{"path":"configs/project_budget_v4.json","sha256":"4"*64},
         "human_authorization":{"path":"human.json","sha256":"1"*64}}
        review={"kind":review_kind,"review_role":"independent_reviewer","verdict":verdict,
          **bindings,"actual_apply_performed":False}
        if wrong_source:review["implementation"]={"path":"scripts/project_budget_v5.py","sha256":"5"*64}
        rp=self.root/"review.json";rp.write_text(json.dumps(review))
        authority={"kind":"project_budget_v6_root_admission","root_authorizer":"root_agent","active":True,
          "operation":"reserve",**bindings,
          "all_budget_writers_use_scheduler_mutex":True,"all_new_consumers_use_v6_guard":True,
          "legacy_and_v4_new_admissions_frozen":True,"no_historical_record_mutation_authorized":True,
          "global_rebase_authorized":False,"GPU_authorized":False,"scientific_execution_authorized":False,
          "independent_review":{"path":str(rp),"sha256":b.v4.file_bytes(self.root,rp)[1]}}
        ap=self.root/"authority.json";ap.write_text(json.dumps(authority))
        return {"path":str(ap),"sha256":b.v4.file_bytes(self.root,ap)[1]},bindings
    def test_published_v5_fail_cannot_qualify(self):
        ref,bindings=self.review_authority_fixture("project_budget_v5_independent_review","FAIL")
        with self.assertRaises(ValueError):b.check_authority(self.root,ref,"reserve",bindings)
    def test_relabelled_pass_wrong_source_cannot_qualify(self):
        ref,bindings=self.review_authority_fixture(wrong_source=True)
        with self.assertRaises(ValueError):b.check_authority(self.root,ref,"reserve",bindings)
    def test_handmade_matching_v6_review_policy(self):
        ref,bindings=self.review_authority_fixture()
        self.assertEqual(b.check_authority(self.root,ref,"reserve",bindings)["kind"],"project_budget_v6_root_admission")
    def terminal_fixture(self, method="root_observed_absence", missing_worker=False, missing_launcher=False):
        lease={"lease_path":str(self.root/".runtime/locks/gpu_2.lock"),"lease_device":0,
          "lease_inode":1,"launcher_metadata":{"launcher_pid":12}}
        runtime={"pid":11,"lease":lease}
        if missing_worker:runtime.pop("pid")
        if missing_launcher:lease["launcher_metadata"].pop("launcher_pid")
        cp=self.scope/"completed.json";cp.write_text(json.dumps({"status":"completed_toy","runtime":runtime}))
        cr={"path":str(cp),"sha256":b.v4.file_bytes(self.root,cp)[1]}
        op=self.root/"observation.json";op.write_text("{}")
        observed={"path":str(op),"sha256":b.v4.file_bytes(self.root,op)[1]}
        terminal={"kind":"project_budget_v6_terminal_receipt","root_authorizer":"root_agent",
          "reservation_sha256":b.digest(self.r),"completed":cr,"pids":[11,12],
          "evidence_method":method,"status":"driver_completed_launcher_exit_unknown",
          "returncode":None,"exit_status_known":False,"observed_utc":"2026-10-11T00:00:00Z",
          "recorded_pids_absent":True,"original_lease_exclusive_acquire_release_verified":True,
          "no_recorded_gpu_context":True,"external_NFS_writers_outside_observation":True,
          "lease_path":lease["lease_path"],"original_lease_identity":{"device":0,"inode":1},
          "process_observation":observed,"lease_observation":observed,"gpu_observation":observed}
        if method=="supervisor_wait":
            wp=self.scope/"launcher_exit_v3.json"
            wait={"kind":"reserved_gpu_v3_supervisor_wait","job_id":"toy","launcher_pid":12,"worker_pid":11,
             "returncode":0,"wait_status_known":True,"completed_unix":1,"reservation_sha256":b.digest(self.r),
             "descendant_drain_or_no_live_writer_claim":False}
            wp.write_text(json.dumps(wait))
            terminal["supervisor_wait_evidence"]={"path":str(wp),"sha256":b.v4.file_bytes(self.root,wp)[1]}
        request={"key":"old","reservation_sha256":b.digest(self.r),"completed":cr,
          "completed_identity":{"status":"completed_toy"},"terminal_receipt":None,
          "scope_paths":[str(self.scope)],"expected_scope_measurement":{}}
        return request,terminal
    def bind_terminal(self,request,terminal):
        tp=self.root/"terminal.json";tp.write_text(json.dumps(terminal))
        request["terminal_receipt"]={"path":str(tp),"sha256":b.v4.file_bytes(self.root,tp)[1]}
        return b.terminal_binding(self.root,request,self.r,[self.scope])
    def test_contradictory_supervisor_unknown_rejected(self):
        request,terminal=self.terminal_fixture("supervisor_wait")
        terminal["returncode"]=0
        with self.assertRaises(ValueError):self.bind_terminal(request,terminal)
    def test_consistent_supervisor_accepted(self):
        request,terminal=self.terminal_fixture("supervisor_wait")
        terminal.update(status="driver_completed_launcher_exit_zero_known",returncode=0,exit_status_known=True)
        self.assertEqual(self.bind_terminal(request,terminal),[11,12])
    def test_supervisor_wrong_recorded_worker_rejected(self):
        request,terminal=self.terminal_fixture("supervisor_wait")
        terminal.update(status="driver_completed_launcher_exit_zero_known",returncode=0,exit_status_known=True)
        wp=self.scope/"launcher_exit_v3.json";wait=json.loads(wp.read_text());wait["worker_pid"]=99
        wp.write_text(json.dumps(wait));terminal["supervisor_wait_evidence"]["sha256"]=b.v4.file_bytes(self.root,wp)[1]
        with self.assertRaises(ValueError):self.bind_terminal(request,terminal)
    def test_missing_recorded_worker_rejected(self):
        request,terminal=self.terminal_fixture(missing_worker=True)
        with self.assertRaises(ValueError):self.bind_terminal(request,terminal)
    def test_missing_recorded_launcher_rejected(self):
        request,terminal=self.terminal_fixture(missing_launcher=True)
        with self.assertRaises(ValueError):self.bind_terminal(request,terminal)
    def test_terminal_exited_state_present_rejected(self):
        with self.assertRaises(ValueError):
            b.no_live_writers([self.scope],[11],proc_root=self.fakeproc(state="X"))
    def lease_fixture(self):
        p=self.root/".runtime/locks/gpu_2.lock";p.parent.mkdir(parents=True);p.write_text("original")
        st=p.stat()
        tp=self.root/"terminal.json";tp.write_text(json.dumps({"lease_path":str(p),
             "original_lease_identity":{"device":st.st_dev,"inode":st.st_ino}}))
        request={"terminal_receipt":{"path":str(tp),"sha256":b.v4.file_bytes(self.root,tp)[1]}}
        return p,request
    def test_lease_path_replaced_after_open_rejected(self):
        p,request=self.lease_fixture()
        real_open=b.os.open
        def replacement_open(path,*args,**kwargs):
            fd=real_open(path,*args,**kwargs)
            if Path(path)==p:
                p.rename(p.with_name("retained_original.lock"))
                p.write_text("replacement")
            return fd
        with mock.patch.object(b.os,"open",side_effect=replacement_open):
            with self.assertRaises(ValueError):
                with b.terminal_lease(self.root,request):
                    self.fail("Replaced canonical pathname must not yield")
        self.assertEqual(p.with_name("retained_original.lock").read_text(),"original")
    def test_lease_path_replaced_before_release_rejected(self):
        p,request=self.lease_fixture()
        with self.assertRaises(ValueError):
            with b.terminal_lease(self.root,request):
                p.rename(p.with_name("retained_original.lock"))
                p.write_text("replacement")
        self.assertEqual(p.with_name("retained_original.lock").read_text(),"original")
    def test_consistent_lease_identity_accepted(self):
        p,request=self.lease_fixture()
        with b.terminal_lease(self.root,request):self.assertEqual(p.read_text(),"original")
    def test_unknown_exit_is_not_zero(self):
        # Strict terminal schema rejects a fabricated zero wait code under the absence method.
        completed={"status":"completed_toy","runtime":{"pid":11,"lease":{"lease_path":str(self.root/".runtime/locks/gpu_2.lock"),
                         "launcher_metadata":{"launcher_pid":12}}}}
        cp=self.scope/"completed.json";cp.write_text(json.dumps(completed))
        cr={"path":str(cp),"sha256":b.v4.file_bytes(self.root,cp)[1]}
        terminal={"kind":"project_budget_v6_terminal_receipt","root_authorizer":"root_agent",
          "reservation_sha256":b.digest(self.r),"completed":cr,"pids":[11,12],
          "evidence_method":"root_observed_absence","status":"driver_completed_launcher_exit_unknown",
          "returncode":0,"exit_status_known":False,"observed_utc":"2026-10-11T00:00:00Z"}
        tp=self.root/"terminal.json";tp.write_text(json.dumps(terminal))
        closure={"key":"old","reservation_sha256":b.digest(self.r),"completed":cr,
          "completed_identity":{"status":"completed_toy"},"terminal_receipt":{"path":str(tp),"sha256":b.v4.file_bytes(self.root,tp)[1]},
          "scope_paths":[str(self.scope)],"expected_scope_measurement":{}}
        with self.assertRaises(ValueError):b.terminal_binding(self.root,closure,self.r,[self.scope])

if __name__=="__main__":
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Cases)
    output=io.StringIO()
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
    print(output.getvalue())
    report={"kind":"project_budget_v6_handmade_test_report","tests":result.testsRun,
        "failures":len(result.failures),"errors":len(result.errors),"handmade_cases_passed":result.wasSuccessful(),
        "live_ledger_or_products_or_processes_inspected":False,"GPU_used":False,
        "global_scan_performed":False,"actual_apply_performed":False,"fixture_directory":str(FIXTURES)}
    (ROOT/"research/reviews/project_budget_v6_author/test_report.json").write_text(json.dumps(report,indent=2)+"\n")
    raise SystemExit(not result.wasSuccessful())
