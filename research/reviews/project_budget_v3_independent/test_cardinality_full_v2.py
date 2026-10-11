"""Strengthened handcrafted4096-entry qualification, retaining earlier isolated test."""
from pathlib import Path
import contextlib,copy,datetime,hashlib,importlib.util,io,json,unittest
from unittest.mock import patch
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");OWN=ROOT/"research/reviews/project_budget_v3_independent"
s=importlib.util.spec_from_file_location("budget_fixture_tests",OWN/"test_budget_independent.py");h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
b=h.b
class FullReview(h.Review):
    def test_full_qualifying_map_plus_one_must_reject_before_write(self):
        root=self.root;policy=b.FIXED_POLICY
        ledger=b.empty_ledger(root,policy,self.bindings)
        for group in range(64):
            requests=[]
            for position in range(64):
                i=group*64+position;q=copy.deepcopy(self.request);q["key"]="existing/"+str(i);q["output"]=str(root/"old_scopes"/str(i));q["auxiliary_scopes"]=[];q["reserved_bytes"]=1;requests.append(q)
            prior=copy.deepcopy(self.authority);prior["reservations"]=requests
            prior_path=root/"research"/("old_admission_"+str(group)+".json");h.dump(prior_path,prior);prior_ref=h.ref(root,prior_path)
            for q in requests:ledger["reservations"][q["key"]]={**q,"root_admission":prior_ref}
        h.dump(root/b.SUPPLEMENTAL,ledger)
        self.authority["expected_before_ledger_pins"]["supplemental_ledger_sha256"]=h.ref(root,root/b.SUPPLEMENTAL)["sha256"];h.dump(self.apath,self.authority)
        before=(root/b.SUPPLEMENTAL).read_bytes();original=b.paths_overlap;prefix=str(root/"old_scopes")+"/"
        def overlap(a,c):
            sa=str(a);sc=str(c)
            # Every accelerated pair is a known flat, distinct-or-equal sibling fixture.
            # This is exactly the production overlap predicate on these paths.
            if sa.startswith(prefix) and sc.startswith(prefix) and "/" not in sa[len(prefix):] and "/" not in sc[len(prefix):]:return sa==sc
            return original(a,c)
        with patch.object(b,"paths_overlap",new=overlap):
            legacy,supplemental,view=b.current(root,policy,self.bindings)
            self.assertEqual(len(supplemental["reservations"]),4096)
            self.assertEqual(view["supplemental_reserved_bytes"],4096)
            try:self.reserve()
            except ValueError:
                self.assertEqual((root/b.SUPPLEMENTAL).read_bytes(),before);return
        written=json.loads((root/b.SUPPLEMENTAL).read_text());self.assertEqual(len(written["reservations"]),4097)
        with self.assertRaisesRegex(ValueError,"bounded supplemental"):b.current(root,policy,self.bindings)
        self.fail("Full original current/schema/authority/consumer validation qualified4096 entries; reserve wrote4097 and poisoned future checks")
if __name__=="__main__":
    import os,resource
    resource.setrlimit(resource.RLIMIT_AS,(1073741824,1073741824))
    resource.setrlimit(resource.RLIMIT_CPU,(90,90))
    (OWN/"cardinality_full_v2_own_process.json").write_text(json.dumps({"own_pid":os.getpid(),"started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"purpose":"handmade4096-entry cardinality review","memory_limit_bytes":1073741824,"cpu_limit_seconds":90})+"\n")
    start=datetime.datetime.now(datetime.timezone.utc).isoformat();buff=io.StringIO()
    result=unittest.TextTestRunner(stream=buff,verbosity=2).run(unittest.TestSuite([FullReview("test_full_qualifying_map_plus_one_must_reject_before_write")]))
    (OWN/"cardinality_full_execution_v2.log").write_text(buff.getvalue())
    out={"started_utc":start,"completed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"blocked_events":h.EVENTS,"all4096_stored_scopes_authorities_and_consumer_bindings_validated_by_original_current":True,"optimization":"Only paths_overlap accelerated for known flat old_scopes sibling pairs with an equivalent predicate; no current/scope_spec/bound/reserve/schema/read/write validation stubbed","actual_live_ledger_access":False,"handmade_fixtures_only":True}
    (OWN/"cardinality_full_execution_v2.json").write_text(json.dumps(out,indent=2)+"\n");print(buff.getvalue());print(json.dumps(out,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
