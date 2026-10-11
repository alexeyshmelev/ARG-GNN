"""Independent source and handmade review. No live ledger or product read."""
from pathlib import Path
import builtins,contextlib,copy,datetime,hashlib,importlib.util,io,json,os,stat,sys,unittest,uuid
from unittest.mock import patch
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OWN=ROOT/"research/reviews/project_budget_v2_independent"
EVENTS=[]
def absolute(v):
    if isinstance(v,int) or v is None:return None
    try:return Path(os.path.abspath(os.fsdecode(v)))
    except TypeError:return None
def inside(p):return p is not None and (p==OWN or OWN in p.parents)
def blocked(reason):
    EVENTS.append(reason);raise RuntimeError("INDEPENDENT REVIEW GUARD: "+reason)
def reads(p):
    if p is None or inside(p):return
    for base in [ROOT/"data",ROOT/"results",ROOT/"experiments/results",ROOT/".runtime/gpup",ROOT/".runtime/locks",ROOT/".runtime/storage",Path("/proc")]:
        if p==base or base in p.parents:blocked("live state read "+str(p))
    if p in [ROOT/"research/experiment_ledger.md",ROOT/"research/experiment_decisions.json"]:blocked("live registry read "+str(p))
def audit(event,args):
    if event=="open":
        p=absolute(args[0]);m=args[1];flags=args[2]
        writing=(isinstance(m,str) and any(c in m for c in "wax+")) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if writing and p is not None and not inside(p):blocked("write outside own review "+str(p))
        if not writing:reads(p)
    elif event in {"os.scandir","os.listdir"}:reads(absolute(args[0]))
    elif event in {"os.mkdir","os.remove","os.rmdir","os.unlink","os.chmod","os.chown","os.truncate","os.utime","os.mkfifo"}:
        p=absolute(args[0])
        if p is not None and not inside(p):blocked("mutation outside own review "+str(p))
    elif event in {"os.rename","os.link","os.symlink"}:
        for v in args[:2]:
            p=absolute(v)
            if p is not None and not inside(p):blocked("mutation outside own review "+str(p))
    elif event in {"subprocess.Popen","os.system","os.posix_spawn","socket.connect","socket.bind"}:blocked("process/network "+event)
original_import=builtins.__import__
def importer(name,*args,**kwargs):
    if name.split(".")[0] in {"numpy","torch","tskit","tsinfer","tsdate","composer","transformers","omegaconf"}:blocked("scientific decoder/model import "+name)
    return original_import(name,*args,**kwargs)
sys.dont_write_bytecode=True;sys.addaudithook(audit);builtins.__import__=importer
spec=importlib.util.spec_from_file_location("independent_budget",ROOT/"src/arggnn/project_budget_v2.py")
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,sort_keys=True)+"\n")
def ref(root,p):return {"path":str(p.relative_to(root)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
class Review(unittest.TestCase):
    def setUp(self):
        self.root=OWN/"fixtures"/uuid.uuid4().hex;self.root.mkdir(parents=True)
        root=self.root;c=copy.deepcopy(b.FIXED_POLICY);c["root"]=str(root)
        source=root/"src/sealed.py";source.parent.mkdir(parents=True);source.write_text("handmade source\n")
        pool=root/"src/pool.py";pool.write_text("handmade legacy source\n")
        human=root/"research/human.json";dump(human,{"kind":"human_project_storage_authorization","project_root":str(root),"authorized_hard_limit_bytes":1000000000000,"future_scheduling_ceiling_bytes":950000000000})
        c["human_authorization"]=ref(root,human);c["legacy_pool_source"]=ref(root,pool)
        config=root/"configs/policy.json";dump(config,c)
        self.patches=[patch.object(b,k,v) for k,v in {"ROOT":root,"SOURCE":source,"CONFIG":config,"FIXED_POLICY":c}.items()]
        for p in self.patches:p.start()
        self.addCleanup(lambda:[p.stop() for p in reversed(self.patches)])
        lock=root/b.LOCK;lock.parent.mkdir(parents=True);lock.write_bytes(b"unchanged handcrafted mutex")
        self.legacy={"root":str(root),**c["legacy_invariants"],"controllers":{"pool":{"pid":101,"reserved_bytes":67108864}},"jobs":{"pool/old":{"output":str(root/"old/output"),"reserved_bytes":60000000000}}}
        dump(root/b.LEGACY,self.legacy)
        consumer=root/"src/consumer.py";consumer.write_text("handmade consumer source\n")
        runtime=root/"research/runtime.json";dump(runtime,{"kind":"handmade_runtime_metadata"})
        self.bindings={"implementation":ref(root,source),"config":ref(root,config),"human_authorization":c["human_authorization"]}
        review=root/"research/review.json";dump(review,{"kind":"project_budget_v2_independent_review","review_role":"independent_reviewer","verdict":"PASS",**self.bindings,"actual_apply_performed":False})
        self.request={"key":"fit/parent","output":str(root/"new/output"),"auxiliary_scopes":[str(root/"new/tmp")],"reserved_bytes":20000000000,"consumer_bindings":[ref(root,consumer),ref(root,runtime)]}
        self.authority={"kind":"project_budget_v2_root_reserve_admission","root_authorizer":"root_agent","active":True,**self.bindings,"all_budget_writers_use_scheduler_mutex":True,"no_legacy_mutation_authorized":True,"GPU_authorized":False,"scientific_execution_authorized":False,"global_rebase_authorized":False,"independent_review":ref(root,review),"expected_before_ledger_pins":{"legacy_ledger_sha256":ref(root,root/b.LEGACY)["sha256"],"supplemental_ledger_sha256":None},"reservations":[copy.deepcopy(self.request)]}
        self.apath=root/"research/root.json";dump(self.apath,self.authority)
    def refresh(self):
        dump(self.root/b.LEGACY,self.legacy)
        self.authority["expected_before_ledger_pins"]["legacy_ledger_sha256"]=ref(self.root,self.root/b.LEGACY)["sha256"]
        dump(self.apath,self.authority)
    def reserve(self):
        return b.reserve(self.root,ref(self.root,self.apath),expected_source_sha256=self.bindings["implementation"]["sha256"],expected_config_sha256=self.bindings["config"]["sha256"])
    def own(self):return self.reserve()["admitted_reservations"]["fit/parent"]
    def check(self,own,**kw):
        return b.check_reservation(self.root,own,expected_source_sha256=self.bindings["implementation"]["sha256"],expected_config_sha256=self.bindings["config"]["sha256"],**kw)
    def reject_reserve(self,pattern=None):
        old=(self.root/b.LEGACY).read_bytes()
        with self.assertRaises(ValueError) if pattern is None else self.assertRaisesRegex(ValueError,pattern):self.reserve()
        self.assertEqual((self.root/b.LEGACY).read_bytes(),old)
        self.assertFalse((self.root/b.SUPPLEMENTAL).exists())
    def test_default_cli_is_source_only(self):
        buff=io.StringIO()
        with contextlib.redirect_stdout(buff):self.assertEqual(b.main([]),0)
        self.assertFalse(json.loads(buff.getvalue())["live_ledger_or_scope_access"])
        self.assertFalse((self.root/b.SUPPLEMENTAL).exists())
    def test_four_charges_and_readonly_dynamic_projection(self):
        incoming=[]
        for i in range(4):
            r=copy.deepcopy(self.request);r["key"]+=str(i);r["output"]+=str(i);r["auxiliary_scopes"]=[r["auxiliary_scopes"][0]+str(i)];incoming.append(r)
        self.authority["reservations"]=incoming;dump(self.apath,self.authority)
        old=(self.root/b.LEGACY).read_bytes();mutex=(self.root/b.LOCK).read_bytes()
        result=self.reserve();own=next(iter(result["admitted_reservations"].values()));view=self.check(own,boundary=True)
        self.assertEqual(view["projected_bytes"],437403749376)
        self.assertEqual(view["supplemental_reserved_bytes"],80000000000)
        sraw=(self.root/b.SUPPLEMENTAL).read_bytes()
        self.legacy["controllers"]["new"]={"pid":102,"reserved_bytes":123456};dump(self.root/b.LEGACY,self.legacy)
        later=self.check(own)
        self.assertEqual(later["projected_bytes"],view["projected_bytes"]+123456)
        self.assertEqual((self.root/b.SUPPLEMENTAL).read_bytes(),sraw)
        self.assertEqual((self.root/b.LOCK).read_bytes(),mutex)
        self.assertEqual(old.count(b"reserved_bytes"),2)
    def test_first_read_hash_binding(self):
        own=self.own();v=self.check(own);pins={k:v[k] for k in ["legacy_ledger_sha256","supplemental_ledger_sha256"]}
        self.check(own,initial_ledger_pins=pins)
        self.legacy["controllers"]["new"]={"pid":102,"reserved_bytes":1};dump(self.root/b.LEGACY,self.legacy)
        with self.assertRaisesRegex(ValueError,"first-read"):self.check(own,initial_ledger_pins=pins)
        self.check(own)
    def test_reject_own_value_mutation(self):
        own=self.own();changed=copy.deepcopy(own);changed["reserved_bytes"]-=1
        with self.assertRaisesRegex(ValueError,"own"):self.check(changed)
    def test_reject_root_authority_rewrite(self):
        own=self.own();self.authority["active"]=False;dump(self.apath,self.authority)
        with self.assertRaisesRegex(ValueError,"bound"):self.check(own)
    def test_source_binding_drift(self):
        (self.root/"src/consumer.py").write_text("changed source")
        self.reject_reserve("bound")
    def test_inactive_admission(self):
        self.authority["active"]=False;dump(self.apath,self.authority);self.reject_reserve("activated")
    def test_noninteger_legacy_charge(self):
        self.legacy["controllers"]["pool"]["reserved_bytes"]=True;self.refresh();self.reject_reserve("integer")
    def test_float_baseline_invariant(self):
        self.legacy["baseline_bytes"]=float(self.legacy["baseline_bytes"]);self.refresh();self.reject_reserve("drift")
    def test_exact_ceiling_blocks(self):
        base=297336640512+60000000000+67108864
        self.authority["reservations"][0]["reserved_bytes"]=950000000000-base
        dump(self.apath,self.authority);self.reject_reserve("ceiling")
    def test_one_below_ceiling_passes(self):
        base=297336640512+60000000000+67108864
        self.authority["reservations"][0]["reserved_bytes"]=950000000000-base-1
        dump(self.apath,self.authority);self.assertEqual(self.reserve()["projected_bytes"],949999999999)
    def test_cross_ledger_ancestor_scope_rejected(self):
        self.authority["reservations"][0]["output"]=str(self.root/"old")
        dump(self.apath,self.authority);self.reject_reserve("exclusively")
    def test_cross_new_scope_aux_overlap_rejected(self):
        q=copy.deepcopy(self.request);q["key"]="fit/second";q["output"]=self.request["auxiliary_scopes"][0]+"/nested";q["auxiliary_scopes"]=[]
        self.authority["reservations"].append(q);dump(self.apath,self.authority);self.reject_reserve("exclusively")
    def test_accounting_scope_rejected(self):
        self.authority["reservations"][0]["output"]=str(self.root/".runtime/storage")
        dump(self.apath,self.authority);self.reject_reserve("namespace")
    def test_metadata_hardlink_rejected(self):
        os.link(self.root/b.LEGACY,self.root/"legacy_alias");self.reject_reserve("singly")
    def test_scope_symlink_rejected(self):
        own=self.own();out=Path(own["output"]);out.mkdir(parents=True);(out/"actual").write_bytes(b"x");(out/"link").symlink_to(out/"actual")
        with self.assertRaisesRegex(ValueError,"canonical"):self.check(own,boundary=True)
    def test_boundary_sparse_and_planned_growth(self):
        own=self.own();out=Path(own["output"]);out.mkdir(parents=True)
        with (out/"opaque").open("wb") as f:f.truncate(100000)
        v=self.check(own,boundary=True);self.assertGreaterEqual(v["scope_measurement"]["conservative_scope_bytes"],100000)
        with self.assertRaisesRegex(ValueError,"exceeds"):self.check(own,boundary=True,required_write_bytes=v["remaining_scope_bytes"]+1)
    def test_unknown_supplemental_header_rejected(self):
        own=self.own();p=self.root/b.SUPPLEMENTAL;value=json.loads(p.read_text());value["new_policy"]=0;dump(p,value)
        with self.assertRaisesRegex(ValueError,"header"):self.check(own)
    def test_legacy_ancestor_descendant_should_reject(self):
        self.legacy["jobs"]["pool/nested"]={"output":str(self.root/"old/output/sub"),"reserved_bytes":1};self.refresh()
        self.reject_reserve()
    def test_unknown_legacy_top_level_should_reject(self):
        self.legacy["unrecognized_reservations"]={"unknown":{"reserved_bytes":700000000000}};self.refresh()
        self.reject_reserve()
    def test_unknown_legacy_job_scope_should_reject(self):
        self.legacy["jobs"]["pool/old"]["auxiliary_scopes"]=[self.request["output"]];self.refresh()
        self.reject_reserve()
    def test_legacy_accounting_namespace_should_reject(self):
        self.legacy["jobs"]["pool/old"]["output"]=str(self.root/".runtime/storage");self.refresh()
        self.reject_reserve()
if __name__=="__main__":
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Review);buff=io.StringIO()
    result=unittest.TextTestRunner(stream=buff,verbosity=2).run(suite)
    (OWN/"test_execution.log").write_text(buff.getvalue())
    execution={"started_utc":started,"completed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped),"blocked_events":EVENTS,"handmade_fixtures_only":True,"actual_live_ledger_access":False,"actual_apply_performed":False,"failure_cases":[test.id() for test,_ in result.failures]}
    (OWN/"test_execution.json").write_text(json.dumps(execution,indent=2)+"\n")
    print(buff.getvalue());print(json.dumps(execution,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
