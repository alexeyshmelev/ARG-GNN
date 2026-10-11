"""Bounded adapter review: source, handmade authorities/ledgers, fake inventory/Popen."""
from pathlib import Path
import builtins,contextlib,copy,datetime,fcntl,hashlib,importlib.util,io,json,os,stat,sys,types,unittest,uuid
from unittest.mock import patch
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN");OWN=ROOT/"research/reviews/budgeted_gpu_launcher_v1_independent";BLOCKED=[]
def pth(v):
    if v is None or isinstance(v,int):return None
    try:return Path(os.path.abspath(os.fsdecode(v)))
    except TypeError:return None
def own(p):return p is not None and (p==OWN or OWN in p.parents)
def deny(s):BLOCKED.append(s);raise RuntimeError("ADAPTER REVIEW GUARD: "+s)
def readcheck(p):
    if p is None or own(p):return
    for base in [ROOT/"data",ROOT/"experiments/results",ROOT/"results",ROOT/".runtime/gpup",ROOT/".runtime/storage",ROOT/".runtime/locks",Path("/proc")]:
        if p==base or base in p.parents:deny("live/data read "+str(p))
    if p in [ROOT/"research/experiment_ledger.md",ROOT/"research/experiment_decisions.json"]:deny("live registry read")
def audit(event,args):
    if event=="open":
        p=pth(args[0]);m=args[1];flags=args[2];write=(isinstance(m,str) and any(x in m for x in "wax+")) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if write and p is not None and not own(p):deny("write outside review "+str(p))
        if not write:readcheck(p)
    elif event in {"os.scandir","os.listdir"}:readcheck(pth(args[0]))
    elif event in {"os.mkdir","os.remove","os.rmdir","os.unlink","os.chmod","os.truncate","os.utime"}:
        p=pth(args[0])
        if p is not None and not own(p):deny("mutation outside review "+str(p))
    elif event in {"os.rename","os.link","os.symlink"}:
        for v in args[:2]:
            p=pth(v)
            if p is not None and not own(p):deny("mutation outside review "+str(p))
    elif event in {"subprocess.Popen","os.system","os.posix_spawn","socket.connect","socket.bind"}:deny("real process/network "+event)
original_import=builtins.__import__
def importer(name,*a,**k):
    if name.split(".")[0] in {"torch","numpy","tskit","tsinfer","tsdate","rdkit","torch_geometric","composer","transformers"}:deny("model/scientific import "+name)
    return original_import(name,*a,**k)
sys.dont_write_bytecode=True;sys.addaudithook(audit);builtins.__import__=importer
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
a=load("adapter_reviewed",ROOT/"src/arggnn/budgeted_gpu_launcher_v1.py")
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v)+"\n")
def ref(root,p):return {"path":str(p.relative_to(root)),"sha256":digest(p)}
UUIDS={i:"GPU-"+str(uuid.UUID(int=i)) for i in (2,3,4,5)}
class Adapter(unittest.TestCase):
    def setUp(self):
        root=OWN/"fixtures"/uuid.uuid4().hex;root.mkdir(parents=True);self.root=root
        for rel in ["src/arggnn/project_budget_v4.py","src/arggnn/budgeted_gpu_launcher_v1.py","scripts/launch_gpu.py"]:
            p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/rel).read_bytes())
        self.b=load("budget_fixture_"+uuid.uuid4().hex,root/a.BUDGET);b=self.b;c=copy.deepcopy(b.FIXED_POLICY);c["root"]=str(root)
        human=root/"research/human.json";dump(human,{"kind":"human_project_storage_authorization","project_root":str(root),"authorized_hard_limit_bytes":1000000000000,"future_scheduling_ceiling_bytes":950000000000})
        pool=root/"src/legacy.py";pool.write_text("handmade legacy source\n");c["human_authorization"]=ref(root,human);c["legacy_pool_source"]=ref(root,pool)
        config=root/"configs/project_budget_v4.json";dump(config,c);b.ROOT=root;b.SOURCE=root/a.BUDGET;b.CONFIG=config;b.FIXED_POLICY=c
        lock=root/b.LOCK;lock.parent.mkdir(parents=True);lock.write_bytes(b"unchanged fixture scheduler")
        legacy={"root":str(root),**c["legacy_invariants"],"controllers":{"pool":{"pid":101,"reserved_bytes":67108864}},"jobs":{"pool/old":{"output":str(root/"old/output"),"reserved_bytes":60000000000}}};dump(root/b.LEGACY,legacy)
        consumer=root/"src/consumer.py";consumer.write_text("handmade scientific consumer source\n");runtime=root/"research/runtime.json";dump(runtime,{"kind":"handmade_runtime_only"})
        bindings={"implementation":ref(root,b.SOURCE),"config":ref(root,config),"human_authorization":c["human_authorization"]}
        br=root/"research/budget_review.json";dump(br,{"kind":"project_budget_v4_independent_review","review_role":"independent_reviewer","verdict":"PASS",**bindings,"actual_apply_performed":False})
        request={"key":"generic/seed202","output":str(root/"new/output"),"auxiliary_scopes":[str(root/"new/tmp")],"reserved_bytes":52321000000,"consumer_bindings":[ref(root,consumer),ref(root,runtime)]}
        ba=root/"research/budget_admission.json";dump(ba,{"kind":"project_budget_v4_root_reserve_admission","root_authorizer":"root_agent","active":True,**bindings,"all_budget_writers_use_scheduler_mutex":True,"no_legacy_mutation_authorized":True,"GPU_authorized":False,"scientific_execution_authorized":False,"global_rebase_authorized":False,"independent_review":ref(root,br),"expected_before_ledger_pins":{"legacy_ledger_sha256":ref(root,root/b.LEGACY)["sha256"],"supplemental_ledger_sha256":None},"reservations":[request]})
        reserved=b.reserve(root,ref(root,ba),expected_source_sha256=a.BUDGET_SHA,expected_config_sha256=bindings["config"]["sha256"]);reservation=reserved["admitted_reservations"][request["key"]]
        review=root/"research/adapter_review.json";implementation=ref(root,root/"src/arggnn/budgeted_gpu_launcher_v1.py");launcher={"path":a.LAUNCHER,"sha256":a.LAUNCHER_SHA}
        dump(review,{"kind":"budgeted_gpu_launcher_v1_independent_review","review_role":"independent_reviewer","verdict":"PASS","implementation":implementation,"launcher":launcher,"budget_source_sha256":a.BUDGET_SHA,"actual_GPU_launch_performed":False})
        self.authority={"kind":"budgeted_gpu_launcher_v1_root_admission","root_authorizer":"root_agent","active":True,"implementation":implementation,"launcher":launcher,"budget_source_sha256":a.BUDGET_SHA,"budget_config_sha256":bindings["config"]["sha256"],"GPU_authorized":True,"scientific_method_changes":False,"storage_preflight":"supplemental_combined_projection_v4","independent_review":ref(root,review),"command":{"gpu":2,"job":"review_job","module":"arggnn.handmade_worker","args":["--literal","a b"]},"reservation":reservation,"initial_ledger_pins":{"legacy_ledger_sha256":reserved["legacy_ledger_sha256"],"supplemental_ledger_sha256":reserved["supplemental_ledger_sha256"]}}
        self.ap=root/"research/adapter_admission.json";dump(self.ap,self.authority)
        self.r=load("original_launcher_fixture_"+uuid.uuid4().hex,root/a.LAUNCHER);self.r.ROOT=root
        def forbidden_original(**kw):raise AssertionError("original du preflight invoked")
        self.original_storage=forbidden_original;self.r.require_storage=forbidden_original
        self.inventory=[{"index":i,"uuid":UUIDS[i],"memory_used_mib":0,"memory_total_mib":24000,"utilization_percent":0,"has_compute_process":False} for i in (2,3,4,5)]
        self.inventory_calls=0;self.spawn=[];self.budget_checks=[];self.free=200000000000
        def inventory():self.inventory_calls+=1;return copy.deepcopy(self.inventory)
        self.r.gpu_inventory=inventory
        def popen(cmd,**kw):
            fd=kw["pass_fds"][0];self.assertEqual(kw["env"]["ARGGNN_GPU_LEASE_FD"],str(fd));self.assertTrue(stat.S_ISREG(os.fstat(fd).st_mode))
            with (root/".runtime/locks/gpu_2.lock").open("r+") as probe:
                with self.assertRaises(BlockingIOError):fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.spawn.append((cmd,kw,fd));return types.SimpleNamespace(pid=919191,poll=lambda:None,wait=lambda:7)
        actual_check=b.check_reservation
        def checked(*args,**kw):self.budget_checks.append(copy.deepcopy(kw));return actual_check(*args,**kw)
        def modules(name,path):
            if path==root/a.BUDGET:return b
            if path==root/a.LAUNCHER:return self.r
            raise AssertionError("Unexpected loader path")
        self.patches=[patch.object(a,"__file__",str(root/"src/arggnn/budgeted_gpu_launcher_v1.py")),patch.object(a,"CONFIG_SHA",bindings["config"]["sha256"]),patch.object(a,"load_module",new=modules),patch.object(a.os,"statvfs",new=lambda p:types.SimpleNamespace(f_bavail=self.free,f_frsize=1)),patch.object(self.r.subprocess,"Popen",new=popen),patch.object(self.r.signal,"signal",new=lambda *x:None),patch.object(b,"check_reservation",new=checked)]
        for p in self.patches:p.start()
        self.addCleanup(lambda:[p.stop() for p in reversed(self.patches)])
    def run_adapter(self):
        self.argv=sys.argv[:]
        with contextlib.redirect_stdout(io.StringIO()):return a.launch(self.root,ref(self.root,self.ap))
    def assert_restored(self):
        self.assertIs(self.r.require_storage,self.original_storage);self.assertEqual(sys.argv,self.argv)
    def reject(self,pattern=None):
        with self.assertRaises(ValueError) if pattern is None else self.assertRaisesRegex(ValueError,pattern):self.run_adapter()
        self.assertFalse(self.spawn);self.assertEqual(self.inventory_calls,0);self.assert_restored()
    def test_default_source_only(self):
        with patch.object(sys,"argv",["adapter"]),patch.object(a,"launch",side_effect=AssertionError("No launch")),contextlib.redirect_stdout(io.StringIO()) as out:a.main()
        self.assertFalse(json.loads(out.getvalue())["GPU_launched"])
    def test_unchanged_child_argv_env_passFD_locks_and_projection(self):
        old=(self.root/self.b.LEGACY).read_bytes();self.assertEqual(self.run_adapter(),7);self.assert_restored();self.assertEqual((self.root/self.b.LEGACY).read_bytes(),old)
        cmd,kw,fd=self.spawn[0];self.assertEqual(cmd,[str(self.root/".runtime/env/bin/python"),"-m","arggnn.handmade_worker","--literal","a b"]);self.assertEqual(kw["cwd"],self.root);self.assertTrue(kw["start_new_session"]);self.assertEqual(kw["pass_fds"],(fd,))
        self.assertEqual(kw["env"]["CUDA_VISIBLE_DEVICES"],UUIDS[2]);self.assertEqual(kw["env"]["ARGGNN_PHYSICAL_GPU"],"2");self.assertEqual(kw["env"]["ARGGNN_JOB_NAME"],"review_job")
        metadata=json.loads((self.root/".runtime/locks/gpu_2.lock").read_text());self.assertEqual(metadata["storage_bytes"],409724749376)
        self.assertTrue(self.budget_checks[0]["boundary"]);self.assertEqual(self.budget_checks[0]["required_write_bytes"],2000000000);self.assertEqual(self.budget_checks[0]["initial_ledger_pins"],self.authority["initial_ledger_pins"])
        with self.assertRaises(OSError):os.fstat(fd)
    def test_bad_review_gate(self):
        p=self.root/self.authority["independent_review"]["path"];r=json.loads(p.read_text());r["review_role"]="author";dump(p,r);self.authority["independent_review"]=ref(self.root,p);dump(self.ap,self.authority);self.reject("independent")
    def test_untyped_active_and_duplicate_authority(self):
        self.authority["active"]=1;dump(self.ap,self.authority);self.reject("authority")
        self.authority["active"]=True;dump(self.ap,self.authority);text=self.ap.read_text();self.ap.write_text(text.replace("{",'{ "active": false,',1));self.reject("duplicate")
    def test_ineligible_or_boolean_GPU_gate(self):
        for value in [0,1,6,True]:
            self.authority["command"]["gpu"]=value;dump(self.ap,self.authority);self.reject("Eligible")
    def test_module_namespace_escape_gate(self):
        self.authority["command"]["module"]="external_pkg.worker";dump(self.ap,self.authority);self.reject("Project module")
    def test_NUL_args_gate(self):
        self.authority["command"]["args"]=["bad\0arg"];dump(self.ap,self.authority);self.reject("argument")
    def test_source_or_launcher_pin_drift_gate(self):
        self.authority["launcher"]["sha256"]="0"*64;dump(self.ap,self.authority);self.reject("authority")
    def test_initial_ledger_pin_gate(self):
        self.authority["initial_ledger_pins"]["legacy_ledger_sha256"]="0"*64;dump(self.ap,self.authority);self.reject("first-read")
    def test_own_reservation_mutation_gate(self):
        self.authority["reservation"]["reserved_bytes"]-=1;dump(self.ap,self.authority);self.reject("own")
    def test_own_output_scope_type_mutation_gate(self):
        out=Path(self.authority["reservation"]["output"]);out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(b"handmade opaque");self.reject("scope directory")
    def test_filesystem_headroom_equal_and_below_reject(self):
        self.free=self.authority["reservation"]["reserved_bytes"]+2000000000;self.reject("Filesystem")
        self.free-=1;self.reject("Filesystem")
    def test_busy_requested_GPU_rejects_and_restores(self):
        self.inventory[0]["has_compute_process"]=True
        with self.assertRaisesRegex(RuntimeError,"eligible GPU"):self.run_adapter()
        self.assertFalse(self.spawn);self.assert_restored()
    def test_two_physical_leases_cap(self):
        handles=[]
        try:
            for i in [3,4]:
                p=self.root/".runtime/locks"/("gpu_"+str(i)+".lock");f=p.open("a+");fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(f)
            with self.assertRaisesRegex(RuntimeError,"Two project"):self.run_adapter()
            self.assertFalse(self.spawn);self.assert_restored()
        finally:
            for f in handles:f.close()
    def test_one_other_physical_lease_allows_second(self):
        p=self.root/".runtime/locks/gpu_3.lock"
        with p.open("a+") as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);self.assertEqual(self.run_adapter(),7)
        self.assert_restored();self.assertEqual(len(self.spawn),1)
    def test_subprocess_failure_closes_lease_and_restores(self):
        with patch.object(self.r.subprocess,"Popen",side_effect=RuntimeError("fake child failure")),self.assertRaisesRegex(RuntimeError,"fake child"):self.run_adapter()
        self.assert_restored()
        with (self.root/".runtime/locks/gpu_2.lock").open("a+") as probe:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def test_preflight_later_check_accepts_unrelated_change_and_omits_first_pins(self):
        def twice():
            x=self.r.require_storage(headroom_bytes=2000000000)
            p=self.root/self.b.LEGACY;l=json.loads(p.read_text());l["controllers"]["new"]={"pid":222,"reserved_bytes":1234};dump(p,l)
            y=self.r.require_storage(headroom_bytes=2000000000);return y-x
        with patch.object(self.r,"main",new=twice):self.assertEqual(self.run_adapter(),1234)
        self.assertEqual(len(self.budget_checks),2);self.assertIsNone(self.budget_checks[1]["initial_ledger_pins"]);self.assert_restored()
    def test_changed_original_headroom_contract_rejected(self):
        with patch.object(self.r,"main",new=lambda:self.r.require_storage(headroom_bytes=1)),self.assertRaisesRegex(ValueError,"headroom changed"):self.run_adapter()
        self.assert_restored();self.assertFalse(self.spawn)
if __name__=="__main__":
    started=datetime.datetime.now(datetime.timezone.utc).isoformat();out=io.StringIO()
    with patch.object(sys,"argv",["adapter"]),contextlib.redirect_stdout(io.StringIO()) as source:a.main()
    (OWN/"source_only_cli.json").write_text(source.getvalue())
    result=unittest.TextTestRunner(stream=out,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Adapter))
    (OWN/"test_execution.log").write_text(out.getvalue())
    record={"started_utc":started,"completed_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped),"blocked_events":BLOCKED,"actual_GPU_launch_performed":False,"actual_live_ledger_or_data_or_device_access":False,"handmade_authority_budget_and_scopes_only":True,"original_launcher_main_exercised_with_fake_inventory_and_fake_Popen":True}
    (OWN/"test_execution.json").write_text(json.dumps(record,indent=2)+"\n");print(out.getvalue());print(json.dumps(record,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
