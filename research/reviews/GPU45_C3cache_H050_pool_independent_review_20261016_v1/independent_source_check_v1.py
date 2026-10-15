"""Independent source/pure-metadata review; no pool/worker/guard imports."""
from pathlib import Path
import argparse,ast,contextlib,copy,datetime,hashlib,io,json,runpy,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=Path(__file__).resolve().parent
PACKET=HERE/"reviewed_source_snapshot"
EXPECTED="904e6621ec39f62e3d7960f08c412d35eb5691ce681aa8629e6220fa88ca2532"
results=[]
def check(name,condition):
    assert condition,name
    results.append({"check":name,"passed":True})
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def function(source,name):
    return next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
def cls(source,name):
    return next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name==name)
def compile_node(node,ns):
    exec(compile(ast.Module(body=[copy.deepcopy(node)],type_ignores=[]),"<independent pure source>","exec"),ns)
def require(ok,message):
    if not ok:raise ValueError(message)
def ref(p):return {"path":str(p),"sha256":hashlib.sha256(str(p).encode()).hexdigest()}
handoff=json.loads((PACKET/"source_handoff_v1.json").read_text())
check("exact_final_author_seal",sha(PACKET/"source_handoff_v1.json")==EXPECTED)
for row in handoff["files"]:check("seal:"+row["path"],sha(PACKET/Path(row["path"]).name)==row["sha256"])
for p in PACKET.glob("*.py"):
    compile(p.read_text(),str(p),"exec")
check("four_author_Python_sources_compile_without_import",len(list(PACKET.glob("*.py")))==4)
pool=(PACKET/"gpu45_c3cache_h050_pool_v1.staged.py").read_text()
adapter=(PACKET/"root_shared_admission_v1.py").read_text()
base=(ROOT/"src/arggnn/gpu_pool_v1.py").read_text()
check("frozen_base_sha",hashlib.sha256(base.encode()).hexdigest()=="ae1e0494ab672cfa3017759460eb06a4b4789eaef54928496c3c80314adcc0e1")
pins_node=next(n.value for n in ast.parse(pool).body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=="MODULE_PINS" for x in n.targets))
for key,value in zip(pins_node.keys,pins_node.values):
    pin={x.arg:ast.literal_eval(x.value) for x in value.keywords if x.arg in {"path","sha256"}}
    check("installed_exact_canonical_shim:"+ast.literal_eval(key),sha(ROOT/pin["path"])==pin["sha256"])
methods={n.name:n for n in cls(pool,"Shared45Controller").body if isinstance(n,ast.FunctionDef)}
old=(ROOT/"research/reviews/GPU45_shared_pool_cancel_fix_20261015_v1/gpu45_shared_pool_v1.staged.py").read_text()
old_methods={n.name:n for n in cls(old,"Shared45Controller").body if isinstance(n,ast.FunctionDef)}
preserved=[]
for name in ("_check","start","status","_persist","_memory","submit_declared","_drain_logs",
             "_reap_owned_adopted","_fail_owned_supervision","_service_termination","tick",
             "cancel_declared","_signal_job","_signal_own","dispatch","_release_handles"):
    check("preserved_ast:"+name,ast.dump(methods[name],include_attributes=False)==ast.dump(old_methods[name],include_attributes=False))
    preserved.append(name)
check("owned_scanner_has_no_foreign_fd_inventory",not any(
    isinstance(n,ast.Constant) and isinstance(n.value,str) and "/fd" in n.value
    for n in ast.walk(cls(pool,"OwnedProcReader"))))
check("owned_scanner_does_not_glob_global_proc_PIDs",not any(
    isinstance(n,ast.Attribute) and n.attr in {"glob","rglob","listdir"}
    for n in ast.walk(cls(pool,"OwnedProcReader"))))
start=ast.get_source_segment(pool,methods["start"])
check("subreaper_before_startup_mutation",start.index("enable_child_subreaper")<start.index("directory.mkdir"))
release=ast.get_source_segment(pool,methods["_release_handles"])
check("lease_close_has_no_explicit_unlock",not any(
    isinstance(n,ast.Attribute) and n.attr=="flock" for n in ast.walk(methods["_release_handles"])))
signal_source=ast.get_source_segment(pool,methods["_signal_job"])
check("pidfd_open_identity_recheck_signal_order",signal_source.index("open_pidfd")<signal_source.index("proc_identity")<signal_source.index("send_pidfd"))
check("no_broadcast_process_group_signal",not any(
    isinstance(n,ast.Attribute) and n.attr in {"kill","killpg"} for n in ast.walk(methods["_signal_job"])))
check("actual_targeted_adopted_wait",any(
    isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="waitpid"
    and isinstance(n.args[0],ast.Name) and n.args[0].id=="pid" for n in ast.walk(methods["_reap_owned_adopted"])))
resource=ast.get_source_segment(pool,methods["_owned_resource_check"])
check("threshold_breach_records_drain_without_expiry_raise","violations.append" in resource and 'if violations:self._fail_owned_supervision' in resource)
check("owned_monitor_runs_at_each_scan", "_owned_resource_check()" in ast.get_source_segment(pool,methods["_process_scan"]))
check("load_cap_bounds_worker_expiry", 'expiry.timestamp()<=cap["expires_unix"]' in pool)
check("original_CPU_caps_and_deadline_bound", 'cpu_bytes=128*2**30' in pool and 'cpu_bytes=64*2**30' in pool and 'operational_deadline_seconds"]==86400' in pool)

# Run only preinspected AST-isolated author fixtures; no scientific/pool imports.
capture=io.StringIO()
with contextlib.redirect_stdout(capture):
    runpy.run_path(str(PACKET/"fixtures_source_only_v1.py"),run_name="_pure_source_fixture_review")
author=json.loads(capture.getvalue())
check("author41_pure_cases_rerun",author["cases_passed"]==41 and author["actual_pool_imports_guard_calls_PIDFDs_processes_jobs_GPU_queries_signals_or_Git"]==0)

# Independently test a known tracker/escaped worker through later adoption.
records={900:dict(pid=900,start_ticks=1,uid=77,ppid=1,pgid=900,sid=900,state="S",rss_bytes=100),
 101:dict(pid=101,start_ticks=10,uid=77,ppid=900,pgid=101,sid=101,state="S",rss_bytes=100),
 102:dict(pid=102,start_ticks=11,uid=77,ppid=101,pgid=101,sid=101,state="S",rss_bytes=100),
 103:dict(pid=103,start_ticks=12,uid=77,ppid=101,pgid=700,sid=700,state="S",rss_bytes=100)}
edges={900:{101},101:{102,103},102:set(),103:set()}
def ident(pid):
    if pid not in records:raise ProcessLookupError(pid)
    return dict(records[pid])
def children(pid):return set(edges.get(pid,set()))
fake_os=types.SimpleNamespace(getpid=lambda:900,getuid=lambda:77)
rn={"base":types.SimpleNamespace(proc_identity=ident),"os":fake_os,"Path":Path,"require":require}
compile_node(cls(pool,"OwnedProcReader"),rn)
reader=rn["OwnedProcReader"](identity=ident,children=children,uid=77)
bn={"os":fake_os};compile_node(next(n for n in cls(base,"Controller").body if isinstance(n,ast.FunctionDef) and n.name=="_process_scan"),bn)
override=ast.get_source_segment(pool,methods["_process_scan"]).replace("super()._process_scan()","base_scan(self)")
sn={"base_scan":bn["_process_scan"],"require":require};exec(compile(override,"<pure shared scan>","exec"),sn)
x=types.SimpleNamespace(lease=None,children={"A":types.SimpleNamespace(pid=101,returncode=None)},
 jobs={"A":{"leader_start_ticks":10,"process_identities":[dict(records[101])]}},
 proc_reader=reader,_owned_resource_check=lambda:None)
sn["_process_scan"](x)
check("tracker_and_setsid_worker_attributed_from_actual_owned_ancestry",x.pid_jobs.get(102)=="A" and x.pid_jobs.get(103)=="A")
del records[101];edges[900]={102,103};records[102]["ppid"]=900;records[103]["ppid"]=900
sn["_process_scan"](x)
check("known_tracker_and_setsid_orphans_survive_adoption",x.pid_jobs=={102:"A",103:"A"} and not x.ambiguous)
records[104]=dict(records[102],pid=104,start_ticks=13,state="Z",pgid=888,sid=888);edges[900].add(104)
sn["_process_scan"](x)
check("only_unattributable_adopted_zombie_blocks",104 in x.ambiguous and 104 not in x.pid_jobs)

# H050 admission transformation compiles in isolation; no admission function call.
hf=function(adapter,"admit_h050")
rep=ast.literal_eval(next(n.value for n in hf.body if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=="replacements" for v in n.targets)))
hd=(ROOT/"research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1/root_bridge_v1/root_driver_v1.py").read_text()
adapted=ast.get_source_segment(hd,function(hd,"admit"))
for before,after in rep.items():
    check("one_exact_H050_admission_boundary:"+before[:45],adapted.count(before)==1)
    adapted=adapted.replace(before,after)
compile(adapted,"<pure H050 transformed admission compile>","exec")
check("H050_expiry_shared_runtime_spec_storage",adapted.count("expires_at_utc=expiry")>=3 and "datetime.fromtimestamp(SHARED_EXPIRES_UNIX" in adapted)
check("fresh_required_refs_before_original_reserve",adapted.index("SHARED_REQUIRED_REFS")<adapted.index("apply_reserve_v1"))
check("original_H050_driver_context_globals_preserved","root_driver_ref=ref(DRIVER)" in adapted and "root_context_module_ref=ref(CONTEXT)" in adapted)

# Fresh source namespaces, not the private admission driver's in-memory globals.
c3path=ROOT/"research/reviews/C3_GNN_prefetch_ramcache_isolated980_root_20261015_v3/worker_entry_v1.py"
c3=c3path.read_text();c3fn=function(c3,"execute")
prefix=[]
for node in c3fn.body:
    if isinstance(node,ast.Assign) and any(isinstance(v,ast.Name) and v.id=="GPU" for v in node.targets):break
    prefix.append(node)
source_closure={"new_pool.py":"1"*64,"new_launcher.py":"2"*64,"canonical_c3.py":"3"*64}
spec={"root_driver_ref":ref("original_C3_driver.py"),"job_id":"C3-GNN-cachev3-shared-TRAIN-v1",
      "phase":"train","output_directory":"experiments/results/C3_GNN_cachev3_shared_train_v1",
      "root_stage_request_ref":ref("fresh_request.json"),"source_sha256":source_closure}
req={"phase":"train"}
def source_ref(path):
    return {"path":path,"sha256":source_closure[path]} if path in source_closure else ref(path)
c=types.SimpleNamespace(read=lambda rec,cap=None:spec if rec==ref("fresh_input.json") else req,
 active=lambda s:None,require=require,DRIVER="original_C3_driver.py",ref=source_ref)
ns={"c":c,"record":ref("fresh_input.json")}
exec(compile(ast.Module(body=prefix,type_ignores=[]),"<fresh frozen C3 pre-GPU metadata>","exec"),ns)
check("fresh_C3_canonical_metadata_accepts_shared_job_output_with_original_driver",True)
check("C3_child_has_no_original_phase_path_gate",not any(isinstance(n,ast.Attribute) and n.attr=="phase_paths" for n in ast.walk(c3fn)))

common_path=ROOT/"research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1/root_bridge_v1/bridge_common_v1.py"
common=common_path.read_text()
context_path=common_path.parent/"context_factory_v1.py";context=context_path.read_text()
newjob="H050-direct-readout-shared-TRAIN-v1";newout="experiments/results/H050_shared_direct_readout_train_seed101_v1"
runtime={"job_id":newjob,"device":"cuda","physical_gpu":4,"output_directory":newout}
h_spec=dict(runtime)
ctxfn=function(context,"build_context_v1")
runtime_check=next(n for n in ctxfn.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
 and any(isinstance(a,ast.Constant) and a.value=="Exact actual H050 runtime" for a in n.value.args))
exec(compile(ast.Module(body=[runtime_check],type_ignores=[]),"<fresh H050 runtime metadata equality>","exec"),{"runtime":runtime,"spec":h_spec,"require":require})
check("fresh_H050_context_accepts_shared_job_output_runtime",True)
cfg={"root_context_module_ref":ref("original_context.py"),"root_driver_ref":ref("original_driver.py"),
 "canonical_dispatcher_ref":ref("original_producer.py"),"producer_ref":ref("original_producer.py"),
 "max_cpu_rss_bytes":64*2**30,"max_cuda_reserved_bytes":28*2**30}
h_spec.update(schema="H050_direct_readout_root_input_v1",hypothesis_id="H050",arm="joint_four_arm",phase="train",
 root_authorizer="root_agent",status="active",fit_authorized=True,pyg_authorized=True,
 **{k:cfg[k] for k in ("root_context_module_ref","root_driver_ref","canonical_dispatcher_ref","producer_ref","max_cpu_rss_bytes","max_cuda_reserved_bytes")})
h_spec.update(main_source_ref=cfg["producer_ref"],worker_cli_ref=cfg["producer_ref"],source_sha256=source_closure,
 consumer_manifest_ref=ref("fresh_manifest.json"),refs={"config":ref("new_shared_config.json"),"runtime":ref("new_runtime.json"),"storage":ref("new_storage.json")},
 observation_entry_refs={"p":ref("old_observed.json")},observation_cache_source_refs={"cache":ref("old_cache.py")},
 train_io_lineage=None)
manifest=dict(schema="H050_exact_transitive_consumer_closure_v1",hypothesis_id="H050",phase="train",
 source_sha256=source_closure,refs={k:v for k,v in h_spec["refs"].items() if k!="storage"},
 observation_entry_refs=h_spec["observation_entry_refs"],observation_cache_source_refs=h_spec["observation_cache_source_refs"])
hn={"require":require,"validate_config_v1":lambda cfg,reader:cfg,"ref":ref,
 "CONTEXT":"original_context.py","DRIVER":"original_driver.py","worker":types.SimpleNamespace(SOURCE="original_producer.py"),
 "exact_sources_v1":lambda reader,cfg:source_closure,"terminal_v1":lambda cfg:{},
 "framework":types.SimpleNamespace(declared_cut_preparation_source_refs_v2=lambda cfg:{}),
 "TRAIN_FIELDS":("train_completion_ref","train_input_ref","train_actual_wait_ref"),"direct_required_refs_v1":lambda cfg,spec:[]}
compile_node(function(common,"validate_spec_v1"),hn)
reader=types.SimpleNamespace(decode_json=lambda r:manifest)
hn["validate_spec_v1"](reader,h_spec,cfg,{"reservation":{"consumer_bindings":[]}})
check("fresh_frozen_H050_spec_validator_accepts_new_shared_names",True)

fit=(ROOT/"src/arggnn/census_tsinfer_fit_v1.py").read_text()
for job in ("C3-GNN-cachev3-shared-TRAIN-v1","H050-direct-readout-shared-TRAIN-v1"):
 env={"ARGGNN_LOGICAL_JOB_NAME":job,"ARGGNN_JOB_NAME":"gpu45_shared_v1:gpu4_c3cache_h050_v1:gpu4"}
 jn={"os":types.SimpleNamespace(environ=env)}
 compile_node(function(fit,"require_logical_job_v1"),jn)
 jn["require_logical_job_v1"]({"launcher_metadata":{"job":env["ARGGNN_JOB_NAME"]}},{"job_id":job})
 check("canonical_logical_job_accepts_shared_environment:"+job,True)

check("seal_unchanged_after_review",sha(PACKET/"source_handoff_v1.json")==EXPECTED)
result={"schema":"GPU45_C3cache_H050_independent_source_metadata_receipt_v1","passed":True,
 "reviewed_handoff_sha256":EXPECTED,"author_fixture_cases_rerun":author["cases_passed"],
 "independent_checks_passed":len(results),"preserved_controller_method_ASTs":preserved,
 "checks":results,"actual_pool_worker_guard_imports_process_GPU_probes_waits_signals_admissions_installs_jobs_or_Git":False,
 "fresh_namespace_scope":"Actual frozen C3 pre-GPU acceptance prefix, frozen H050 spec validator with metadata-only immutable preparation/source providers, actual H050 context runtime equality and actual logical-job validator; no native scientific execution."}
(HERE/"independent_receipt.v1.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
(HERE/"author_fixtures_rerun.v1.json").write_text(json.dumps(author,indent=2,sort_keys=True)+"\n")
print(json.dumps({k:v for k,v in result.items() if k not in ("checks","preserved_controller_method_ASTs")},indent=2))
