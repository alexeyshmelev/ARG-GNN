"""Focused v4 AST and in-memory inventory policy checks; no driver/filesystem/process import."""
from pathlib import Path
import ast,hashlib,json,errno,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/H018_native_Relate_qualification_source_v1"
checks=[]
def check(name,ok):
    if not ok:raise AssertionError(name)
    checks.append(name)
def require(ok,message):
    if not ok:raise ValueError(message)
def node(source,name):
    return next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name==name)
new=(HERE/"root_qualifier_v4.py").read_text();old=(HERE/"root_qualifier_v3.py").read_text()
compile(new,"root_qualifier_v4.py","exec");checks.append("AST compile")
for name in ("require","utc","load","common","syncdir","save","identity","bounded_file","error_record",
 "direct_authority_refs","write_bytes","group_absent","direct_wait_owned","limit_process","native_limits",
 "owned_rss","supervise","verify_request","admit","admitted_inputs","generated_project_pycache_path_v2",
 "verify_runtime_packages","runtime_preflight_v2","parse_split_headers_v3","read_split_headers_v3",
 "split_cli_stages_v3","deadline","main"):
    check("unchanged "+name,ast.dump(node(new,name),include_attributes=False)==ast.dump(node(old,name),include_attributes=False))
cfg=json.loads((HERE/"config_v4.json").read_text());prior=json.loads((HERE/"config_v3.json").read_text())
nskeys={"job_id","output_directory","packet_directory","auxiliary_directory"}
added=set(cfg)-set(prior)
check("only namespace and inventory policy fields changed",added=={"inventory_max_attempts","inventory_retry_delay_seconds","inventory_transient_errnos","inventory_terminal_equal_complete_scans"} and {k:v for k,v in cfg.items() if k not in nskeys|added}=={k:v for k,v in prior.items() if k not in nskeys})
check("fresh v4 namespaces",all(cfg[k].endswith("v4") for k in nskeys))
check("same50MB and fixed scientific options/binaries",cfg["reserved_bytes"]==50000000 and cfg["native_binary_refs"]==prior["native_binary_refs"] and cfg["helper_ref"]==prior["helper_ref"] and cfg["fixture_profile_ref"]==prior["fixture_profile_ref"] and cfg["expected_native_headers"]==prior["expected_native_headers"])
s=ast.get_source_segment(new,node(new,"_inventory_once_v4"))
check("perfile size/count/symlink caps retained",'info.st_size<=SETTINGS["native_file_cap_bytes"]' in s and 'len(refs)<SETTINGS["native_output_file_count_cap"]' in s and "not stat.S_ISLNK" in s)
check("partial scan no missing-entry continue", "except FileNotFoundError" not in s and "continue" not in s)
check("terminal hashing requires unchanged identity","identity(info)==identity(path.lstat())" in s)
sleeps=[];calls=[];queue=[]
def once(folder,retain):
    calls.append((folder,retain));value=queue.pop(0)
    if isinstance(value,BaseException):raise value
    return value
env=dict(Path=Path,ROOT=ROOT,OUTPUT=cfg["output_directory"],AUX=cfg["auxiliary_directory"],SETTINGS=cfg,
 errno=errno,time=types.SimpleNamespace(sleep=lambda v:sleeps.append(v)),require=require,_inventory_once_v4=once)
exec(compile(ast.Module(body=[node(new,"file_inventory")],type_ignores=[]),"<pure-inventory>","exec"),env)
inventory=env["file_inventory"];folder=ROOT/cfg["output_directory"]
def run(values,*,stable=False,retain=False,scope=folder):
    queue[:]=values;calls.clear();sleeps.clear()
    return inventory(scope,retain,stable=stable)
a=(13,{"a":{"bytes":13}},("a",));b=(17,{"b":{"bytes":17}},("b",));b2=(17,{"b":{"bytes":17}},("b",));c=(19,{"c":{"bytes":19}},("c",))
def err(number,path=folder/"work/native/chunk_0"):
    return OSError(number,"synthetic inventory error",str(path)) if path is not None else OSError(number,"synthetic inventory error")
check("complete active scan retained",run([a])==a[:2] and len(calls)==1 and not sleeps)
for label,number in [("ENOENT",errno.ENOENT),("ESTALE",errno.ESTALE)]:
    check("owned "+label+" retries fresh and retains true bytes",run([err(number),b])==b[:2] and len(calls)==2 and sleeps==[.01])
def rejects(label,fn,count=None):
    try:fn()
    except (OSError,ValueError):
        check(label,count is None or len(calls)==count);return
    raise AssertionError(label+" accepted")
rejects("persistent ESTALE fails after exactly3",lambda:run([err(errno.ESTALE)]*3),3)
check("retry delay is bounded",sleeps==[.01,.01])
for label,exc in [("EACCES",err(errno.EACCES)),("EIO",err(errno.EIO)),("outside_owned_ESTale",err(errno.ESTALE,ROOT/"unrelated/file")),("unknown_path_ESTale",err(errno.ESTALE,None))]:
    rejects(label+" fatal immediately",lambda exc=exc:run([exc]),1)
rejects("outside owned root rejected",lambda:run([a],scope=ROOT/"unrelated"),0)
check("stable needs two identical complete snapshots",run([a,a],stable=True)==a[:2] and len(calls)==2)
check("stable rescan discards changed earlier count",run([a,b,b2],stable=True)==b[:2] and len(calls)==3)
rejects("three changing terminal snapshots never succeed",lambda:run([a,b,c],stable=True),3)
check("retry then two matching complete terminal snapshots",run([err(errno.ENOENT),a,a],stable=True)==a[:2] and len(calls)==3)
rejects("terminal error breaks consecutive pair",lambda:run([a,err(errno.ESTALE),a],stable=True),3)
rejects("hashing active inventory forbidden",lambda:run([a],retain=True),0)
launch=ast.get_source_segment(new,node(new,"launch"))
check("wait/drain/control gating before terminal scan",launch.index('observation["actual_exit_code"]==0')<launch.index("terminal_output,terminal_files=file_inventory(output,stable=True)"))
check("terminal scan cap gate before next native stage",launch.index("terminal_output,terminal_files=file_inventory(output,stable=True)")<launch.index("Terminal stable output/aux byte caps before next phase")<launch.index('terminal_inventory_refs[name]=save'))
check("final stable hashed inventory precedes completed artifact",'file_inventory(output,True,stable=True)' in launch and launch.index('file_inventory(output,True,stable=True)')<launch.index('write_bytes(output,"completed.json"'))
def ref(path):return dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
m=json.loads((HERE/"source_manifest_v4.json").read_text());r=json.loads((HERE/"root_request_inactive_v4.json").read_text())
check("actual frozen source/config/manifest refs",r["driver_ref"]==m["source_refs"]["driver"]==ref(HERE/"root_qualifier_v4.py") and r["config_ref"]==m["source_refs"]["config"]==ref(HERE/"config_v4.json") and r["source_manifest_ref"]==ref(HERE/"source_manifest_v4.json"))
check("inert authority template",r["native_execution_authorized"] is False and r["root_authorizer"] is None and r["status"]=="inactive_source_plan" and r["registration_ref"] is None)
pins=json.loads((HERE/"preserved_v1_v2_v3_pins_v4.json").read_text())["files_sha256"]
check("all named v1/v2/v3 sources failures preserved",all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha for p,sha in pins.items()))
print(json.dumps(dict(schema="H018_Relate_operational_v4_focused_source_checks",status="passed_source_and_pure_cases_only",
 checks=len(checks),check_names=checks,actual_filesystem_or_process_probe=False,
 driver_guard_native_runtime_import_or_admission_executed=False),indent=2,sort_keys=True))
