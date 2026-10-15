"""Focused v3 source/metadata and pure argument/header checks. No driver import."""
from pathlib import Path
import ast,hashlib,json,struct,math,argparse,contextlib,io
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/H018_native_Relate_qualification_source_v1"
checks=[]
def check(name,ok):
    if not ok:raise AssertionError(name)
    checks.append(name)
def require(ok,message):
    if not ok:raise ValueError(message)
def reject(name,fn):
    try:fn()
    except (ValueError,TypeError,struct.error):checks.append(name);return
    raise AssertionError(name+" accepted")
def node(source,name):
    return next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name==name)
def pure(source,name,ns):
    exec(compile(ast.Module(body=[node(source,name)],type_ignores=[]),"<pure-"+name+">","exec"),ns)
    return ns[name]
def ref(path):
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
new=(HERE/"root_qualifier_v3.py").read_text();old=(HERE/"root_qualifier_v2.py").read_text()
compile(new,"root_qualifier_v3.py","exec");checks.append("AST compile")
unchanged=("require","utc","load","common","syncdir","save","identity","bounded_file","error_record",
 "direct_authority_refs","write_bytes","group_absent","direct_wait_owned","limit_process","native_limits",
 "file_inventory","owned_rss","verify_request","admit","admitted_inputs","generated_project_pycache_path_v2",
 "verify_runtime_packages","runtime_preflight_v2","deadline","main")
for name in unchanged:
    check("unchanged "+name,ast.dump(node(new,name),include_attributes=False)==ast.dump(node(old,name),include_attributes=False))
normalized=new.replace("def supervise(proc,seconds,output,aux,log_remaining):","def supervise(proc,seconds,output,aux):")
normalized=normalized.replace('    log_cap=min(SETTINGS["per_stage_log_cap_bytes"],log_remaining)\n',"").replace("log_cap-len(captured)",'SETTINGS["per_stage_log_cap_bytes"]-len(captured)').replace("total<=log_cap",'total<=SETTINGS["per_stage_log_cap_bytes"]')
check("supervision unchanged except tighter aggregate remaining log allowance",ast.dump(node(normalized,"supervise"),include_attributes=False)==ast.dump(node(old,"supervise"),include_attributes=False))
cfg=json.loads((HERE/"config_v3.json").read_text());prior=json.loads((HERE/"config_v2.json").read_text())
added=set(cfg)-set(prior);namespace={"job_id","output_directory","packet_directory","auxiliary_directory"}
check("only explicit operational fields and fresh namespace changed",{k:v for k,v in cfg.items() if k not in namespace|added}=={k:v for k,v in prior.items() if k not in namespace})
check("same50MB/time/RSS/file/loglimits",all(cfg[k]==prior[k] for k in ("reserved_bytes","Relate_seconds","converter_seconds","total_controller_seconds","combined_RSS_cap_bytes","native_file_cap_bytes","output_cap_bytes","combined_log_cap_bytes","CPU_seconds_per_process")))
check("fresh v3 namespaces",all(cfg[k].endswith("v3") for k in namespace))
literal=next(x.value for x in ast.parse(new).body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="SETTINGS" for t in x.targets))
check("literal exact config",ast.literal_eval(literal)==cfg)
manifest=json.loads((HERE/"source_manifest_v3.json").read_text());request=json.loads((HERE/"root_request_inactive_v3.json").read_text())
check("inactive request",request["status"]=="inactive_source_plan" and request["root_authorizer"] is None and request["native_execution_authorized"] is False and request["registration_ref"] is None and request["registered_before_execution"] is False)
check("actual current driver/config/manifest refs",request["driver_ref"]==manifest["source_refs"]["driver"]==ref(HERE/"root_qualifier_v3.py") and request["config_ref"]==manifest["source_refs"]["config"]==ref(HERE/"config_v3.json") and request["source_manifest_ref"]==ref(HERE/"source_manifest_v3.json"))
check("installed author CLI source pinned",manifest["source_refs"]["author_split_CLI"]==cfg["author_split_CLI_ref"]==ref(ROOT/cfg["author_split_CLI_ref"]["path"]))
check("sourcebuild identity uncertainty retained",manifest["installed_binary_source_build_identity_proven"] is False and manifest["public_CXX_source_commit"]=="b54ede259cbb0be095bc9c9a8bd18cdaf7e88b74")
check("complete excluded refs retained",manifest["transitive_only_refs"]=={g:{n:r for n,r in manifest[g].items() if Path(r["path"]).suffix not in manifest["direct_binding_suffixes"]} for g in ("source_refs","metadata_refs")})
check("binary/fixture/helper/age/runtime refs unchanged",all(cfg[k]==prior[k] for k in ("native_binary_refs","fixture_profile_ref","fixture_expected_ref","helper_ref","runtime_package_RECORD_refs","license_ref")))
profile=json.loads((HERE/"fixture_profile.json").read_text())
ns=dict(require=require,struct=struct,math=math,hashlib=hashlib)
parse=pure(new,"parse_split_headers_v3",ns)
global_bytes=struct.pack("<iiidii",40,512,1,0.00083,0,512)
chunk_bytes=struct.pack("<iiiii",40,512,2,0,512)
decoded=parse(global_bytes,chunk_bytes)
check("exact header decoded",decoded["N"]==40 and decoded["L"]==512 and decoded["num_chunks"]==1 and decoded["section_boundaries"]==[0,512] and decoded["first_section"]==decoded["last_section"]==0)
for label,g,c in [
 ("shortglobal",global_bytes[:-1],chunk_bytes),("longglobal",global_bytes+b"x",chunk_bytes),
 ("shortchunk",global_bytes,chunk_bytes[:-1]),("longchunk",global_bytes,chunk_bytes+b"x"),
 ("wrongN",struct.pack("<iiidii",39,512,1,0.00083,0,512),chunk_bytes),
 ("wrongL",struct.pack("<iiidii",40,511,1,0.00083,0,512),chunk_bytes),
 ("twochunks",struct.pack("<iiidii",40,512,2,0.00083,0,512),chunk_bytes),
 ("wrongspan",struct.pack("<iiidii",40,512,1,0.00083,1,512),chunk_bytes),
 ("nanmemory",struct.pack("<iiidii",40,512,1,float("nan"),0,512),chunk_bytes),
 ("wrongboundaries",global_bytes,struct.pack("<iiiii",40,512,2,0,511)),
 ("twosections",global_bytes,struct.pack("<iiiii",40,512,3,0,512))]:
    reject("header rejects "+label,lambda g=g,c=c:parse(g,c))
binaries={"Relate":"/synthetic/Relate","RelateFileFormats":"/synthetic/RelateFileFormats"};inputs=Path("/synthetic/fresh/inputs")
plan=pure(new,"split_cli_stages_v3",{})(binaries,inputs,profile)
old_assign=next(x for x in ast.walk(node(old,"launch")) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="stages" for t in x.targets))
prior_ns=dict(binaries=binaries,inputs=inputs,profile=profile,SETTINGS=prior)
exec(compile(ast.Module(body=[old_assign],type_ignores=[]),"<prior-argv-only>","exec"),prior_ns)
old_args=prior_ns["stages"];common_args=old_args[0][1][3:]
check("exact eight stage order",[x[0] for x in plan]==cfg["expected_native_direct_wait_stages"])
check("seven supported author CLI modes",[x[1][2] for x in plan[:-1]]==["MakeChunks","Paint","BuildTopology","FindEquivalentBranches","InferBranchLengths","CombineSections","Finalize"])
for name,args,kind in plan[:-1]:
    check("identical scientific options "+name,args[3:3+len(common_args)]==common_args and kind=="relate")
check("same converter argv",plan[-1][1]==old_args[-1][1] and plan[-1][2]=="convert")
check("only intended chunk and fullrange args",all(args[3+len(common_args):]==((["--chunk_index","0"] if 1<=i<=5 else [])+(["--first_section","0","--last_section","0"] if i in (2,4) else [])) for i,(_,args,_) in enumerate(plan[:-1])))
launch=ast.get_source_segment(new,node(new,"launch"))
check("single aggregate Relate deadline before loop",launch.count("relate_deadline=time.monotonic()")==1 and launch.index("relate_deadline=time.monotonic()")<launch.index("for name,argv,kind in stages"))
check("seconds and remaining logs defined before spawn",launch.index("seconds=(relate_deadline-time.monotonic())")<launch.index('save(b,packet,name+"_prelaunch.json"')<launch.index("subprocess.Popen") and launch.index("log_remaining=SETTINGS")<launch.index("subprocess.Popen"))
check("headers gate later native stages",launch.index("headers=read_split_headers_v3(work)")<launch.index("subprocess.Popen") and 'name=="make_chunks" or parameter_headers_ref is not None' in launch)
check("all waits before nativefacts",launch.index("All exact eight waits before native support verification")<launch.index("import tskit")<launch.index("helper.verify_native"))
check("wait0/drain/error gates preserved",'observation["actual_exit_code"]==0' in launch and 'observation["process_group_drained"] is True' in launch and 'observation["control_error"] is None' in launch)
check("actual wait receipt before logs or guards",launch.index('saved=save(b,packet,name+"_direct_wait.json"')<launch.index('write_bytes(aux,name+".log"'))
main=pure(new,"main",dict(argparse=argparse,json=json,require=require))
with contextlib.redirect_stdout(io.StringIO()):status=main([])
check("default inert",type(status) is int and status==0)
preserved=json.loads((HERE/"preserved_v1_v2_pins_v3.json").read_text())
check("all named v1/v2 sources and failed evidence preserved",all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha for name,sha in preserved["files_sha256"].items()))
print(json.dumps(dict(schema="H018_Relate_operational_v3_focused_source_checks",status="passed_source_and_pure_cases_only",
 checks=len(checks),check_names=checks,actual_runtime_preflight_executed=False,
 driver_or_guard_or_native_or_package_imported=False,actual_process_or_admission_executed=False,
 fixture_science_or_date_policy_changed=False),sort_keys=True,indent=2))
