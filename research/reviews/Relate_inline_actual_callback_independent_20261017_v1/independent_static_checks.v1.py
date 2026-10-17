"""Independent static whole-join checks. Never imports or invokes project modules or operational callables."""
import ast,copy,hashlib,json,os,pathlib,stat
ROOT=pathlib.Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=ROOT/"research/reviews/Relate_inline_actual_callback_independent_20261017_v1"
HELPER="research/reviews/Relate_retirement_inline_ROOT_evidence_source_20261017_v1"
ACTUAL="research/reviews/Relate_retention_inline_ROOT_actual_root01_oct17_v1"
def need(value,message):
 if not value:raise ValueError(message)
def read(name):
 p=ROOT/name;s=p.lstat()
 need(p.resolve()==p and ROOT in p.parents and stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_size<=2**20,"Exact static metadata/source "+name)
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,"rb") as f:
  opened=os.fstat(f.fileno());data=f.read(2**20+1);after=os.fstat(f.fileno())
 identity=lambda x:(x.st_dev,x.st_ino,x.st_mode,x.st_nlink,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
 need(identity(s)==identity(opened)==identity(after)==identity(p.lstat()) and len(data)==s.st_size,"Stable static bytes "+name)
 return data
def digest(raw):return hashlib.sha256(raw).hexdigest()
def ref(name):return dict(path=name,sha256=digest(read(name)))
def bound(record):
 raw=read(record["path"]);need(digest(raw)==record["sha256"],"Exact SHA "+record["path"]);return raw
def fn(tree,name):
 nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name]
 need(len(nodes)==1,"One named source AST "+name);return nodes[0]
def dump(tree):return ast.dump(tree,include_attributes=False)
def calls(tree):return [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
def method(cls,name):return fn(cls,name)
request_name=ACTUAL+"/whole_join_review_request.root_v2.json"
request_raw=read(request_name);need(digest(request_raw)=="cb6aa3af6fd9b93b507679aaa78e932c3ef9b31edda7bbec5d5c42defdbd32bc","Exact v2 request")
request=json.loads(request_raw);manifest=json.loads(bound(request["source_manifest_ref"]))
need(request["source_manifest_ref"]==dict(path=ACTUAL+"/source_manifest.root_v2.json",sha256="b343a6cdbc2e91425c271ff50a1c5072278bd58da925ff3953ab84ff7fa0d908"),"Actual ROOT v2 manifest binding")
helpermanifest=json.loads(bound(request["helper_source_manifest_ref"]))
need(all(manifest["source_sha256"].get(k)==v for k,v in helpermanifest["source_sha256"].items()),"Entire exact helper closure retained")
need(len(manifest["source_sha256"])==manifest["source_sha256_count"]==1139 and manifest["actual_ROOT_callback_supplied"] is True,"Truthful actual manifest")
total=python_count=0
for name,sha in manifest["source_sha256"].items():
 raw=bound(dict(path=name,sha256=sha));total+=len(raw)
 if pathlib.Path(name).suffix==".py":ast.parse(raw,filename=name);python_count+=1
helper_ref=ref(HELPER+"/inline_ROOT_retire_v1.py");memo_ref=ref("research/reviews/ROOT_mutation_transaction_memo_source_20261017_v2/root_mutation_memo_v1.py")
guard_ref=ref("research/reviews/Relate_failed64GiB_retention_successor_source_20261017_v1/relate_failed64_retention_v1.py")
need(helper_ref["sha256"]=="df0616bab64e85ead53aa1358289e26eaae0f85edd4f98bfb3d8146d168f112d" and guard_ref["sha256"]=="b8b696a0df45c12b931e563600c2b52e89e73dd19c7d160351063687dcfb5164" and memo_ref["sha256"]=="487fcd93b73f6668126522b67720041abbbaa8b95ee54c3bc69adffba1a765b2","Original unchanged component pins")
ht=ast.parse(bound(helper_ref));gt=ast.parse(bound(guard_ref));mt=ast.parse(bound(memo_ref));original=fn(gt,"apply_retire_stopped_v1")
derived=copy.deepcopy(original);derived.name="_inline_ROOT_exact_retirement"
append_calls=[n for n in ast.walk(derived.body[-1]) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="_append"]
need(len(append_calls)==1,"One original append")
ap=append_calls[0];ap.func.id="_retirement_append";ap.args[0:0]=[ast.Name(id="finite",ctx=ast.Load()),ast.Name(id="auth",ctx=ast.Load())]
derived.body[-1].body.insert(2,ast.parse("authority_ref = _inline_ROOT_authority(state, finite)").body[0])
derived.body[-1].body=[ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Name(id="_retirement_transaction",ctx=ast.Load()),args=[ast.Name(id="finite",ctx=ast.Load())],keywords=[]),optional_vars=None)],body=derived.body[-1].body)]
restored=copy.deepcopy(derived);restored.name=original.name;restored.body[-1].body=restored.body[-1].body[0].body
seam=restored.body[-1].body.pop(2)
need(ast.unparse(seam)=="authority_ref = _inline_ROOT_authority(state, finite)","Exact fresh-state seam")
for n in ast.walk(restored):
 if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="_retirement_append":
  need([a.id for a in n.args[:2]]==["finite","auth"],"Original finite/auth");n.func.id="_append";n.args=n.args[2:]
need(dump(restored)==dump(original),"Exact reversal of only three additions")
# Compile/call only the AST-to-AST derivation function. The returned retirement
# function is never invoked. No project import, guard, callback or real operation.
namespace=dict(ast=ast,copy=copy,need=need,SOURCE=helper_ref["path"],__name__="independent_static_AST_only")
transform=fn(ht,"derive_inline_retirement")
exec(compile(ast.Module(body=[copy.deepcopy(transform)],type_ignores=[]),"independent_transform_definition","exec"),namespace)
actual=namespace["derive_inline_retirement"](copy.deepcopy(original),{},None,None,None)
manual={}
exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(derived)],type_ignores=[])),actual.__code__.co_filename,"exec"),manual)
need(actual.__code__==manual[derived.name].__code__,"Actual derivation is exactly independent three-change AST")
preservation=json.loads(read(HELPER+"/preservation.v1.json"))
need(digest(dump(original).encode())==preservation["original_AST_sha256"] and digest(dump(derived).encode())==preservation["derived_AST_sha256"],"Preserved whole AST hashes")
need(ast.unparse(original.body[1])=="started = time.monotonic()" and ast.unparse(original.body[2])=="deadline = started + OPERATION_WALL_SECONDS" and isinstance(original.body[-1],ast.With),"Deadline remains before original mutex")
ct=ast.parse(bound(request["actual_callback_source_ref"]));callback=fn(ct,request["callback_name"])
need(len(ct.body)==2 and isinstance(ct.body[0],ast.Expr) and isinstance(ct.body[0].value,ast.Constant) and isinstance(ct.body[0].value.value,str),"Inert callback module contains docstring and one definition")
args=callback.args
need(not callback.decorator_list and callback.returns is None and not args.posonlyargs and not args.args and args.vararg is None and args.kwarg is None and not args.defaults and [x.arg for x in args.kwonlyargs]==["guard","context","finite"] and args.kw_defaults==[None,None,None] and all(x.annotation is None for x in args.kwonlyargs),"Exact inert callback signature")
imports=[]
for n in ast.walk(callback):
 if isinstance(n,ast.Import):imports.extend(a.name.split(".")[0] for a in n.names)
 if isinstance(n,ast.ImportFrom):need(n.level==0,"No relative callback import");imports.append(n.module.split(".")[0])
need(set(imports)=={"pathlib","datetime","hashlib","json","os","stat"},"Only explicit stdlib callback imports")
callback_calls=sorted(set(calls(callback)))
guard_calls=sorted(x for x in callback_calls if x.startswith("guard."))
need(guard_calls==["guard._json","guard._observe_targets","guard._target","guard._utc","guard.budget.digest","guard.budget.v4.canonical","guard.operation_draft_v1","guard.stopped_observation_draft_v1"],"Only statically reviewed metadata/observation guard capabilities")
writes=[n for n in ast.walk(callback) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="write_new"]
need([n.args[0].value for n in writes]==["observation","authority"],"Only two predetermined ROOT metadata writes")
need(not any(x in callback_calls for x in ("exec","eval","compile","__import__","getattr","setattr","open","os.system","os.remove","os.unlink","os.rename","os.kill","os.killpg")),"No hidden imports/launch/signal/cleanup/mutation")
write_fn=fn(callback,"write_new")
open_calls=[n for n in ast.walk(write_fn) if isinstance(n,ast.Call) and ast.unparse(n.func)=="os.open"]
need(len(open_calls)==2 and ast.unparse(open_calls[0].args[1])=="os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW" and ast.unparse(open_calls[1].args[1])=="os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW","Only exclusive no-follow new metadata and parent fsync")
hclass=fn(ht,"InlineRootRetireStoppedMemo")
tx=method(hclass,"_transaction");txlines=[ast.unparse(n) for n in tx.body]
need(next(i for i,n in enumerate(txlines) if "uncached_paths=exclusions" in n)<next(i for i,n in enumerate(txlines) if "self._put_attempt(finite)" in n),"Exclusions constructed before first write")
constructor_calls=calls(method(hclass,"__init__"))
need(not any("_observe_targets" in x or "_state_locked" in x or "apply_retire" in x or "mutex" in x for x in constructor_calls),"Constructor metadata authentication is not an operation")
finish_src=ast.unparse(method(hclass,"_append"))
need(finish_src.index("tx.finish()")<finish_src.index("self.guard._expire(auth)")<finish_src.index("finite()")<finish_src.index("self._original_append("),"Final original raw validation then expiry/deadline/original append")
gm=json.loads(read("research/reviews/Relate_failed64GiB_retention_successor_source_20261017_v1/source_manifest.v1.json"));target=gm["targets"]["ARGMix/prospective1611-accepted-TRAIN000-Relate-v4"]
own=target["original_reservation"];need(own["key"]==request.get("key",own["key"])=="ARGMix/prospective1611-accepted-TRAIN000-Relate-v4","Exact own key exists")
need(target["reserved_bytes"]==own["reserved_bytes"]==68719476736 and target["original_expiry_unchanged"]=="2026-10-16T21:07:10.474316+00:00","Original cap and clock")
rev=json.loads(bound(request["permanent_revocation_ref"]))
need(rev["key"]==own["key"] and rev["unknown_historical_descendants_absent_claimed"] is False and rev["all_original_launch_resume_retry_write_aliases_for_exact_key_permanently_cooperatively_revoked"] is True,"Permanent cooperative exact-key revocation")
need(set(request["fresh_metadata_paths"])=={"attempt","observation","authority"} and len(set(request["fresh_metadata_paths"].values()))==3,"Three distinct paths")
fresh={}
for k,name in request["fresh_metadata_paths"].items():
 p=ROOT/name;need(p.parent==ROOT/request["metadata_directory"] and not p.exists(),"Never-created concrete path "+name);fresh[k]=False
need(not any("instruction" in p.name.lower() for p in (ROOT/request["metadata_directory"]).iterdir()),"No actual instruction exists")
result=dict(schema="Relate64_actual_callback_independent_static_checks_v1",status="STATIC_SOURCE_PASS_ONLY",request_ref=ref(request_name),source_manifest_ref=request["source_manifest_ref"],source_refs_verified=len(manifest["source_sha256"]),static_bytes_verified=total,python_ASTs_parsed=python_count,entire_actual_callback_bound=True,exact_three_change_derivation_code_matches=True,original_whole_AST_restored=True,original_AST_sha256=preservation["original_AST_sha256"],derived_AST_sha256=preservation["derived_AST_sha256"],original_deadline_before_mutex=True,actual_callback_import_roots=sorted(set(imports)),actual_callback_guard_calls=guard_calls,callback_writes_only=["observation","authority"],fresh_paths_present=fresh,instruction_present=False,original_reservation_has_key=True,original_reservation_sha256=target["original_reservation_sha256"],project_modules_imported=False,retirement_or_callback_invoked=False,PID_or_group_observation_performed=False,resource_or_mutex_or_probe_or_accounting_operations=False,scientific_payloads_opened=False,Git_operations=False)
path=OUT/"independent_static_checks.v1.json"
with path.open("x") as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
print(json.dumps(dict(path=str(path),sha256=digest(path.read_bytes()),status=result["status"],source_refs_verified=result["source_refs_verified"],python_ASTs_parsed=python_count)))
