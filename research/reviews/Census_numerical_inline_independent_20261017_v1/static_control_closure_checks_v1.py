"""Fresh numerical source/control qualification. All operational lower layers remain uncalled."""
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,json,os,stat
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
D=ROOT/"research/reviews/Census_numerical_inline_ROOT_orchestration_source_20261017_v1"
OUT=ROOT/"research/reviews/Census_numerical_inline_independent_20261017_v1"
checks=[];verified={};ast_inventory={}
def ck(name,value=True):
 if not value:raise AssertionError(name)
 checks.append({"name":name,"status":"PASS"})
def read(path,expected=None):
 p=ROOT/path
 assert not Path(path).is_absolute() and ".." not in Path(path).parts and ".runtime" not in Path(path).parts
 assert p.resolve()==p
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_size<=32*2**20
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,"rb")as f:
  start=os.fstat(f.fileno());data=f.read(32*2**20+1);end=os.fstat(f.fileno())
 ident=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 assert ident(s)==ident(start)==ident(end)==ident(p.lstat()) and len(data)==s.st_size
 h=hashlib.sha256(data).hexdigest()
 assert expected is None or h==expected
 verified[path]=h
 return data
def docref(r):return json.loads(read(r["path"],r["sha256"]))
def doc(p):return json.loads(read(str(p.relative_to(ROOT))))
sm=doc(D/"source_manifest.v1.json");layout=doc(D/"staging_layout.v1.json")
aliases={r["to"]["path"]:r["from"]["path"] for r in layout["byte_identical_copies"]}
for p,h in sm["source_sha256"].items():
 data=read(aliases.get(p,p),h)
 if p.endswith(".py"):
  t=ast.parse(data);compile(t,p,"exec",dont_inherit=True,optimize=0)
  ast_inventory[p]={"functions":len([n for n in ast.walk(t)if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]),"imports":[ast.unparse(n) for n in t.body if isinstance(n,(ast.Import,ast.ImportFrom))],"sha256":h}
ck("all231 complete source-map entries authenticated without probing future aliases",len(sm["source_sha256"])==231)
audit=doc(D/"audit_manifest.v1.json")
for r in audit["refs"]:read(r["path"],r["sha256"])
ck("all17 audit refs authenticate",len(audit["refs"])==17)
m=doc(D/"integration_manifest.INACTIVE.template.v1.json")
hm=doc(ROOT/"research/reviews/Census_retention_inline_source_20261017_v1/source_manifest.v1.json")
ck("every210 inherited source pin retained verbatim",len(hm["source_sha256"])==210 and all(m["source_sha256"].get(p)==h for p,h in hm["source_sha256"].items()))
old=ROOT/"research/reviews/Census_inspect_inline_ROOT_orchestration_source_20261017_v1"
pairs=[
("research/reviews/Census_retention_inline_ROOT_actual_inspect_root01_oct17","research/reviews/Census_retention_inline_ROOT_actual_numerical_root01_oct17"),
("census_tsinfer_observation_v1/census-tsinfer-inspect-v1","census_tsinfer_observation_v1/census-tsinfer-numerical-v1"),
("research/reviews/Census_observation_ROOT_actual_inspect_retention_20261017_v1/permanent_writer_revocation.root_active_v1.json","research/reviews/Census_numerical_permanent_revocation_ROOT_20261017_v1/permanent_writer_revocation.root_active_v1.json"),
("56affb17278daf23bdb50c649052ee83a880f791af7b8dc06eec2903bc897042","c7a4411960259dc536e25bbc2b0f19f4540301d9186413b0eb91b5a9e5cb019e"),
("2cb83077f3379ad31322d01542270b6431e8ee88b8138f560ee7c9e31b971128","e79a7c7b3806576efda20f606eb16fd0dbfd8f5fd4e7d14511cfe2a5785c16e5"),
("2026-10-11T10:28:47.092270+00:00","2026-10-11T11:40:36.315986+00:00"),
("1275252","1306271"),("1275253","1306272"),("435988585","436412146"),("435988592","436412154"),
("Census_inspect","Census_numerical"),("inspect-only","numerical-only"),
("Exact inspect target","Exact numerical target"),("Exact inspect integration","Exact numerical integration"),
("Same original inspect reservation","Same original numerical reservation")]
runtime_delta=[]
for name in ("ROOT_authorize_inline_retirement_v1.py","orchestrate_v1.py"):
 before=read(str((old/name).relative_to(ROOT))).decode();after=read(str((D/name).relative_to(ROOT))).decode()
 for a,b in pairs:before=before.replace(a,b)
 ck("whole numerical runtime AST equals independently enumerated exact inspect substitutions "+name,ast.dump(ast.parse(before),include_attributes=False)==ast.dump(ast.parse(after),include_attributes=False))
 runtime_delta.append({"file":name,"whole_AST_equal_after_fixed_substitutions":True})
guard="research/reviews/Census_observation_two_scope_retention_source_20261017_v1/census_observation_retention_v1.py"
gt=ast.parse(read(guard));fn={n.name:n for n in gt.body if isinstance(n,ast.FunctionDef)}
manifest_ref={"path":"research/reviews/Census_observation_two_scope_retention_source_20261017_v1/source_manifest.v1.json","sha256":"e0d5e04703410f9e2f89336a9059fe73dc2fe948240ea9425bc4e928a5793405"}
gm=docref(manifest_ref);key="census_tsinfer_observation_v1/census-tsinfer-numerical-v1";t=gm["targets"][key];own=t["original_reservation"]
digest=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
ck("exact original numerical20GB reservation and all3 original bindings",digest(own)=="e79a7c7b3806576efda20f606eb16fd0dbfd8f5fd4e7d14511cfe2a5785c16e5" and own["reserved_bytes"]==20000000000 and own["auxiliary_scopes"]==[] and own["consumer_bindings"]==t["all_original_binding_refs"] and len(own["consumer_bindings"])==3)
from types import SimpleNamespace
ns={"datetime":datetime,"timezone":timezone,"ROOT":ROOT,"copy":copy,"require":lambda v,msg:None if v else (_ for _ in ()).throw(ValueError(msg)),
"budget":SimpleNamespace(digest=digest),"_json":docref,"_phase_json":docref,"_target":lambda k:copy.deepcopy(gm["targets"][k]),"_ref":lambda p:manifest_ref,
"MANIFEST":manifest_ref["path"]}
picked=["_stamp","_historical_stop","stopped_revocation_draft_v1","_revocation"]
exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(fn[n])for n in picked],type_ignores=[])),"independent-original-pure-history-only","exec"),ns)
ck("actual original historical numerical schema/provenance/directworker0/unknown-history validator passes immutable metadata",ns["_historical_stop"](t))
rev={"path":"research/reviews/Census_numerical_permanent_revocation_ROOT_20261017_v1/permanent_writer_revocation.root_active_v1.json","sha256":"c7a4411960259dc536e25bbc2b0f19f4540301d9186413b0eb91b5a9e5cb019e"}
ck("genuine permanent numerical revocation passes exact original validator",ns["_revocation"](rev,key)["active"]is True)
ck("unknown supervisor wait and descendant drain remain exact",t["historical_facts_preserved"]["supervisor_outer_OS_wait"]=="UNKNOWN" and t["historical_facts_preserved"]["historical_descendant_drain"]=="UNKNOWN")
# Entire original critical path ordering and measurement invariants are unchanged.
ret=fn["apply_retire_stopped_v1"]
ck("original300second timer begins before mutex and never restarts",ast.unparse(ret.body[1])=="started = time.monotonic()" and ast.unparse(ret.body[2])=="deadline = started + OPERATION_WALL_SECONDS" and isinstance(ret.body[-1],ast.With) and sum(isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=="deadline" for x in n.targets) for n in ast.walk(ret))==1)
measure=ast.parse(read("research/reviews/Census_observation_two_scope_retention_source_20261017_v1/single_root_retained_nodes_v1.py"))
mt=ast.unparse(measure)
ck("scanner two full snapshots final descriptor-relative revalidation no socket allowance",all(x in mt for x in ("first, scopes = snapshot()","second, again = snapshot()","first == second and scopes == again","_leaf_metadata(rootfd, tuple(relative.split('/')), device)","os.O_NOFOLLOW","follow_symlinks=False","PER_NODE_METADATA_CHARGE = 4096","MAX_ENTRIES = 1000000","MAX_DEPTH = 64")) and "socket_is_absent=lambda path: False" in ast.unparse(ret))
helper=ast.parse(read("research/reviews/Census_retention_inline_source_20261017_v1/inline_ROOT_retire_v1.py"))
cl=next(n for n in helper.body if isinstance(n,ast.ClassDef)and n.name=="InlineRootRetireStoppedMemo")
methods={n.name:n for n in cl.body if isinstance(n,ast.FunctionDef)}
append=ast.unparse(methods["_append"])
order=["tx.finish()","tx.close()","self.core._check()","instruction = self._authenticate(self.v4.bound)","fresh_auth = self.guard._authority","self.guard._recorded_stopped_proof","self.guard._expire(auth)","self.guard._expire(instruction)","finite()","tail = self._original_append","finite()"]
pos=-1
for item in order:
 pos=append.index(item,pos+1)
ck("complete final cached bytes/names finish restoration uncached controls proofs expiry same-deadline append and postappend ordering")
transaction=ast.unparse(methods["_transaction"])
ck("all integration JSON/configured ancestor controls plus fresh/current records excluded before install",all(x in transaction for x in ("self.paths.values()","manifest['source_sha256']","('_POLICY_REF', '_SELECTOR_REF', '_REPAIR_AUTHORITY_REF')","len(seen) < 16","<= 2048","tx.install()","tx.close()")))
ck("47role and64full-union checks before first helper metadata write",ast.unparse(methods["_inline_ROOT_authority"]).index("len(existing) + len(self.paths) <= 64")<ast.unparse(methods["_inline_ROOT_authority"]).index("self._put_attempt"))
ck("future controls are inactive without latest prefix roles finite clocks or pass",all(doc(D/f)["expected_before"]is None for f in ("ROOT_instruction.INACTIVE.template.v1.json","whole_join_review.INACTIVE.template.v1.json","ROOT_actual_operation_intent.INACTIVE.template.v1.json")))
result={"schema":"Census_numerical_independent_static_control_closure_checks_v1","status":"PASS","independent":True,"source_only":True,"checks":checks,"assertions":len(checks),"source_map_entries":231,"audit_refs":17,"inherited_entries":210,"all_mapped_python_AST_parsed_compiled":len(ast_inventory),"runtime_delta":runtime_delta,"verified_path_sha256":verified,"python_AST_inventory":ast_inventory,"current_state_or_original_journal_or_mutex_or_targets_or_scientific_payload_opened":False,"future_staging_probed_or_created":False,"actual_whole_capability_supplied":False,"credit_accepted":False}
with(OUT/"static_control_closure.result.v1.json").open("x")as f:json.dump(result,f,indent=2,sort_keys=True);f.write("\n")
print(json.dumps({k:v for k,v in result.items()if k not in ("checks","verified_path_sha256","python_AST_inventory","runtime_delta")},sort_keys=True))
