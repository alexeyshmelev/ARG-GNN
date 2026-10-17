"""Whole candidate module with RAM source loader, journal, observers and measurement filesystem only."""
from pathlib import Path as RealPath
from datetime import datetime,timezone,timedelta
import ast,copy,hashlib,json,types,contextlib,io,os as RealOS,stat,posixpath,sys
ROOT=RealPath("/workspace-SR003.nfs2/ARG-GNN")
D=ROOT/"research/reviews/Census_observation_two_scope_retention_source_20261017_v1"
def encoded(v):return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
def digest(v):return hashlib.sha256(encoded(v)).hexdigest()
checks=[];negative=[];journal={};directories=set();metadata={};records={};pid_calls=[];group_calls=[]
def need(v,label):
 if not v:raise AssertionError(label)
 checks.append(label)
def expect(call,label):
 try:call()
 except (ValueError,KeyError,TypeError,AssertionError,FileNotFoundError):negative.append(label);return
 raise AssertionError("Accepted invalid fixture: "+label)
def actual_ref(path):
 p=RealPath(path);p=p if p.is_absolute()else ROOT/p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def put(path,value):
 raw=encoded(value)+b"\n";r=dict(path=path,sha256=hashlib.sha256(raw).hexdigest())
 metadata[path]=raw;records[path]=r;return r
def seed(r,read=False):
 records[r["path"]]=r
 if read:
  raw=(ROOT/r["path"]).read_bytes();need(hashlib.sha256(raw).hexdigest()==r["sha256"],"Actual targeted metadata pin "+r["path"])
  metadata[r["path"]]=raw
manifest=json.loads((D/"source_manifest.v1.json").read_text())
seed(actual_ref(D/"source_manifest.v1.json"),True)
for p,h in manifest["source_sha256"].items():seed(dict(path=p,sha256=h))
for r in manifest["runtime_authenticated_historical_source_and_metadata_refs"]:seed(r)
for k in("predecessor_policy_ref","predecessor_selector_ref"):seed(manifest[k],True)
for t in manifest["targets"].values():
 for field in("phase_evidence_ref","admission_ref","storage_ref","started_ref","launch_ref","direct_worker_wait_ref","config_ref","inspection_ROOT_freeze_ref"):
  seed(t[field],True)
# Direct role snapshot is existing ROOT JSON only, not a guard call.
snap=json.loads((ROOT/"research/reviews/FLARE_VAL001_conversion_ROOT_close_20261017_v2/final_original_journal_acceptance.root_v1.json").read_text())
def find_bindings(v):
 if isinstance(v,dict):
  if isinstance(v.get("bindings"),dict)and len(v["bindings"])==46:return v["bindings"]
  for x in v.values():
   r=find_bindings(x)
   if r is not None:return r
 elif isinstance(v,list):
  for x in v:
   r=find_bindings(x)
   if r is not None:return r
parentbindings=find_bindings(snap)
need(parentbindings is not None,"Actual immutable46-role metadata snapshot located")
for r in parentbindings.values():seed(r)
class RP:
 def __init__(self,p):self.p=str(p)
 def __str__(self):return self.p
 def __fspath__(self):return self.p
 def __truediv__(self,p):return RP(self.p.rstrip("/")+"/"+str(p).lstrip("/"))
 def __eq__(self,o):return self.p==str(o)
 def __hash__(self):return hash(self.p)
 @property
 def name(self):return posixpath.basename(self.p)
 @property
 def parents(self):return tuple(RP(x)for x in RealPath(self.p).parents)
 def relative_to(self,root):return RP(str(RealPath(self.p).relative_to(str(root))))
 def exists(self):return self.p in directories or self.p in journal
 def is_dir(self):return self.p in directories
 def mkdir(self,**kw):directories.add(self.p)
 def iterdir(self):return [RP(p)for p in sorted(journal)if posixpath.dirname(p)==self.p]
 def lstat(self):return types.SimpleNamespace(st_dev=1,st_ino=1,st_mtime_ns=1,st_ctime_ns=1)
def canonical(root,p):
 p=str(p);p=p if p.startswith("/")else str(root)+"/"+p
 if posixpath.normpath(p)!=p or not p.startswith(str(root)+"/"):raise ValueError("RAM canonical containment")
 return RP(p)
def reference(root,path):
 p=str(path);p=str(RealPath(p).relative_to(str(root)))if p.startswith("/")else p
 if p not in records:raise ValueError("Unknown RAM reference "+p)
 return copy.deepcopy(records[p])
def bound(root,r):
 if records.get(r.get("path"))!=r:raise ValueError("RAM pin mismatch")
 return metadata.get(r["path"],b"PINNED_SOURCE_RAM_DOUBLE_NOT_EXECUTED")
def file_bytes(root,p):
 p=str(p)
 if p in journal:b=journal[p];return b,hashlib.sha256(b).hexdigest()
 r=reference(root,p);b=bound(root,r);return b,r["sha256"]
@contextlib.contextmanager
def mutex(root):yield
for f in(canonical,file_bytes,mutex):f.__module__="RAM_v4"
def integer(v,name,minimum=0):
 if type(v)is not int or v<minimum:raise ValueError(name)
 return v
v4=types.SimpleNamespace(__name__="RAM_v4",canonical=canonical,file_bytes=file_bytes,mutex=mutex,
 reference=reference,bound=bound,parse=lambda b:json.loads(b),encoded=encoded,integer=integer,LEGACY="RAM_LEGACY",
 read_json=lambda root,path:(dict(jobs={}),"RAM"),paths_overlap=lambda a,b:str(a)==str(b)or str(a).startswith(str(b)+"/")or str(b).startswith(str(a)+"/"))
budget=types.SimpleNamespace(v4=v4,digest=digest,SOURCE_REL="scripts/project_budget_v9.py",LAUNCHER_REL="scripts/launch_reserved_gpu_v6.py")
clock=types.SimpleNamespace(value=0.0,monotonic=lambda:clock.value)
class RamOS:
 O_WRONLY=RealOS.O_WRONLY;O_CREAT=RealOS.O_CREAT;O_EXCL=RealOS.O_EXCL;O_NOFOLLOW=RealOS.O_NOFOLLOW
 O_RDONLY=RealOS.O_RDONLY;O_DIRECTORY=RealOS.O_DIRECTORY
 def __init__(self):self.fds={};self.nextfd=20;self.live_pid=None;self.live_group=None
 def open(self,p,flags,mode=0o400):
  p=str(p)
  if flags&self.O_CREAT:
   if p in journal:raise FileExistsError(p)
   self.nextfd+=1;self.fds[self.nextfd]=p;return self.nextfd
  if p not in directories:raise AssertionError("Only RAM journal dirs open")
  self.nextfd+=1;self.fds[self.nextfd]=p;return self.nextfd
 def fdopen(self,fd,mode):
  p=self.fds[fd];outer=self
  class B(io.BytesIO):
   def __exit__(self,*args):journal[p]=self.getvalue();self.close()
   def fileno(self):return fd
  return B()
 def fsync(self,fd):pass
 def close(self,fd):self.fds.pop(fd,None)
 def kill(self,pid,sig):
  need(sig==0,"RAM numeric PID observation only")
  pid_calls.append(pid)
  if pid==self.live_pid:return
  raise ProcessLookupError()
 def killpg(self,pgid,sig):
  need(sig==0,"RAM numeric group observation only")
  group_calls.append(pgid)
  if pgid==self.live_group:return
  raise ProcessLookupError()
ram_os=RamOS()
# Execute the complete candidate module. Only its project-guard import is replaced with RAM budget.
source=(D/"census_observation_retention_v1.py").read_text();tree=ast.parse(source)
oldimports=[n for n in tree.body if isinstance(n,ast.ImportFrom)and n.module=="scripts"]
need(len(oldimports)==1,"Only one project guard import needs RAM replacement")
tree.body=[ast.Assign(targets=[ast.Name(id="budget",ctx=ast.Store())],value=ast.Name(id="__RAM_BUDGET__",ctx=ast.Load()))if n in oldimports else n for n in tree.body]
ast.fix_missing_locations(tree)
module=types.ModuleType("Census_observation_RAM_candidate");env=module.__dict__;env["__RAM_BUDGET__"]=budget
exec(compile(tree,"<whole_candidate_module_RAM_guard_import>","exec"),env)
env.update(ROOT=RP(ROOT),Path=RP,os=ram_os,time=clock)
sys.modules[module.__name__]=module
# One-root helper executes its full production scanner only against a virtual filesystem.
class MPath(RP):
 def is_absolute(self):return self.p.startswith("/")
 def as_posix(self):return self.p
 @property
 def parts(self):return RealPath(self.p).parts
 @property
 def parents(self):return tuple(MPath(x)for x in RealPath(self.p).parents)
 def __truediv__(self,p):return MPath(super().__truediv__(p))
 def relative_to(self,r):return MPath(str(RealPath(self.p).relative_to(str(r))))
 def lstat(self):return measurefs.stat_path(self.p)
class MeasureFS:
 O_RDONLY=RealOS.O_RDONLY;O_DIRECTORY=RealOS.O_DIRECTORY;O_NOFOLLOW=RealOS.O_NOFOLLOW
 def __init__(self):self.nodes={};self.fds={};self.nextfd=0;self.opens=0;self.payload_opens=0;self.unstable=False
 def add(self,p,kind="directory",size=0,link=None):
  p=str(p);mode={"directory":stat.S_IFDIR,"regular":stat.S_IFREG,"symlink":stat.S_IFLNK,"socket":stat.S_IFSOCK,"fifo":stat.S_IFIFO}[kind]|0o700
  self.nodes[p]=types.SimpleNamespace(st_dev=1,st_ino=len(self.nodes)+1,st_mode=mode,st_uid=1000,st_gid=1000,st_nlink=1,
   st_size=len(link.encode())if link is not None else size,st_blocks=0,st_mtime_ns=1,st_ctime_ns=1,link=link)
 def path(self,p,fd=None):return posixpath.join(self.fds[fd],str(p))if fd is not None else str(p)
 def stat_path(self,p):
  if p not in self.nodes:raise FileNotFoundError(p)
  return copy.copy(self.nodes[p])
 def stat(self,p,dir_fd=None,follow_symlinks=False):
  need(follow_symlinks is False,"RAM measurement never follows stored links")
  out=self.stat_path(self.path(p,dir_fd))
  if self.unstable:out.st_mtime_ns+=self.opens
  return out
 def open(self,p,flags,dir_fd=None):
  p=self.path(p,dir_fd)
  if flags&(self.O_DIRECTORY|self.O_NOFOLLOW)!=(self.O_DIRECTORY|self.O_NOFOLLOW):raise AssertionError("Missing no-follow directory open")
  if not stat.S_ISDIR(self.nodes[p].st_mode):self.payload_opens+=1;raise AssertionError("Payload opening")
  self.nextfd+=1;self.fds[self.nextfd]=p;self.opens+=1;return self.nextfd
 def dup(self,fd):self.nextfd+=1;self.fds[self.nextfd]=self.fds[fd];return self.nextfd
 def close(self,fd):self.fds.pop(fd,None)
 def fstat(self,fd):return self.stat_path(self.fds[fd])
 def readlink(self,p,dir_fd=None):return self.nodes[self.path(p,dir_fd)].link
 def fsencode(self,text):return text.encode()
 def scandir(self,fd):
  p=self.fds[fd]
  class Entries:
   def __enter__(x):return [types.SimpleNamespace(name=posixpath.basename(n))for n in sorted(self.nodes)if posixpath.dirname(n)==p and n!=p]
   def __exit__(x,*a):pass
  return Entries()
measurefs=MeasureFS()
def setup_measure_scope(path):
 for p in reversed((RealPath(path),*RealPath(path).parents)):
  if str(p)=="/":continue
  if str(p)not in measurefs.nodes:measurefs.add(str(p))
 for p,kind,size,link in[(path+"/nested","directory",0,None),(path+"/nested/preserved.bin","regular",123,None),(path+"/stored_link","symlink",0,"../../unopened_external")]:
  measurefs.add(p,kind,size,link)
for t in manifest["targets"].values():setup_measure_scope(t["scope_paths"][0])
helper=types.ModuleType("RAM_single_root_measure")
exec(compile((D/"single_root_retained_nodes_v1.py").read_text(),"<whole_single_root_helper_RAM_FS>","exec"),helper.__dict__)
helper.Path=MPath;helper.os=measurefs
helper_mode=types.SimpleNamespace(value="normal")
def helper_module():
 out=types.SimpleNamespace(**helper.__dict__)
 original=out.measure_retained_nodes_v1
 # Function globals live in the helper module: bind the candidate's finite require at call time.
 def measured(*args,**kwargs):
  helper.require=out.require
  if helper_mode.value=="timeout":clock.value=301.0
  return original(*args,**kwargs)
 out.measure_retained_nodes_v1=measured
 return out
parentpins=dict(base_pins={"RAM":"immutable_parent"},completed_overlay_prefix=dict(events=8,tail_sha256="b"*64))
baseview=dict(accounting_mode=env["BASE_MODE"],limit_bytes=980000000000,hard_limit_bytes=1000000000000,
 projected_bytes=900000000000,base_pins=parentpins["base_pins"],completed_overlay_prefix=parentpins["completed_overlay_prefix"],
 new_reserved_bytes=0,completed_credit_bytes=0)
parentcalls=[]
def parent_state(*a,**k):return(copy.deepcopy({k:t["original_reservation"]for k,t in manifest["targets"].items()}),set(),copy.deepcopy(baseview),copy.deepcopy(parentbindings),[],[],{})
parent=types.SimpleNamespace(configure_v2=lambda *a,**k:parentcalls.append(k),_pins=lambda v:copy.deepcopy(parentpins),
 _state_locked=parent_state,_paths=lambda own:[RP(own["output"])]+[RP(p)for p in own["auxiliary_scopes"]])
renewal=types.SimpleNamespace(install_renewal_v1=lambda r:None)
class Loader:
 def __init__(self,path):self.path=str(path)
 def exec_module(self,m):
  target=parent if self.path.endswith(env["BASE_REF"]["path"])else renewal if self.path.endswith(env["RENEWAL_REF"]["path"])else helper_module()if self.path.endswith(env["MEASURE_SOURCE"])else None
  if target is None:raise AssertionError("Unexpected RAM module source "+self.path)
  m.__dict__.update(target.__dict__)
env["importlib"]=types.SimpleNamespace(util=types.SimpleNamespace(spec_from_file_location=lambda n,p:types.SimpleNamespace(name=n,loader=Loader(p)),
 module_from_spec=lambda sp:types.ModuleType(sp.name)))
now=datetime.now(timezone.utc);stamp=lambda seconds:(now+timedelta(seconds=seconds)).isoformat()
fake=lambda name:put("research/reviews/RAM_Census_NOT_AUTHORITY/"+name+".json",{})
repair=fake("repair")
# Build real candidate factory outputs in RAM, with explicit inactive-to-test-active changes only in memory.
review=put("research/reviews/RAM_Census_NOT_AUTHORITY/review.json",dict(independent=True,verdict="SOURCE_PASS",blockers=[],
 source_ref=records[env["SOURCE"]],source_manifest_ref=records[env["MANIFEST"]],measurement_source_ref=records[env["MEASURE_SOURCE"]]))
termrefs=[]
for row in env["PREREQUISITE_ADMISSIONS"].values():
 for value in row.values():
  if isinstance(value,dict)and set(value)=={"path","sha256"}:seed(value,True)
 termrefs.append(row["terminal_ref"])
pre=env["prerequisite_admissions_draft_v1"](H058v5_terminal_ref=termrefs[0],FLARE_native_terminal_ref=termrefs[1],released_at_utc=stamp(-30));pre["active"]=True
preref=put("research/reviews/RAM_Census_NOT_AUTHORITY/prerequisites.json",pre)
rev=put("research/reviews/RAM_Census_NOT_AUTHORITY/parent_revocation.json",dict(schema="Census_observation_parent_future_mutation_revocation_v1",
 root_authorizer="root_agent",active=True,parent_source_ref=env["BASE_REF"],parent_policy_ref=env["BASE_POLICY_REF"],parent_selector_ref=env["BASE_SELECTOR_REF"],
 actual_parent_pins=parentpins,prerequisite_admissions_ref=preref,parent_selected_at_utc=stamp(-20),
 all_future_parent_reserve_close_retire_mutations_permanently_cooperatively_revoked=True,existing_parent_read_check_and_fully_charged_writers_unchanged=True,no_historical_mutation=True))
activation=env["root_activation_draft_v1"](base_pins=parentpins,base_projected_bytes=baseview["projected_bytes"],permanent_revocation_ref=rev,
 source_review_ref=review,prerequisite_admissions_ref=preref,parent_selected_at_utc=stamp(-20));activation["active"]=True
actref=put("research/reviews/RAM_Census_NOT_AUTHORITY/activation.json",activation)
policy=env["activation_policy_draft_v1"](repair_authority_ref=repair,base_pins=parentpins,base_projected_bytes=baseview["projected_bytes"],
 implementation_ref=records[budget.SOURCE_REL],launcher_ref=records[budget.LAUNCHER_REL],root_activation_ref=actref,
 permanent_revocation_ref=rev,source_review_ref=review,prerequisite_admissions_ref=preref,parent_selected_at_utc=stamp(-20));policy["active"]=True
policyref=put("research/reviews/RAM_Census_NOT_AUTHORITY/policy.json",policy)
selector=env["selector_draft_v1"](policy_ref=policyref,source_review_ref=review,repair_authority_ref=repair,
 frozen_parent_lineage=dict(base_ref=env["BASE_REF"],base_policy_ref=env["BASE_POLICY_REF"],base_pins=parentpins,
 base_projected_bytes=baseview["projected_bytes"],base_accounting_mode=env["BASE_MODE"]));selector["active"]=True
selectorref=put("research/reviews/RAM_Census_NOT_AUTHORITY/selector.json",selector)
need(env["load_selected_guard_v1"](selectorref)is module,"Whole selector/configure/policy/prerequisite/manifest path returns same singleton")
need(parentcalls==[dict(policy_ref=env["BASE_POLICY_REF"],selector_ref=env["BASE_SELECTOR_REF"])],"Exact parent configuration includes actual policy and selector")
need(env["load_selected_guard_v1"](selectorref)is module,"Same selector accepted for existing singleton")
expect(lambda:env["load_selected_guard_v1"](fake("other_selector")),"different_lifetime_selector")
S=records[budget.SOURCE_REL]["sha256"];L=records[budget.LAUNCHER_REL]["sha256"]
state=env["_state_locked"](S,L,fresh=True)
need(state[2]["projected_bytes"]==baseview["projected_bytes"]+2097152,"Activation adds2MiB only and zero credit")
need(len(state[3])==47 and all(state[3][k]==v for k,v in parentbindings.items()),"All46 actual parent roles plus one direct capsule preserved")
need(state[2]["completed_credit_bytes"]==0,"No credit by activation")
need(env["_manifest"]()["target_order"]==manifest["target_order"],"Actual candidate manifest verifier accepts both original target joins")
for key,t in manifest["targets"].items():
 need(env["_historical_stop"](t)is True,"Full original evidence/admission/wait/source joins "+t["phase"])
expect(lambda:env["_target"]("other/key"),"unrelated_retention_key")
overflow_refs=[dict(path="RAM/"+str(i),sha256="a"*64)for i in range(65)]
for r in overflow_refs:records[r["path"]]=r
expect(lambda:env["consumer_bindings_v1"](overflow_refs),"valid_bound_overflow65_refs")
# Pin-aware mutation tests use the actual verifier and RAM I/O boundary to inject bad decoded metadata.
original_json=env["_json"];original_phase=env["_phase_json"]
def mutate_doc(record,field,value,call,phase=False):
 old=env["_phase_json"if phase else"_json"]
 def altered(r):
  o=copy.deepcopy(old(r))
  if r==record:o[field]=value
  return o
 env["_phase_json"if phase else"_json"]=altered
 try:expect(call,field+"_invalid")
 finally:env["_phase_json"if phase else"_json"]=old
for t in manifest["targets"].values():
 mutate_doc(t["direct_worker_wait_ref"],"returncode",1,lambda t=t:env["_historical_stop"](t))
 mutate_doc(t["direct_worker_wait_ref"],"watchdog_triggered",True,lambda t=t:env["_historical_stop"](t))
 mutate_doc(t["phase_evidence_ref"],"failed_cells",["unexpected"],lambda t=t:env["_historical_stop"](t),True)
 mutate_doc(t["phase_evidence_ref"],"status","completed_manufactured",lambda t=t:env["_historical_stop"](t),True)
 mutate_doc(t["admission_ref"],"expires_at_utc",stamp(100),lambda t=t:env["_historical_stop"](t))
 mutate_doc(t["admission_ref"],"source_sha256",{},lambda t=t:env["_historical_stop"](t))
mutate_doc(termrefs[0],"actual_admission_terminal_consumed",False,lambda:env["_policy"]())
mutate_doc(termrefs[1],"outer_exit_code",1,lambda:env["_policy"]())
mutate_doc(preref,"ROOT_explicitly_releases_current_selector_after_both_admissions",False,lambda:env["_policy"]())
mutate_doc(preref,"released_at_utc",stamp(-10),lambda:env["_policy"]())
# Full request -> authority -> guarded apply -> RAM scanner -> append -> original journal replay.
receipts=[];operation_refs=[]
def prepare_operation(key,stale=False):
 t=manifest["targets"][key]
 rv=env["stopped_revocation_draft_v1"](key);rv.update(active=True,revoked_at_utc=stamp(-15))
 rr=put("research/reviews/RAM_Census_NOT_AUTHORITY/"+t["phase"]+"_writer_revocation.json",rv)
 ob=env["stopped_observation_draft_v1"](key,revocation_ref=rr)
 at=stamp(-121)if stale else stamp(-5)
 ob.update(active=True,observed_at_utc=at,PID_observations=[[r["pid"],r["start_ticks"],True,at]for r in t["known_PID_start_rows"]],
 group_observations=[[g,True,at]for g in t["known_process_groups"]])
 obs=put("research/reviews/RAM_Census_NOT_AUTHORITY/"+t["phase"]+"_observations.json",ob)
 request=dict(key=key,reservation_sha256=t["original_reservation_sha256"],revocation_ref=rr,observation_ref=obs)
 st=env["_state_locked"](S,L,fresh=True)
 op=env["operation_draft_v1"](env["RETIRE_OPERATION"],request,expected_before=env["_pins"](st[2]),bindings=st[3])
 op.update(active=True,expires_at_utc=stamp(240))
 return put("research/reviews/RAM_Census_NOT_AUTHORITY/"+t["phase"]+"_operation.json",op),request
keys=list(manifest["targets"])
op,req=prepare_operation(keys[0])
ordinary=env["operation_draft_v1"]("close_completed",req,expected_before=env["_pins"](env["_state_locked"](S,L,fresh=True)[2]),bindings=state[3])
ordinary.update(active=True,expires_at_utc=stamp(240))
ordinary_ref=put("research/reviews/RAM_Census_NOT_AUTHORITY/forbidden_ordinary_close.json",ordinary)
expect(lambda:env["apply_close_completed_v1"](ordinary_ref,expected_source_sha256=S,expected_launcher_sha256=L),"Census_original_evidence_cannot_be_relabelled_ordinary_success")
ram_os.live_pid=manifest["targets"][keys[0]]["known_PID_start_rows"][0]["pid"]
expect(lambda:env["apply_retire_stopped_v1"](op,expected_source_sha256=S,expected_launcher_sha256=L),"live_or_reused_PID_blocks")
ram_os.live_pid=None
ram_os.live_group=manifest["targets"][keys[0]]["known_process_groups"][0]
expect(lambda:env["apply_retire_stopped_v1"](op,expected_source_sha256=S,expected_launcher_sha256=L),"live_or_reused_group_blocks")
ram_os.live_group=None
helper_mode.value="timeout"
expect(lambda:env["apply_retire_stopped_v1"](op,expected_source_sha256=S,expected_launcher_sha256=L),"300second_deadline_blocks_without_append")
helper_mode.value="normal";clock.value=0.0
need(not journal,"Rejected live/deadline fixtures append no events")
for key in keys:
 op,req=prepare_operation(key)
 receipt=env["apply_retire_stopped_v1"](op,expected_source_sha256=S,expected_launcher_sha256=L)
 receipts.append(receipt);operation_refs.append(op)
 need(receipt["credit_bytes"]==20000000000-receipt["retained_bytes"]and receipt["artifacts_deleted"]==0,"Exact20GB unused allowance only "+key)
 need(receipt["scientific_success_asserted"]is False and receipt["value"]["proof"]["historical_facts_preserved"]["supervisor_outer_OS_wait"]=="UNKNOWN",
  "No scientific completion or fabricated supervisor wait "+key)
 need(receipt["value"]["measurement"]["scope_paths"]==manifest["targets"][key]["scope_paths"]and len(receipt["value"]["measurement"]["per_scope"])==1,
  "Whole single original root measured in RAM "+key)
 expect(lambda op=op:env["apply_retire_stopped_v1"](op,expected_source_sha256=S,expected_launcher_sha256=L),"same_operation_or_key_replay_blocked")
final=env["_state_locked"](S,L,fresh=True)
need(len(final[4])==2 and set(keys)<=final[1],"Both distinct retention events replay with exact inherited reservations")
need(final[2]["completed_credit_bytes"]==sum(r["credit_bytes"]for r in receipts),"Replay credit equals only two measured unused allowances")
need(all(final[0][k]==manifest["targets"][k]["original_reservation"]for k in keys),"Original reservations remain byte-value exact after replay")
need(measurefs.payload_opens==0,"Full helper opened no regular file or stored link target")
# Actual measurement helper negatives operate only on this virtual filesystem.
one=manifest["targets"][keys[0]]["scope_paths"][0]
measure=lambda paths=[one],expected=[one]:helper.measure_retained_nodes_v1(MPath(ROOT),[MPath(p)for p in paths],expected_paths=expected,socket_is_absent=lambda p:False)
helper.require=lambda ok,message:None if ok else (_ for _ in ()).throw(ValueError(message))
expect(lambda:measure([one,one+"/dummy"],[one,one+"/dummy"]),"dummy_second_scope_rejected")
expect(lambda:measure([one+"/other"],[one]),"wrong_single_scope_rejected")
baseline=copy.deepcopy(measurefs.nodes)
for kind in("socket","fifo"):
 measurefs.add(one+"/unsafe",kind)
 expect(measure,"retained_"+kind+"_blocks")
 measurefs.nodes=copy.deepcopy(baseline)
measurefs.nodes[one+"/nested/preserved.bin"].st_nlink=2
expect(measure,"hardlinked_regular_blocks");measurefs.nodes=copy.deepcopy(baseline)
measurefs.nodes[one+"/nested/preserved.bin"].st_dev=2
expect(measure,"cross_device_node_blocks");measurefs.nodes=copy.deepcopy(baseline)
measurefs.nodes[one+"/nested/preserved.bin"].st_ino=measurefs.nodes[one].st_ino
expect(measure,"aliased_node_blocks");measurefs.nodes=copy.deepcopy(baseline)
measurefs.nodes[str(RealPath(one).parent)].st_mode=stat.S_IFLNK|0o700
expect(measure,"linked_ancestor_or_unsafe_descent_blocks");measurefs.nodes=copy.deepcopy(baseline)
measurefs.unstable=True;expect(measure,"unstable_metadata_blocks");measurefs.unstable=False
need(measurefs.payload_opens==0,"All virtual scanner rejection fixtures preserve payload-opening prohibition")
# AST deltas: core ordinary accounting and guard methods remain exact to Relate64 pattern.
base_ast=ast.parse((ROOT/"research/reviews/Relate_failed64GiB_retention_successor_source_20261017_v1/relate_failed64_retention_v1.py").read_text())
def fn(t,name):return next(n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
preserved=("require","charge","warning_fields","check_prefix","assert_open","_proof","_absent","_measurement","_pins","consumer_bindings_v1",
 "check_storage_locked_v1","budget_check_v9","validate_storage_v9","PilotOutputV9","_stamp","_validate_targets","_observe_targets","_recorded_stopped_proof",
 "apply_reserve_v1","apply_close_completed_v1")
# Message-only Census naming changes do not alter any guard operands.
for name in preserved:
 a=ast.dump(fn(ast.parse(source),name),include_attributes=False);b=ast.dump(fn(base_ast,name),include_attributes=False)
 need(a==b,"Original accounting/guard AST exact "+name)
old_measure=(ROOT/"research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/retained_nodes_v1.py").read_text()
expected_measure=old_measure.replace('len(paths)==2,"Both exact charged parents, in original order"','len(paths)==1,"One exact charged parent, never a dummy auxiliary scope"').replace("two_key_exact_stopped_retained_node_measurement_v1","Census_observation_exact_single_root_retained_node_measurement_v1")
need((D/"single_root_retained_nodes_v1.py").read_text()==expected_measure,"Single-root helper has only exact arity/error/schema delta")
parent_manifest=json.loads((ROOT/"research/reviews/Relate_failed64GiB_retention_successor_source_20261017_v1/source_manifest.v1.json").read_text())

need(manifest["inherited_parent_source_pins_exact"]==parent_manifest["source_sha256"]and
 all(manifest["source_sha256"][p]==h for p,h in parent_manifest["source_sha256"].items()),"All17 actual predecessor source pins preserved")
need(manifest["runtime_authenticated_historical_source_and_metadata_refs"][:73]==parent_manifest["runtime_authenticated_historical_source_and_metadata_refs"],
 "All73 predecessor metadata refs preserved in original order")
return_value=dict(schema="Census_observation_whole_module_and_single_root_RAM_check_v1",status="PASS_SOURCE_AND_RAM_ONLY",
 positive_checks=len(checks),negative_fixture_count=len(negative),negative_fixtures=negative,checks=checks,
 full_candidate_module_executed_with_single_RAM_project_guard_import=True,
 actual_candidate_policy_selector_manifest_original_metadata_proof_request_apply_append_and_replay_paths=True,
 actual_single_root_scanner_executed_only_on_virtual_filesystem=True,original_guard_AST_exact=list(preserved),
 parent_roles_preserved=46,new_direct_capsule_roles=1,original64_role_ceiling_unchanged=True,
 native_or_FLARE_future_union_fit_claimed=False,
 actual_guard_imports_or_state_calls=0,actual_process_or_GPU_observations=0,actual_retained_measurements=0,
 actual_journal_mutations_or_activation_or_admission_or_credit=0,scientific_array_truth_or_checkpoint_reads=0,
 RAM_journal_events=len(journal),RAM_single_root_receipt_count=len(receipts),real_payload_opens=0)
print(json.dumps(return_value,sort_keys=True,indent=2))
