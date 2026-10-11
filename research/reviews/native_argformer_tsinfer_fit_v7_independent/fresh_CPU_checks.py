"""Fresh independent v7 stat-policy probes; all real objects stay in this review."""
from pathlib import Path
import ast,builtins,copy,datetime,hashlib,importlib.util,json,os,re,socket,stat,sys,types,uuid
from unittest.mock import patch
R=Path("/workspace-SR003.nfs2/ARG-GNN")
O=R/"research/reviews/native_argformer_tsinfer_fit_v7_independent"
O.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location("_independent_native_v7",R/"src/arggnn/native_argformer_tsinfer_fit_v7.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
C=m.policy();checks=[];events=[];diagnostics=[];manifests=[];fixtures=[];handles=[]
assert os.environ["CUDA_VISIBLE_DEVICES"]==""
def check(name,condition=True):
 if not condition:raise AssertionError(name)
 checks.append(name)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def owned(p):return p==O or O in p.parents
def audit(event,args):
 if event=="open" and not isinstance(args[0],int):
  p=Path(os.path.abspath(os.fsdecode(args[0])));mode,flags=args[1:3]
  writing=(isinstance(mode,str) and any(x in mode for x in "wax+")) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
  if writing and not owned(p):events.append("outside write "+str(p));raise RuntimeError(events[-1])
  if not writing and not owned(p):
   for base in (R/"data",R/"experiments/results",R/".runtime/conda/envs",R/".runtime/gpup",R/".runtime/locks",R/".runtime/storage",Path("/proc")):
    if p==base or base in p.parents:events.append("actual state read "+str(p));raise RuntimeError(events[-1])
 if event in ("socket.connect","subprocess.Popen","os.system","os.posix_spawn"):
  events.append(event);raise RuntimeError(event)
 if event=="socket.bind":
  sock,address=args[:2];p=Path(os.path.abspath(os.fsdecode(address)))
  if sock.family!=socket.AF_UNIX or not owned(p):events.append("outside own AF_UNIX bind");raise RuntimeError(events[-1])
sys.addaudithook(audit)
orig_import=builtins.__import__
def imported(name,*a,**k):
 if name.split(".")[0] in ("torch","numpy","tskit","tsinfer","composer","transformers","streaming"):
  events.append("scientific import "+name);raise RuntimeError(events[-1])
 return orig_import(name,*a,**k)
builtins.__import__=imported
author=R/"research/reviews/native_argformer_tsinfer_fit_v7_author"
receipt=json.loads((author/"source_receipt.json").read_text())
check("assigned_source",sha(m.SCRIPT)=="b94a4e16a6a9dc230256974b9a0e45a6edd1b41b9ecc8ac4171c4461ec514712")
check("assigned_config",sha(m.CONFIG)=="d744aad4883062d23f874d15be65068b5f1b5e8a44db20ed2e8a425a4fd336c1")
check("assigned_author_receipt",sha(author/"source_receipt.json")=="dec4fbc22577a858846e9467e392b63932b89e52ad535975306e6592d6e9ca34")
pins={x["path"]:x["sha256"] for x in receipt["files"]}
pins.update(json.loads((author/"baseline_pins.json").read_text()))
registration=R/"research/reviews/native_argformer_tsinfer_fit_v7_registration_root.json"
pins[str(registration.relative_to(R))]="3daa7546ece9778b0494612a7731b629fd596ba42e4bdbf1c991f59ad309a3e7"
for p,h in pins.items():check("pin_before_"+p,sha(R/p)==h)
S=m.SCRIPT.read_text();old=(R/"src/arggnn/native_argformer_tsinfer_fit_v6.py").read_text()
inverse=S.replace("_v7","_v6").replace("afts7","afts6").replace(m.CONFIG_SHA256,"e0dfe65b93ce614a13bbfc390b9f79988956e3724e3c697ad840fc2dc0196318")
start=inverse.index("class NativeOwnedScopeError");end=inverse.index("class Storage:",start)
inverse=inverse[:start-1]+inverse[end:]
inverse=inverse.replace("boundary=False)","boundary=True)").replace("        measured=native_owned_scope_measurement(self.scope,self.tmp)\n        used=measured['conservative_scope_bytes']","        used=local_bytes(self.scope)+local_bytes(self.tmp)").replace("'stage_boundary_cap_not_OS_quota':True,\n          'native_owned_scope_measurement':measured}","'stage_boundary_cap_not_OS_quota':True}")
check("entire_source_bytes_inverse_identical",inverse==old)
previous=json.loads((R/"configs/native_argformer_tsinfer_fit_v6.json").read_text());cfg=copy.deepcopy(C)
cfg.pop("namespace_correction_diagnosis");cfg.pop("namespace_correction_scope");cfg["storage"].pop("native_owned_namespace_policy")
check("entire_config_inverse_identical",json.loads(json.dumps(cfg).replace("_v7","_v6").replace("afts7","afts6"))==previous)
proof=json.loads((author/"narrow_successor_proof.json").read_text())
for p,names in proof["native_scientific_ASTs"].items():
 check("frozen_scientific_source_"+p,sha(R/p)==C["frozen_files_sha256"][p])
 nodes={n.name:n for n in ast.parse((R/p).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 for name,h in names.items():check("original_scientific_AST_"+name,hashlib.sha256(ast.dump(nodes[name],include_attributes=False).encode()).hexdigest()==h)
check("native_runtime_contract_and_both_capture_passes_preserved",C["runtime"]==previous["runtime"] and C["launcher_contract"]==previous["launcher_contract"] and C["runtime_inventory_contract"]["prefix_import_bytes"]==previous["runtime_inventory_contract"]["prefix_import_bytes"])
check("inactive20GB_and_actual_measured500GB_launcher",C["execution_enabled"] is False and C["storage"]["reserved_bytes_per_parent"]==20000000000 and C["launcher_contract"]["measured_project_hard_limit_bytes"]==500000000000)
class F:
 def __init__(self):
  self.base=O/("f_"+uuid.uuid4().hex[:8]);self.base.mkdir();self.result=self.base/"r";self.tmp=self.base/"t";self.result.mkdir();self.tmp.mkdir();fixtures.append(self.base)
 def sock(self,relative="pymp-abcdefgh/listener-12345678",result=False,mode=0o700):
  p=(self.result if result else self.tmp)/relative;p.parent.mkdir(parents=True,exist_ok=True);p.parent.chmod(mode)
  s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);handles.append(s);cwd=Path.cwd()
  try:os.chdir(self.base);s.bind(str(p.relative_to(self.base)))
  finally:os.chdir(cwd)
  st=p.lstat();manifests.append({"real_path":str(p),"absolute_bytes":len(os.fsencode(p)),"uid":st.st_uid,"nlink":st.st_nlink,"inode":st.st_ino,"family":"AF_UNIX","own_relative_bind":True});return p
 def measure(self):return m.native_owned_scope_measurement(self.result,self.tmp)
def rejected(name,call,contains=None):
 try:call()
 except m.NativeOwnedScopeError as e:
  x=e.namespace_metadata;check(name,contains is None or contains in x["reason"]);check(name+"_metadata",set(x)=={"kind","reason","path","object"} and "present" in x["object"])
  if x["object"]["present"]:check(name+"_stat_fields",all(k in x["object"] for k in ("type","dev","inode","uid","gid","nlink","mode","size","blocks","mtime_ns","ctime_ns")))
  diagnostics.append({"case":name,"error":x});return x
 raise AssertionError("namespace accepted "+name)
# Real own result/tmp regular objects are measured without payload opens.
f=F();p=f.result/"data";p.write_bytes(b"handmade");r=f.measure()
charge=sum(max(x.lstat().st_size,x.lstat().st_blocks*512) for x in (f.result,f.tmp,p))
check("real_regular_and_directory_exact_stat_charge",r["conservative_scope_bytes"]==charge and r["entries"]==3)
with patch.object(Path,"open",side_effect=AssertionError("payload open")),patch.object(os,"open",side_effect=AssertionError("payload FD")):
 check("measurement_never_opens_payload",f.measure()["stat_only"])
# The exclusive reviewer directory is longer than107; bind relative and verify real rejection.
f=F();p=f.sock();check("real_own_socket_path_exceeds107",len(os.fsencode(p))>107);rejected("real_overlong_AF_UNIX_rejected",f.measure,"107bytes")
# Exercise accepted socket rules using a virtual short address backed only by this real own inode.
# This is explicitly a path/stat provider fixture; no real short address outside ownership is created.
orig_resolve=Path.resolve;orig_lstat=Path.lstat;orig_scandir=os.scandir;orig_lexists=os.path.lexists
def virtual_measure(f,vroot="/v",stat_transform=None):
 V=Path(vroot)
 def backing(p):
  p=Path(p)
  if p==V:return f.base
  if V in p.parents:return f.base/p.relative_to(V)
  return p
 def resolved(p,*a,**k):return Path(p) if Path(p)==V or V in Path(p).parents else orig_resolve(p,*a,**k)
 def lstat(p,*a,**k):
  st=orig_lstat(backing(p),*a,**k)
  return stat_transform(Path(p),st) if stat_transform else st
 with patch.object(m,"ROOT",V),patch.object(Path,"resolve",resolved),patch.object(Path,"lstat",lstat),patch.object(os,"scandir",side_effect=lambda p:orig_scandir(backing(p))),patch.object(os.path,"lexists",side_effect=lambda p:orig_lexists(backing(p))):
  return m.native_owned_scope_measurement(V/"r",V/"t")
f=F();p=f.sock();r=virtual_measure(f)
check("real_socket_inode_short_provider_accepted_and_charged",r["accepted_AF_UNIX_socket_inodes"]==1 and r["conservative_scope_bytes"]==sum(max(x.lstat().st_size,x.lstat().st_blocks*512) for x in (f.result,f.tmp,p.parent))+max(p.lstat().st_size,p.lstat().st_blocks*512,4096))
for size in (107,108):
 f=F();f.sock();base=len(os.fsencode(Path("/v")/"t/pymp-abcdefgh/listener-12345678"));vroot="/"+("v"*(1+size-base))
 check("provider_absolute_length_"+str(size),len(os.fsencode(Path(vroot)/"t/pymp-abcdefgh/listener-12345678"))==size)
 if size==107:check("107_byte_provider_socket_accepted",virtual_measure(f,vroot)["accepted_AF_UNIX_socket_inodes"]==1)
 else:rejected("108_byte_provider_socket_rejected",lambda:virtual_measure(f,vroot),"107bytes")
for name,relative,result,mode in (("outside_tmp","pymp-abcdefgh/listener-12345678",True,0o700),("bad_parent_name","pymp-abcdefg/listener-12345678",False,0o700),("bad_listener_name","pymp-abcdefgh/listener-1234567A",False,0o700),("deeper_socket","extra/pymp-abcdefgh/listener-12345678",False,0o700),("bad_parent_mode","pymp-abcdefgh/listener-12345678",False,0o750),("parent_special_mode","pymp-abcdefgh/listener-12345678",False,0o1700)):
 f=F();f.sock(relative,result,mode);rejected(name,lambda:virtual_measure(f))
def fields(st,**values):
 attrs=("st_dev","st_ino","st_mode","st_uid","st_gid","st_nlink","st_size","st_blocks","st_mtime_ns","st_ctime_ns")
 return types.SimpleNamespace(**{k:values.get(k,getattr(st,k)) for k in attrs})
f=F();f.sock()
rejected("wrong_socket_UID",lambda:virtual_measure(f,stat_transform=lambda p,s:fields(s,st_uid=s.st_uid+1) if stat.S_ISSOCK(s.st_mode) else s),"UID")
f=F();f.sock()
rejected("wrong_socket_parent_UID",lambda:virtual_measure(f,stat_transform=lambda p,s:fields(s,st_uid=s.st_uid+1) if p.name=="pymp-abcdefgh" else s),"owned0700")
f=F();p=f.sock();os.link(p,f.tmp/"duplicate")
rejected("socket_hardlink",lambda:virtual_measure(f))
for name in ("FIFO","regular_hardlink","file_symlink","directory_symlink","dangling_link","root_regular","overlapping"):
 f=F();p=f.result/"a"
 if name=="FIFO":os.mkfifo(p)
 elif name=="regular_hardlink":p.write_bytes(b"a");os.link(p,f.result/"b")
 elif name=="file_symlink":p.write_bytes(b"a");(f.result/"alias").symlink_to(p)
 elif name=="directory_symlink":(f.result/"alias").symlink_to(f.tmp,target_is_directory=True)
 elif name=="dangling_link":p.symlink_to(f.base/"absent")
 elif name=="root_regular":f.result.rmdir();f.result.write_bytes(b"a")
 else:f.tmp=f.result/"nested";f.tmp.mkdir()
 rejected("real_"+name,f.measure)
f=F();rejected("outside_project_scope",lambda:m.native_owned_scope_measurement(Path("/etc"),f.tmp),"project namespace")
# Directory membership mutation after listing rejects before returning.
f=F()
class Listing:
 def __init__(self,p):self.p=Path(p);self.ctx=orig_scandir(p)
 def __enter__(self):return self.ctx.__enter__()
 def __exit__(self,*a):
  self.ctx.__exit__(*a)
  if self.p==f.result:(self.p/"appeared").write_bytes(b"x")
with patch.object(os,"scandir",side_effect=Listing):rejected("directory_membership_race",f.measure,"membership")
# Regular nlink change and disappearing-entry metadata are separately rechecked.
for mode in ("hardlink","disappear"):
 f=F();p=f.result/"file";p.write_bytes(b"x");seen=[]
 def racing(path,*a,**k):
  if Path(path)==p:
   seen.append(True)
   if len(seen)==2:
    if mode=="hardlink":os.link(p,f.result/"link")
    else:p.unlink()
  return orig_lstat(path,*a,**k)
 with patch.object(Path,"lstat",racing):x=rejected("regular_"+mode+"_race",f.measure)
 if mode=="disappear":check("last_successful_lstat_retained",x["object"]["currently_present"] is False and x["object"]["last_successful_lstat_retained"] is True and x["object"]["type"]=="regular")
f=F();f.result.rmdir();done=[]
def appearing(path):
 manager=orig_scandir(path)
 if Path(path)==f.tmp and not done:f.result.mkdir();done.append(True)
 return manager
with patch.object(os,"scandir",side_effect=appearing):rejected("absent_scope_appearance_race",f.measure,"appeared")
f=F();f.result.rmdir();f.tmp.rmdir();check("fresh_missing_scopes_zero_charge",f.measure()["entries"]==0)
# Existing unmodified Storage.check with an inert accounting provider, never real ledgers.
f=F();st=object.__new__(m.Storage);st.c=copy.deepcopy(C);st.receipt={"reservation":{"key":"fixture","output":str(f.result),"auxiliary_scopes":[str(f.tmp)],"reserved_bytes":20000000000},"companion":{"implementation":{"sha256":"a"*64},"config":{"sha256":"b"*64}},"legacy_ledger_sha256_at_admission":"c"*64,"supplemental_ledger_sha256_at_admission":"d"*64};st.key="fixture";st.scope=f.result;st.tmp=f.tmp;st.first=True;st.high_water=0;calls=[]
def account(*a,**k):calls.append(k);return {"projected_bytes":600000000000,"limit_bytes":950000000000,"scope_measurement":None}
st.companion=types.SimpleNamespace(check_reservation=account)
r=st.check();check("accounting_then_private_measurement_boundary_false",calls[0]["boundary"] is False and calls[0]["initial_ledger_pins"] is not None and r["native_owned_scope_measurement"]["stat_only"])
p=f.result/"growth";p.write_bytes(b"x"*20000);high=st.check()["high_water_bytes"];p.write_bytes(b"");r=st.check();check("high_water_remains_after_truncate",r["high_water_bytes"]==high and calls[-1]["initial_ledger_pins"] is None)
st.c["storage"]["reserved_bytes_per_parent"]=high-1;st.receipt["reservation"]["reserved_bytes"]=high-1
try:st.check()
except ValueError as e:check("combined_conservative_sampled_cap_rejects","20GB sampled cap" in str(e))
else:raise AssertionError("cap ignored")
st.c["storage"]["reserved_bytes_per_parent"]=20000000000;st.receipt["reservation"]["reserved_bytes"]=20000000000
st.companion.check_reservation=lambda *a,**k:{"projected_bytes":950000000000,"limit_bytes":950000000000}
with patch.object(m,"native_owned_scope_measurement",side_effect=AssertionError("private scan before accounting rejection")):
 try:st.check()
 except ValueError as e:check("projection_rejects_before_private_scan","projection" in str(e))
 else:raise AssertionError("bad accounting accepted")
# New error metadata survives the unchanged terminal failure writer.
f=F();os.mkfifo(f.result/"blocked");x=rejected("terminal_FIFO_diagnostic",f.measure)
out=O/"terminal_fixture";out.mkdir()
m.terminal_receipt(C,out,{"status":"failed_partial_preserved","exit_code":1,"failure":{"type":"NativeOwnedScopeError","message":json.dumps(x)}})
check("existing_terminal_writer_preserves_new_offender_metadata",json.loads(json.loads((out/"receipt.json").read_text())["failure"]["message"])==x)
for p,h in pins.items():check("pin_after_"+p,sha(R/p)==h)
check("no_forbidden_access_events",not events)
for s in handles:s.close()
def clean(p):
 for child in p.iterdir():
  if stat.S_ISDIR(child.lstat().st_mode):clean(child)
  else:child.unlink()
 p.rmdir()
for p in fixtures:clean(p)
clean(out);check("all_real_fixtures_and_socket_endpoints_removed",all(not p.exists() for p in fixtures))
report={"kind":"native_argformer_tsinfer_fit_v7_fresh_independent_CPU_checks","verdict":"PASS","count":len(checks),"checks":checks,"diagnostics":diagnostics,"real_AF_UNIX_manifests":manifests,"audit_events":events,"reviewed_pins":pins,"whole_source_bytes_and_config_inverse_delta_identical":True,"actual_real_short_socket_outside_reviewer_ownership_created":False,"accepted_socket_and107_108_provider_tests_disclosure":"Virtual short canonical path/stat providers use metadata from real AF_UNIX inodes created only by relative binds under this long reviewer directory. Real absolute own over107 path independently rejects. No pathname-length function monkeypatch, external fixture writes, or contents/peers opened.","scientific_imports_actual_data_TEST_models_GPU_native_prefix_budget_ledger_lease_process_or_Git_operations":False}
(O/"fresh_CPU_checks.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
print(json.dumps({"verdict":"PASS","count":len(checks),"audit_events":events}))
