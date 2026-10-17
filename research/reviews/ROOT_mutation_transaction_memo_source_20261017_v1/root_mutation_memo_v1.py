"""Inactive, reserve-only ROOT transaction memo. Import performs no project IO."""
from pathlib import Path
from types import FunctionType,SimpleNamespace
import ast,copy,hashlib,json,os,stat,sys,threading
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/ROOT_mutation_transaction_memo_source_20261017_v1"
SOURCE=HERE+"/root_mutation_memo_v1.py"
PINS={
 "guard":dict(path="research/reviews/Relate_failed64GiB_retention_successor_source_20261017_v1/relate_failed64_retention_v1.py",sha256="b8b696a0df45c12b931e563600c2b52e89e73dd19c7d160351063687dcfb5164"),
 "v4":dict(path="src/arggnn/project_budget_v4.py",sha256="a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47"),
 "budget":dict(path="scripts/project_budget_v9.py",sha256="483d78bf2ef3c917d65ecf1dde5a7809bfadc21194c9f9a90394ce5c532fdd5e")}
_ACTIVE=False
def need(ok,message):
 if not ok:raise ValueError(message)
def source_bytes(record):
 p=ROOT/record["path"];need(p.resolve()==p and ROOT in p.parents,"Canonical pinned source")
 before=p.lstat();need(stat.S_ISREG(before.st_mode)and before.st_nlink==1 and before.st_size<=2**20,"Bounded source only")
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,"rb")as f:opened=os.fstat(f.fileno());data=f.read(2**20+1);after=os.fstat(f.fileno())
 named=p.lstat();identity=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
 need(identity(before)==identity(opened)==identity(after)==identity(named)and len(data)==before.st_size
  and hashlib.sha256(data).hexdigest()==record["sha256"],"Exact stable source SHA")
 return data
def functions_from_source(module,record,names):
 data=source_bytes(record);path=ROOT/record["path"]
 need(Path(module.__file__)==path,"Exact supported module pathname")
 tree=ast.parse(data);compiled=compile(data,str(path),"exec",dont_inherit=True,optimize=0);result={}
 for name in names:
  nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name]
  codes=[c for c in compiled.co_consts if hasattr(c,"co_name")and c.co_name==name]
  need(len(nodes)==len(codes)==1,"One exact supported callable "+name)
  fn=getattr(module,name);original=fn.__wrapped__ if name=="mutex"and hasattr(fn,"__wrapped__")else fn
  need(original.__code__==codes[0]and original.__globals__ is module.__dict__ and fn.__module__==module.__name__,
   "Original unmodified callable "+name)
  result[name]=(fn,nodes[0])
 return result
def clone(fn,changes):
 scope=dict(fn.__globals__);scope.update(changes)
 result=FunctionType(fn.__code__,scope,fn.__name__,fn.__defaults__,fn.__closure__)
 result.__kwdefaults__=copy.copy(fn.__kwdefaults__);result.__module__=__name__
 return result
def critical_reserve(node,scope,append):
 need(len(node.body)==1 and isinstance(node.body[0],ast.With),"Exact single reserve critical section")
 block=node.body[0]
 need(len(block.items)==1 and block.items[0].optional_vars is None
  and ast.unparse(block.items[0].context_expr)=="budget.v4.mutex(ROOT)","Original actual reserve mutex")
 derived=copy.deepcopy(node);derived.name="_memo_reserve_critical";derived.body=copy.deepcopy(block.body)
 derived.decorator_list=[];namespace=dict(scope,_append=append,__name__=__name__)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[derived],type_ignores=[])),
  SOURCE+"#exact-reserve-critical-body","exec"),namespace)
 return namespace[derived.name]
class _Transaction:
 """Private core; caller must enter only after the actual original mutex yields."""
 def __init__(self,root,v4,budget,allow,max_entries=2048,max_bytes=64*2**20):
  self.root=root;self.v4=v4;self.budget=budget;self.allow=allow
  self.max_entries=max_entries;self.max_bytes=max_bytes;self.sources={};self.names={};self.bytes=0;self.enabled=True
  self.original={n:getattr(v4,n)for n in("bound","reference","scope_spec","canonical","file_bytes","mutex")}
  self.successor=budget.successor_paths;self.patches=[]
  self.counts=dict(source_hits=0,source_misses=0,namespace_hits=0,namespace_misses=0,revalidated_sources=0,revalidated_names=0)
 def relative(self,root,name):
  need(root==self.root,"Exact transaction root")
  p=Path(name);p=p if p.is_absolute()else root/p
  need(p!=root and root in p.parents and ".."not in p.parts,"Contained lexical path")
  return str(p.relative_to(root))
 def file_bytes(self,root,name):
  relative=self.relative(root,name);expected=self.allow.get(relative)
  if not self.enabled or expected is None:return self.original["file_bytes"](root,name)
  if relative in self.sources:self.counts["source_hits"]+=1;return self.sources[relative]
  raw,digest=self.original["file_bytes"](root,name)
  need(digest==expected,"Explicit immutable source SHA")
  self.counts["source_misses"]+=1
  if len(self.sources)<self.max_entries and self.bytes+len(raw)<=self.max_bytes:
   self.sources[relative]=(raw,digest);self.bytes+=len(raw)
  return raw,digest
 def canonical(self,root,name):
  relative=self.relative(root,name)
  if not self.enabled:return self.original["canonical"](root,name)
  if relative in self.names:self.counts["namespace_hits"]+=1;return self.names[relative]
  result=self.original["canonical"](root,name);self.counts["namespace_misses"]+=1
  if len(self.names)<self.max_entries:self.names[relative]=result
  return result
 def install(self):
  bound=clone(self.original["bound"],dict(file_bytes=self.file_bytes))
  reference=clone(self.original["reference"],dict(file_bytes=self.file_bytes,canonical=self.canonical))
  scope=clone(self.original["scope_spec"],dict(bound=bound,reference=reference,canonical=self.canonical))
  view=SimpleNamespace(**self.v4.__dict__);view.scope_spec=scope;view.canonical=self.canonical
  successor=clone(self.successor,dict(v4=view))
  for owner,name,value in[(self.v4,"bound",bound),(self.v4,"reference",reference),(self.v4,"scope_spec",scope),
    (self.budget,"successor_paths",successor)]:
   self.patches.append((owner,name,getattr(owner,name),value));setattr(owner,name,value)
 def finish(self):
  """Disable hits first; original uncached checks precede return/first append."""
  need(self.enabled,"One uncached final validation only");self.enabled=False
  for relative,(raw,digest)in self.sources.items():
   need(self.original["file_bytes"](self.root,relative)==(raw,digest),"Cached source changed before commit")
   self.counts["revalidated_sources"]+=1
  for relative,expected in self.names.items():
   need(self.original["canonical"](self.root,relative)==expected,"Cached namespace changed before commit")
   self.counts["revalidated_names"]+=1
 def close(self):
  error=None
  for owner,name,original,installed in reversed(self.patches):
   if getattr(owner,name)is not installed:error="Unexpected transaction patch replacement"
   setattr(owner,name,original)
  self.patches.clear();self.sources.clear();self.names.clear();self.bytes=0;self.enabled=False
  if error:raise ValueError(error)
class RootMutationMemo:
 """Only read_state_v1 and reserve_v1, only the exact pinned Relate guard."""
 def __init__(self,guard,*,integration_manifest_ref,adapter_ref,cache_refs):
  need(sys.dont_write_bytecode and sys.flags.no_user_site and sys.flags.optimize==0
   and Path.cwd()==ROOT and threading.active_count()==1,"Fresh single-thread ROOT -B -s interpreter")
  self.guard=guard;self.budget=guard.budget;self.v4=self.budget.v4
  self.manifest_ref=copy.deepcopy(integration_manifest_ref);self.adapter_ref=copy.deepcopy(adapter_ref)
  need(self.adapter_ref["path"]==SOURCE,"Exact memo source path")
  source_bytes(self.adapter_ref)
  self.allow={}
  for record in cache_refs:
   need(isinstance(record,dict)and set(record)=={"path","sha256"}and isinstance(record["path"],str)
    and Path(record["path"]).suffix in(".py",".sh")and not Path(record["path"]).is_absolute()
    and ".."not in Path(record["path"]).parts and str(Path(record["path"]))==record["path"]
    and record["path"].split("/")[0]in("src","scripts","research")
    and isinstance(record["sha256"],str)and len(record["sha256"])==64
    and all(c in"0123456789abcdef"for c in record["sha256"]),"Explicit immutable code refs only; JSON/data/accounting never cached")
   need(record["path"]not in self.allow or self.allow[record["path"]]==record["sha256"],"No conflicting source pins")
   self.allow[record["path"]]=record["sha256"]
  need(len(self.allow)<=2048,"Bounded explicit source cache")
  self.counts={};self._check()
 def _check(self):
  need(threading.active_count()==1 and not hasattr(self.guard,"_outer_memo"),"No concurrent consumer memo or threads")
  self.guard_functions=functions_from_source(self.guard,PINS["guard"],("apply_reserve_v1","_state_locked","_append"))
  functions_from_source(self.v4,PINS["v4"],("bound","reference","scope_spec","canonical","file_bytes","mutex"))
  functions_from_source(self.budget,PINS["budget"],("successor_paths",))
  manifest=self.v4.bound(ROOT,self.manifest_ref,True)
  need(manifest.get("source_sha256",{}).get(SOURCE)==self.adapter_ref["sha256"],
   "Already directly bound integration manifest transitively authenticates exact memo source")
 def _run(self,mode,authority_ref=None,*,expected_source_sha256,expected_launcher_sha256):
  global _ACTIVE
  need(not _ACTIVE and mode in("state","reserve"),"No nested or unsupported mutation memo")
  self._check();original_mutex=self.v4.mutex;original_append=self.guard._append
  _ACTIVE=True;tx=None
  try:
   with original_mutex(ROOT):
    tx=_Transaction(ROOT,self.v4,self.budget,self.allow)
    try:
     tx.install()
     if mode=="state":
      result=self.guard._state_locked(expected_source_sha256,expected_launcher_sha256,fresh=True)
      source_bytes(self.adapter_ref);tx.finish();return result
     calls=[0]
     def append(operation,record,value,state):
      need(calls[0]==0 and operation=="reserve","One exact reserve append only");calls[0]+=1
      need(self.manifest_ref in value["consumer_bindings"],"Transitive memo integration manifest remains a DIRECT role")
      # JSON manifest is uncached and must still authenticate this exact implementation.
      manifest=tx.original["bound"](ROOT,self.manifest_ref,True)
      need(manifest.get("source_sha256",{}).get(SOURCE)==self.adapter_ref["sha256"],"Exact immutable integration join")
      source_bytes(self.adapter_ref);tx.finish()
      # Caching is disabled throughout original append, fsync and post-write read.
      return original_append(operation,record,value,state)
     critical=critical_reserve(self.guard_functions["apply_reserve_v1"][1],self.guard.__dict__,append)
     result=critical(authority_ref,expected_source_sha256=expected_source_sha256,expected_launcher_sha256=expected_launcher_sha256)
     need(calls[0]==1,"Exact reserve must append once");return result
    finally:
     self.counts=dict(tx.counts);tx.close()
  finally:_ACTIVE=False
 def read_state_v1(self,*,expected_source_sha256,expected_launcher_sha256):
  return self._run("state",expected_source_sha256=expected_source_sha256,expected_launcher_sha256=expected_launcher_sha256)
 def reserve_v1(self,authority_ref,*,expected_source_sha256,expected_launcher_sha256):
  return self._run("reserve",authority_ref,expected_source_sha256=expected_source_sha256,expected_launcher_sha256=expected_launcher_sha256)
if __name__=="__main__":
 print(json.dumps(dict(status="INACTIVE_SOURCE_ONLY",guard_imported=False,operations=False)))
