"""Inactive strict-JSON computation reuse for separately reviewed future consumers.

No guard, process, resource, lock or accounting operation runs at import.
The old namespace memo and all required guard boundaries remain unchanged.
"""
from contextlib import contextmanager
from pathlib import Path
from types import CodeType, FunctionType
import ast,copy,hashlib,json,os,stat,sys,threading

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/consumer_strict_JSON_parse_memo_source_20261017_v2/parse_memo_v2.py"
V4_REF={"path":"src/arggnn/project_budget_v4.py","sha256":"a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47"}
MEMO_REF={"path":"research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py","sha256":"c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364"}
NS_REF={"path":"research/reviews/current_mutex_contention_20261015_v1/consumer_namespace_memo_v1.py","sha256":"17a1377320f61c3c0cc8d3816d0259f9dae3121ed91bf7206a7ee63fcab8eb7f"}
FORBIDDEN_WORDS=("authority","instruction","policy","selector","activation","revocation","request","input","runtime","config","manifest","review","journal","ledger","inventory","fingerprint","observation","proof","intent","directive")
MAX_ENTRIES=256
MAX_RAW_BYTES=4*2**20
MAX_PARSED_BYTES=32*2**20

def need(ok,message):
 if not ok:raise ValueError(message)

def _relative(path):
 p=Path(path)
 need(not p.is_absolute() and str(p)==path and ".."not in p.parts and p!=Path("."),"Exact repository-relative path")
 return path

def _record(record):
 need(type(record)is dict and set(record)=={"path","sha256"},"Exact immutable ref")
 _relative(record["path"]);sha=record["sha256"]
 need(type(sha)is str and len(sha)==64 and all(c in"0123456789abcdef"for c in sha),"Exact lowercase SHA")
 return record

def _bytes(record):
 _record(record);p=ROOT/record["path"];need(p.resolve()==p,"No source/control symlink")
 before=p.lstat();need(stat.S_ISREG(before.st_mode)and before.st_nlink==1 and before.st_size<=2**21,"Bounded regular source/metadata")
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 with os.fdopen(fd,"rb")as f:
  opened=os.fstat(f.fileno());raw=f.read(2**21+1);after=os.fstat(f.fileno())
 identity=lambda x:(x.st_dev,x.st_ino,x.st_mode,x.st_nlink,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
 need(identity(before)==identity(opened)==identity(after)==identity(p.lstat())
  and len(raw)==before.st_size and hashlib.sha256(raw).hexdigest()==record["sha256"],"Stable exact source/control bytes")
 return raw

def _code(raw,path,name,owner=None):
 root=compile(raw,str(ROOT/path),"exec",dont_inherit=True,optimize=0)
 if owner:
  rows=[x for x in root.co_consts if isinstance(x,CodeType)and x.co_name==owner]
  need(len(rows)==1,"Exact original class");root=rows[0]
 rows=[x for x in root.co_consts if isinstance(x,CodeType)and x.co_name==name]
 need(len(rows)==1,"Exact single source-derived function")
 return rows[0]

def _function(module,name,record,raw):
 fn=getattr(module,name,None)
 need(fn is not None and type(fn)is FunctionType and not hasattr(fn,"__wrapped__")
  and Path(module.__file__).resolve()==ROOT/record["path"] and fn.__globals__ is module.__dict__
  and fn.__module__==module.__name__ and fn.__code__==_code(raw,record["path"],name),"Original exact SHA-source function required")
 return fn

def _charge(value):
 # Conservative repeated-occurrence charge. It is not a resource-fit claim.
 total=0;stack=[value]
 while stack:
  v=stack.pop();need(type(v)in(dict,list,str,int,float,bool,type(None)),"Original JSON types only")
  total+=sys.getsizeof(v)
  if total>MAX_PARSED_BYTES:return total
  if type(v)is dict:
   for k,item in v.items():stack.extend((k,item))
  elif type(v)is list:stack.extend(v)
 return total

def _derive_json_function(fn,record,raw,parse_for_record):
 tree=ast.parse(raw);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==fn.__name__]
 need(len(nodes)==1,"One original JSON reader definition")
 node=copy.deepcopy(nodes[0])
 need(len(node.args.args)==1 and node.args.args[0].arg=="record"
  and not node.args.defaults and not node.decorator_list and len(node.body)==3
  and ast.unparse(node.body[0])=="raw = _raw(record)"
  and isinstance(node.body[1],ast.Expr) and isinstance(node.body[1].value,ast.Call)
  and ast.unparse(node.body[1].value.func)=="require"
  and ast.unparse(node.body[2])=="return budget.v4.parse(raw)","Exact unchanged original raw/size-check/parse shape")
 node.body[2].value=ast.Call(func=ast.Name(id="_strict_parse_owned_record",ctx=ast.Load()),
  args=[ast.Name(id="record",ctx=ast.Load()),ast.Name(id="raw",ctx=ast.Load())],keywords=[])
 scope=dict(fn.__globals__,_strict_parse_owned_record=parse_for_record)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
  str(ROOT/record["path"]),"exec",dont_inherit=True,optimize=0),scope)
 return scope[fn.__name__]

class StrictJSONParseMemo:
 """A component only. Actual consumer/source/control integration needs whole review."""
 def __init__(self,consumer_handle,v4_module,budget_module,*,guard_source_entries,
              frozen_history_manifest_refs,eligible_historical_refs,current_control_refs):
  need(sys.dont_write_bytecode and sys.flags.no_user_site and sys.flags.optimize==0
   and threading.active_count()==1 and Path.cwd()==ROOT,"Fresh single-thread -B -s future consumer")
  self.handle=consumer_handle;self.v4=v4_module;self.budget=budget_module
  memo_raw=_bytes(MEMO_REF);v4raw=_bytes(V4_REF);_bytes(NS_REF)
  need(consumer_handle.__class__.__name__=="TransactionMemoAdapter"
   and consumer_handle.__class__.__module__==consumer_handle._make_mutex.__globals__["__name__"]
   and consumer_handle.module is v4_module and budget_module.v4 is v4_module,"Exact existing original consumer handle")
  for name in("_key","_state","_make_mutex","_make_file_bytes","_make_canonical","_validate_snapshot"):
   need(getattr(consumer_handle,name).__func__.__code__==_code(memo_raw,MEMO_REF["path"],name,"TransactionMemoAdapter"),"Pinned unchanged existing consumer frame")
  receipt=consumer_handle.receipt
  need(receipt.get("installed")is True and receipt.get("consumer_role")=="read_only_consumer"
   and receipt.get("namespace_extension_ref")==NS_REF and receipt.get("original_memo2_ref")==MEMO_REF,
   "Existing exact read-only namespace memo installed")
  need(v4_module.mutex is consumer_handle.mutex_wrapper and v4_module.file_bytes is consumer_handle.file_bytes_wrapper
   and v4_module.canonical is consumer_handle.canonical_wrapper,"No stacking or changed original memo functions")
  self.original_bound=_function(v4_module,"bound",V4_REF,v4raw)
  self.original_parse=_function(v4_module,"parse",V4_REF,v4raw)
  parser_node=next(n for n in ast.parse(v4raw).body if isinstance(n,ast.FunctionDef)and n.name=="parse")
  need([a.arg for a in parser_node.args.args]==["raw"] and not parser_node.args.defaults
   and not parser_node.decorator_list and not self.original_parse.__code__.co_freevars
   and self.original_parse.__globals__.get("json")is json,"Exact original pure raw-to-JSON parser signature/library")
  self.expected_mutex=v4_module.mutex
  controls={_record(r)["path"] for r in current_control_refs}
  need(controls and len(controls)<=2048,"Explicit mandatory current-control closure required")
  known={}
  for manifest_ref in frozen_history_manifest_refs:
   _record(manifest_ref);controls.add(manifest_ref["path"])
   # Original strict parser, never a parsed-object authority cache.
   m=self.original_parse(_bytes(manifest_ref))
   need(type(m)is dict and type(m.get("targets"))is dict
    and type(m.get("runtime_authenticated_historical_source_and_metadata_refs"))is list,
    "Exact historical-ref-bearing sealed manifest")
   rows=list(m["runtime_authenticated_historical_source_and_metadata_refs"])
   rows.extend(t["storage_ref"]for t in m["targets"].values())
   for r in rows:
    _record(r)
    need(r["path"]not in known or known[r["path"]]==r["sha256"],"No historical-ref conflict")
    known[r["path"]]=r["sha256"]
  need(known and len(frozen_history_manifest_refs)<=16,"Explicit bounded frozen historical provenance")
  self.eligible={}
  for r in eligible_historical_refs:
   _record(r);p=Path(r["path"]);name=p.name.lower()
   need(p.suffix==".json" and p.parts[0]in("experiments","research")and ".runtime"not in p.parts
    and not any(w in name for w in FORBIDDEN_WORDS)and r["path"]not in controls
    and known.get(r["path"])==r["sha256"],"Only explicitly eligible sealed historical JSON, never current controls")
   need(r["path"]not in self.eligible,"No duplicate historical eligibility")
   self.eligible[r["path"]]=r["sha256"]
  need(0<len(self.eligible)<=MAX_ENTRIES,"Bounded nonempty explicit historical allowlist")
  self.controls=controls;self.guard_readers=[];self.guard_raw=[];modules=set()
  for module,record in guard_source_entries:
   _record(record);raw=_bytes(record)
   need(id(module)not in modules and getattr(module,"budget",None)is budget_module,"Exact common original budget singleton")
   modules.add(id(module))
   for name in("_POLICY_REF","_SELECTOR_REF","_REPAIR_AUTHORITY_REF"):
    r=getattr(module,name,None)
    if r is not None:controls.add(_record(r)["path"])
   raw_fn=_function(module,"_raw",record,raw)
   raw_node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)and n.name=="_raw")
   need(len(raw_node.body)==1 and ast.unparse(raw_node.body[0])=="return budget.v4.bound(ROOT, record)","Exact authenticated original raw reader")
   self.guard_raw.append((module,raw_fn))
   for name in("_json","_completed_json"):
    if hasattr(module,name):
     fn=_function(module,name,record,raw)
     # Shape is derived now. No module is patched by construction.
     derived=_derive_json_function(fn,record,raw,self.parse_record)
     self.guard_readers.append((module,name,fn,derived))
  need(self.guard_readers and not(set(self.eligible)&controls),"Current guard/control overlap blocks")
  self.active=None;self.installed=False;self.last_counts={}
  self.wrapper=self._mutex_wrapper()

 def _owned(self,record,raw):
  if self.active is None or type(record)is not dict or set(record)!={"path","sha256"}:return None
  relative=record.get("path")
  if self.eligible.get(relative)!=record.get("sha256")or relative in self.controls:return None
  state=self.handle._state()
  need(state is self.active["consumer_state"],"Same exact original lock transaction")
  item=self.handle._key(ROOT,relative)
  if item is None:return None
  entry=state["raw"].get(item[0])
  # Exact old memo tuple layout from pinned _make_file_bytes. Capacity bypass is original parse.
  if entry is None:return None
  need(len(entry)==4 and entry[0]==ROOT and str(Path(entry[1]))==relative
   and type(entry[3])is tuple and len(entry[3])==2 and entry[3][0]is raw
   and entry[3][1]==record["sha256"],"Eligible raw belongs to this exact original path/SHA transaction entry")
  return relative

 def _mark_copy_uncacheable(self,relative):
  # Retain only immutable bytes/SHA for the same final validation, never a parsed value.
  entry=self.active["parsed"].pop(relative)
  self.active["parsed_bytes"]-=entry["parsed_charge"]
  self.active["uncacheable"][relative]=dict(raw=entry["raw"],sha256=entry["sha256"])
  self.active["counts"]["deepcopy_recursion_fallbacks"]+=1

 def parse_record(self,record,raw):
  relative=self._owned(record,raw)
  if relative is None:
   if self.active is not None:self.active["counts"]["uncached_parses"]+=1
   return self.original_parse(raw)
  if relative in self.active["uncacheable"]:
   self.active["counts"]["uncached_parses"]+=1
   return self.original_parse(raw)
  cache=self.active["parsed"];prior=cache.get(relative)
  if prior is not None:
   need(prior["raw"]is raw and prior["sha256"]==record["sha256"],"No raw/SHA replacement in parsed memo")
   self.active["counts"]["parse_hits"]+=1
   try:return copy.deepcopy(prior["parsed"])
   except RecursionError:
    self._mark_copy_uncacheable(relative)
    return self.original_parse(raw)
  value=self.original_parse(raw);self.active["counts"]["parse_misses"]+=1
  charge=_charge(value)
  if len(cache)>=MAX_ENTRIES or self.active["raw_bytes"]+len(raw)>MAX_RAW_BYTES or self.active["parsed_bytes"]+charge>MAX_PARSED_BYTES:
   self.active["counts"]["capacity_bypasses"]+=1
   return value
  cache[relative]=dict(raw=raw,sha256=record["sha256"],parsed=value,parsed_charge=charge)
  self.active["raw_bytes"]+=len(raw);self.active["parsed_bytes"]+=charge
  try:return copy.deepcopy(value)
  except RecursionError:
   self._mark_copy_uncacheable(relative)
   return value

 def _bound(self,root,record,as_json=False):
  # Only the original parser operand differs, and only after original ref/SHA checks.
  if not as_json:return self.original_bound(root,record,as_json)
  scope=dict(self.original_bound.__globals__,parse=lambda raw:self.parse_record(record,raw))
  fn=FunctionType(self.original_bound.__code__,scope,self.original_bound.__name__,self.original_bound.__defaults__)
  return fn(root,record,as_json)

 def _final(self):
  # Additional original byte validation. Original memo snapshot still runs next under same mutex.
  for relative,entry in (*self.active["parsed"].items(),*self.active["uncacheable"].items()):
   need(self.handle.original["file_bytes"](ROOT,relative)==(entry["raw"],entry["sha256"]),"Parsed historical bytes changed before original boundary returned")
   self.active["counts"]["final_uncached_bytes"]+=1

 def _mutex_wrapper(self):
  @contextmanager
  def mutex(root):
   need(self.installed and self.active is None and threading.active_count()==1,"One fresh single-thread original consumer transaction")
   need(self.v4.bound is self.original_bound and self.v4.parse is self.original_parse,"Unchanged original byte/ref/parser seam")
   with self.expected_mutex(root):
    state=self.handle._state();need(state is not None,"Original mutex acquired and old raw cache active")
    self.active=dict(consumer_state=state,parsed={},uncacheable={},raw_bytes=0,parsed_bytes=0,
     counts=dict(parse_hits=0,parse_misses=0,uncached_parses=0,capacity_bypasses=0,final_uncached_bytes=0,deepcopy_recursion_fallbacks=0))
    patches=[]
    try:
     for module,raw_fn in self.guard_raw:need(module._raw is raw_fn and module.budget is self.budget,"Original common-budget raw reader unchanged")
     self.v4.bound=self._bound;patches.append((self.v4,"bound",self.original_bound,self.v4.bound))
     for module,name,original,derived in self.guard_readers:
      need(getattr(module,name)is original,"Exact original JSON reader before transaction")
      setattr(module,name,derived);patches.append((module,name,original,derived))
     yield
     self._final()
    finally:
     drift=False
     for owner,name,original,installed in reversed(patches):
      drift=drift or getattr(owner,name)is not installed
      setattr(owner,name,original)
     self.last_counts=dict(self.active["counts"]);self.active["parsed"].clear();self.active["uncacheable"].clear();self.active=None
     need(not drift,"Unexpected transaction JSON seam replacement")
    # Old memo performs its exact uncached namespace/leaf validation after this yield.
  return mutex

 def install(self):
  need(not self.installed and self.v4.mutex is self.expected_mutex and self.active is None,"Install once before future consumer work")
  self.v4.mutex=self.wrapper;self.installed=True;return self

 def uninstall(self):
  need(self.installed and self.active is None and self.v4.mutex is self.wrapper
   and self.v4.bound is self.original_bound and self.v4.parse is self.original_parse,"No live operation or replacement during uninstall")
  for module,name,original,_ in self.guard_readers:need(getattr(module,name)is original,"Original JSON functions restored")
  self.v4.mutex=self.expected_mutex;self.installed=False

if __name__=="__main__":
 print(json.dumps({"status":"INACTIVE_SOURCE_ONLY","guard_imported":False,"operations":0,"actual_consumer_adoption":False}))
