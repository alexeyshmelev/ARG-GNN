"""Inactive parse-only consumer candidate. Import is inert; no guard is loaded.

For a separately reviewed future consumer only, after exact current Relate64
configuration and the existing namespace/memo2 installation, before threads.
Every original byte/SHA/length/guard predicate still executes. This only reuses
pure JSON decoding of the fixed immutable refs when an existing transaction raw
entry already covers those exact bytes. No state/authority/journal/resource cache.
No current integration installs this component. ROOT mutations are unsupported.
"""
from contextlib import contextmanager
from pathlib import Path
import copy,sys,types
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
SOURCE='research/reviews/H058_consumer_guard_hotpath_source_audit_20261017_v1/immutable_parse_reuse.INACTIVE.v1.py'
AUTH_REF=dict(path='research/reviews/ROOT_mutation_transaction_memo_source_20261017_v1/root_mutation_memo_v1.py',sha256='2923df9f5a48e76da0b59e6394f31d0b8f901c214a76cc62d65969d010655d98')
MEMO_REF=dict(path='research/reviews/resource_guard_hotpath_source_audit_v1/transaction_memo_adapter_v2.py',sha256='c972e9a779d424ceafdb258848614bf40994d1004e2c73fc5cd225ae99621364')
V4_REF=dict(path='src/arggnn/project_budget_v4.py',sha256='a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47')
CACHE_FIELD='_H058_exact_immutable_json_parse_reuse'
MAX_PARSE_ENTRIES=128
MAX_PARSE_BYTES=16*2**20
READERS=[{'parse_readers': {'_completed_json': {'end': 63, 'line': 61, 'record_argument': 'record'}, '_json': {'end': 59, 'line': 57, 'record_argument': 'record'}}, 'policy_ref': {'path': 'research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1/activation_policy.root_active_v1.json', 'sha256': 'fdec761f8cae04022093299955655f8ffa27166829653817165126d7ada6a65c'}, 'source_ref': {'path': 'research/reviews/Relate_failed64GiB_retention_successor_source_20261017_v1/relate_failed64_retention_v1.py', 'sha256': 'b8b696a0df45c12b931e563600c2b52e89e73dd19c7d160351063687dcfb5164'}}, {'parse_readers': {'_completed_json': {'end': 61, 'line': 59, 'record_argument': 'record'}, '_json': {'end': 57, 'line': 55, 'record_argument': 'record'}}, 'policy_ref': {'path': 'research/reviews/native7_H050_stopped_retention_ROOT_activation_20261016_v1/activation_policy.root_active_v1.json', 'sha256': '99a347af4c18c242c21b6cb84df05a36959c478b622bfcd775261a0272cb9974'}, 'source_ref': {'path': 'research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/stopped_retention_v1.py', 'sha256': 'c9392ed88b76a5a6c1394d13ff78b55df83c1b16d8186d23b4d3110b46008e8f'}}, {'parse_readers': {'_completed_json': {'end': 58, 'line': 56, 'record_argument': 'record'}, '_json': {'end': 54, 'line': 52, 'record_argument': 'record'}}, 'policy_ref': {'path': 'research/reviews/native14_stopped_retention_ROOT_activation_20261016_v3/activation_policy.root_active_v1.json', 'sha256': '72ed727feb1a824fa8f9c2b31ce374dca6728ca4e655e9562b4dedc84122f417'}, 'source_ref': {'path': 'research/reviews/native14_stopped_retention_successor_source_20261016_v3/native14_stopped_retention_v1.py', 'sha256': '60d010e393bda25f662381051b3fd3d4e4decfa034b10c3baad3c3f30a6a196c'}}, {'parse_readers': {'_completed_json': {'end': 155, 'line': 153, 'record_argument': 'record'}, '_json': {'end': 151, 'line': 149, 'record_argument': 'record'}}, 'policy_ref': {'path': 'research/reviews/prospective980_Census_fit2_root_activation_20261016_v1/activation_policy.root_active_v1.json', 'sha256': '72ff346c43f96ba8430411cfe6b2250480a3aa66fa8e3b8c904a625a0399932a'}, 'source_ref': {'path': 'research/reviews/prospective980_Census_fit2_retention_source_20261016_v1/prospective_980gb_Census_fit2_v1.py', 'sha256': 'fab6bb67bc1f4be989df9bb16e7a3a17d24652ca8238617ca060a56cb8c89d43'}}, {'parse_readers': {'_completed_json': {'end': 58, 'line': 56, 'record_argument': 'record'}, '_json': {'end': 55, 'line': 53, 'record_argument': 'record'}}, 'policy_ref': {'path': 'research/reviews/prospective980_H020_root_activation_20261016_v3/activation_policy.root_active_v1.json', 'sha256': '7360bd28f57c8eaef19336ccd77eba193afae554e98d2ef0726e8ec879d7e582'}, 'source_ref': {'path': 'research/reviews/prospective980_H020_legacy_source_20261016_v3/prospective_980gb_H020_v3.py', 'sha256': 'ec475ecf59f0335397ee8b310cd0cb406ce4e03161e7c791861c6583799e6635'}}, {'parse_readers': {'_completed_json': {'end': 50, 'line': 48, 'record_argument': 'record'}, '_json': {'end': 47, 'line': 45, 'record_argument': 'record'}}, 'policy_ref': {'path': 'research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json', 'sha256': 'eb3408004a508136aa434c0a2e02a9d14a5598a96b8483c264c93026f9326018'}, 'source_ref': {'path': 'research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py', 'sha256': 'fc71030865e0a3d5da31120d4daa2caa0b2d39ec7dd212b8c46c476b5a866d56'}}, {'parse_readers': {'_completed_json': {'end': 45, 'line': 43, 'record_argument': 'record'}, '_json': {'end': 42, 'line': 40, 'record_argument': 'record'}}, 'policy_ref': {'path': 'research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json', 'sha256': 'a0885b5209172dd79a835d442e0ad3495f5c1a0eac61a1c5c08154ec877ecf85'}, 'source_ref': {'path': 'research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py', 'sha256': 'ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef'}}]
ALLOW_REFS=[{'path': 'research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1/ROOT_activation.root_active_v1.json', 'sha256': '68d282d8aba8ad08436f60895f7e80ce5079ce6363cd141925fc0233652ae802'}, {'path': 'research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1/activation_policy.root_active_v1.json', 'sha256': 'fdec761f8cae04022093299955655f8ffa27166829653817165126d7ada6a65c'}, {'path': 'research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1/explicit_child_selector.root_active_v1.json', 'sha256': 'e27d47a22d0da1b2e8f0bdf0818cb2c6c473d8c9ddb5424618bf97e704928642'}, {'path': 'research/reviews/Relate_failed64GiB_retention_ROOT_activation_20261017_v1/parent_mutation_revocation.root_active_v1.json', 'sha256': '3a9be3b0786df2ba6489f809bb0aee58cdaa8f20cdf9f793885cc9fdc20bfe6d'}, {'path': 'research/reviews/Relate_failed64GiB_retention_successor_independent_review_20261017_v1/review.v1.json', 'sha256': '1705946c29ef199fe7e359ca33656f1604dd5091262405b7fa022c82a2c15ddb'}, {'path': 'research/reviews/Relate_failed64GiB_retention_successor_source_20261017_v1/source_manifest.v1.json', 'sha256': '8c9339104cff95bbada5f8a36c41908ede6b2eccb6aef53640a913de14ad2774'}, {'path': 'research/reviews/completed_scope_credits_root_activation_20261015_v1/activation_policy.json', 'sha256': 'a0885b5209172dd79a835d442e0ad3495f5c1a0eac61a1c5c08154ec877ecf85'}, {'path': 'research/reviews/native14_stopped_retention_ROOT_activation_20261016_v3/activation_policy.root_active_v1.json', 'sha256': '72ed727feb1a824fa8f9c2b31ce374dca6728ca4e655e9562b4dedc84122f417'}, {'path': 'research/reviews/native14_stopped_retention_ROOT_activation_20261016_v3/frozen_parent_activation.root_active_v1.json', 'sha256': '675c7a6d44ba98dc7f473ecd3f99a9008da9e8f5c34ce4ef97d655f31cb9448d'}, {'path': 'research/reviews/native14_stopped_retention_ROOT_activation_20261016_v3/permanent_revocation.root_active_v1.json', 'sha256': '219771261fc438f62e21e28cc1a2f8c735e5f64d9bf17bb6b45a961f4b97b53c'}, {'path': 'research/reviews/native14_stopped_retention_independent_ROOT_review_20261016_v3/review.root_v1.json', 'sha256': '64f9fbf997ec15e558b0bf46c4aa29e28d41132db800ef133a93ea2b7dd28b29'}, {'path': 'research/reviews/native14_stopped_retention_successor_source_20261016_v3/source_manifest.v1.json', 'sha256': 'c2bae5575ae0e8e68e152b0edde0b7b2d53e4fc7398a3a8501ce278e23a84fa6'}, {'path': 'research/reviews/native7_H050_stopped_retention_ROOT_activation_20261016_v1/ROOT_activation.root_active_v1.json', 'sha256': '65a29c7a0e642b93f27f670a5794fa07b282849e9e222434977262c74d9358a2'}, {'path': 'research/reviews/native7_H050_stopped_retention_ROOT_activation_20261016_v1/activation_policy.root_active_v1.json', 'sha256': '99a347af4c18c242c21b6cb84df05a36959c478b622bfcd775261a0272cb9974'}, {'path': 'research/reviews/native7_H050_stopped_retention_ROOT_activation_20261016_v1/explicit_child_selector.root_active_v1.json', 'sha256': '5bcee3d8b53222ee45119dc51e1a35850e7e04c20d84c702080d5685d77b5120'}, {'path': 'research/reviews/native7_H050_stopped_retention_ROOT_activation_20261016_v1/permanent_parent_mutation_revocation.root_v1.json', 'sha256': 'cff542d03df547cf0b66869ab41b1f331f8ddcf126d89cd5ae24bb49c544f9f4'}, {'path': 'research/reviews/native7_H050_stopped_retention_independent_review_20261016_v1/review.v1.json', 'sha256': 'da3fc5ba5ace3f06c465d7f4788960d5cc748d902d84bf5b3832c2afb4645579'}, {'path': 'research/reviews/native7_H050_stopped_retention_successor_source_20261016_v1/source_manifest.v1.json', 'sha256': 'cb6c3af2858d007d74edb08742f8633ddef5f2a9cb4c6cd5b3682205fee32d88'}, {'path': 'research/reviews/prospective980_Census_fit2_root_activation_20261016_v1/activation_policy.root_active_v1.json', 'sha256': '72ff346c43f96ba8430411cfe6b2250480a3aa66fa8e3b8c904a625a0399932a'}, {'path': 'research/reviews/prospective980_Census_fit2_root_activation_20261016_v1/frozen_parent_activation.root_active_v1.json', 'sha256': '5ff21b0772b507356512a39e661812d0c8eecf6ada284f5047147aff5db99943'}, {'path': 'research/reviews/prospective980_H020_root_activation_20261016_v3/activation_policy.root_active_v1.json', 'sha256': '7360bd28f57c8eaef19336ccd77eba193afae554e98d2ef0726e8ec879d7e582'}, {'path': 'research/reviews/prospective980_H020_root_activation_20261016_v3/frozen_parent_activation.root_active_v1.json', 'sha256': 'e6124d48789510d30012818354b42d0c427d073ca447b99dfe585c58a0a6cd66'}, {'path': 'research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json', 'sha256': 'eb3408004a508136aa434c0a2e02a9d14a5598a96b8483c264c93026f9326018'}]

def need(ok,message):
 if not ok:raise ValueError(message)

def retained_charge(value):
 """Conservative retained Python JSON-tree/key overhead, never a RSS promise."""
 pending=[value];total=0
 while pending:
  item=pending.pop();need(type(item)in(dict,list,str,int,float,bool,type(None)),'Original JSON builtins only')
  total+=sys.getsizeof(item)
  if type(item)is dict:
   pending.extend(item.keys());pending.extend(item.values())
  elif type(item)is list:pending.extend(item)
  if total>MAX_PARSE_BYTES:return total
 return total

class ImmutableParseReuse:
 """Finite extension to the already installed read-only memo; never install live."""
 def __init__(self,guard,namespace_memo,authenticated_base,*,consumer_role):
  need(consumer_role=='read_only_consumer','No ROOT mutation or other role')
  auth=authenticated_base
  # The future integration must load this existing helper through its exact
  # SHA-pinned source closure, as H058 already does. Do not construct RootMutationMemo.
  auth.functions_from_source(auth,AUTH_REF,('source_bytes','functions_from_source'))
  self.v4=guard.budget.v4;self.memo=namespace_memo;self.guard=guard
  need(getattr(guard,'_outer_memo',None)is namespace_memo and namespace_memo.module is self.v4,
   'Existing exact selected namespace memo handle required')
  receipt=namespace_memo.receipt
  need(receipt.get('installed')is True and receipt.get('consumer_role')=='read_only_consumer'
   and receipt.get('namespace_extension_schema')=='resource_guard_consumer_namespace_memo_v1'
   and receipt.get('original_memo2_ref')==MEMO_REF and receipt.get('cross_operation_cache')is False,
   'Already installed original namespace/memo2 consumer; no duplicate adoption')
  need(namespace_memo.active_count==0,'Install before consumer operations/threads')
  # Verify the exact frozen memo class methods; no memo source module is executed here.
  data=auth.source_bytes(MEMO_REF);compiled=compile(data,str(ROOT/MEMO_REF['path']),'exec',dont_inherit=True,optimize=0)
  cls=next(c for c in compiled.co_consts if isinstance(c,types.CodeType)and c.co_name=='TransactionMemoAdapter')
  self.memo_methods={}
  for name in ('_state','_key','_room','_make_mutex','_make_canonical','_make_file_bytes','_validate_snapshot','install','uninstall'):
   expected=[c for c in cls.co_consts if isinstance(c,types.CodeType)and c.co_name==name]
   actual=getattr(type(namespace_memo),name)
   need(len(expected)==1 and actual.__code__==expected[0],'Exact original memo method '+name)
   self.memo_methods[name]=(actual,actual.__code__)
  # Do not trust receipt fields alone: verify each installed wrapper's exact
  # source-derived code AND its original-function/self closure relationships.
  original_compiled=compile(auth.source_bytes(V4_REF),str(ROOT/V4_REF['path']),'exec',dont_inherit=True,optimize=0)
  for name in ('canonical','file_bytes','mutex'):
   original=namespace_memo.original[name]
   original_body=getattr(original,'__wrapped__',original)
   codes=[c for c in original_compiled.co_consts if isinstance(c,types.CodeType)and c.co_name==name]
   need(len(codes)==1 and original_body.__code__==codes[0]
    and original_body.__globals__ is self.v4.__dict__ and original.__module__==self.v4.__name__,
    'Exact original v4 callable behind memo '+name)
   maker=getattr(type(namespace_memo),'_make_'+name)
   nested=[c for c in maker.__code__.co_consts if isinstance(c,types.CodeType)and c.co_name==name]
   wrapper=getattr(namespace_memo,name+'_wrapper');body=getattr(wrapper,'__wrapped__',wrapper)
   closure=dict(zip(body.__code__.co_freevars,[c.cell_contents for c in(body.__closure__ or())]))
   need(len(nested)==1 and body.__code__==nested[0]and body.__globals__ is maker.__globals__
    and closure.get('self')is namespace_memo and closure.get('original')is original,
    'Exact existing memo wrapper and captured original '+name)
  checked=auth.functions_from_source(self.v4,V4_REF,('parse','bound'))
  self.old_parse=checked['parse'][0];self.old_mutex=self.v4.mutex
  need(self.old_mutex is namespace_memo.mutex_wrapper and self.v4.canonical is namespace_memo.canonical_wrapper
   and self.v4.file_bytes is namespace_memo.file_bytes_wrapper,'One existing installed raw/namespace memo only')
  self.callers={};self.reader_functions=[];current=guard
  for row in READERS:
   names=tuple(row['parse_readers']);verified=auth.functions_from_source(current,row['source_ref'],names)
   for name,(fn,node)in verified.items():
    need(node.args.args[0].arg=='record','Exact SHA-bound record reader shape')
    self.callers[fn.__code__]='record';self.reader_functions.append((current,name,fn,fn.__code__))
   current=current._BASE
  self.allow={(r['path'],r['sha256'])for r in ALLOW_REFS}
  need(len(self.allow)==len(ALLOW_REFS)==23,'Only exact finite current immutable metadata refs')
  self.installed=False;self.parse_wrapper=self._make_parse();self.mutex_wrapper=self._make_mutex()

 def _check(self):
  need(self.v4.canonical is self.memo.canonical_wrapper and self.v4.file_bytes is self.memo.file_bytes_wrapper,
   'Existing raw/namespace memo must remain exact')
  need(self.v4.parse is(self.parse_wrapper if self.installed else self.old_parse)
   and self.v4.mutex is(self.mutex_wrapper if self.installed else self.old_mutex),'Exact parse/mutex extension lifetime')
  for name,(fn,code)in self.memo_methods.items():
   need(getattr(type(self.memo),name)is fn and fn.__code__ is code,'Original memo method drift')
  for module,name,fn,code in self.reader_functions:
   need(getattr(module,name)is fn and fn.__code__ is code and fn.__globals__ is module.__dict__,
    'Original SHA-bound reader callable drift')

 def _parse(self,raw,frame):
  state=self.memo._state();field=self.callers.get(frame.f_code)
  if state is None or field is None or CACHE_FIELD not in state:return self.old_parse(raw)
  record=frame.f_locals.get(field)
  if type(record)is not dict or set(record)!={'path','sha256'}:return self.old_parse(raw)
  identity=(record['path'],record['sha256'])
  if identity not in self.allow:return self.old_parse(raw)
  item=self.memo._key(ROOT,record['path'])
  if item is None:return self.old_parse(raw)
  entry=state['raw'].get(item[0])
  # Hits require the exact already-validated raw object AND digest already covered
  # by memo2's unchanged final no-follow namespace/leaf identity revalidation.
  if entry is None or entry[3][0]is not raw or entry[3][1]!=record['sha256']:
   return self.old_parse(raw)
  cache=state[CACHE_FIELD]
  if identity in cache['values']:
   state['immutable_json_parse_hits']+=1
   return copy.deepcopy(cache['values'][identity])
  result=self.old_parse(raw);state['immutable_json_parse_misses']+=1
  charge=retained_charge(result)+len(raw)+len(record['path'].encode())+len(record['sha256'])+512
  if len(cache['values'])<MAX_PARSE_ENTRIES and cache['bytes']+charge<=MAX_PARSE_BYTES and self.memo._room(state,charge):
   cache['values'][identity]=copy.deepcopy(result);cache['bytes']+=charge
   state['entries']+=1;state['charge']+=charge
  else:state['immutable_json_parse_bypasses']+=1
  return result

 def _make_parse(self):
  def parse(raw):return self._parse(raw,sys._getframe(1))
  return parse

 def _make_mutex(self):
  @contextmanager
  def mutex(root):
   self._check()
   # The existing wrapper acquires the actual original mutex first, creates its
   # fresh transaction state, validates its snapshot last, and unlocks unchanged.
   with self.old_mutex(root):
    state=self.memo._state();need(state is not None and CACHE_FIELD not in state,'One fresh existing consumer transaction')
    state[CACHE_FIELD]=dict(values={},bytes=0)
    state.update(immutable_json_parse_hits=0,immutable_json_parse_misses=0,immutable_json_parse_bypasses=0)
    try:
     yield
     self._check()
    finally:
     cache=state.pop(CACHE_FIELD);cache['values'].clear();cache['bytes']=0
     # This runs on exceptions too, before the original snapshot validation and
     # before the actual original mutex unlock. No cache survives an operation.
  return mutex

 def install(self):
  need(not self.installed and self.memo.active_count==0,'One explicit pre-thread installation only')
  self._check();self.v4.parse=self.parse_wrapper;self.v4.mutex=self.mutex_wrapper;self.installed=True
  return self

 def uninstall(self):
  need(self.installed and self.memo.active_count==0,'No live operation during uninstall')
  self._check();self.v4.parse=self.old_parse;self.v4.mutex=self.old_mutex;self.installed=False
  # Existing namespace_memo.uninstall() may run only after this restores its
  # exact original mutex wrapper. This method leaves that existing memo installed.

if __name__=='__main__':print('INACTIVE_SOURCE_ONLY_NO_GUARD_OR_INSTALLATION')
