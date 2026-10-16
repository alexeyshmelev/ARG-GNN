"""Exact public-metadata prefix and command-parser checks; no preparation branch."""
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout
import ast,copy,hashlib,importlib,io,json,sys,types
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');OUT=ROOT/'research/reviews/native_VAL000_full_state_resume_independent_review_20261016_v10'
PREP='research/reviews/native_v10_ROOT_preparation_script_source_20261016_v1'
SRC='research/reviews/native_VAL000_full_state_resume_source_20261016_v10'
def ref(p):
 p=ROOT/p if isinstance(p,str) else p
 return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
handoff=ROOT/PREP/'source_handoff.v1.json'
assert ref(handoff)['sha256']=='d10304469a6f0beba42ef9438bff8022ddc6ec7aaeabfe373582f4e9752d50ed'
h=json.loads(handoff.read_text())
refs=[]
def walk(v):
 if isinstance(v,dict):
  if set(v)=={'path','sha256'}:refs.append(v)
  else:
   for item in v.values():walk(item)
 elif isinstance(v,list):
  for item in v:walk(item)
walk(h)
for r in refs:assert ref(r['path'])==r
p=ROOT/PREP/'prepare_root_v1.py';source=p.read_text();tree=ast.parse(source)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare_root_v1')
# Precisely stop the real public metadata prefix before the first original
# validate_input/runtime/census/guard or write operation could occur.
index=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Tuple) and any(isinstance(x,ast.Name) and x.id=='original_spec' for x in t.elts) for t in n.targets))
prefix=copy.deepcopy(fn);prefix.body=prefix.body[:index]+[ast.Return(value=ast.Constant(value='STOP_BEFORE_ANY_RUNTIME_GUARD_OR_PREPARATION'))]
prefix.name='metadata_only_prefix';ast.fix_missing_locations(prefix)
module=importlib.import_module(PREP.replace('/','.')+'.prepare_root_v1')
c=importlib.import_module(SRC.replace('/','.')+'.amendment_contract_v1')
ns=dict(module.__dict__);exec(compile(ast.Module(body=[prefix],type_ignores=[]),str(p)+':public-metadata-prefix-only','exec'),ns)
review=ref(OUT/'review.v10.json')
args=types.SimpleNamespace(root_directive=module.DIRECTIVE,review_path=review['path'],review_sha256=review['sha256'],additional_known_source_ref=[])
assert ns['metadata_only_prefix'](args)=='STOP_BEFORE_ANY_RUNTIME_GUARD_OR_PREPARATION'
checks=[dict(name='exact_handoff_packet_template_history_and_actual_v10_review_prefix',status='PASS_METADATA_ONLY')]
for name,updates in [('wrong_directive',dict(root_directive='WRONG')),
 ('wrong_review_digest',dict(review_sha256='0'*64)),
 ('wrong_v9_review',dict(review_path='research/reviews/native_VAL000_full_state_resume_independent_review_20261016_v9/review.v9.json',
 review_sha256='0635aa2bfe723a1839487d4ad70178fd9358b823cf0f39c8287b4e83fe7c430c'))]:
 bad=copy.copy(args);bad.__dict__.update(updates)
 try:ns['metadata_only_prefix'](bad)
 except (AssertionError,ValueError):pass
 else:raise AssertionError(name)
 checks.append(dict(name=name,status='PASS_REJECTED_BEFORE_RUNTIME_GUARD_OR_WRITE'))
# Nested prewrite function is bytecode-identical to corrected v9 preparation.
old=importlib.import_module('research.reviews.native_v9_ROOT_preparation_script_source_20261016_v2.prepare_root_v1')
get=lambda f:next(k for k in f.__code__.co_consts if isinstance(k,types.CodeType) and k.co_name=='save')
a,b=get(old.prepare_root_v1),get(module.prepare_root_v1)
assert a.co_code==b.co_code and a.co_consts==b.co_consts and a.co_names==b.co_names and a.co_freevars==b.co_freevars
checks.append(dict(name='corrected_v9_nested_prewrite_save_bytecode_exact',status='PASS_SOURCE_CHECK'))
# Exact returned command assignment, parsed through actual executor main with
# fake set-amendment/execute endpoints. No command/process is launched.
commandnode=next(n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='execution_command' for t in n.targets))
marker=dict(path='SYNTHETIC_ACTIVE_V10',sha256='0'*64)
ns=dict(module.__dict__,c=c,amendment_ref=marker)
exec(compile(ast.Module(body=[copy.deepcopy(commandnode)],type_ignores=[]),str(p)+':command-construction-only','exec'),ns)
command=ns['execution_command']
assert command[:5]==[str(ROOT/'.runtime/env/bin/python'),'-B','-s','-m',SRC.replace('/','.')+'.execution_only_v1']
e=importlib.import_module(SRC.replace('/','.')+'.execution_only_v1')
events=[];before=list(sys.path)
def fake_execute(r):events.append(('fake_execute',r));return dict(returncode=37)
try:
 with patch.object(e,'set_amendment_v1',lambda r:events.append(('fake_set_amendment',r))),patch.object(e,'execute_v1',fake_execute),redirect_stdout(io.StringIO()):
  rc=e.main(command[5:])
finally:sys.path[:]=before
assert rc==37 and events==[('fake_set_amendment',marker),('fake_execute',c.ORIGINAL_INPUT)]
checks.append(dict(name='exact_emitted_execution_argv_real_parser_with_fake_execution',status='PASS_SOFTWARE_ONLY'))
# Exact deadline tick AST retains strict/anchor/expiry and margin.
oldtree=ast.parse((ROOT/'research/reviews/native_v9_ROOT_preparation_script_source_20261016_v2/prepare_root_v1.py').read_text())
oldfn=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare_root_v1')
gettick=lambda f:next(n for n in f.body if isinstance(n,ast.FunctionDef) and n.name=='tick')
assert ast.dump(gettick(oldfn),include_attributes=False)==ast.dump(gettick(fn),include_attributes=False)
checks.append(dict(name='original_deadline_tick_AST_exact',status='PASS_SOURCE_CHECK'))
assert not any(n in sys.modules for n in ('torch','numpy','composer','torch_geometric','tskit','tsinfer'))
value=dict(schema='native_v10_ROOT_prep_independent_metadata_and_route_checks',status='PASS_SOFTWARE_ONLY',
 checks=checks,declared_prep_refs_verified=len(refs),actual_command=command,
 preparation_runtime_guard_capacity_probe_accounting_activation_or_file_writes_never_executed=True,
 exact_public_metadata_prefix_stopped_before_original_validate_input=True)
(OUT/'ROOT_prep_metadata_checks.v10.json').write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(status=value['status'],checks=len(checks),refs=len(refs))))
