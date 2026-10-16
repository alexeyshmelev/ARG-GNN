"""Independent bounded v2 ROOT-prep real-interface regressions; fake endpoints only."""
from pathlib import Path
from unittest.mock import patch
import ast,copy,hashlib,importlib,json,subprocess,sys,types
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OUT=ROOT/'research/reviews/native_VAL000_full_state_resume_independent_review_20261016_v9'
OLD='research/reviews/native_v9_ROOT_preparation_script_source_20261016_v1/prepare_root_v1.py'
NEW='research/reviews/native_v9_ROOT_preparation_script_source_20261016_v2/prepare_root_v1.py'
assert hashlib.sha256((ROOT/NEW).read_bytes()).hexdigest()=='0ca842e8c2170bf45bf8554054dba1dd95e1687799222647f54bc977bc9c09b8'
old=(ROOT/OLD).read_text();source=(ROOT/NEW).read_text()
expected=old.replace('g.validate_storage_v9(original_spec,cfg,storage,source_ref=', 'g.validate_storage_v9(original_spec,dict(output_cap_bytes=c.CAPS["reservation_bytes"]),storage,source_ref=').replace('        g.budget_check_v9(storage,boundary=True,required_write_bytes=len(raw)+8192)','        c.original.charged_census_v1(original_spec,required_write_bytes=len(raw)+8192)\n        g.budget_check_v9(storage,boundary=True,required_write_bytes=len(raw)+8192)')
assert source==expected,'Unexpected source delta'
prep=importlib.import_module('research.reviews.native_v9_ROOT_preparation_script_source_20261016_v2.prepare_root_v1')
c=importlib.import_module('research.reviews.native_VAL000_full_state_resume_source_20261016_v9.amendment_contract_v1')
checks=[]
spec=c.read(c.ORIGINAL_INPUT);storage=c.read(spec['refs']['storage'])
def require(value,message):
 if not value:raise ValueError(message)
# Exact actual guard function with faked I/O boundary and actual public binding metadata.
gpath=ROOT/c.GUARD['path'];tree=ast.parse(gpath.read_text())
validate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_storage_v9')
events=[];ns=dict(require=require,_raw=lambda r:events.append('source_binding'),budget_check_v9=lambda *a,**k:events.append(('boundary',k)) or 'FAKE_BOUNDARY_OK')
exec(compile(ast.Module(body=[validate],type_ignores=[]),str(gpath)+':inert_extracted_AST','exec'),ns)
args=dict(source_ref=c.original.ref(c.original.DRIVER),config_ref=spec['refs']['config'],runtime_ref=spec['refs']['runtime'])
assert ns['validate_storage_v9'](spec,dict(output_cap_bytes=c.CAPS['reservation_bytes']),storage,**args)=='FAKE_BOUNDARY_OK'
assert events==['source_binding']*3+[('boundary',dict(first=True,boundary=True))]
checks.append(dict(name='fixed_config_view_satisfies_exact_actual_guard_interface',status='PASS'))
# Exact actual original census function code, substituting sizes/local only.
nested=next(x for x in prep.prepare_root_v1.__code__.co_consts if isinstance(x,types.CodeType) and x.co_name=='save')
def cell(value):
 def inner():return value
 return inner.__closure__[0]
def make_save(auxbytes,outputbytes=100,checkpointbytes=None,guard_failure=False):
 seen=[];written=[];counts=[auxbytes]
 def sizes(path,**kw):
  seen.append('census_output' if str(path)==spec['output_directory'] else 'census_AUX')
  if str(path)==spec['output_directory']:
   return [(('checkpoints','SYNTHETIC.pt'),checkpointbytes+4096,'regular')] if checkpointbytes else [(('SYNTHETIC_META',),outputbytes,'regular')]
  return [(('SYNTHETIC_AUX',),counts[0],'regular')]
 nc=dict(c.original.charged_census_v1.__globals__,_sizes=sizes,local=lambda p:Path(p))
 census=types.FunctionType(c.original.charged_census_v1.__code__,nc)
 def save(aux,name,value):
  seen.append('fake_save');written.append(name);counts[0]+=len((json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())+4096
  return dict(path='SYNTHETIC/'+name,sha256='0'*64)
 def boundary(*a,**k):
  seen.append('fake_strict_boundary')
  assert k['boundary'] is True and k['required_write_bytes']==len((json.dumps(dict(SYNTHETIC=True),sort_keys=True,indent=2,allow_nan=False)+'\n').encode())+8192
  if guard_failure:raise ValueError('SYNTHETIC_BOUNDARY_FAILURE')
 cc=types.SimpleNamespace(original=types.SimpleNamespace(charged_census_v1=census,save=save))
 gg=types.SimpleNamespace(budget_check_v9=boundary)
 vals=dict(aux=Path('SYNTHETIC_AUX'),c=cc,g=gg,original_spec=spec,require=require,storage=storage,tick=lambda:seen.append('tick'))
 fakeos=types.SimpleNamespace(open=lambda *a:seen.append('fake_open_dir') or 88,fsync=lambda fd:seen.append('fake_dir_fsync'),close=lambda fd:None,O_RDONLY=0,O_DIRECTORY=0)
 nsave=types.FunctionType(nested,dict(prep.prepare_root_v1.__globals__,os=fakeos),closure=tuple(cell(vals[n]) for n in nested.co_freevars))
 return nsave,seen,written
sample=dict(SYNTHETIC=True);planned=len((json.dumps(sample,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())+8192
save,seen,written=make_save(c.CAPS['auxiliary_bytes']-planned)
save('synthetic_0.json',sample)
assert seen[:5]==['tick','census_output','census_AUX','fake_strict_boundary','tick'] and written==['synthetic_0.json']
checks.append(dict(name='exact_AUX_planned_growth_boundary_accepts_and_precedes_strict_guard_write',status='PASS'))
for index in (1,2,3):
 save,seen,written=make_save(c.CAPS['auxiliary_bytes']-planned+1)
 try:save('synthetic_'+str(index)+'.json',sample)
 except ValueError:pass
 else:raise AssertionError('AUXoverflow accepted')
 assert not written and 'fake_strict_boundary' not in seen
 assert c.CAPS['auxiliary_bytes']-planned+1+100+planned<c.CAPS['reservation_bytes']
 checks.append(dict(name='AUX_overflow_blocks_before_metadata_write_'+str(index),status='PASS_REJECTED'))
for name,kwargs in [('scientific_output_overflow',dict(auxbytes=100,outputbytes=c.CAPS['scientific_output_bytes']-planned+1)),
 ('checkpoint_original_file_cap',dict(auxbytes=100,checkpointbytes=3*2**30+1)),
 ('strict_original_boundary_failure',dict(auxbytes=100,guard_failure=True))]:
 save,seen,written=make_save(**kwargs)
 try:save('synthetic.json',sample)
 except ValueError:pass
 else:raise AssertionError(name)
 assert not written
 checks.append(dict(name=name,status='PASS_REJECTED'))
save,seen,written=make_save(100)
for i in range(3):save('synthetic_'+str(i)+'.json',sample)
assert len(written)==3 and seen.count('census_AUX')==3 and seen.count('fake_strict_boundary')==3
checks.append(dict(name='all_three_metadata_writes_each_use_census_then_strict_guard_then_write',status='PASS'))
cli=[]
for args in ([],['--help'],['--prepare-root-v9'],['--prepare-root-v9','--review-path','SYNTHETIC','--review-sha256','0'*64,'--root-directive','WRONG']):
 r=subprocess.run([str(ROOT/'.runtime/env/bin/python'),'-B','-s','-m','research.reviews.native_v9_ROOT_preparation_script_source_20261016_v2.prepare_root_v1',*args],cwd=ROOT,text=True,capture_output=True,timeout=30)
 assert r.returncode==(0 if not args or args==['--help'] else 2)
 if not args:assert json.loads(r.stdout)==dict(status='SOURCE_ONLY_INACTIVE',authority_written=False,execution=False)
 cli.append(dict(args=args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
checks.append(dict(name='actual_pinned_inert_default_help_and_invalid_directive_CLI',status='PASS'))
assert not any(n in sys.modules for n in ('torch','numpy','composer','torch_geometric','tskit','tsinfer'))
result=dict(schema='native_v9_ROOT_prep_v2_independent_bounded_regression',status='PASS_SOFTWARE_ONLY',
 exact_source_delta_only_two_reviewed_fixes=True,checks=checks,inert_CLI=cli,
 actual_nested_save_code_object_used=True,actual_original_census_code_object_with_synthetic_sizes_only=True,
 exact_guard_interface_AST_with_fake_boundary_only=True,active_preparation_real_guard_resource_probe_or_authority_writes=False)
(OUT/'ROOT_prep_v2_independent_checks.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],checks=len(checks))))
