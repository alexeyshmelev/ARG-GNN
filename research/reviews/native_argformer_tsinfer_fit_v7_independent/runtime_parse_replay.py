"""Guard counterexamples and genuine full retrieval on handmade fixtures only."""
from __future__ import annotations
import ast
import copy
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import types
from unittest.mock import patch
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');REVIEW=ROOT/'research/reviews/native_argformer_tsinfer_fit_v7_independent'
sys.dont_write_bytecode=True;os.environ['CUDA_VISIBLE_DEVICES']=''
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
m=load('_author_guard_v3',ROOT/'src/arggnn/native_argformer_tsinfer_fit_v7.py');c=m.policy();checks=[];details={};sha=m.sha
def check(name,value):
 if not value:raise AssertionError(name)
 checks.append(name)
def reject(name,call):
 try:call()
 except (ValueError,RuntimeError,KeyError,TypeError,FileNotFoundError):checks.append(name);return
 raise AssertionError('unexpected acceptance: '+name)
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n');return m.reference(path)
with tempfile.TemporaryDirectory(prefix='guards_',dir=REVIEW) as temp:
 directory=Path(temp);fake=directory/'runtime_fixture';prefix=fake/c['runtime']['prefix'];site=prefix/'lib/python3.11/site-packages';site.mkdir(parents=True)
 binary=prefix/'bin/python3.11';binary.parent.mkdir();binary.write_bytes(b'handmade executable identity, never executed');binary.chmod(0o755)
 alias=prefix/'bin/python';alias.symlink_to(binary.name)
 distributions={m.distribution_name(k):v for k,v in c['runtime_inventory_contract']['expected_scientific_distributions'].items()}
 modules={};objects={}
 for name,dist in c['runtime_inventory_contract']['scientific_modules'].items():
  p=site/(name+'.py');p.write_text('# handmade package identity only\n')
  obj=types.ModuleType(name);obj.__file__=str(p);obj.__version__=distributions[m.distribution_name(dist)];obj.__spec__=types.SimpleNamespace(loader=importlib.machinery.SourceFileLoader(name,str(p)));objects[name]=obj
  modules[name]={'distribution':m.distribution_name(dist),'distribution_version':distributions[m.distribution_name(dist)],'module_version':obj.__version__,'origin':str(p),'sha256':m.sha(p),'loader':'_frozen_importlib_external.SourceFileLoader'}
 with patch.object(m,'ROOT',fake):
  fc=copy.deepcopy(c);fc['runtime_inventory_contract']['expected_additive_helpers']=[];fc['runtime_inventory_contract']['helper_provenance']=None
  for field in ('directory_alias_layout','absent_executable_alias_layout'):
   fc['runtime_inventory_contract'][field]['provenance']=None;fc['runtime_inventory_contract'][field]['aliases']=[]
  runtime={'python_executable_sha256':m.sha(alias),'additive_helpers':[]}
  inventory={'kind':c['runtime_inventory_contract']['kind'],'status':c['runtime_inventory_contract']['status'],'prefix':str(prefix),
   'python':{'version':[3,11,9],'executable':str(alias),'resolved_executable':str(binary),'sha256':m.sha(alias)},'distributions':distributions,'scientific_modules':modules,'import_file_sha256':m.native_import_bytes(prefix),'additive_helpers':[],'directory_symlinks':{},'absent_executable_symlinks':{}}
  iref=write(fake/'inventory.json',inventory);runtime['environment_inventory_receipt']=iref
  check('strict_complete_handmade_inventory_parses',m.parse_runtime_inventory(fc,inventory,runtime)==inventory)
  aliased_contract=copy.deepcopy(fc)
  fixture_dir_shape={'path':'lib/python3.1','target':'python3.11','resolved_path':'lib/python3.11','target_is_directory':True,'lstat_mode':41471,'lstat_size':10,'lstat_device':67,'lstat_inode':12345}
  fixture_absent_shape={'path':'lib/absent.so','target':'absent_target.so','resolved_path':'lib/absent_target.so','target_absent':True,'lstat_mode':41471,'lstat_size':16,'lstat_device':67,'lstat_inode':23456}
  aliased_contract['runtime_inventory_contract']['directory_alias_layout']['aliases']=[fixture_dir_shape]
  aliased_contract['runtime_inventory_contract']['absent_executable_alias_layout']['aliases']=[fixture_absent_shape]
  aliased_inventory=copy.deepcopy(inventory);aliased_inventory['directory_symlinks']=m.native_alias_records([fixture_dir_shape],True);aliased_inventory['absent_executable_symlinks']=m.native_alias_records([fixture_absent_shape],False)
  check('parser_separately_binds_alias_layout_without_invented_bytes',m.parse_runtime_inventory(aliased_contract,aliased_inventory,runtime)==aliased_inventory)
  for relative in ('lib/absent.so','lib/python3.1/site-packages/invented.py'):
   bad=copy.deepcopy(aliased_inventory);bad['import_file_sha256'][relative]={'resolved_path':'lib/absent_target.so','sha256':'1'*64}
   reject('parser_rejects_absent_or_duplicate_alias_bytes_'+relative,lambda bad=bad:m.parse_runtime_inventory(aliased_contract,bad,runtime))

  for layout_name in ('directory_symlinks','absent_executable_symlinks'):
   bad=copy.deepcopy(inventory);bad[layout_name]={'unbound':{}}
   reject('inventory_unbound_'+layout_name,lambda bad=bad:m.parse_runtime_inventory(fc,bad,runtime))
   bad=copy.deepcopy(inventory);del bad[layout_name]
   reject('inventory_missing_'+layout_name,lambda bad=bad:m.parse_runtime_inventory(fc,bad,runtime))

  for name,bad in [('empty',{}),('missing_modules',{k:v for k,v in inventory.items() if k!='scientific_modules'}),('empty_files',dict(inventory,import_file_sha256={})),('wrong_prefix',dict(inventory,prefix=str(directory)) )]:
   reject('inventory_reject_'+name,lambda bad=bad:m.parse_runtime_inventory(fc,bad,runtime))
  bad=copy.deepcopy(inventory);bad['distributions']['numpy']='99.0'
  reject('pinned_numpy_distribution_version_mismatch',lambda:m.parse_runtime_inventory(fc,bad,runtime))
  bad=copy.deepcopy(inventory);bad['scientific_modules']['numpy']['origin']=str(directory/'shadow_numpy.py')
  reject('inventory_module_origin_outside_native_prefix',lambda:m.parse_runtime_inventory(fc,bad,runtime))
  bad=copy.deepcopy(inventory);bad['scientific_modules']['numpy']['sha256']='9'*64
  reject('inventory_module_import_bytes_unbound',lambda:m.parse_runtime_inventory(fc,bad,runtime))
  def imported(name):return objects[name]
  with patch.object(sys,'prefix',str(prefix)),patch.object(sys,'executable',str(alias)),patch.object(sys,'version_info',(3,11,9)),patch.dict(os.environ,{'PYTHONPATH':'','CUDA_VISIBLE_DEVICES':''}),patch.object(m,'native_distributions',return_value=distributions),patch.object(m.importlib,'import_module',side_effect=imported):
   proof=m.runtime_preflight(fc,runtime,True)
   check('actual_child_runtime_compares_complete_import_bytes_versions_modules',proof['inventory_matches_actual_child_scientific_runtime'])
   objects['torch'].cuda=types.SimpleNamespace(is_initialized=lambda:False)
   with patch.dict(sys.modules,{'torch':objects['torch']}):
    measured=m.capture_runtime_inventory(fc,runtime)
   check('root_capture_measures_complete_handmade_package_identities_without_model',measured==inventory)
   with patch.object(sys,'executable',str(directory/'different_interpreter')):
    reject('root_capture_rejects_actual_wrong_interpreter',lambda:m.capture_runtime_inventory(fc,runtime))
   shadow=types.ModuleType('numpy');shadow.__file__=str(directory/'outside_numpy.py');Path(shadow.__file__).write_text('# shadow');shadow.__version__='1.26.4';shadow.__spec__=objects['numpy'].__spec__
   objects['numpy']=shadow
   reject('actual_shadowed_package_rejected_even_with_expected_version',lambda:m.runtime_preflight(fc,runtime,True))
   objects['numpy']=types.SimpleNamespace(__file__=modules['numpy']['origin'],__version__=modules['numpy']['module_version'],__spec__=types.SimpleNamespace(loader=importlib.machinery.SourceFileLoader('numpy',modules['numpy']['origin'])))
   objects['numpy'].__version__='wrong_module_version'
   reject('actual_module_version_mismatch_rejected',lambda:m.runtime_preflight(fc,runtime,True));objects['numpy'].__version__=modules['numpy']['module_version']
   p=site/'numpy.py';original=p.read_text();p.write_text('# changed import bytes')
   reject('actual_prefix_import_byte_drift_rejected_before_package_import',lambda:m.runtime_preflight(fc,runtime,True));p.write_text(original)
   p=site/'numpy.pyc';p.write_bytes(b'handmade never executed bytecode')
   reject('unrecorded_actual_native_bytecode_shadow_rejected',lambda:m.runtime_preflight(fc,runtime,True));p.unlink()
   with patch.object(m,'native_distributions',return_value=dict(distributions,numpy='wrong')):
    reject('actual_complete_distribution_map_mismatch_rejected',lambda:m.runtime_preflight(fc,runtime,True))
report={'kind':'native_argformer_tsinfer_fit_v7_handmade_runtime_parse_checks','status':'PASS','count':len(checks),'checks':checks,
 'source_sha256':m.sha(m.SCRIPT),'config_sha256':m.sha(m.CONFIG),'test_source_sha256':m.sha(__file__),
 'native_prefix_or_actual_runtime_inventory_captured':False,'all_import_providers_and_native_roots_handmade':True,'scientific_modules_instantiated':False,'models_or_GPU':False}
(REVIEW/'runtime_parse_replay_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'PASS','checks':len(checks)},indent=2))
