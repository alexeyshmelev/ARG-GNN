"""Independent v6 delta audit. Only pinned source/JSON and handmade prefixes."""
from __future__ import annotations
import ast,copy,hashlib,importlib.util,json,os,re,stat,sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');OUT=ROOT/'research/reviews/native_argformer_tsinfer_fit_v6_independent'
sys.dont_write_bytecode=True
assert os.environ.get('PYTHONDONTWRITEBYTECODE')=='1' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
checks=[];details={};rejections={}
def check(name,value):
 if not value:raise AssertionError(name)
 checks.append(name)
def reject(name,call,exception=(ValueError,OSError,RuntimeError)):
 try:call()
 except exception as error:checks.append(name);rejections[name]={'type':type(error).__name__,'message':str(error)};return
 raise AssertionError('unexpected acceptance: '+name)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return {'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
def write(name,value):
 with (OUT/name).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
pins={
'src/arggnn/native_argformer_tsinfer_fit_v6.py':'024b98957be0b86d86e97a7c9fef1f22a492709f652eaf3f7900f6f651d4c77f',
'configs/native_argformer_tsinfer_fit_v6.json':'e0dfe65b93ce614a13bbfc390b9f79988956e3724e3c697ad840fc2dc0196318',
'research/reviews/native_argformer_tsinfer_fit_v6_author/source_receipt.json':'d2f383dc1ef2eb85a72cfb24aecc1a664061943769cb10d4d398d322cc31c507',
'src/arggnn/native_argformer_tsinfer_fit_v5.py':'1f05220ddfe7ba2fd3e55b079088eb361c6901323a9520145b0f037f6cfd22e0',
'configs/native_argformer_tsinfer_fit_v5.json':'6d58f126a9fa16f9dfc04956215e85569f9696cb3b12c06f63fa458d8bb0986f',
'research/reviews/native_argformer_tsinfer_fit_v5_independent/review_receipt.json':'3fbe3ceb65dad2d4036276d3f249f2a9827c3855f121f5bcdab4791d19ca082f',
'research/reviews/native_argformer_tsinfer_fit_v5_independent/validator_receipt.json':'cc26dcba1506dcb6ab5beaee6f7767348fdb2dbcac2cabfd03f123c37e02636f',
'experiments/results/native_argformer_tsinfer_fit_v5_root/directory_symlink_metadata.json':'bbd7b8fb18d107582d5b67d81803d37bf447da68c44c154c1b97e68ec9781ee3',
'experiments/results/native_argformer_tsinfer_fit_v5_root/dangling_file_symlink_metadata.json':'1d9c1b5b4e09e020d30802da36390b14c9485838d419a362fb9fac1fc6337049'}
for p,h in pins.items():check('requested_sealed_pin_'+p,sha(ROOT/p)==h)
author=json.loads((ROOT/'research/reviews/native_argformer_tsinfer_fit_v6_author/source_receipt.json').read_text())
for r in author['files']:check('sealed_v6_author_file_'+r['path'],ref(ROOT/r['path'])==r)
v5review=json.loads((ROOT/'research/reviews/native_argformer_tsinfer_fit_v5_independent/review_receipt.json').read_text())
for r in v5review['files']:check('preserved_v5_independent_file_'+r['path'],ref(ROOT/r['path'])==r)
v5author=json.loads((ROOT/v5review['author_source_receipt']['path']).read_text())
check('v5_author_receipt_unchanged',ref(ROOT/v5review['author_source_receipt']['path'])==v5review['author_source_receipt'])
for r in v5author['files']:check('preserved_v5_author_file_'+r['path'],ref(ROOT/r['path'])==r)
check('v5_source_PASS_preserved',v5review['verdict']=='PASS')
m=load('_independent_native_v6_source',ROOT/'src/arggnn/native_argformer_tsinfer_fit_v6.py');c=m.policy()
v5=load('_independent_native_v5_source',ROOT/'src/arggnn/native_argformer_tsinfer_fit_v5.py');c5=v5.policy()
check('inactive_exact_source_config',c['execution_enabled'] is False and c['status']==c5['status'])
for p,h in c['frozen_files_sha256'].items():check('unchanged_frozen_authority_'+p,sha(ROOT/p)==h)
proof=m.ast_proof(c)
check('all30_native_scientific_ASTs_preserved',proof==v5.ast_proof(c5) and sum(map(len,proof.values()))==30)
s5=v5.SCRIPT.read_text();s6=m.SCRIPT.read_text()
normalized=s6.replace('native_argformer_tsinfer_fit_v6','native_argformer_tsinfer_fit_v5').replace('_admitted_native_tsinfer_fit_v6','_admitted_native_tsinfer_fit_v5').replace('_admitted_native_tsinfer_retrieval_v6','_admitted_native_tsinfer_retrieval_v5').replace(m.CONFIG_SHA256,v5.CONFIG_SHA256).replace('.runtime/tmp/afts6','.runtime/tmp/afts5')
def region(text,start,end):
 a=text.index('\ndef '+start+'(')+1;b=text.index('\ndef '+end+'(',a)+1;return text[a:b]
for a,b in [('parse_runtime_inventory','native_distributions'),('capture_runtime_inventory','prepare_environment')]:
 normalized=normalized.replace(region(normalized,a,b),region(s5,a,b))
check('complete_executor_bytes_equal_after_only_bindings_and_inventory_regions',normalized==s5)
t5=ast.parse(s5);t6=ast.parse(s6)
def defs(t):return {n.name:n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
d5=defs(t5);d6=defs(t6)
added={'native_alias_records','native_alias_contract','native_alias_cycles','native_inventory_scan'}
changed={'parse_runtime_inventory','native_import_bytes','capture_runtime_inventory','runtime_preflight'}
check('only_four_added_inventory_helpers',set(d6)-set(d5)==added and not (set(d5)-set(d6)))
normalized_ast=ast.parse(s6.replace('native_argformer_tsinfer_fit_v6','native_argformer_tsinfer_fit_v5').replace('_admitted_native_tsinfer_fit_v6','_admitted_native_tsinfer_fit_v5').replace('_admitted_native_tsinfer_retrieval_v6','_admitted_native_tsinfer_retrieval_v5').replace(m.CONFIG_SHA256,v5.CONFIG_SHA256).replace('.runtime/tmp/afts6','.runtime/tmp/afts5'))
nd=defs(normalized_ast)
for name in sorted(set(d5)-changed):check('unchanged_non_inventory_AST_'+name,ast.dump(d5[name],include_attributes=False)==ast.dump(nd[name],include_attributes=False))
nc=copy.deepcopy(c)
for key in ('kind','result_root','short_tmp_root'):nc[key]=c5[key]
nc['storage']['reservation_key_prefix']=c5['storage']['reservation_key_prefix'];nc['runtime_inventory_contract']['kind']=c5['runtime_inventory_contract']['kind']
for key in ('directory_alias_layout','absent_executable_alias_layout'):del nc['runtime_inventory_contract'][key]
check('entire_config_equal_after_only_bindings_and_two_alias_contracts',nc==c5)
check('exclusive_route_and_budget_helper_pins_unchanged',c['launcher_contract']==c5['launcher_contract'] and c['storage']['companion_contract']==c5['storage']['companion_contract'] and c['runtime_inventory_contract']['expected_additive_helpers']==c5['runtime_inventory_contract']['expected_additive_helpers'])
real_dirs,real_absent=m.native_alias_contract(c)
check('real_metadata_gate_seven_dirs_three_absent',len(real_dirs)==7 and len(real_absent)==3)
check('absent_metadata_never_invents_SHA',all('sha256' not in r for r in real_absent.values()))
check('root_prefix_unchanged_metadata_only_contract',all(json.loads((ROOT/c['runtime_inventory_contract'][k]['provenance']['path']).read_text())['prefix_modified'] is False for k in ('directory_alias_layout','absent_executable_alias_layout')))
for name in ('capture_runtime_inventory','runtime_preflight'):
 node=d6[name];calls=[n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
 scans=sorted(n.lineno for n in calls if n.func.id=='native_inventory_scan');imports=sorted(n.lineno for n in calls if n.func.id=='actual_scientific_modules')
 check('before_after_bytes_and_layout_scan_'+name,len(scans)==2 and len(imports)==1 and scans[0]<imports[0]<scans[1])
check('scanner_never_follows_directory_aliases','followlinks=False' in ast.unparse(d6['native_import_bytes']))
check('capture_keeps_no_CUDA_context_and_exact_prefix_guards',all(x in ast.unparse(d6['capture_runtime_inventory']) for x in ('torch.cuda.is_initialized()',"os.environ.get('CUDA_VISIBLE_DEVICES') != ''",'not sys.dont_write_bytecode',"os.environ.get('PYTHONPATH', '')")))

F=OUT/'f';F.mkdir()
def alias_record(prefix,path,directory):
 p=prefix/path;st=p.lstat();record={'path':path,'target':os.readlink(p),'resolved_path':str(p.resolve().relative_to(prefix)),'lstat_mode':st.st_mode,'lstat_size':st.st_size,'lstat_device':st.st_dev,'lstat_inode':st.st_ino}
 record['target_is_directory' if directory else 'target_absent']=True;return record
def fixture(name):
 prefix=F/name;prefix.mkdir()
 for r in real_dirs.values():
  (prefix/r['resolved_path']).mkdir(parents=True,exist_ok=True)
 for index,r in enumerate(real_dirs.values()):
  (prefix/r['resolved_path']/('handmade_'+str(index)+'.py')).write_bytes(('explicit handmade code bytes '+str(index)+'\n').encode())
 for r in real_dirs.values():
  p=prefix/r['path'];p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(r['target'])
 for r in real_absent.values():
  p=prefix/r['path'];p.parent.mkdir(parents=True,exist_ok=True);p.symlink_to(r['target'])
 python=prefix/'bin/python';python.parent.mkdir();python.write_bytes(b'explicit handmade interpreter token\n');python.chmod(0o700)
 mod=prefix/'lib/python3.11/site-packages/fixturemodule/__init__.py';mod.parent.mkdir(parents=True);mod.write_bytes(b'explicit handmade module token\n')
 for suffix in ('.pyc','.pyo','.pth','.so.7'):(prefix/('canonical'+suffix)).write_bytes(('handmade '+suffix+'\n').encode())
 live=prefix/'live.py';live.symlink_to('lib/python3.11/site-packages/fixturemodule/__init__.py')
 directories=[alias_record(prefix,r['path'],True) for r in real_dirs.values()]
 absent=[alias_record(prefix,r['path'],False) for r in real_absent.values()]
 return prefix,directories,absent

def scan(prefix,dirs,absent):return m.native_import_bytes(prefix,directory_symlinks=dirs,absent_symlinks=absent,return_layout=True)
p,d,a=fixture('positive');inventory,dirs,absent=scan(p,d,a)
check('handmade_all7_directory_shapes_all3_absent_shapes_pass',len(dirs)==7 and len(absent)==3)
check('complete_canonical_executable_records_not_alias_subtree_duplicates',len(inventory)==13 and all(not any(Path(alias)==Path(k) or Path(alias) in Path(k).parents for alias in dirs) for k in inventory))
check('canonical_existing_bytecode_pyo_pth_numbered_library_included',all('canonical'+x in inventory for x in ('.pyc','.pyo','.pth','.so.7')))
check('valid_internal_live_file_alias_bytes_still_bound',inventory['live.py']['sha256']==inventory['lib/python3.11/site-packages/fixturemodule/__init__.py']['sha256'])
check('absent_aliases_have_layout_only_no_byte_hash',not set(absent)&set(inventory))
reject('v5_reproduces_ordinary_directory_alias_failure',lambda:v5.native_import_bytes(p))
reject('v6_unbound_all_aliases_still_fails',lambda:m.native_import_bytes(p))
for index,record in enumerate(d):
 for field in ('target','resolved_path','lstat_mode','lstat_size','lstat_device','lstat_inode','target_is_directory'):
  mutated=copy.deepcopy(d)
  if field=='target':mutated[index][field]='73.3' if len(record[field])==4 else record[field]+'x'
  elif field=='resolved_path':mutated[index][field]='other/handmade_directory'
  elif field=='target_is_directory':mutated[index][field]=False
  else:mutated[index][field]+=1
  reject('directory_bound_metadata_mutation_'+str(index)+'_'+field,lambda z=mutated:scan(p,z,a))
for index,record in enumerate(a):
 for field in ('target','resolved_path','lstat_mode','lstat_size','lstat_device','lstat_inode','target_absent'):
  mutated=copy.deepcopy(a)
  if field=='target':mutated[index][field]=record[field]+'x'
  elif field=='resolved_path':mutated[index][field]='lib/other_absent.so'
  elif field=='target_absent':mutated[index][field]=False
  else:mutated[index][field]+=1
  reject('absent_bound_metadata_mutation_'+str(index)+'_'+field,lambda z=mutated:scan(p,d,z))
for kind in ('missing','regular_directory','regular_file','retargeted','unexpected_directory_alias','unexpected_dangling_alias','expected_absent_now_file','expected_absent_now_directory','escape_live_directory','escape_dangling','symlink_resolution_cycle','directory_traversal_cycle'):
 p2,d2,a2=fixture(kind);link=p2/d2[0]['path']
 if kind=='missing':link.unlink()
 elif kind=='regular_directory':link.unlink();link.mkdir()
 elif kind=='regular_file':link.unlink();link.write_bytes(b'handmade regular replacement')
 elif kind=='retargeted':link.unlink();link.symlink_to(d2[1]['resolved_path'])
 elif kind=='unexpected_directory_alias':(p2/'new_alias').symlink_to('lib/python3.11')
 elif kind=='unexpected_dangling_alias':(p2/'unbound.py').symlink_to('missing.py')
 elif kind.startswith('expected_absent_now_'):
  target=p2/a2[0]['resolved_path']
  if kind.endswith('file'):target.write_bytes(b'now unexpectedly present')
  else:target.mkdir()
 elif kind=='escape_live_directory':(p2/'escape').symlink_to(F)
 elif kind=='escape_dangling':(p2/'escape.py').symlink_to('../../external_missing.py')
 elif kind=='symlink_resolution_cycle':(p2/'cycleA.py').symlink_to('cycleB.py');(p2/'cycleB.py').symlink_to('cycleA.py')
 elif kind=='directory_traversal_cycle':
  target=p2/'cycle_real';target.mkdir();link=target/'back';link.symlink_to('../cycle_real');d2.append(alias_record(p2,'cycle_real/back',True))
 reject('fresh_actual_alias_mutation_'+kind,lambda:scan(p2,d2,a2))
for kind in ('missing_absent','regular_absent','absent_intermediate_symlink'):
 p2,d2,a2=fixture(kind);link=p2/a2[0]['path']
 if kind=='missing_absent':link.unlink()
 elif kind=='regular_absent':link.unlink();link.write_bytes(b'handmade replacement')
 else:(p2/a2[0]['resolved_path']).symlink_to('different_missing.so')
 reject('fresh_actual_absent_mutation_'+kind,lambda:scan(p2,d2,a2))
for kind in ('new_byte','deleted_byte','changed_byte'):
 p2,d2,a2=fixture(kind);before=scan(p2,d2,a2);file=p2/'canonical.pyc'
 if kind=='new_byte':(p2/'new_real.py').write_bytes(b'new handmade code')
 elif kind=='deleted_byte':file.unlink()
 else:file.write_bytes(b'changed handmade code')
 check('complete_bytes_detect_'+kind,not m.exact(before,scan(p2,d2,a2)))
reject('strict_alias_duplicate_path',lambda:m.native_alias_records(d+d[:1],True))
for field,value in [('path','../escape'),('path','absolute//odd'),('resolved_path','.'),('target','/absolute'),('lstat_inode',True),('extra','forbidden')]:
 v=copy.deepcopy(d[:1]);v[0][field]=value
 reject('strict_alias_schema_'+field+'_'+str(value),lambda:m.native_alias_records(v,True))
cycles={'a':{'resolved_path':'b'},'b':{'resolved_path':'a'}}
reject('explicit_cross_alias_directory_cycle',lambda:m.native_alias_cycles(cycles))
# Error propagation is tested by explicit in-memory os.walk error injection.
def denied_walk(prefix,**kw):kw['onerror'](PermissionError('handmade inventory traversal denial'));return iter(())
with patch.object(m.os,'walk',side_effect=denied_walk):reject('walk_permission_error_preserved',lambda:scan(p,d,a),PermissionError)

# Strict inventory parser on synthetic origins and maps, with no actual module import.
fc=copy.deepcopy(c);fc['runtime']={'prefix':str(p.relative_to(ROOT)),'interpreter':str((p/'bin/python').relative_to(ROOT)),'python_version':[3,11,9]}
for field,aliases,key,kind in [('directory_alias_layout',d,'directory_symlinks','handmade_dir'),('absent_executable_alias_layout',a,'aliases','handmade_absent')]:fc['runtime_inventory_contract'][field]={'provenance':None,'kind':kind,'records_key':key,'aliases':aliases}
fc['runtime_inventory_contract']['scientific_modules']={'fixturemodule':'fixturemodule'};fc['runtime_inventory_contract']['expected_scientific_distributions']={'fixturemodule':'1.0'}
runtime={'python_executable_sha256':sha(p/'bin/python'),'additive_helpers':[]}
origin='lib/python3.11/site-packages/fixturemodule/__init__.py'
value={'kind':fc['runtime_inventory_contract']['kind'],'status':fc['runtime_inventory_contract']['status'],'prefix':str(p),'python':{'version':[3,11,9],'executable':str(p/'bin/python'),'resolved_executable':str(p/'bin/python'),'sha256':runtime['python_executable_sha256']},'distributions':{'fixturemodule':'1.0'},'scientific_modules':{'fixturemodule':{'distribution':'fixturemodule','distribution_version':'1.0','module_version':'1.0','origin':str(p/origin),'sha256':inventory[origin]['sha256'],'loader':'explicit.HandmadeLoader'}},'import_file_sha256':inventory,'additive_helpers':[],'directory_symlinks':dirs,'absent_executable_symlinks':absent}
check('strict_handmade_inventory_parser_pass',m.parse_runtime_inventory(fc,value,runtime)==value)
for kind in ('missing_dir_map','missing_absent_map','dir_state_drift','absent_state_drift','invent_absent_SHA','duplicate_directory_subtree_SHA','module_origin_alias','wrong_module_SHA','wrong_distribution','wrong_loader','extra_key'):
 bad=copy.deepcopy(value)
 if kind=='missing_dir_map':del bad['directory_symlinks']
 elif kind=='missing_absent_map':del bad['absent_executable_symlinks']
 elif kind=='dir_state_drift':next(iter(bad['directory_symlinks'].values()))['lstat_inode']+=1
 elif kind=='absent_state_drift':next(iter(bad['absent_executable_symlinks'].values()))['target_absent']=False
 elif kind=='invent_absent_SHA':bad['import_file_sha256'][a[0]['path']]={'resolved_path':a[0]['resolved_path'],'sha256':'a'*64}
 elif kind=='duplicate_directory_subtree_SHA':bad['import_file_sha256'][d[0]['path']+'/invented.py']={'resolved_path':d[0]['resolved_path']+'/invented.py','sha256':'a'*64}
 elif kind=='module_origin_alias':bad['scientific_modules']['fixturemodule']['origin']=str(p/'lib/python3.1/site-packages/fixturemodule/__init__.py')
 elif kind=='wrong_module_SHA':bad['scientific_modules']['fixturemodule']['sha256']='a'*64
 elif kind=='wrong_distribution':bad['distributions']['fixturemodule']='2.0'
 elif kind=='wrong_loader':bad['scientific_modules']['fixturemodule']['loader']=''
 else:bad['unregistered_field']=True
 reject('strict_inventory_parser_'+kind,lambda:m.parse_runtime_inventory(fc,bad,runtime))
# Production scan helper joins the bound metadata and actual handmade filesystem scan.
check('joined_handmade_inventory_scan_pass',m.native_inventory_scan(fc,p)==(inventory,dirs,absent))
# Keep inherited terminal lstat behavior visible without touching root execution outputs.
terminal=F/'terminal';terminal.mkdir()
check('empty_terminal_directory_denies_presence',m.terminal_entries_present(terminal) is False)
for name in ('receipt.json','exit_status'):
 link=terminal/name;link.symlink_to('explicit_missing_terminal')
 check('inherited_dangling_terminal_detected_'+name,m.terminal_entries_present(terminal) is True);link.unlink()
for pth,h in pins.items():check('final_sealed_pin_'+pth,sha(ROOT/pth)==h)
for r in author['files']:check('final_author_file_'+r['path'],ref(ROOT/r['path'])==r)
details={'scientific_ASTs':proof,'source_regions_changed_only':[['parse_runtime_inventory','native_distributions'],['capture_runtime_inventory','prepare_environment']],'changed_existing_functions':sorted(changed),'added_helpers':sorted(added),'real_root_metadata_maps':{'directory_alias_count':len(real_dirs),'absent_alias_count':len(real_absent),'actual_prefix_scanned':False},'positive_handmade_inventory':{'canonical_executable_records':len(inventory),'directory_aliases':len(dirs),'absent_executable_aliases':len(absent),'actual_scientific_imports':False},'before_after_scans_source_AST_verified':True,'strict_parser_mutations':11,'rejections':rejections,'preserved_v5_PASS':ref(ROOT/'research/reviews/native_argformer_tsinfer_fit_v5_independent/review_receipt.json')}
report={'kind':'native_argformer_tsinfer_fit_v6_independent_delta_checks','status':'passed','check_count':len(checks),'checks':checks,'details':details,'issues':[],'actual_scientific_arrays_opened':False,'released_model_instantiated':False,'actual_model_fit_executed':False,'actual_native_runtime_inventory_captured':False,'actual_native_scientific_imports':False,'actual_GPU_inventory_context_or_live_ledger_read':False,'authority_activation_performed':False,'runtime_prefix_modified':False,'install_or_Git_performed':False,'pinned_inputs':pins,'CPU_threads':1}
write('handmade_checks.json',report)
print(json.dumps({'status':'passed','check_count':len(checks),'independent_rejection_cases':len(rejections),'issues':[]},indent=2))
