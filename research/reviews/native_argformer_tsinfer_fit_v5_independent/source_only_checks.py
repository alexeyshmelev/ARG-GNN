"""Bounded independent v5 delta: terminal directory entries and exact science pins."""
from __future__ import annotations
import ast,copy,hashlib,importlib.util,json,os,sys,types
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');REVIEW=ROOT/'research/reviews/native_argformer_tsinfer_fit_v5_independent'
sys.dont_write_bytecode=True;os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONPATH']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
m=load('_independent_native_v5',ROOT/'src/arggnn/native_argformer_tsinfer_fit_v5.py');c=m.policy()
v4=load('_independent_sealed_native_v4',ROOT/'src/arggnn/native_argformer_tsinfer_fit_v4.py');c4=v4.policy()
checks=[];details={}
def check(name,value):
 if not value:raise AssertionError(name)
 checks.append(name)
def reject(name,call,exception=(ValueError,RuntimeError,OSError)):
 try:call()
 except exception as error:
  checks.append(name);return {'type':type(error).__name__,'message':str(error)}
 raise AssertionError('unexpected acceptance: '+name)
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');return m.reference(path)
pins={
 'src/arggnn/native_argformer_tsinfer_fit_v5.py':'1f05220ddfe7ba2fd3e55b079088eb361c6901323a9520145b0f037f6cfd22e0',
 'configs/native_argformer_tsinfer_fit_v5.json':'6d58f126a9fa16f9dfc04956215e85569f9696cb3b12c06f63fa458d8bb0986f',
 'research/reviews/native_argformer_tsinfer_fit_v5_author/source_receipt.json':'f457ba22a72c9c139e2689788021d37cffd5bf4ac4e780c061626a999e7790b8',
 'research/reviews/native_argformer_tsinfer_fit_v4_independent/review_receipt.json':'ba99eb8fdfd871454ef7542b57746f12e2f8d4b1d3ce7e6496dc98cd2403b132'}
for p,h in pins.items():check('requested_pin_'+p,m.sha(ROOT/p)==h)
author=json.loads((ROOT/'research/reviews/native_argformer_tsinfer_fit_v5_author/source_receipt.json').read_text())
for r in author['files']:check('sealed_author_file_'+r['path'],m.sha(ROOT/r['path'])==r['sha256'])
proof=m.ast_proof(c)
check('all30_scientific_ASTs_exact',proof==v4.ast_proof(c4) and sum(len(x) for x in proof.values())==30)
s=m.SCRIPT.read_text().replace('native_argformer_tsinfer_fit_v5','native_argformer_tsinfer_fit_v4').replace('_admitted_native_tsinfer_fit_v5','_admitted_native_tsinfer_fit_v4').replace('_admitted_native_tsinfer_retrieval_v5','_admitted_native_tsinfer_retrieval_v4').replace('_v5_child_title_before_scientific_import','_v4_child_title_before_scientific_import').replace(m.CONFIG_SHA256,v4.CONFIG_SHA256).replace('.runtime/tmp/afts5','.runtime/tmp/afts4')
old=ast.parse(v4.SCRIPT.read_text());new=ast.parse(s)
olddefs={n.name:n for n in old.body if isinstance(n,ast.FunctionDef)};newdefs={n.name:n for n in new.body if isinstance(n,ast.FunctionDef)}
check('only_one_new_terminal_lstat_helper',set(newdefs)-set(olddefs)=={'terminal_entries_present'} and set(olddefs)-set(newdefs)==set())
for name in ('execute_fit','execute_retrieval','verify_completed_fit','verify_memory','retrieval_parent','runtime_preflight','parse_runtime_inventory','load_exact_helpers','additive_helper_paths'):
 check('unchanged_AST_'+name,ast.dump(olddefs[name],include_attributes=False)==ast.dump(newdefs[name],include_attributes=False))
def expression(value):return ast.dump(ast.parse(value,mode='eval').body,include_attributes=False)
term=newdefs['terminal_receipt'].body[1]
check('terminal_writer_uses_presence_helper',ast.dump(term.test,include_attributes=False)==expression('terminal_entries_present(output)'))
term.test=copy.deepcopy(olddefs['terminal_receipt'].body[1].test)
newowned=next(n for n in newdefs['execute'].body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='owned' for t in n.targets))
oldowned=next(n for n in olddefs['execute'].body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='owned' for t in n.targets))
check('execute_uses_presence_helper',ast.dump(newowned.value,include_attributes=False)==expression('bool(args.native and output.is_dir() and not terminal_entries_present(output))'))
newowned.value=copy.deepcopy(oldowned.value)
guard=next(n for n in ast.walk(newdefs['existing_entry_ownership']) if isinstance(n,ast.If) and ast.dump(n.test,include_attributes=False)==expression('not output.is_dir() or terminal_entries_present(output)'))
oldguard=next(n for n in ast.walk(olddefs['existing_entry_ownership']) if isinstance(n,ast.If) and ast.dump(n.test,include_attributes=False)==expression("not output.is_dir() or (output/'receipt.json').exists() or (output/'exit_status').exists()"))
guard.test=copy.deepcopy(oldguard.test)
new.body=[n for n in new.body if not (isinstance(n,ast.FunctionDef) and n.name=='terminal_entries_present')]
check('entire_executor_equal_after_only_declared_delta',ast.dump(new,include_attributes=False)==ast.dump(old,include_attributes=False))
nc=copy.deepcopy(c)
for k in ('kind','result_root','short_tmp_root'):nc[k]=c4[k]
nc['storage']['reservation_key_prefix']=c4['storage']['reservation_key_prefix'];nc['runtime_inventory_contract']['kind']=c4['runtime_inventory_contract']['kind']
check('entire_config_equal_after_only_bindings',nc==c4)
check('helper_and_budget_pins_exact',c['runtime_inventory_contract']['expected_additive_helpers']==c4['runtime_inventory_contract']['expected_additive_helpers'] and c['storage']['companion_contract']==c4['storage']['companion_contract'])
for r in c['storage']['companion_contract'].values():check('budget_source_config_bound_'+r['path'],m.reference(ROOT/r['path'])==r)
check('source_contract_scope_and64KiB_limit',all(m.ENTRY_OWNERSHIP[k]==c[k] for k in ('parent_ids','result_root','short_tmp_root','condition')) and m.ENTRY_OWNERSHIP['storage']['failure_receipt_max_bytes']==65536)
check('inactive_source_config',c['execution_enabled'] is False)

f=REVIEW/'f';f.mkdir()
entry=copy.deepcopy(m.ENTRY_OWNERSHIP);entry['result_root']=str(f/'children')
parent=c['parent_ids'][0];cases={}
def snapshot(out):
 records={}
 for p in out.iterdir():
  if p.is_symlink():records[p.name]=['symlink',str(p.readlink())]
  elif p.is_dir():records[p.name]=['directory']
  elif p.is_file():records[p.name]=['file',m.sha(p)]
  else:records[p.name]=['special',p.lstat().st_mode]
 return records
def case(mode,terminal=None,kind=None):
 directory=f/mode;directory.mkdir()
 fc=copy.deepcopy(c);fc['result_root']=entry['result_root']
 cfg=directory/'config.json';cr=write(cfg,fc);crc=cr['sha256']
 args=types.SimpleNamespace(execute=True,native=True,parent=parent,stage='retrieve',run_id=mode,expected_source_sha256=m.sha(m.SCRIPT),expected_config_sha256=crc,admission='',expected_admission_sha256='')
 with ExitStack() as stack:
  stack.enter_context(patch.object(m,'ENTRY_OWNERSHIP',entry));stack.enter_context(patch.object(m,'CONFIG',cfg));stack.enter_context(patch.object(m,'CONFIG_SHA256',crc))
  scope,out,tmp,log=m.scopes(entry,parent,'retrieve',mode);out.mkdir(parents=True)
  rr=write(directory/'review.json',{'kind':'native_argformer_tsinfer_fit_v5_independent_review','review_role':'independent_reviewer','verdict':'PASS','implementation':m.reference(m.SCRIPT),'config':cr,'fixture_only_not_real_review':True})
  ar=write(directory/'root.json',{'kind':'native_argformer_tsinfer_fit_v5_root_admission','authorizer':'root_agent','active':True,'status':'root_authorized_exact_native_stage','condition':entry['condition'],'scope_parents':[parent],'stage':'retrieve','run_id':mode,'implementation':m.reference(m.SCRIPT),'config':cr,'parent_storage_scope':str(scope),'output':str(out),'short_tmp':str(tmp),'launch_log':str(log),'CPU_only':True,'query_truth_authorized':False,'scientific_fallback':False,'paper_reproduction_claimed':False,'contrastive_execution':'full_batch','independent_review':rr,'fixture_only_not_real_authority':True})
  args.admission=str(ROOT/ar['path']);args.expected_admission_sha256=ar['sha256']
  write(out/'exec_admission.json',{'implementation_sha256':args.expected_source_sha256,'stage':'retrieve','root_admission':ar,'TMPDIR':str(tmp)})
  argv=['--execute','--native','--parent',parent,'--stage','retrieve','--run-id',mode,'--expected-source-sha256',args.expected_source_sha256,'--expected-config-sha256',crc,'--admission',args.admission,'--expected-admission-sha256',args.expected_admission_sha256]
  check('real_policy_initially_exact_'+mode,m.policy()==fc)
  if terminal:
   p=out/terminal
   if kind=='regular':p.write_text('PRESERVE')
   elif kind=='directory':p.mkdir()
   elif kind=='dangling_symlink':p.symlink_to('missing_terminal_target')
   elif kind=='live_symlink':
    target=directory/'target.txt';target.write_text('PRESERVE');p.symlink_to(target)
   elif kind=='fifo':os.mkfifo(p)
  if mode=='missing_config':cfg.unlink()
  else:cfg.write_bytes(cfg.read_bytes()+b'\n')
  if mode=='injected_policy':stack.enter_context(patch.object(m,'policy',side_effect=RuntimeError('fresh independent policy failure')))
  before=snapshot(out)
  with patch.dict(os.environ,{'TMPDIR':str(tmp)}):
   if terminal:
    check('lstat_detects_terminal_'+mode,m.terminal_entries_present(out))
    check('owned_entry_denied_'+mode,m.existing_entry_ownership(args) is None)
    failure=reject('main_preserves_policy_failure_'+mode,lambda:m.main(argv))
    check('main_never_amends_terminal_output_'+mode,snapshot(out)==before)
    reject('terminal_writer_denies_entry_'+mode,lambda:m.terminal_receipt(fc,out,{'exit_code':1}))
    check('terminal_writer_never_amends_'+mode,snapshot(out)==before)
    reject('execute_does_not_adopt_terminal_'+mode,lambda:m.execute(fc,args))
    check('execute_never_amends_'+mode,snapshot(out)==before)
    cases[mode]={'entry_detected':True,'ownership_denied':True,'output_amended':False,'failure':failure}
   else:
    check('fresh_owned_entry_proven_'+mode,m.existing_entry_ownership(args) is not None)
    code=m.main(argv);r=json.loads((out/'receipt.json').read_text())
    check('owned_policy_failure_terminal_'+mode,code==1 and (out/'exit_status').read_text()=='1\n' and r['root_actual_execution'] is False and r['rejected_config_used_for_scope_or_limit'] is False and (out/'receipt.json').stat().st_size<=65536)
    if mode=='missing_config':check('missing_config_observed_hash_null',r['config_sha256'] is None)
    cases[mode]={'receipt':m.reference(out/'receipt.json'),'exit_status':m.reference(out/'exit_status'),'native_success':False}
for mode in ('byte_drift','missing_config','injected_policy'):case(mode)
for terminal in ('receipt.json','exit_status'):
 for kind in ('regular','directory','dangling_symlink','live_symlink','fifo'):case(('receipt_' if terminal=='receipt.json' else 'exit_')+kind,terminal,kind)
empty=f/'empty';empty.mkdir()
check('no_terminal_entries_false',m.terminal_entries_present(empty) is False)
original_lstat=Path.lstat
def denied(path,*args,**kwargs):
 if path==empty/'receipt.json':raise PermissionError('fresh independent terminal stat denial')
 return original_lstat(path,*args,**kwargs)
with patch.object(Path,'lstat',autospec=True,side_effect=denied):
 reject('terminal_stat_permission_failure_preserved',lambda:m.terminal_entries_present(empty),PermissionError)
details['entry_cases']=cases
details['scientific_ASTs']=proof
details['native_capture_and_GPU4096_probe_pending']=True
reject('empty_inventory_still_rejected',lambda:m.parse_runtime_inventory(c,{},{}))
reject('unregistered_helper_still_rejected',lambda:m.additive_helper_paths(c,{'additive_helpers':[]}))
for p,h in pins.items():check('final_immutable_pin_'+p,m.sha(ROOT/p)==h)
report={'kind':'native_argformer_tsinfer_fit_v5_independent_delta_checks','status':'passed','check_count':len(checks),'checks':checks,'details':details,'issues':[],'actual_model_fit_executed':False,'released_model_instantiated':False,'actual_scientific_arrays_opened':False,'actual_GPU_inventory_context_or_live_ledger_read':False,'actual_native_runtime_inventory_captured':False,'authority_activation_performed':False,'runtime_prefix_modified':False,'install_or_Git_performed':False,'contained_private_constants_only':True}
write(REVIEW/'handmade_checks.json',report)
print(json.dumps({'status':'passed','check_count':len(checks),'entry_cases':len(cases),'issues':[]},indent=2))
