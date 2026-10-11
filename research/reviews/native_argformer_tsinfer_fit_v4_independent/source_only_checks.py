"""Fresh narrow independent v4 entrypoint, binding and handmade kernel checks."""
from __future__ import annotations
import ast,copy,hashlib,importlib.util,json,os,sys,types
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');REVIEW=ROOT/'research/reviews/native_argformer_tsinfer_fit_v4_independent'
sys.dont_write_bytecode=True;os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONPATH']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='4'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
m=load('_independent_native_v4',ROOT/'src/arggnn/native_argformer_tsinfer_fit_v4.py');c=m.policy()
checks=[];details={};issues=[]
def check(name,value):
 if not value:raise AssertionError(name)
 checks.append(name)
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');return m.reference(path)
def reject(name,call):
 try:call()
 except (ValueError,RuntimeError,TypeError,KeyError,FileNotFoundError,FileExistsError) as error:
  checks.append(name);return {'type':type(error).__name__,'message':str(error)}
 raise AssertionError('unexpected acceptance: '+name)
pins={
 'src/arggnn/native_argformer_tsinfer_fit_v4.py':'0cc1a9d47d69e1533a00de4ce8a8cefb2db2a8b342cba112072e64e24cc40457',
 'configs/native_argformer_tsinfer_fit_v4.json':'04aa7e623575c97b527ebbd45e3d68e5fdcbd6db16d71b41b9bf6ba3a4813061',
 'research/reviews/native_argformer_tsinfer_fit_v4_author/source_receipt.json':'8970d517d2d641ea8b552af6ce1d5043e2df16f5da6878b6e0bce24d658cce86',
 'research/reviews/native_argformer_tsinfer_fit_v3_independent/review_receipt.json':'b6eb0cbf37ab0143ce69aebcd81cca7c3feaa7e592a5bb814dc875c1076ffe3f'}
for p,h in pins.items():check('exact_requested_pin_'+p,m.sha(ROOT/p)==h)
author=json.loads((ROOT/'research/reviews/native_argformer_tsinfer_fit_v4_author/source_receipt.json').read_text())
v3author=json.loads((ROOT/'research/reviews/native_argformer_tsinfer_fit_v3_author/source_receipt.json').read_text())
for record in author['files']+v3author['files']:check('sealed_file_'+record['path'],m.sha(ROOT/record['path'])==record['sha256'])
v3=load('_independent_previous_native_v3',ROOT/'src/arggnn/native_argformer_tsinfer_fit_v3.py');c3=v3.policy()
proof=m.ast_proof(c)
check('all30_native_scientific_ASTs_unchanged',proof==v3author['native_ASTs'] and sum(len(v) for v in proof.values())==30)
s4=m.SCRIPT.read_text().replace('native_argformer_tsinfer_fit_v4','native_argformer_tsinfer_fit_v3').replace('_admitted_native_tsinfer_fit_v4','_admitted_native_tsinfer_fit_v3').replace('_admitted_native_tsinfer_retrieval_v4','_admitted_native_tsinfer_retrieval_v3').replace('_v4_child_title_before_scientific_import','_v3_child_title_before_scientific_import').replace(m.CONFIG_SHA256,v3.CONFIG_SHA256)
def remainder(tree,new=False):
 tree=ast.parse(tree)
 tree.body=[n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name in {'main','existing_entry_ownership','finalize_entry_policy_failure'}) and not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ENTRY_OWNERSHIP' for t in n.targets))]
 return ast.dump(tree,include_attributes=False)
check('entire_remainder_AST_exact_after_declared_bindings',remainder(s4)==remainder(v3.SCRIPT.read_text()))
normalized=copy.deepcopy(c)
for field in ('kind','result_root','short_tmp_root'):normalized[field]=c3[field]
normalized['storage']['reservation_key_prefix']=c3['storage']['reservation_key_prefix']
normalized['runtime_inventory_contract']['kind']=c3['runtime_inventory_contract']['kind']
check('entire_config_exact_after_declared_bindings',normalized==c3)
trees=[ast.parse(v3.SCRIPT.read_text()),ast.parse(m.SCRIPT.read_text())]
for name in ('execute_fit','execute_retrieval','verify_completed_fit','verify_memory','retrieval_parent'):
 nodes=[next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name) for tree in trees]
 check('unchanged_scientific_wrapper_'+name,ast.dump(nodes[0],include_attributes=False)==ast.dump(nodes[1],include_attributes=False))
for name in ('runtime_preflight','parse_runtime_inventory','load_exact_helpers','additive_helper_paths','native_import_bytes'):
 nodes=[next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name) for tree in trees]
 check('unchanged_strict_runtime_helper_guard_'+name,ast.dump(nodes[0],include_attributes=False)==ast.dump(nodes[1],include_attributes=False))
entry=m.ENTRY_OWNERSHIP
check('ENTRY_source_contract_matches_exact_reviewed_scope_limit',all(entry[k]==c[k] for k in ('parent_ids','result_root','short_tmp_root','condition')) and entry['storage']['failure_receipt_max_bytes']==c['storage']['failure_receipt_max_bytes']==65536)
prior=json.loads((ROOT/'research/reviews/native_argformer_tsinfer_fit_v3_independent/handmade_checks.json').read_text())
check('prior134_source_and_small_kernel_evidence_bound',prior['check_count']==134 and m.sha(ROOT/'research/reviews/native_argformer_tsinfer_fit_v3_independent/handmade_checks.json')=='36b5665ae8a37535004a2e13667648da08eec150ea21a32db449528c9dc4a042')
check('inactive_recipe_and_all_stage_gates',c['execution_enabled'] is False and all(m.templates(c,p,s,'fresh_review')['admission']['active'] is False for p in c['parent_ids'] for s in m.STAGES))
check('four20GB_fresh_parent_scopes',len(c['parent_ids'])==4 and c['storage']['reserved_bytes_per_parent']==20000000000)
for p in c['parent_ids']:
 _,_,tmp,_=m.scopes(c,p,'probe-memory','fresh_review')
 check('source_short_socket_limit_'+p,len(os.fsencode(tmp/'pymp-12345678/listener-12345678'))<=107)

f=REVIEW/'f';f.mkdir()
entry=copy.deepcopy(m.ENTRY_OWNERSHIP);entry['result_root']=str(f/'children')
parent=c['parent_ids'][0];case_results={}
def snapshot(path):
 if not path.exists() and not path.is_symlink():return {'absent':True}
 result={'self_symlink':str(path.readlink()) if path.is_symlink() else None}
 if path.is_dir():
  for p in sorted(path.rglob('*')):
   relative=str(p.relative_to(path))
   if p.is_symlink():result[relative]=['symlink',str(p.readlink())]
   elif p.is_file():result[relative]=['file',m.sha(p)]
   else:result[relative]=['directory']
 return result
def main_case(mode):
 directory=f/mode;directory.mkdir()
 cfg=directory/'config.json';fc=copy.deepcopy(c);fc['result_root']=entry['result_root']
 cfgref=write(cfg,fc);crc=cfgref['sha256']
 args=types.SimpleNamespace(execute=True,native=True,parent=parent,stage='retrieve',run_id=mode,
  expected_source_sha256=m.sha(m.SCRIPT),expected_config_sha256=crc,admission='',expected_admission_sha256='')
 with ExitStack() as stack:
  stack.enter_context(patch.object(m,'ENTRY_OWNERSHIP',entry));stack.enter_context(patch.object(m,'CONFIG',cfg));stack.enter_context(patch.object(m,'CONFIG_SHA256',crc))
  scope,out,tmp,log=m.scopes(entry,parent,'retrieve',mode);out.mkdir(parents=True)
  review={'kind':'native_argformer_tsinfer_fit_v4_independent_review','review_role':'independent_reviewer','verdict':'PASS','implementation':m.reference(m.SCRIPT),'config':cfgref,'fixture_only_not_actual_review':True}
  rr=write(directory/'review.json',review)
  a={'kind':'native_argformer_tsinfer_fit_v4_root_admission','authorizer':'root_agent','active':True,
   'status':'root_authorized_exact_native_stage','condition':entry['condition'],'scope_parents':[parent],
   'stage':'retrieve','run_id':mode,'implementation':m.reference(m.SCRIPT),'config':cfgref,
   'parent_storage_scope':str(scope),'output':str(out),'short_tmp':str(tmp),'launch_log':str(log),
   'CPU_only':True,'query_truth_authorized':False,'scientific_fallback':False,'paper_reproduction_claimed':False,
   'contrastive_execution':'full_batch','independent_review':rr,'fixture_only_not_actual_authority':True}
  if mode=='wrong_root':a['authorizer']='not_root'
  if mode=='wrong_review':
   review['verdict']='PENDING';a['independent_review']=write(directory/'review.json',review)
  if mode=='wrong_config_reference':a['config']={'path':str((directory/'wrong_config.json').relative_to(ROOT)),'sha256':crc}
  if mode=='symlink_review':
   link=directory/'review_link.json';link.symlink_to('review.json');a['independent_review']={'path':str(link.relative_to(ROOT)),'sha256':rr['sha256']}
  ar=write(directory/'root.json',a)
  args.admission=str(ROOT/ar['path']);args.expected_admission_sha256=ar['sha256']
  marker={'implementation_sha256':args.expected_source_sha256,'stage':'retrieve','root_admission':ar,'TMPDIR':str(tmp)}
  write(out/'exec_admission.json',marker)
  if mode=='missing_marker':(out/'exec_admission.json').unlink()
  if mode=='wrong_marker':
   marker['stage']='fit';write(out/'exec_admission.json',marker)
  if mode=='symlink_marker':
   (out/'exec_admission.json').rename(out/'real_marker.json');(out/'exec_admission.json').symlink_to('real_marker.json')
  if mode=='missing_root':(directory/'root.json').unlink()
  if mode=='symlink_root':
   link=directory/'root_link.json';link.symlink_to('root.json');args.admission=str(link)
  if mode=='missing_review':(directory/'review.json').unlink()
  if mode=='wrong_source':args.expected_source_sha256='f'*64
  if mode=='missing_source':args.expected_source_sha256=None
  if mode=='symlink_source':
   link=directory/'source.py';link.symlink_to(m.SCRIPT);stack.enter_context(patch.object(m,'SCRIPT',link))
  if mode=='wrong_config_argument':args.expected_config_sha256='f'*64
  if mode=='missing_config_argument':args.expected_config_sha256=None
  if mode=='wrong_tmp':envtmp=str(tmp)+'_wrong'
  else:envtmp=str(tmp)
  if mode=='not_native':args.native=False
  if mode=='not_execute':args.execute=False
  if mode=='existing_receipt':(out/'receipt.json').write_text('KEEP_EXISTING_RECEIPT')
  if mode=='existing_exit':(out/'exit_status').write_text('KEEP_EXISTING_EXIT')
  if mode=='dangling_receipt':(out/'receipt.json').symlink_to('missing_receipt_target')
  if mode=='dangling_exit':(out/'exit_status').symlink_to('missing_exit_target')
  if mode=='missing_output':
   (out/'exec_admission.json').unlink();out.rmdir()
  if mode=='symlink_output':
   target=out.with_name(out.name+'_real');out.rename(target);out.symlink_to(target.name,target_is_directory=True)
  argv=[]
  if args.execute:argv+=['--execute']
  if args.native:argv+=['--native']
  argv+=['--parent',args.parent,'--stage',args.stage,'--run-id',args.run_id]
  for key in ('expected_source_sha256','expected_config_sha256','admission','expected_admission_sha256'):
   value=getattr(args,key)
   if value is not None:argv+=['--'+key.replace('_','-'),value]
  check('real_initial_policy_'+mode,m.policy()==fc)
  if mode=='missing_config':cfg.unlink()
  else:cfg.write_bytes(cfg.read_bytes()+b'\n')
  if mode=='injected_policy':stack.enter_context(patch.object(m,'policy',side_effect=RuntimeError('fresh independent early policy rejection')))
  before=snapshot(out)
  with patch.dict(os.environ,{'TMPDIR':envtmp}):
   owner=m.existing_entry_ownership(args)
   if mode in ('byte_drift','missing_config','injected_policy'):
    check('secure_owned_scope_proven_'+mode,owner is not None)
    code=m.main(argv);result=json.loads((out/'receipt.json').read_text())
    check('real_main_owned_policy_terminal_'+mode,code==1 and (out/'exit_status').read_text()=='1\n' and result['failure_stage']=='main_policy_preflight' and result['root_actual_execution'] is False)
    check('rejected_config_never_supplies_scope_or_limit_'+mode,result['scope_established_from_reviewed_source_contract'] is True and result['rejected_config_used_for_scope_or_limit'] is False and (out/'receipt.json').stat().st_size<=65536)
    if mode=='missing_config':check('missing_config_observed_digest_null',result['config_sha256'] is None)
    else:check('policy_failure_reviewed_and_observed_config_digests_separate_'+mode,result['reviewed_config_sha256']==crc and result['config_sha256']!=crc)
    reject('owned_terminal_not_reused_'+mode,lambda:m.main(argv))
    case_results[mode]={'terminal_receipt':m.reference(out/'receipt.json'),'exit_status':m.reference(out/'exit_status'),'ownership_proven':True}
   else:
    failure=reject('real_main_unproven_or_terminal_rejected_'+mode,lambda:m.main(argv))
    after=snapshot(out)
    if mode=='dangling_exit':
     check('counterexample_existing_dangling_exit_owner_accepted',owner is not None)
     check('counterexample_existing_dangling_exit_creates_receipt',after!=before and (out/'receipt.json').is_file() and failure['type']=='FileExistsError')
     issues.append({'id':'DANGLING_EXIT_STATUS_NOT_TREATED_AS_EXISTING_TERMINAL','priority':2,
      'source':'src/arggnn/native_argformer_tsinfer_fit_v4.py','lines':[893,778,807],
      'failure':failure,'fixture_output':str(out.relative_to(ROOT)),
      'observation':'An existing dangling exit_status symlink passes .exists() checks. Real main and real policy rejection create receipt.json, then open(x) for exit_status fails. Thus existing terminal directory entries do not reliably prevent amendment.'})
    elif mode=='dangling_receipt':
     check('dangling_receipt_writer_does_not_amend_output',after==before)
     check('counterexample_existing_dangling_receipt_owner_accepted',owner is not None)
    else:
     check('secure_ownership_denied_'+mode,owner is None)
     check('no_output_creation_or_amendment_'+mode,after==before)
    case_results[mode]={'ownership_proven':owner is not None,'output_amended':after!=before,'failure':failure}
for mode in ('byte_drift','missing_config','injected_policy','missing_marker','wrong_marker','symlink_marker','wrong_root','missing_root','symlink_root','wrong_source','missing_source','symlink_source','wrong_config_argument','missing_config_argument','wrong_config_reference','wrong_review','missing_review','symlink_review','wrong_tmp','not_native','not_execute','existing_receipt','existing_exit','missing_output','symlink_output','dangling_receipt','dangling_exit'):
 main_case(mode)
details['entry_cases']=case_results

# Fresh small original retrieval kernel; all references retained, raw k1/k5.
import numpy as np
ns=m.compile_retrieval(c)
query=np.array([[1,0],[0,0],[0,1]],dtype=np.float32)
corpus=np.array([[1,0],[1,0],[1,0],[0,1],[0,1],[-1,0],[0,-1],[0,0]],dtype=np.float32)
ids=np.array([17,2,9,11,8,4,30,5],dtype=np.int64);labels=np.array([0,1,2,3,0,1,2,3],dtype=np.int64)
scores,selected,ties=ns['exhaustive_neighbors'](query,corpus,ids,labels,1,3,6)
dense=query@corpus.T
expected=np.array([ids[np.lexsort((ids,-row))][:6] for row in dense])
check('fresh_full_reference_exhaustive_kernel_chunked_ties',np.array_equal(selected,expected))
p1=ns['hard_votes'](selected,ids,labels,1);p5=ns['hard_votes'](selected,ids,labels,5)
check('fresh_raw_k1_k5_votes_and_no_floor',np.array_equal(p1[0],[0,1,0,0]) and np.any(p1==0) and np.allclose(p5.sum(1),1))
check('zero_query_exact_ties_original_IDs',selected[1].tolist()==sorted(ids.tolist())[:6] and ties[1].sum()==8)
reject('strict_v4_empty_inventory_rejected',lambda:m.parse_runtime_inventory(c,{},{}))
reject('strict_v4_helper_unregistered_runtime_rejected',lambda:m.additive_helper_paths(c,{'additive_helpers':[]}))
for name in ('numpy.pyc','composer.so','sitecustomize.pth','torch.pyo'):check('executable_shadow_suffix_still_denied_'+name,m.import_executable_name(name))
details['fresh_handmade_kernel']={'query_rows':3,'pure_reference_rows':8,'exhaustive_pairs':24,'native_fit_evidence':False,'all_reference_rows_used':True,'primary_k':1,'sensitivity_k':5}
details['native_scientific_ASTs']=proof
details['capture_and_execution_boundary']={'genuine_native_inventory_capture_performed':False,'actual_GPU4096_probe_performed':False,
 'next_root_capture_proposal':m.reference(ROOT/'research/reviews/native_argformer_tsinfer_fit_v4_author/ready_runtime_capture_invocation.txt'),
 'next_root_probe_proposal':m.reference(ROOT/'research/reviews/native_argformer_tsinfer_fit_v4_author/ready_probe_invocation.txt'),
 'requires_final_independent_PASS_and_fresh_root_admissions':True}
for p,h in pins.items():check('final_immutable_pin_'+p,m.sha(ROOT/p)==h)
report={'kind':'native_argformer_tsinfer_fit_v4_independent_delta_checks','status':'completed_one_terminal_presence_blocker',
 'check_count':len(checks),'checks':checks,'issues':issues,'details':details,
 'actual_model_fit_executed':False,'released_model_instantiated':False,'actual_scientific_arrays_opened':False,
 'actual_GPU_inventory_context_or_live_ledger_read':False,'actual_native_runtime_inventory_captured':False,
 'authority_activation_performed':False,'runtime_prefix_modified':False,'Git_or_install_performed':False,
 'handmade_module_constants_relocated_only_for_contained_entry_fixtures':True}
write(REVIEW/'handmade_checks.json',report)
print(json.dumps({'status':report['status'],'check_count':len(checks),'issues':issues,'fresh_handmade_kernel':details['fresh_handmade_kernel']},indent=2))
