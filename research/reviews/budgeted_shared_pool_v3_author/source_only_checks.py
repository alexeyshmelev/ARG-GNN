"""Author bounded source/fake-root tests inherited from sealed v2; no production runtime state."""
from __future__ import annotations
import copy,datetime as dt,fcntl,hashlib,json,os,shutil,signal,sys,time,types
from pathlib import Path
from unittest.mock import patch
PROJECT=Path('/workspace-SR003.nfs2/ARG-GNN')
sys.dont_write_bytecode=True;sys.path.insert(0,str(PROJECT/'src'))
from arggnn import budgeted_shared_pool_v3 as s,gpu_pool_v1 as g,project_budget_v4 as b
REVIEW=PROJECT/'research/reviews/budgeted_shared_pool_v3_author'
F=REVIEW/'f';CHECKS=[];DETAILS={};MIB=1048576
NATIVE='arggnn.native_argformer_tsinfer_fit_v5';CENSUS='arggnn.census_biological_fit_v1';CENSUS2='arggnn.census_biological_fit_v2';PROFILE='arggnn.census_biological_device_profile_v1'
PINS = {'src/arggnn/budgeted_shared_pool_v1.py': '89e7da1e312e88aad18c3e09a3d50b1c31ecd0d4d1d62ef775bfdd0d15498da2', 'configs/budgeted_shared_pool_v1.json': '89f1847eb05c69d17332966d36f65a9ba55d1891bcb0e824282e1344275cfa69', 'research/reviews/budgeted_shared_pool_v1_author/author_receipt.json': 'c8e616697a2f54bd2cb9c4769596842913d84c5775fc0db8d0572cf834a36939', 'src/arggnn/gpu_pool_v1.py': 'ae1e0494ab672cfa3017759460eb06a4b4789eaef54928496c3c80314adcc0e1', 'src/arggnn/project_budget_v4.py': 'a093c7d86f57fee569cb99f4712250e9505a1119f3950fa44282c7ba8c57df47', 'configs/project_budget_v4.json': '5788f87d11825cf4fbe94a25abe056132107b155848111354e70637334e91d63', 'src/arggnn/budgeted_shared_pool_v2.py': '16812adfc6a672ad0be157ff20b089597c9f0134ede24a14e5d7522bc4f32f72', 'configs/budgeted_shared_pool_v2.json': '42769ba1fc4e654c5aeba6370b74708549f05ff99a73f09022b208821a330327', 'src/arggnn/native_argformer_tsinfer_fit_v5.py': '1f05220ddfe7ba2fd3e55b079088eb361c6901323a9520145b0f037f6cfd22e0', 'configs/native_argformer_tsinfer_fit_v5.json': '6d58f126a9fa16f9dfc04956215e85569f9696cb3b12c06f63fa458d8bb0986f', 'src/arggnn/census_biological_fit_v1.py': '84f7f39c94003256cb22c905f610d5ce4a965d1ca733f483e22367f91dbc90ce', 'configs/census_biological_fit_v1.json': 'b6f70c6d76421662713028b23deedaef071fb6f0d82a0ef6a9aedaa43f5fcd9e', 'src/arggnn/budgeted_shared_pool_v3.py': '2dd01da79462e0c469a1671bd509f0f0ce0b7ac700e63dcd4922db9fc9e9ad13', 'configs/budgeted_shared_pool_v3.json': 'd0e3dba441ecad299ff61b53a262e38b767e1d78118c7d351c0f96888b077c28', 'src/arggnn/census_biological_fit_v2.py': 'c21884ad90a2c940b49968353545edc52d345c0c7bc10fd87c492c54d4b38c34', 'configs/census_biological_fit_v2.json': 'a512816bbab61e80052e7466613d2de4c861b77ef61d0d35cd50fbf81905c621', 'src/arggnn/census_biological_device_profile_v1.py': 'f8915b311b0c1fbb219ff7265a37c46b037e1c49a95d2be85bd754623fd3d69f', 'configs/census_biological_device_profile_v1.json': '36c782b162881391822a528f7aef039a2c61fa4d232bfb5591da7bf9fe87f8d0', 'research/reviews/budgeted_shared_pool_v2_author/author_receipt.json': '553bda39af966ed6177d7fc802817a63afc4b4d99ba91d6b97615dc4e6a815cb'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check(name,value=True):
 if not value:raise AssertionError(name)
 CHECKS.append(name)
def rejected(name,call,contains=None):
 try:call()
 except (ValueError,RuntimeError,FileExistsError,OSError) as e:
  if contains and contains not in str(e):raise AssertionError(name+': '+str(e))
  CHECKS.append(name);return str(e)
 raise AssertionError(name+': unexpected acceptance')
def immutable():
 for path,digest in PINS.items():check('immutable_'+path,sha(PROJECT/path)==digest)
class Child:
 all=[]
 def __init__(self,argv,**kw):
  assert kw['shell'] is False and kw['close_fds'] is True and kw['start_new_session'] is True
  assert kw['env']['CUDA_VISIBLE_DEVICES']=='GPU-00000000-0000-0000-0000-000000000002'
  assert kw['env']['ARGGNN_LOGICAL_JOB_NAME'] in ('j0','j1','j2','j3')
  assert kw['env']['ARGGNN_PHYSICAL_GPU']=='2' and kw['env']['PYTHONNOUSERSITE']=='1'
  assert kw['pass_fds']==(int(kw['env']['ARGGNN_GPU_LEASE_FD']),)
  self.pid=900000+len(Child.all);self.argv=list(argv);self.kw=kw;self.returncode=None
  a,z=os.pipe();os.close(z);self.stdout=os.fdopen(a,'rb',buffering=0);Child.all.append(self)
 def poll(self):return self.returncode
def processes(_lease,_controller,_known=None):
 table={q.pid:{'pid':q.pid,'uid':os.getuid(),'state':'S','ppid':os.getpid(),'pgid':q.pid,'sid':q.pid,'start_ticks':q.pid,'lease_fd_present':True,'controller_descendant':True} for q in Child.all if q.returncode is None}
 table.update(copy.deepcopy(EXTRA_PROC))
 return table
EXTRA_PROC={}
def main():
 immutable();check('no_scientific_libraries_imported','torch' not in sys.modules and 'numpy' not in sys.modules);assert not F.exists();F.mkdir()
 oldalarm=signal.getsignal(signal.SIGALRM);controllers=[];held=[];started=time.monotonic()
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('35-second author fixture ceiling')))
 signal.alarm(35)
 try:
  def text(path,value):
   q=F/path;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(value)
   return b.reference(F,q)
  def data(path,value):return text(path,json.dumps(value,indent=2)+'\n')
  def ref(path):return b.reference(F,F/path)
  for path in ('src/arggnn/gpu_pool_v1.py','src/arggnn/project_budget_v4.py','src/arggnn/budgeted_shared_pool_v3.py'):text(path,(PROJECT/path).read_text())
  cs=text('src/arggnn/biological_bank_cache_v1.py','# inert never imported source fixture\n')
  cc=data('configs/biological_bank_cache_v1.json',{'GPU_budget_bytes':4*MIB})
  additional={}
  for module,contract in s.FIXED_POLICY['additional_worker_contracts'].items():
   for field in ('source','config'):
    record=contract[field];text(record['path'],(PROJECT/record['path']).read_text())
   additional[module]=copy.deepcopy(contract)
  profile_source=text('src/arggnn/census_biological_device_profile_v1.py','# inert profile fixture source, never imported\n')
  rb=text('scripts/project_env.sh','# handmade inert runtime reference\n')
  now=dt.datetime.now(dt.timezone.utc)
  human=data('meta/human.json',{'kind':'human_project_storage_authorization','project_root':str(F),'authorized_hard_limit_bytes':1000000000000,'future_scheduling_ceiling_bytes':950000000000})
  snapshot=data('meta/snapshot.json',{'schema':'biological_candidate_storage_snapshot_v1','status':'complete','returncode':0,'project_root':str(F),'command':['du','-s','-B1',str(F)],'used_bytes':1000,'completed_utc':(now-dt.timedelta(seconds=1)).isoformat()})
  api=types.ModuleType('_independent_private_budget')
  exec(compile((PROJECT/'src/arggnn/project_budget_v4.py').read_text(),'<unchanged-private-budget-code>','exec'),api.__dict__)
  api.ROOT=F;api.SOURCE=F/'src/arggnn/project_budget_v4.py';api.CONFIG=F/'configs/project_budget_v4.json'
  api.FIXED_POLICY=copy.deepcopy(b.FIXED_POLICY);api.FIXED_POLICY.update(root=str(F),human_authorization=human)
  api.FIXED_POLICY['legacy_invariants'].update(snapshot_sha256=snapshot['sha256'],baseline_bytes=1000,external_reserve_bytes=2000)
  bc=data('configs/project_budget_v4.json',api.FIXED_POLICY)
  policy=copy.deepcopy(s.FIXED_POLICY);policy.update(root=str(F),human_authorization=human,budget_config=bc,cache_source=cs,cache_config=cc,socket_relative_template='.runtime/gpup/{pool}/s',additional_worker_contracts=additional)
  cfg=data('configs/budgeted_shared_pool_v3.json',policy);impl=ref('src/arggnn/budgeted_shared_pool_v3.py')
  common={'implementation':ref('src/arggnn/project_budget_v4.py'),'config':bc,'human_authorization':human}
  br=data('meta/budget_review.json',{'kind':'project_budget_v4_independent_review','review_role':'independent_reviewer','verdict':'PASS',**common,'actual_apply_performed':False,'handmade_fixture_only':True})
  def reserve(path,requests):return data(path,{'kind':'project_budget_v4_root_reserve_admission','root_authorizer':'root_agent','active':True,**common,'all_budget_writers_use_scheduler_mutex':True,'no_legacy_mutation_authorized':True,'GPU_authorized':False,'scientific_execution_authorized':False,'global_rebase_authorized':False,'reservations':requests,'independent_review':br,'handmade_fixture_only':True})
  requests=[];requests_by={}
  for name in ('a','b','c'):
   rows=[{'key':'budgeted_shared_pool_v3/'+name+'/controller','output':str(F/'.runtime/gpup'/name),'auxiliary_scopes':[],'reserved_bytes':g.MIN_CONTROLLER_RESERVE+4000,'consumer_bindings':[impl,cfg]}]
   for j in range(4):
    rows.append({'key':'budgeted_shared_pool_v3/'+name+'/j'+str(j),'output':str(F/'outputs'/(name+'_j'+str(j))),'auxiliary_scopes':[],'reserved_bytes':5000,'consumer_bindings':[impl,cfg,cs,cc,profile_source,*[r for v in additional.values() for k,r in v.items() if k in ('source','config')]]})
   requests.extend(rows);requests_by[name]=rows
  reserve_ref=reserve('meta/reserve.json',requests);stored={x['key']:{**x,'root_admission':reserve_ref} for x in requests}
  legacy={**api.FIXED_POLICY['legacy_invariants'],'root':str(F),'controllers':{},'jobs':{}}
  lr=data(api.LEGACY,legacy);supp=api.empty_ledger(F,api.FIXED_POLICY,common);supp['reservations']=stored;sr=data(api.SUPPLEMENTAL,supp)
  legacy_bytes=(F/api.LEGACY).read_bytes();supp_bytes=(F/api.SUPPLEMENTAL).read_bytes()
  (F/'.runtime/locks').mkdir(parents=True);(F/api.LOCK).touch();(F/'outputs').mkdir()
  plans={}
  def pool_authority(name,index):
   jobs=[]
   for j,peak in enumerate((40,35,30,10)):
    module=CENSUS2 if j==3 else CENSUS if j==1 else PROFILE if j==2 else s.CACHE_MODULE
    worker_source=additional[module]['source'] if module in (CENSUS,CENSUS2) else profile_source if module==PROFILE else cs
    worker_config=additional[module]['config'] if module in (CENSUS,CENSUS2) else additional[CENSUS]['config'] if module==PROFILE else cc
    spec={'job':'j'+str(j),'model':'independent inert full-cache worker','seed':17+j,'module':module,'args':(['--admission','handmade/'+module+'.admission.json','--expected-admission-sha256','a'*64] if module in (CENSUS,CENSUS2) else ['--inert-fixture',str(j)]),'output':requests_by[name][j+1]['output'],'peak_mib':peak,'storage_bytes':5000,'policy':'queue','log_limit_bytes':1000}
    memory={'model_optimizer_baseline':1,'cached_bank':4,'uncached_bank_and_retained_references':1,'autograd':peak-7,'runtime_allocator':1}
    jobs.append({'spec':spec,'reservation':stored[requests_by[name][j+1]['key']],'module_source':worker_source,'worker_bindings':[worker_config],'memory_components_mib':memory,'argv_sha256':s.digest([str(F/'.runtime/env/bin/python'),'-m',module,*spec['args']])})
   ap='meta/pool_'+name+'.json'
   recipe=[str(F/'.runtime/env/bin/python'),'-m','arggnn.budgeted_shared_pool_v3','--start','--admission',ap,'--expected-admission-sha256','<ROOT_ADMISSION_SHA256>','--expected-source-sha256',impl['sha256'],'--expected-config-sha256',cfg['sha256']]
   plan={'pool':{'name':name,'physical_gpu':index,'headroom_mib':10,'controller_reservation':stored[requests_by[name][0]['key']],'storage_snapshot':snapshot,'python':str(F/'.runtime/env/bin/python'),'start_recipe_sha256':s.digest(recipe)},'jobs':jobs,'initial_ledger_pins':{'legacy_ledger_sha256':lr['sha256'],'supplemental_ledger_sha256':sr['sha256']},'runtime_bindings':[rb]}
   review=data('meta/review_'+name+'.json',{'kind':'budgeted_shared_pool_v3_independent_review','review_role':'independent_reviewer','verdict':'PASS','implementation':impl,'config':cfg,'legacy_pool_source':policy['legacy_pool_source'],'budget_source':policy['budget_source'],'budget_config':bc,'human_authorization':human,'reviewed_plan_sha256':s.digest(plan),'actual_live_lock_or_gpu_execution_performed':False,'original_sources_unchanged':True,'handmade_fixture_only':True})
   auth=data(ap,{'kind':'budgeted_shared_pool_v3_root_admission','root_authorizer':'root_agent','active':True,'root':str(F),'implementation':impl,'config':cfg,'legacy_pool_source':policy['legacy_pool_source'],'budget_source':policy['budget_source'],'budget_config':bc,'human_authorization':human,'future_pools_only':True,'no_live_pool_or_worker_patch':True,'no_legacy_ledger_mutation':True,'no_duplicate_legacy_reservations':True,'scientific_execution_requires_worker_authority':True,'not_before_utc':(now-dt.timedelta(minutes=1)).isoformat(),'expires_utc':(now+dt.timedelta(hours=1)).isoformat(),'plan':plan,'independent_review':review})
   return s.ReviewedPlan(F,policy,impl,cfg,auth,api=api)
  for name,index in zip(('a','b','c'),(2,3,4)):plans[name]=pool_authority(name,index)
  # Root metadata counterexamples use fresh immutable fake refs only.
  auth=copy.deepcopy(plans['a'].authority)
  auth['expires_utc']=(now-dt.timedelta(seconds=1)).isoformat()
  ar=data('meta/expired_authority.json',auth)
  rejected('expired_root_authority_rejected',lambda:s.ReviewedPlan(F,policy,impl,cfg,ar,api=api),'outside')
  auth=copy.deepcopy(plans['a'].authority);rev=b.bound(F,auth['independent_review'],True);rev['verdict']='PENDING'
  auth['independent_review']=data('meta/pending_review.json',rev);ar=data('meta/pending_authority.json',auth)
  rejected('pending_independent_review_cannot_activate',lambda:s.ReviewedPlan(F,policy,impl,cfg,ar,api=api),'PASS')
  auth=copy.deepcopy(plans['a'].authority);auth['plan']['jobs'][0]['spec']['args'].append('--unreviewed')
  ar=data('meta/changed_plan.json',auth)
  rejected('changed_plan_digest_rejected',lambda:s.ReviewedPlan(F,policy,impl,cfg,ar,api=api),'PASS')
  auth=copy.deepcopy(plans['a'].authority);auth['plan']['jobs'][0]['argv_sha256']='f'*64
  rev=b.bound(F,auth['independent_review'],True);rev['reviewed_plan_sha256']=s.digest(auth['plan'])
  auth['independent_review']=data('meta/bad_command_review.json',rev);ar=data('meta/bad_command_authority.json',auth)
  rejected('incorrect_exact_worker_command_hash_rejected',lambda:s.ReviewedPlan(F,policy,impl,cfg,ar,api=api),'command hash')
  check('finite_allowlist_preserves_v2_and_adds_only_census_fit2',set(s.FIXED_POLICY['allowed_modules'])==set(g.MODULES)|{s.CACHE_MODULE,NATIVE,CENSUS,CENSUS2,PROFILE})
  auth=copy.deepcopy(plans['a'].authority);item=auth['plan']['jobs'][0]
  item['spec']['module']=NATIVE;item['module_source']=additional[NATIVE]['source'];item['worker_bindings']=[additional[NATIVE]['config']]
  item['argv_sha256']=s.digest([auth['plan']['pool']['python'],'-m',NATIVE,*item['spec']['args']])
  rev=b.bound(F,auth['independent_review'],True);rev['reviewed_plan_sha256']=s.digest(auth['plan'])
  auth['independent_review']=data('meta/native_transport_only_review.json',rev);ar=data('meta/native_not_shared.json',auth)
  rejected('native5_exact_source_is_eligible_but_shared_provenance_blocks_plan',lambda:s.ReviewedPlan(F,policy,impl,cfg,ar,api=api),'exclusive launcher provenance')
  auth=copy.deepcopy(plans['a'].authority);auth['plan']['jobs'][1]['worker_bindings']=[cc]
  rev=b.bound(F,auth['independent_review'],True);rev['reviewed_plan_sha256']=s.digest(auth['plan']);auth['independent_review']=data('meta/missing_census_config_review.json',rev);ar=data('meta/missing_census_config.json',auth)
  rejected('census_exact_config_binding_is_required',lambda:s.ReviewedPlan(F,policy,impl,cfg,ar,api=api),'additional worker config')
  auth=copy.deepcopy(plans['a'].authority);auth['plan']['jobs'][3]['worker_bindings']=[cc]
  rev=b.bound(F,auth['independent_review'],True);rev['reviewed_plan_sha256']=s.digest(auth['plan']);auth['independent_review']=data('meta/missing_census2_config_review.json',rev);ar=data('meta/missing_census2_config.json',auth)
  rejected('census2_exact_config_binding_is_required',lambda:s.ReviewedPlan(F,policy,impl,cfg,ar,api=api),'additional worker config')
  check('original_methods_inherited',all(getattr(s.BudgetedController,k) is getattr(g.Controller,k) for k in ('start','_process_scan','_memory','submit','_launch_queue','_drain_logs','_signal_own','tick','shutdown','_release_handles','close','dispatch','run')))
  # Independent descriptor proof/lifetime checks on the contained scheduler.
  rejected('unlocked_scheduler_descriptor_not_accepted',lambda:next(s._owned_scheduler_mutex(F).__enter__() for _ in [0]),'not proved')
  scheduler=F/api.LOCK;fd=os.open(scheduler,os.O_RDWR);held.append(fd)
  fcntl.flock(fd,fcntl.LOCK_SH)
  rejected('shared_FLOCK_not_accepted',lambda:s._owned_scheduler_mutex(F).__enter__(),'not proved')
  fcntl.flock(fd,fcntl.LOCK_EX)
  check('private_checker_preserves_code_object_and_global_bindings',s._owned_checker(api).__code__ is api.check_reservation.__code__ and api.check_reservation.__globals__['mutex'] is api.mutex and b.check_reservation.__globals__['mutex'] is b.mutex)
  probe=os.open(scheduler,os.O_RDWR)
  try:
   with s._owned_scheduler_mutex(F):
    rejected('borrowed_EX_lock_blocks_separate_open_description',lambda:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB))
   rejected('closing_duplicate_does_not_unlock_original_OFD',lambda:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB))
   try:
    with s._owned_scheduler_mutex(F):raise ValueError('fixture owned boundary failure')
   except ValueError:pass
   rejected('exception_cleanup_does_not_unlock_original_OFD',lambda:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB))
  finally:os.close(probe);os.close(fd);held.remove(fd)
  fd=os.open(scheduler,os.O_RDWR);held.append(fd);fcntl.flock(fd,fcntl.LOCK_EX)
  os.link(scheduler,F/'meta/hardlink')
  rejected('multiply_linked_scheduler_rejected',lambda:s._owned_scheduler_mutex(F).__enter__(),'singly linked')
  (F/'meta/hardlink').unlink();os.close(fd);held.remove(fd)
  rows=[{'index':i,'uuid':'GPU-00000000-0000-0000-0000-'+('%012d'%i),'used_mib':0,'total_mib':100,'utilization':0,'compute':[]} for i in g.ALLOWED_GPUS]
  free=[1000000000000]
  controllers_all=[s.BudgetedController(plans[n],inventory=lambda:copy.deepcopy(rows),proc_reader=processes if n=='a' else lambda *_:{},available_bytes=lambda:free[0]) for n in ('a','b','c')]
  a,z,c=controllers_all
  before_bmutex=api.check_reservation.__globals__['mutex']
  with patch.object(g,'enable_child_subreaper',lambda:None),patch.object(g.subprocess,'Popen',Child):
   a.storage.preflight()
   calls=[];original_flock=fcntl.flock
   def record_flock(fd,operation):calls.append(operation);return original_flock(fd,operation)
   with patch.object(fcntl,'flock',side_effect=record_flock):a.start()
   controllers.append(a)
   check('original_acquire_device_has_only_one_scheduler_EX_and_four_device_attempts',calls==[fcntl.LOCK_EX]+[fcntl.LOCK_EX|fcntl.LOCK_NB]*4)
   saved_lease=(a.lease.fileno(),os.fstat(a.lease.fileno()).st_ino,a.lease_sha)
   calls.clear()
   with patch.object(fcntl,'flock',side_effect=record_flock):
    with api.mutex(F):
     mark=len(calls);a.storage._check(plans['a'].reservations()[0],owned=True)
     check('owned_storage_checker_never_nests_flock_or_unlock',len(calls)==mark)

   z.storage.preflight();z.start();controllers.append(z)
   rejected('third_physical_device_rejected_by_original_acquire_device',c.start,'Two distinct')
   check('failed_third_start_closes_socket_and_does_not_keep_lease',c.listener is None and c.lease is None and not c.socket_path.exists())
   check('start_preserves_original_budget_mutex_binding',api.check_reservation.__globals__['mutex'] is before_bmutex)
   for name in ('j0','j1','j2','j3'):a.submit(copy.deepcopy(plans['a'].jobs[name]['spec']))
   check('cache_and_census_transport_share_original_device_with_pending_CUDA_FIFO',len(Child.all)==2 and a._memory()['active_peak_mib']==75 and a.jobs['j2']['state']=='queued' and a.jobs['j3']['state']=='queued')
   check('exact_worker_argv_and_inherited_lease_fd',all(q.argv==[plans['a'].plan['pool']['python'],'-m',plans['a'].jobs[q.kw['env']['ARGGNN_LOGICAL_JOB_NAME']]['spec']['module'],*plans['a'].jobs[q.kw['env']['ARGGNN_LOGICAL_JOB_NAME']]['spec']['args']] for q in Child.all))
   for label,field,value in [('module','module','arggnn.biological_candidate'),('argv','args',['--altered']),('output','output',str(F/'outputs/foreign')),('memory','peak_mib',1)]:
    v=copy.deepcopy(plans['a'].jobs['j2']['spec']);v[field]=value
    rejected('raw_protocol_unregistered_'+label,lambda v=v:a.dispatch({'v':1,'op':'submit','spec':v}),'root job/output/args')
   row=rows[0];original_uuid=row['uuid'];row['uuid']='GPU-00000000-0000-0000-0000-000000000222'
   rejected('actual_UUID_change_rejected',a._memory,'UUID');row['uuid']=original_uuid
   row.update(used_mib=10,compute=[{'pid':999999,'name':'foreign','used_mib':10}])
   check('foreign_GPU_PID_blocks_admission','foreign_compute_pids' in a._memory()['blocked_by'])
   row.update(used_mib=41,compute=[{'pid':Child.all[0].pid,'name':'owned','used_mib':41}]);a._process_scan()
   check('attributed_worker_peak_overrun_blocks_admission','worker_peak_overrun' in a._memory()['blocked_by'])
   row.update(used_mib=0,compute=[])
   EXTRA_PROC[987654]={'pid':987654,'uid':os.getuid(),'state':'S','ppid':1,'pgid':987654,'sid':987654,'start_ticks':987654,'lease_fd_present':True,'controller_descendant':False}
   a._process_scan();check('unknown_lease_holder_blocks_admission','unattributed_descendants_or_lease_holders' in a._memory()['blocked_by']);EXTRA_PROC.clear();a._process_scan()
   # Actual native lease-validator code on a contained fake held FD/env, no CUDA/title calls.
   import importlib.util
   sp=importlib.util.spec_from_file_location('_fake_pool_native_lease_guard',PROJECT/'src/arggnn/native_argformer_exec.py')
   ng=importlib.util.module_from_spec(sp);sp.loader.exec_module(ng)
   with patch.object(ng,'ROOT',F):transport=ng.validate_lease(Child.all[0].kw['env'])
   check('native_guard_accepts_shared_FD_UUID_pool_job_transport_metadata',transport['launcher_metadata']['job']==a.pool_job and transport['lease_fd']==a.lease.fileno())
   for label,path in [('sealed_native_source',additional[NATIVE]['source']['path']),('census_source',additional[CENSUS]['source']['path']),('census_config',additional[CENSUS]['config']['path']),('census2_source',additional[CENSUS2]['source']['path']),('census2_config',additional[CENSUS2]['config']['path'])]:
    p=F/path;raw=p.read_bytes();p.write_bytes(raw+b'\n')
    rejected('additional_pinned_'+label+'_drift_blocks_guard',a._lease_check,'bytes changed');p.write_bytes(raw)
   # Byte-bound capability and source changes are rejected at the real lease boundary.
   for label,path in [('worker_source',cs['path']),('companion_config',cfg['path']),('runtime_binding',rb['path']),('worker_config',cc['path']),('independent_review',plans['a'].authority['independent_review']['path'])]:
    p=F/path;raw=p.read_bytes();p.write_bytes(raw+b'\n')
    rejected('prelaunch_bound_'+label+'_drift_rejected',a._lease_check,'bytes changed');p.write_bytes(raw)
   changed=copy.deepcopy(supp);del changed['reservations'][plans['a'].jobs['j3']['reservation']['key']];data(api.SUPPLEMENTAL,changed)
   rejected('all_own_reservations_checked_even_for_unlaunched_job',a.storage.revalidate,'own registered');(F/api.SUPPLEMENTAL).write_bytes(supp_bytes)
   extra={'key':'unrelated','output':str(F/'outputs/unrelated'),'auxiliary_scopes':[],'reserved_bytes':12345,'consumer_bindings':[impl,cfg]}
   exref=reserve('meta/unrelated.json',[extra]);changed=copy.deepcopy(supp);changed['reservations'][extra['key']]={**extra,'root_admission':exref};data(api.SUPPLEMENTAL,changed)
   view=a.storage.revalidate();check('legitimate_unrelated_global_charge_reaccounted',view['supplemental_reserved_bytes']==sum(v['reserved_bytes'] for v in stored.values())+12345)
   rejected('first_ledger_hash_pair_cannot_be_rebased',a.storage.preflight,'first-read');(F/api.SUPPLEMENTAL).write_bytes(supp_bytes)
   huge={**extra,'key':'huge','reserved_bytes':950000000000};huref=reserve('meta/huge.json',[huge]);changed=copy.deepcopy(supp);changed['reservations']['huge']={**huge,'root_admission':huref};data(api.SUPPLEMENTAL,changed)
   rejected('current_global_950GB_ceiling_blocks',a.storage.revalidate,'ceiling');(F/api.SUPPLEMENTAL).write_bytes(supp_bytes)
   free[0]=1;rejected('current_all_reservations_free_space_blocks',a.storage.revalidate,'free-space');free[0]=1000000000000
   # Mark leaders ended but retain a genuine recorded descendant in fake process metadata.
   child=Child.all[0];child.returncode=0
   EXTRA_PROC[900010]={'pid':900010,'uid':os.getuid(),'state':'S','ppid':child.pid,'pgid':child.pid,'sid':child.pid,'start_ticks':900010,'lease_fd_present':True,'controller_descendant':True}
   a.jobs['j0']['process_identities'].append(copy.deepcopy(EXTRA_PROC[900010]))
   a.tick();check('leader_exit_retains_reservation_until_descendant_drains',a.jobs['j0']['state']=='draining' and len(Child.all)==2)
   EXTRA_PROC.clear();a.tick()
   check('natural_drain_launches_FIFO_jobs_with_same_lease',a.jobs['j0']['state']=='succeeded' and a.jobs['j2']['state']=='running' and a.jobs['j3']['state']=='running' and len(Child.all)==4 and saved_lease==(a.lease.fileno(),os.fstat(a.lease.fileno()).st_ino,a.lease_sha))
   for q in Child.all:q.returncode=0
   a.tick();check('all_fake_workers_drain_successfully',all(j['state']=='succeeded' for j in a.jobs.values()))
   for ctl in controllers:check('natural_shutdown_closes_original_handles_'+ctl.pool,ctl.shutdown()['lifecycle']=='closed');ctl.close()
   controllers.clear()
   check('normal_cleanup_removes_only_owned_sockets',not a.socket_path.exists() and not z.socket_path.exists())
   repeated=s.BudgetedController(plans['a'],inventory=lambda:copy.deepcopy(rows),proc_reader=lambda *_:{},available_bytes=lambda:free[0])
   rejected('existing_pool_is_never_adopted',repeated.start)
  check('legacy_and_supplemental_bytes_unchanged_after_all_flow',(F/api.LEGACY).read_bytes()==legacy_bytes and (F/api.SUPPLEMENTAL).read_bytes()==supp_bytes)
  check('private_mutex_binding_restored_after_success_and_error',api.check_reservation.__globals__['mutex'] is api.mutex and b.check_reservation.__globals__['mutex'] is b.mutex)
  DETAILS.update(fake_registered_pools=3,fake_registered_jobs_per_pool=4,inert_Popen_objects=len(Child.all),actual_worker_processes=0,handmade_private_budget_constants_only=True,cache_census_fit1_fit2_and_profile_Popen_transport_only=True,native5_shared_plan_explicitly_blocked=True,actual_CUDA_title_or_scientific_worker_guards_executed=False,fit2_literal_admission_argv_transport_only=True,seconds=time.monotonic()-started)
 finally:
  signal.alarm(0);signal.signal(signal.SIGALRM,oldalarm)
  for ctl in controllers:ctl._release_handles()
  for fd in held:os.close(fd)
  for q in Child.all:
   if not q.stdout.closed:q.stdout.close()
  shutil.rmtree(F)
 immutable()
 return {'kind':'budgeted_shared_pool_v3_author_source_checks','status':'PASS','check_count':len(CHECKS),'checks':CHECKS,'details':DETAILS,'actual_live_lock_or_gpu_execution_performed':False,'actual_live_ledgers_or_device_queries_performed':False,'actual_model_arrays_read':False,'source_edits_or_installs_or_Git':False,'fresh_fake_resources_only':True,'original_sources_unchanged':True}
if __name__=='__main__':
 result=main()
 (REVIEW/'source_checks.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
