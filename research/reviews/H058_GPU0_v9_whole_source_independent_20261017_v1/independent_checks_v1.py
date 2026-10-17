"""Independent bounded source/RAM review. No actual operational or numerical calls."""
from pathlib import Path,PurePosixPath
from types import SimpleNamespace
from datetime import datetime,timezone,timedelta
import ast,builtins,collections,contextlib,copy,hashlib,io,json,os,runpy,signal,sys,time
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');D='research/reviews/H058_GPU0_full_training_successor_source_20261017_v9';OUT='research/reviews/H058_GPU0_v9_whole_source_independent_20261017_v1'
sys.path.insert(0,str(ROOT));checks=[];negatives=0
def ck(name,value=True):
 if not value:raise AssertionError(name)
 checks.append(dict(name=name,status='PASS'))
def reject(name,call,match=None):
 global negatives
 try:call()
 except (ValueError,PermissionError,RuntimeError) as exc:
  if match and match not in str(exc):raise AssertionError((name,str(exc)))
  negatives+=1;ck(name)
 else:raise AssertionError(name+' accepted')
def rf(p):return dict(path=p,sha256=hashlib.sha256((ROOT/p).read_bytes()).hexdigest())
def read(p):return json.loads((ROOT/p).read_bytes())
def emit(nodes,scope):exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes),type_ignores=[])),'independent_exact_AST_RAM','exec'),scope)
def fn(tree,name):return next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name==name)
m=read(D+'/source_manifest.v1.json');h=read(D+'/source_handoff.v9.json')
ck('exact requested handoff',rf(D+'/source_handoff.v9.json')['sha256']=='52834d36c7442766f67316d2b9292801ea0c34c1e03c1c17ed3c7306da6fa24a')
ck('exact requested runtime manifest',rf(D+'/source_manifest.v1.json')['sha256']=='196b662fcfb7ea2d6de57595b830c2e9109be14cbd861ee503bb6789af170004')
for r in m['refs']:
 p=ROOT/r['path'];assert p.resolve()==p and p.is_file()and not p.is_symlink()and p.stat().st_size<=32*2**20 and p.suffix in('.py','.sh','.json','.patch','.txt','.md')and rf(r['path'])==r
ck('all427 sealed source and bounded operational metadata refs exact',len(m['refs'])==427 and m['source_sha256']=={r['path']:r['sha256']for r in m['refs']})
for r in read(D+'/MANIFEST.v9.json')['files']:assert rf(r['path'])=={k:r[k]for k in('path','sha256')}
ck('all final audit files exact')
ot=ast.parse((ROOT/D/'operational_v1.py').read_text());e={n.targets[0].id:ast.literal_eval(n.value)for n in ot.body if isinstance(n,ast.Assign)and isinstance(n.value,ast.Constant)and isinstance(n.value.value,str)}
for k,f in [('_DRIVER_SOURCE','driver_role.source.txt'),('_PRODUCER_SOURCE','producer_role.source.txt'),('_LEAN_MATH_SOURCE','lean_math_reviewed.source.txt'),('_CENSUS_MEMO_BRIDGE_SOURCE','Census_memo_bridge_reviewed.source.txt')]:assert e[k]==(ROOT/D/f).read_text()
ck('all embedded source mirrors exact')
v8=D[:-1]+'8'
for f in('context_factory_v1.py','direct_gpu0_v1.py','observations_gpu_v1.py','producer_role.source.txt'):
 s=(ROOT/D/f).read_text().replace(D,v8).replace(D.replace('/','.'),v8.replace('/','.')).replace('_H058_v9_','_H058_v8_')
 ck('whole AST preserved from v8: '+f,ast.dump(ast.parse(s),include_attributes=False)==ast.dump(ast.parse((ROOT/v8/f).read_text()),include_attributes=False))
ck('exact reviewed math helper',hashlib.sha256(e['_LEAN_MATH_SOURCE'].encode()).hexdigest()=='fa5bddb450d2e531859f8d0c62bc6a1e49f60e82359baf97adede3e5cdb86966')
ck('exact passed Census bridge',e['_CENSUS_MEMO_BRIDGE_SOURCE']==(ROOT/'research/reviews/FLARE_original_VAL002004_conversion_execution_source_20261017_v1/root_memo_bridge_v1.py').read_text())
# Two short author RAM scaffolds reproduce material joins only; no writer flag.
with contextlib.redirect_stdout(io.StringIO()):
 a=runpy.run_path(str(ROOT/D/'full_admission_RAM_checks_v9.py'),run_name='_author_bounded_admission')
 t=runpy.run_path(str(ROOT/D/'targeted_checks_v9.py'),run_name='_author_bounded_groups')
ck('bounded complete author admission scaffold passed',a['result']['status']=='PASS')
ck('bounded author group cleanup scaffold passed',t['result']['status']=='PASS')
c=a['c'];d=t['d'];q=a['q'];cfg=a['cfg'];sc=a['scope'];g=a['g'];cs=a['common_scope'];v4=a['v4'];budget=a['budget'];mem=a['memory']
for label,mutate,match in [
 ('old packet scope',lambda x:x.update(packet_directory='research/reviews/H058_GPU0_v8_ROOT_actual_old'),'Separate fresh'),
 ('old job identity',lambda x:x.update(job_id='H058-GPU0-v8-fulltrain-ram_only'),'Separate fresh'),
 ('VAL truth enabled',lambda x:x.update(VAL_truth_authorized=True),'TRAIN-only'),
 ('old selector',lambda x:x.update(storage_policy_selector_ref={'path':'old.json','sha256':'0'*64}),'storage selector'),
 ('CUDA cap raised',lambda x:x['caps'].update(max_cuda_reserved_bytes=29*2**30),'Exact narrow caps'),
 ('wrong26h interval',lambda x:x.update(expires_at_utc=(datetime.fromisoformat(q['created_at_utc'])+timedelta(hours=26)).isoformat()),'fresh27h'),
 ('historical clock reset',lambda x:x['historical_ROOT_clocks_v8']['v7'].update(created_at_utc=q['created_at_utc']),'original clocks')]:
 candidate=copy.deepcopy(q);mutate(candidate);r=a['putram'](q['preparation_directory']+'/'+label.replace(' ','_')+'.json',candidate)
 reject('complete request rejects '+label,lambda r=r:cs['validate_request_v1'](r),match)
original=g._state_locked;g._state_locked=lambda *args,**kwargs:None
reject('memo rejects modified guard callable at correct current pathname',lambda:sc['mutation_memo_v1'](g,q,cfg,a['request_ref']),'Original unmodified callable')
g._state_locked=original
u=sc['current_uncached_paths_v9'](q,cfg,a['request_ref'])
ck('current request and exact future staging records uncached',a['request_ref']['path']in u and all(q['preparation_directory']+'/'+q['instance_id']+s in u for s in('.runtime.json','.consumer_manifest.json','.reserve_authority.json')))
base=a['loaded'][0];r=a['putram']('RAM_ONLY/independent.json',{'value':1})
with a['primitive_originals']['mutex'](a['ramroot']):
 tx=base._Transaction(a['ramroot'],v4,budget,{})
 try:
  tx.install();one=v4.bound(a['ramroot'],r,True);one['value']=99;two=v4.bound(a['ramroot'],r,True)
  ck('memo reparses JSON every call',two=={'value':1}and tx.counts['json_hits']==1)
  mem[r['path']]=b'{"value":2}\n'
  reject('original final uncached validation catches changed cached content',tx.finish,'Cached bound content changed')
 finally:tx.close()
ck('failed final validation restores all original primitives',all(getattr(v4,k)is v for k,v in a['primitive_originals'].items())and budget.successor_paths is a['successor_original']and not a['locked'][0])
# Independently constructed ancestry and pidfd fixtures using exact original AST bodies.
bt=ast.parse((ROOT/c.OWNED['path']).read_text());bs={'os':SimpleNamespace(getpid=lambda:1,close=lambda fd:None),'_stat':None,'_children':None}
emit([fn(bt,n)for n in('_same','signal_identity','OwnedTree')],bs)
def row(pid,start,ppid=1100,group=1100,state='R'):return dict(pid=pid,start=start,ppid=ppid,group=group,session=1100,state=state)
def fixture():
 table={1100:row(1100,10,ppid=1),1200:row(1200,20)};kids={1:[1100],1100:[1200]}
 back=SimpleNamespace(OwnedTree=bs['OwnedTree'],_same=bs['_same'])
 tr=d.OwnedLifecycleV8(back,1100,owner=1,read_stat=lambda pid:table.get(pid),read_children=lambda pid:kids.get(pid,[]));tr.refresh();return tr,table,kids
tr,table,kids=fixture()
for group in(1200,1300,1400):table[1200]=row(1200,20,group=group);tr.refresh()
ck('serial samePID start group changes retain all groups without extra discoveries',tr.groups=={1100,1200,1300,1400}and tr.discovered.total==2)
tr,table,kids=fixture();count=[0];prior=tr._original_read_stat
def staged(pid):
 if pid!=1200:return prior(pid)
 count[0]+=1
 if count[0]==3:return row(1200,20,group=1200)
 return None if count[0]>3 else prior(pid)
tr._original_read_stat=staged;tr.refresh()
ck('post-child-enumeration-only group observation retained before disappearance',1200 in tr.groups and 1200 not in tr.known)
tr,table,kids=fixture();limits=dict(d.OWNED_LIMITS_V8);d.OWNED_LIMITS_V8['groups']=1;table[1200]=row(1200,20,group=1200);snapshot=(copy.deepcopy(dict(tr.known)),set(tr.groups),tr.discovered.total,tr.retired.total)
reject('strict external group overflow fails closed',lambda:tr.read_stat_v9(1200),'group bound')
ck('overflow row group and history atomicity',snapshot==(dict(tr.known),set(tr.groups),tr.discovered.total,tr.retired.total)and tr.complete is False)
sent=[];closed=[]
ck('original two-stat pidfd signalling survives overflow cleanup',bs['signal_identity'](tr.known[1200],15,read_stat=tr.cleanup_read_stat_v9,open_fd=lambda p,f:77,send_fd=lambda fd,s,*args:sent.append((fd,s)),close_fd=closed.append)and sent==[(77,15)]and closed==[77])
table.clear();kids.clear();ck('incomplete never recovers after all identities disappear',tr.identities_absent()is False)
d.OWNED_LIMITS_V8.clear();d.OWNED_LIMITS_V8.update(limits)
tr,table,kids=fixture();table[1200]=row(1200,21,ppid=999,group=999);kids[1100]=[];tr.refresh();ck('unrelated PID reuse does not enroll its group',999 not in tr.groups and 1200 not in tr.known)
def denied(pid):raise PermissionError('independent denied stat')
tr._original_read_stat=denied
reject('cleanup-only stat preserves nongroup permission errors',lambda:tr.cleanup_read_stat_v9(1100),'independent denied stat')
ticks=iter([0.,0.,2.]);ns=dict(vars(d),time=SimpleNamespace(monotonic=lambda:next(ticks)),os=SimpleNamespace(killpg=lambda group,sig:(_ for _ in()).throw(ProcessLookupError())))
emit([fn(ast.parse(e['_DRIVER_SOURCE']),'observe_groups_v8')],ns);scan=ns['observe_groups_v8'](SimpleNamespace(groups={1100,1200}),1.)
ck('absent groups after aggregate deadline cannot claim drain',scan['absent_group_count']==2 and not scan['all_owned_groups_observed_absent']and not scan['scan_finished_within_deadline'])
ck('no scientific frameworks imported',not any(n=='torch'or n.startswith(('torch.','torch_geometric','numpy'))for n in sys.modules))
result=dict(schema='H058_v9_independent_bounded_source_RAM_checks_v1',status='PASS',independent=True,source_only=True,assertions=len(checks),negative_fixtures=negatives,checks=checks,source_ref=rf(D+'/operational_v1.py'),source_manifest_ref=rf(D+'/source_manifest.v1.json'),source_handoff_ref=rf(D+'/source_handoff.v9.json'),author_scaffold_reproduced_assertions=a['result']['assertions']+t['result']['assertions'],author_scaffolds_independent=False,actual_guard_state_ledger_process_resource_GPU_or_numerical_operations=False)
with(ROOT/OUT/'checks.result.v1.json').open('x')as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps({k:v for k,v in result.items()if k!='checks'},sort_keys=True))

