"""Independent C3v3 source/negative review; partial RAM fixtures only, no actual operations."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace,ModuleType,FunctionType,CodeType
import ast,builtins,collections,contextlib,copy,hashlib,io,json,marshal,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');D=ROOT/'research/reviews/C3_remaining_cells_GPU0_source_20261017_v3';OUT=ROOT/'research/reviews/C3_remaining_cells_GPU0_independent_20261017_v3'
checks=[];negative=[]
def ck(name,value=True):
 if not value:raise AssertionError(name)
 checks.append(dict(name=name,status='PASS'))
def reject(name,f):
 try:f()
 except(ValueError,KeyError,TypeError,AssertionError):negative.append(name);ck(name)
 else:raise AssertionError(name+' accepted')
def rf(p):return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def doc(p):return json.loads(p.read_bytes())
def nodes(p):return {n.name:n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def execute_defs(scope,ns,names,label=None):exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(ns[n])for n in names],type_ignores=[])),'independent_C3_exact_AST_RAM','exec'),scope)
def require(ok,message):
 if not ok:raise ValueError(message)
expected={'source_handoff.v3.json':'326aae328321111c3bac27a924bc543d0eca3d076b7cb53050bbb4ae6f2e0093','source_manifest.v1.json':'dbf53b9573ec305bcfac3836c71cc98bfa8d67d0e845481406c8a4bf8e9569d9','audit_manifest.v3.json':'b2c9ad6ea9f822da856ca04a7bf9103830252c9440c345815d96e3efb670e27f'}
for f,h in expected.items():ck('requested exact '+f,rf(D/f)['sha256']==h)
refs={}
def collect(v):
 if isinstance(v,dict):
  if 'path'in v and 'sha256'in v and isinstance(v['path'],str):refs[v['path']]=v['sha256']
  for x in v.values():collect(x)
 elif isinstance(v,list):
  for x in v:collect(x)
m=doc(D/'source_manifest.v1.json')
for x in(m,doc(D/'audit_manifest.v3.json'),doc(D/'source_handoff.v3.json'),doc(D/'preserved_v2.v3.json')):collect(x)
for p,h in refs.items():
 f=ROOT/p;assert f.resolve()==f and not f.is_symlink()and f.is_file()and f.stat().st_size<=32*2**20 and f.suffix in('.py','.json','.md','.txt','.patch')and rf(f)['sha256']==h
ck('all sealed source and bounded metadata refs exact',len(m['source_authentication_refs'])==449)
prior=ROOT/'research/reviews/C3_remaining_cells_GPU0_independent_20261017_v2/review.v2.json'
ck('prior blocked review preserved',rf(prior)['sha256']=='c465f652afc52ec9062ff42c4102874be008c0c9ff5c0da61c11240c01a93669'and doc(prior)['verdict']=='SOURCE_BLOCKED')
for r in m['executed_source_refs']:compile((ROOT/r['path']).read_bytes(),r['path'],'exec')
ck('all5 executed modules compile inertly',len(m['executed_source_refs'])==5)
v2=ROOT/'research/reviews/C3_remaining_cells_GPU0_source_20261017_v2'
for f in('worker_v1.py','direct_gpu0_v1.py','contract_v1.py'):
 a=(D/f).read_text().replace(D.name,v2.name)
 ck('whole prior AST unchanged '+f,ast.dump(ast.parse(a),include_attributes=False)==ast.dump(ast.parse((v2/f).read_text()),include_attributes=False))
hcontract=doc(ROOT/'research/reviews/H058_GPU0_full_training_successor_source_20261017_v9/LIFECYCLE_REUSE_CONTRACT.v9.json');ex=(D/'execution_v1.py').read_text();ens=nodes(D/'execution_v1.py')
for n,h in hcontract['source_slice_sha256'].items():ck('exact shared helper bytes '+n,hashlib.sha256(ast.get_source_segment(ex,ens[n]).encode()).hexdigest()==h)
ck('strict RSS and cleanup-only pidfd paths actually wired',"_stat=tree.read_stat_v9"in ex and "rss_identity_v1(rss_owned, r)"in ex and "read_stat=tree.cleanup_read_stat_v9 if tree is not None else owned._stat"in ex)
# Reproduce only setup + 3 prediction admissions + matrix admission, not massive author campaign.
# All modules are AST-rebound to RAM IO; no source writer statement is included.
tree=ast.parse((D/'source_checks_v1.py').read_text());prefix=[n for n in tree.body if n.end_lineno<=221]
a={'__name__':'independent_C3_partial_author_RAM'}
with contextlib.redirect_stdout(io.StringIO()):exec(compile(ast.Module(body=prefix,type_ignores=[]),'partial_author_RAM_through_complete_admission','exec'),a)
c=a['c'];put=a['put'];read=a['read'];ret=a['ret'];pending=a['pending'];election=a['election']
ck('complete actual builder request reserve input joins all4stages in RAM',len(a['prepared'])==4 and set(a['unions'].values())=={64,62})
ck('all47 current roles retained and GPU64 matrix62 direct ceiling',len(a['roles'])==47 and all(a['unions'][p]==64 for p in c.GPU_PHASES)and a['unions']['matrix']==62)
ck('election retains26 adds4 and never relabels original failed waits',sum(map(len,ret.values()))==26 and sum(map(len,pending.values()))==4 and all(read(a['parents'][p]['original_disposition']['wait_ref'])['returncode']==-15 for p in c.VAL[:3]))
# Own authority and lifecycle contract negatives.
for label,edit in[
 ('truth capability',lambda q:q.update(truth={})),('wrong source seal',lambda q:q.update(source_manifest_ref={'path':'wrong','sha256':'0'*64})),
 ('wrong CUDA cap',lambda q:q['caps'].update(max_cuda_reserved_bytes=9*2**30)),('empty GPU subset',lambda q:q.update(pending_epochs=[])),
 ('wrong parent',lambda q:q.update(parent_id=c.VAL[4])),('automatic retry',lambda q:q.update(automatic_restart_or_fallback=True)),
 ('clock extension',lambda q:q.update(expires_at_utc=(datetime.fromisoformat(q['created_at_utc'])+timedelta(hours=25)).isoformat()))]:
 q=copy.deepcopy(a['requests'][c.GPU_PHASES[0]]);edit(q);r=put('RAM/independent/'+label+'.json',q);reject('request rejects '+label,lambda r=r:c.validate_request_v1(r))
base_ev=a['requests']['matrix']['new_stage_evidence'][c.VAL[0]];parent=c.VAL[0]
for label,edit in[
 ('adopted nonzero',lambda w:w['adopted_waits'].update(nonzero_returncodes=1)),
 ('retired mismatch',lambda w:w['ownership']['retired_identity_history'].update(count=0)),
 ('group deadline overrun',lambda w:w['group_absence_summary'].update(scan_finished_within_deadline=False)),
 ('group count omission',lambda w:w['group_absence_summary'].update(expected_group_count=2)),
 ('group probe error',lambda w:w['group_absence_summary'].update(errors=['denied'])),
 ('incomplete ownership',lambda w:w['ownership'].update(complete_owned_roster_known=False)),
 ('unknown raw wait',lambda w:w.update(actual_OS_wait_status=None)),('boolean returncode',lambda w:w.update(returncode=False))]:
 w=read(base_ev['wait_ref']);edit(w);wr=put('RAM/independent/'+label+'.wait.json',w);o=read(base_ev['outer_ref']);o['actual_wait_ref']=wr
 ev=dict(base_ev,wait_ref=wr,outer_ref=put('RAM/independent/'+label+'.outer.json',o))
 reject('new cells reject '+label,lambda ev=ev:c.new_stage_values_v1(ev,parent,election,pending[parent]))
for label,edit in[
 ('duplicate retained cell',lambda e:e['parents'][parent]['retained_prediction_seal_refs'].append(e['parents'][parent]['retained_prediction_seal_refs'][0])),
 ('answered epoch20 pending',lambda e:e['parents'][parent].update(pending_epochs=[20,30])),
 ('historical nonzero omitted',lambda e:e.update(original_nonzero_waits_preserved=False)),
 ('truth opened',lambda e:e.update(private_truth_opened=True))]:
 v=read(election);edit(v);r=put('RAM/independent/'+label+'.election.json',v);reject('election rejects '+label,lambda r=r:c.election_values_v1(r))
# Concrete original writer revocation must precede retained inventory.
v=read(election);r0=v['parents'][parent]['original_disposition']['permanent_revocation_ref'];rev=read(r0);rev['future_writes_and_relaunch_permanently_revoked']=False
v['parents'][parent]['original_disposition']['permanent_revocation_ref']=put('RAM/independent/not_revoked.json',rev)
reject('live original writer cannot enter election',lambda:c.election_values_v1(put('RAM/independent/live_election.json',v)))
reject('original direct65 union refused',lambda:a['guard'].consumer_bindings_v1([{'path':'RAM/'+str(i),'sha256':'0'*64}for i in range(65)]))
# Actual matrix assembly reaches unchanged original barrier only with all30 chosen cells.
chosen=[];written=[]
class StopBarrier(Exception):pass
class WPath:
 def exists(self):return False
 def mkdir(self):pass
class Writer:
 def __init__(self,*args):self.path=WPath()
 def boundary(self,*args,**kw):pass
 def save(self,*args):written.append(args);return {}
def barrier(refs,**kw):chosen.extend(refs);raise StopBarrier()
realimport=builtins.__import__
def imported(name,*args,**kw):
 if name=='arggnn.american_c3_gnn_predict_isolated_v2':return SimpleNamespace(seal_validation_matrix_v1=barrier)
 return realimport(name,*args,**kw)
c.guard_v1=lambda q:a['guard']
worker_scope=dict(vars(a['worker']),GuardedWriter=Writer,sys=SimpleNamespace(modules={},path=[]),os=SimpleNamespace(environ={'CUDA_VISIBLE_DEVICES':''}),__builtins__=dict(vars(builtins),__import__=imported))
execute_defs(worker_scope,nodes(D/'worker_v1.py'),('run_matrix_v1',))
q=copy.deepcopy(a['requests']['matrix'])
try:worker_scope['run_matrix_v1']({}, {},q,{}, {})
except StopBarrier:pass
else:raise AssertionError('matrix did not reach original barrier')
ck('actual matrix chooses exact retained26 new4 all30 before payload boundary',len(chosen)==30 and len({json.dumps(r,sort_keys=True)for r in chosen})==30 and not written)
q2=copy.deepcopy(q);q2['new_stage_evidence'].pop(parent)
reject('matrix missing new-stage terminal cannot reach original barrier',lambda:worker_scope['run_matrix_v1']({}, {},q2,{}, {}))
# Source-only exact code-object join for original consumer namespace memo, no installation IO.
oldchecks=ast.parse((ROOT/'research/reviews/C3_remaining_cells_GPU0_independent_20261017_v2/checks_v2.py').read_text())
memo_scope=dict(globals(),need=lambda ok,label:ck('memo source join: '+label,ok))
with contextlib.redirect_stdout(io.StringIO()):exec(compile(ast.Module(body=[copy.deepcopy(n)for n in oldchecks.body if 91<=n.lineno and n.end_lineno<=112],type_ignores=[]),'prior_independent_exact_memo_RAM_join','exec'),memo_scope)
# Only definitions from author full-supervisor scaffolding; no initial campaign or writer.
gt=ast.parse((D/'group_checks_v3.py').read_text());selected=[n for n in gt.body if n.end_lineno<=54 or (n.lineno>=109 and n.end_lineno<=229)]
gs={'__name__':'partial_C3_supervisor_RAM'}
with contextlib.redirect_stdout(io.StringIO()):exec(compile(ast.Module(body=selected,type_ignores=[]),'bounded_author_supervisor_definitions_only','exec'),gs)
for mode in('normal','changed-groups-absent','changed-groups-live'):
 with contextlib.redirect_stdout(io.StringIO()):r=gs['execute_case'](mode)
 ck('actual C3 supervisor to drain-contract join: '+mode,r['drained']is(mode!='changed-groups-live'))
ck('no Torch PyG NumPy imported',not any(n=='torch'or n.startswith(('torch.','torch_geometric','numpy'))for n in sys.modules))
result=dict(schema='C3v3_independent_bounded_whole_source_checks_v1',source_only=True,independent=True,status='PASS',assertions=len(checks),negative_fixtures=len(negative),checks=checks,negatives=negative,authenticated_unique_refs=len(refs),source_authentication_entries=449,author_massive_campaign_executed=False,partial_author_RAM_fixture_reuse=True,author_writers_suppressed=True,actual_guard_process_lease_census_resource_GPU_scientific_payload_truth_or_numerical_operations=False)
with(OUT/'checks.result.v1.json').open('x')as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items()if k not in('checks','negatives')},sort_keys=True))

