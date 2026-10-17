"""Independent FLARE bridge source/AST/RAM review. No guard or science execution."""
from pathlib import Path
import ast,copy,hashlib,json,sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');CAND=ROOT/'research/reviews/FLARE_VAL001_native_seal_predecessor_bridge_source_20261017_v1';OUT=Path(__file__).resolve().parent
REF=lambda p:{'path':str(p.resolve().relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def fn(source,name):return copy.deepcopy(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name==name))
# Redirect only candidate result destination in its private RAM AST.
tree=ast.parse((CAND/'source_check_v1.py').read_text())
outputs=[n for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='p'for t in n.targets)and isinstance(n.value,ast.BinOp)and isinstance(n.value.right,ast.Constant)and n.value.right.value=='source_check.result.v1.json'];assert len(outputs)==1
outputs[0].value=ast.parse("REVIEW_OUT/'candidate_source_check_reproduced.v1.json'",mode='eval').body
ast.fix_missing_locations(tree);ns={'__name__':'_candidate_source_checks_private_output','REVIEW_OUT':OUT}
exec(compile(tree,'<candidate tests with own result destination only>','exec'),ns);assert ns['result']['case_count']==34
b=ns['ns'];contract_source=ns['src']['contract'];manifest=json.loads((CAND/'source_manifest.v1.json').read_text());inherited=json.loads((ROOT/manifest['inherited_source_manifest_ref']['path']).read_text())
TESTS=[];FINDINGS=[];EVIDENCE=[]
def test(name,action):action();TESTS.append({'name':name,'status':'PASS'})
def rejects(action):
 try:action()
 except (ValueError,KeyError,TypeError):return
 raise AssertionError('Expected rejection')
# Genuine pinned small metadata, never raw input/VCF/NPZ payloads.
actual={name:json.loads((ROOT/b[name]['path']).read_text())for name in ('INPUT_REF','WAIT_REF','COMPLETED_REF','V4_MANIFEST_REF')}
i,w,converted,v4=[actual[k]for k in ('INPUT_REF','WAIT_REF','COMPLETED_REF','V4_MANIFEST_REF')]
ns['old_manifest']=v4['source_sha256'];ns['new_manifest']=manifest['source_sha256'];ns['CURRENT_DRIVER']=REF(CAND/'root_driver_v1.py');ns['CURRENT_ENTRY']=REF(CAND/'worker_entry_v1.py')
def actual_source_records():
 records=ns['base_records']();key=ns['key']
 records[key(b['INPUT_REF'])]=copy.deepcopy(i);records[key(b['WAIT_REF'])]=copy.deepcopy(w)
 records[key(ns['CLOSE'])]['key']=i['storage']['reservation']['key']
 records[key(ns['CLOSE'])]['value']['reservation_sha256']=hashlib.sha256(json.dumps(i['storage']['reservation'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 return records

def genuine_join_positive():
 assert i['root_driver_ref']==b['V4_DRIVER_REF']and i['main_source_ref']==b['V4_ENTRY_REF']and i['source_sha256']==v4['source_sha256']
 assert converted['input_sites']==227316 and converted['native_expected_markers']==26609 and converted['common_focal_records']==10000
 assert converted['reference_haplotypes']==600 and converted['query_haplotypes']==1000 and converted['query_ancestry_truth_opened']is False
 assert w['returncode']==0 and w['actual_direct_worker_wait']is True and w['all_owned_processes_and_groups_drained']is True and w['operational_failure']is None
 records=actual_source_records();assert ns['run'](records)==records[ns['key'](ns['E'])]
 EVIDENCE.append({'case':'actual_v4_source_input_wait_joins_with_future_close_certificates_in_RAM_only','pass':True,'actual_close96129_remains_pending':True,'future_close_certificates_are_synthetic_not_actual':True})
test('exact real v4 source/input/wait metadata satisfies unchanged bridge only with synthetic future successful-close proof',genuine_join_positive)

for label,record,key,value in [('future close pending',ns['ACCEPT'],'accepted',False),('future close unconsumed',ns['CLOSE_OUTER'],'actual_terminal_consumed',False),('future close exit1',ns['CLOSE_OUTER'],'actual_exit_code',1),('future journal unaccepted',ns['FINAL'],'final_original_journal_accepted',False),('wrong current selector',ns['FINAL'],'storage_policy_selector_ref',ns['E']),('wrong original key',ns['CLOSE'],'key','other'),('conversion wait1',b['WAIT_REF'],'returncode',1),('conversion unknown drain',b['WAIT_REF'],'all_owned_processes_and_groups_drained',False),('conversion unconsumed outer',ns['OUTER'],'actual_terminal_consumed',False),('conversion outer1',ns['OUTER'],'actual_exit_code',1),('wrong exact v4 worker',b['INPUT_REF'],'main_source_ref',ns['CURRENT_ENTRY']),('wrong exact v4 closure',b['INPUT_REF'],'source_sha256',manifest['source_sha256'])]:
 def check(record=record,key=key,value=value):
  records=actual_source_records();records[ns['key'](record)][key]=value;rejects(lambda:ns['run'](records))
 test('actual-source graph rejects '+label,check)

def missing_close():
 records=actual_source_records();records[ns['key'](ns['E'])].pop('ordinary_close_terminal_acceptance_ref');rejects(lambda:ns['run'](records))
 records=actual_source_records();records[ns['key'](ns['ACCEPT'])].pop('final_original_journal_acceptance_ref');rejects(lambda:ns['run'](records))
test('no pending ordinary close or final journal proof can be omitted',missing_close)

# The full request path invokes inherited sources() with the NEW candidate manifest.
# Reproduce without importing contract top-level helpers or invoking any guard.
manifest_ref=REF(CAND/'source_manifest.v1.json');request_ref={'path':'RAM/never_active_request.json','sha256':'0'*64};events=[]
source_scope={'MANIFEST':manifest_ref['path'],'ref':lambda p:manifest_ref,'raw':lambda r:events.append(('source_RAM_validation',r['path'])),'scientific_review':lambda:events.append(('UNREACHED_scientific_review',None))}
def read(r):
 if r==manifest_ref:return copy.deepcopy(manifest)
 if r==request_ref:return {'RAM_NOT_ACTUAL':True}
 raise AssertionError('Unexpected RAM metadata request')
source_scope['read']=read
for name in ('sources','validate_request'):
 n=fn(contract_source,name);exec(compile(ast.Module(body=[n],type_ignores=[]),'<exact inherited contract source RAM>','exec'),source_scope)
def reproduce_manifest_failure():
 events.clear()
 try:source_scope['validate_request'](request_ref)
 except KeyError as e:assert e.args==('metadata_refs',)
 else:raise AssertionError('Expected missing exact inherited manifest field')
 assert not any(name=='UNREACHED_scientific_review'for name,_ in events)
 FINDINGS.append({'id':'B1','source':'root_contract_v1.sources called by original validate_request','error_type':'KeyError','error_key':'metadata_refs','candidate_field_absent':True,'inherited_metadata_ref_count':len(inherited['metadata_refs']),'before_predecessor_or_guard_or_admission':True,'exact_original_source_executed_with_RAM_IO_doubles':True})
test('B1 entire inherited validate_request source path fails on candidate missing metadata_refs before later validation',reproduce_manifest_failure)

def minimal_schema_correction_demo():
 fixed=copy.deepcopy(manifest);fixed['metadata_refs']=copy.deepcopy(inherited['metadata_refs']);original_read=source_scope['read'];events.clear()
 source_scope['read']=lambda r:fixed if r==manifest_ref else original_read(r)
 try:assert source_scope['sources']()is fixed
 finally:source_scope['read']=original_read
 assert len(events)==len(manifest['source_sha256'])+len(inherited['metadata_refs'])
test('RAM-only original sources validator accepts unchanged inherited metadata_refs field without bypass',minimal_schema_correction_demo)

# Verify actual marker-count derivation with exact original config-builder code.
core_path=ROOT/'research/reviews/FLARE_original_VAL001004_completion_source_20261017_v1/completion_v1.py';core_src=core_path.read_text()
basis_path=core_path.parent/'VAL000_basis.v1.json';basis_ref=next(r for r in inherited['metadata_refs']if r['path']==str(basis_path.relative_to(ROOT)));assert REF(basis_path)==basis_ref
basis=json.loads(basis_path.read_text())
row={'input_sites':227316,'count_manifest_ref':{'path':'RAM/count.json','sha256':'1'*64},'focal_registry_ref':converted['focal_registry_ref'],'observation_ref':converted['observation_ref'],'simulation_config_ref':{'path':'RAM/simcfg.json','sha256':'2'*64}}
core={'copy':copy,'require':b['require'],'MISSING':tuple('american_c3_val_%03d'%k for k in range(1,5)),'OPTIONS':converted['native_options'],'COMMIT':converted['native_source_commit'],'_sources':lambda:None,'_context':lambda r,parent_id=None:{'targets':{b['PARENT']:row}},'_local':lambda name:basis,'_tools':lambda:(lambda r:copy.deepcopy(converted),None,None)}
for name in ('_gzip_bound','_conversion_bound','_native_output_bound','inactive_stage_configs_v1','_missing_keys'):
 n=fn(core_src,name);exec(compile(ast.Module(body=[n],type_ignores=[]),'<exact inactive scientific metadata builder RAM>','exec'),core)
configs=core['inactive_stage_configs_v1']({'path':'RAM/context.json','sha256':'3'*64},b['PARENT'],conversion_ref=b['COMPLETED_REF'])
def marker_bounds():
 expected=core['_native_output_bound'](26609);wrong=core['_native_output_bound'](227316)
 assert configs['native']['max_native_vcf_bytes']==expected['vcf_bytes']and configs['native']['output_cap_bytes']==expected['total_bytes']
 assert expected!=wrong and configs['seal']['max_markers']==26609 and configs['native']['max_sites_bound']==227316
 assert configs['conversion']['max_sites']==227316 and configs['native']['native_options']==basis['configurations']['native_config']['native_options']
 assert configs['native']['heap_gib']==16 and configs['native']['timeout_seconds']==14400 and configs['native']['external_truth_ref']is None
 EVIDENCE.append({'case':'actual26609native_markers_distinct_from227316inputsites','native_vcf_cap':expected['vcf_bytes'],'native_output_cap':expected['total_bytes'],'native_scope_with_8MiB_packet':expected['total_bytes']+8*2**20,'basis_ref':basis_ref})
test('native/seal configs preserve original settings and derive caps from26609 actual native markers',marker_bounds)

def caps_and_phase():
 scope={'require':b['require'],'STAGES':('native','seal')};n=fn(contract_source,'stage_caps');exec(compile(ast.Module(body=[n],type_ignores=[]),'<original stage caps RAM>','exec'),scope)
 native=scope['stage_caps']('native',configs['native']);seal=scope['stage_caps']('seal',configs['seal'])
 assert native['reservation_bytes']==configs['native']['output_cap_bytes']+8*2**20 and native['cpu_threads']==4 and native['rss_bytes']==24*2**30
 assert native['scientific_seconds']==14400 and native['wall_seconds']==15000 and native['cleanup_seconds']==30
 assert seal['rss_bytes']==4*2**30 and seal['scientific_seconds']==3600 and seal['wall_seconds']==4200
 env={'need':b['require'],'SETUP_SECONDS':7200};n=fn(ns['src']['engine'],'phase_budget');exec(compile(ast.Module(body=[n],type_ignores=[]),'<original phase budget RAM>','exec'),env)
 assert env['phase_budget']('native')['combined_seconds']==22200 and env['phase_budget']('seal')['combined_seconds']==11400
 assert env['phase_budget']('native')['evidence_reserve_seconds']==300 and env['phase_budget']('native')['authority_margin_seconds']==600
test('original4CPU/RSS/output+packet/scientific/setup/cleanup/evidence caps remain exact',caps_and_phase)

def four_pretruth():
 seals={parent:{'path':'RAM/'+parent+'.json','sha256':'4'*64}for parent in core['MISSING']}
 assert core['_missing_keys'](seals)==seals
 incomplete=dict(seals);incomplete.pop(b['PARENT']);rejects(lambda:core['_missing_keys'](incomplete))
 extra=dict(seals,american_c3_val_000={});rejects(lambda:core['_missing_keys'](extra))
 score=fn(core_src,'score_five_original_VAL_v1')
 loops=[n for n in score.body if isinstance(n,ast.For)]
 assert len(loops)>=2 and '_cell('in ast.unparse(loops[0])and 'resolve_val_truth('not in ast.unparse(loops[0])
 assert 'resolve_val_truth('in ast.unparse(loops[1])
 assert 'all4_remaining_sealed_before_their_truth' in core_src and 'TEST_authorized' in core_src
test('all four new seals required and all prediction validation precedes remaining VAL truth resolution',four_pretruth)

def final_integrity():
 closure=json.loads((OUT/'source_closure.v1.json').read_text())
 for path,sha in closure['verified_source_and_small_metadata_sha256'].items():assert REF(ROOT/path)['sha256']==sha,path
 assert not any(k=='torch'or k.startswith(('arggnn','numpy','pandas','composer','omegaconf'))for k in sys.modules)
test('all candidate/inherited pins unchanged; no scientific/runtime/guard imports',final_integrity)
result={'schema':'independent_FLARE_VAL001_predecessor_bridge_RAM_AST_v1','status':'BLOCKING_MANIFEST_SCHEMA_FINDING_REPRODUCED','candidate_cases_reproduced':34,'independent_cases':len(TESTS),'total_cases':34+len(TESTS),'tests':TESTS,'findings':FINDINGS,'source_metadata_evidence':EVIDENCE,'test_source_ref':REF(Path(__file__)),'ordinary_close96129_completed_or_accepted_by_review':False,'real_payload_runtime_guard_process_GPU_resource_admission_launch_Git_action':False}
with(OUT/'independent_source_ram_results.v1.json').open('x')as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({'status':result['status'],'candidate_cases':34,'independent_cases':len(TESTS),'total_cases':result['total_cases'],'finding':FINDINGS,'results_ref':REF(OUT/'independent_source_ram_results.v1.json')},sort_keys=True))
