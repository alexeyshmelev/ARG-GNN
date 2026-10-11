"""Adaptation-only independent source/handmade CPU checks. No actual arrays or live budgets."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime,timezone,timedelta
from pathlib import Path
import ast,hashlib,importlib.util,json,os,subprocess,sys,types
import torch
from arggnn import census_tsinfer_fit_v1 as fit
from arggnn import census_tsinfer_device_profile_v1 as profile
from arggnn import census_tsinfer_observation_v1 as bridge
ROOT=fit.ROOT;DIR=ROOT/'research/reviews/census_tsinfer_consumers_v1_independent'
torch.set_num_threads(1)
assert os.environ['CUDA_VISIBLE_DEVICES']==''
checks=[];details={}
def check(name,value=True):
    if not bool(value):raise AssertionError(name)
    checks.append(name)
def reject(name,call):
    try:call()
    except (ValueError,RuntimeError,TypeError,KeyError,IndexError):check(name);return
    raise AssertionError('unexpected acceptance: '+name)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tree(p):return ast.parse((ROOT/p).read_text())
def node(t,name):return next(n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
def dump(n):return ast.dump(n,include_attributes=False)
paths=[fit.SOURCE,fit.CONFIG,'src/arggnn/census_tsinfer_device_profile_v1.py','configs/census_tsinfer_device_profile_v1.json']
initial={p:sha(ROOT/p) for p in paths}
fixed={
 'src/arggnn/census_biological_fit_v2.py':'c21884ad90a2c940b49968353545edc52d345c0c7bc10fd87c492c54d4b38c34',
 'configs/census_biological_fit_v2.json':'a512816bbab61e80052e7466613d2de4c861b77ef61d0d35cd50fbf81905c621',
 'src/arggnn/census_tsinfer_observation_v1.py':'c32a0ed65500d6ea4a3f9ea9373463dbc7bda88eedace84a2d6538a3b3c8980a',
 'research/reviews/census_tsinfer_actual32_numerical_v1_root/inspection_freeze.json':'b8417f462bad48a18077d82ddbdcc6040e46049441502bac72041ba35ff96e45',
 'research/reviews/census_tsinfer_actual32_numerical_v1_root/inspection_lineage.json':'4ab0009e5909392347cc5d8e789b80a8bff62af523e3434617ec36c3be7943d9'}
for p,h in fixed.items():check('fixed_reference_pin_'+p,sha(ROOT/p)==h)
code="import sys; from arggnn import census_tsinfer_fit_v1 as f, census_tsinfer_device_profile_v1 as p; f.source_report_v1(); p.source_report(); assert not any(k in sys.modules for k in ('torch','numpy','tskit'))"
run=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=os.environ.copy(),capture_output=True,text=True)
check('default_source_reports_standard_library_only',run.returncode==0)
old=tree('src/arggnn/census_biological_fit_v2.py');new=tree(fit.SOURCE)
cfg=fit.load_config_v1()
for name in cfg['unchanged_fit_AST']:check('exact_preserved_fit2_AST_'+name,dump(node(old,name))==dump(node(new,name)))
before=deepcopy(node(old,'run_fit_v1'));after=node(new,'run_fit_v1')
for n in ast.walk(before):
    if isinstance(n,ast.Constant) and n.value=='census_biological_fit_evidence_v2':n.value='census_tsinfer_fit_evidence_v1'
check('entire_fit2_loop_only_receipt_schema_literal_changed',dump(before)==dump(after))
profile_old=tree('src/arggnn/census_biological_device_profile_v1.py');profile_new=tree('src/arggnn/census_tsinfer_device_profile_v1.py')
for name in ['compare_certified_tile_v1','run_profile']:
    a=deepcopy(node(profile_old,name));b=deepcopy(node(profile_new,name))
    for n in ast.walk(a):
        if isinstance(n,ast.Constant) and n.value=='census_biological_device_profile_evidence_v1':n.value='census_tsinfer_device_profile_evidence_v1'
    for n in (a,b):n.body=[x for x in n.body if not isinstance(x,(ast.Import,ast.ImportFrom))]
    check('exact_profile_scientific_body_except_transport_imports_'+name,dump(a)==dump(b))
for k in ['training','model','controls','metrics','seeds','readout_chunk_size','external_target_manifest']:
    check('unchanged_fit2_configuration_'+k,cfg[k]==json.loads((ROOT/'configs/census_biological_fit_v2.json').read_text())[k])
pconfig=json.loads(profile.CONFIG.read_text())
check('profile_exact_prespecified446_550_extrema',pconfig['actual_TRAIN_extrema']==[
 {'parent_id':'bio_paths_pilot_train_001','tile_id':7,'original_tree_count':446},
 {'parent_id':'bio_paths_pilot_train_000','tile_id':1,'original_tree_count':550}])
for module,t in [(fit,new),(profile,profile_new)]:
    main=node(t,'main');flat=list(ast.walk(main))
    calls=[n.func.id for n in flat if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    check('main_'+module.__name__+'_lease_title_before_execution',calls.index('validate_lease')<calls.index('apply_and_validate_title')<calls.index('run_fit_v1' if module is fit else 'run_profile'))
# Handmade fixture provider is the sealed observation author's tiny, native TS.
fixturepath=ROOT/'research/reviews/census_tsinfer_observation_v1_author/test_census_tsinfer_observation_v1.py'
spec=importlib.util.spec_from_file_location('independent_handmade_tsinfer_provider',fixturepath)
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
fit._science_v1();capsule=fit.bridge_capsule_v1()
check('private_capsule_not_original_module_namespace',capsule is not vars(bridge))
check('original_dependency_guard_kept',bridge._dependencies_v1 is not capsule['_dependencies_v1'])
check('private_dependency_supplier_only_own_function',capsule['_dependencies_v1'] is fit._dependencies_v1)
for name in ['_load_parent_v1','role_join_v1','physical_contract_v1','extract_native_tsinfer_tile_v1','raw_sequence_bytes_v1',
             'certify_native_tile_v1','validate_tsinfer_certificate_v1','forward_certified_tsinfer_v1','_producer_payload_v1']:
    check('capsule_'+name+'_private_globals',capsule[name].__globals__ is capsule)
    check('capsule_'+name+'_same_compiled_body',capsule[name].__code__.co_code==getattr(bridge,name).__code__.co_code)
check('capsule_original_file_identity',capsule['__file__']==bridge.__file__)
original_guard=bridge._dependencies_v1
# A bogus visibility string tests only the guard; no device is queried/created.
os.environ['CUDA_VISIBLE_DEVICES']='independent_handmade_no_GPU'
reject('original_bridge_CPU_visibility_guard_unchanged',lambda:bridge._dependencies_v1())
check('consumer_scientific_supplier_works_without_GPU_query',fit._dependencies_v1()[1] is torch)
os.environ['CUDA_VISIBLE_DEVICES']=''
check('original_bridge_dependency_function_unchanged_after',bridge._dependencies_v1 is original_guard)
ts,roles,original,derived,mapping=fixture.handmade(full=True)
outer,raw=bridge.certify_native_tile_v1(ts,roles,fixture.positions(),[0.,6400.],0,fixture.BINDINGS,fixture.LINEAGE,fixture.CFG)
private_outer,private_raw=capsule['certify_native_tile_v1'](ts,roles,fixture.positions(),[0.,6400.],0,fixture.BINDINGS,fixture.LINEAGE,fixture.CFG)
check('private_outer_inner_certificates_match_unchanged_bridge',outer.certificate_sha256==private_outer.certificate_sha256 and
      outer.native_certificate.certificate_sha256==private_outer.native_certificate.certificate_sha256)
check('private_materialized_x48_value_fingerprint_match',fit.fingerprint_sha256_v1(raw)==fit.fingerprint_sha256_v1(private_raw))
cert=fit.ConsumerCertificateV1(private_outer)
reject('inner_only_certificate_cannot_pass_new_consumer',lambda:fit.validate_census_biological_certificate_v1(outer.native_certificate,outer.native_certificate.certificate_sha256))
wrong=fit.ConsumerCertificateV1(replace(private_outer,lineage={**private_outer.lineage,'parent_id':'other'}))
reject('changed_outer_lineage_rejected',lambda:fit.validate_census_biological_certificate_v1(wrong,private_outer.certificate_sha256))
permutation=torch.randperm(184,generator=torch.Generator().manual_seed(851))
for dtype in [torch.float64,torch.float32]:
    torch.manual_seed(337)
    model=fit.CensusTreeGNNV1(**cfg['model']).to(dtype)
    initial_model=fit.fingerprint_sha256_v1(model.state_dict())
    for control in fit.CONTROLS:
        a=deepcopy(model);b=deepcopy(model)
        kw={'group_permutation':permutation} if control=='permuted_groups' else {}
        left=bridge.forward_certified_tsinfer_v1(a,outer,expected_certificate_sha256=outer.certificate_sha256,
              control=control,readout_chunk_size=17,**kw)
        right=fit.forward_certified_census_biological_v1(b,cert,expected_certificate_sha256=private_outer.certificate_sha256,
              control=control,readout_chunk_size=17,**kw)
        check('outer_consumer_exact_output_'+str(dtype)+'_'+control,torch.equal(left.logits,right.logits))
        check('outer_consumer_exact_scored_ID_order_'+str(dtype)+'_'+control,torch.equal(left.scored_query_ids,right.scored_query_ids))
        coefficient=torch.linspace(-1.13,.91,left.logits.numel(),dtype=dtype).reshape_as(left.logits)
        lg=torch.autograd.grad((left.logits*coefficient).mean(),tuple(a.parameters()),allow_unused=True)
        rg=torch.autograd.grad((right.logits*coefficient).mean(),tuple(b.parameters()),allow_unused=True)
        check('outer_consumer_exact_all_parameter_VJP_'+str(dtype)+'_'+control,
              all((x is None and y is None) or (x is not None and y is not None and torch.equal(x,y)) for x,y in zip(lg,rg)))
        del a,b,left,right,lg,rg
    check('five_wrapper_controls_did_not_mutate_initial_'+str(dtype),fit.fingerprint_sha256_v1(model.state_dict())==initial_model)
# Test target-column/source joins and support on observed-only handmade metadata.
import numpy as np
class MemoryReader:
    def __init__(self,value):self.value=value;self.calls=[]
    def read(self,ref):self.calls.append(ref['path']);return self.value
ref={'path':'explicit_handmade_source','sha256':'a'*64}
reader=MemoryReader(json.dumps(original).encode())
binding={'query_sample_indexes':list(range(120)),'query_raw_node_ids':[r['observed_node_id'] for r in original[:120]]}
for coverage in ([1.,6389.],[51.,6331.],[273.,6321.],[116.,6224.]):
    loaded=(None,roles,fixture.positions(),coverage[0],coverage,{'source':ref},{})
    scored=fit.scoring_parent_v1({'parent_id':'handmade','split':'train'},{},loaded,reader,binding)
    retained=np.flatnonzero((fixture.positions()>=coverage[0])&(fixture.positions()<coverage[1]))
    check('exact_original_retained_center_indexes_'+str(coverage),np.array_equal(scored['retained_indexes'],retained))
    check('exact_fixed100bp_clipped_support_'+str(coverage),scored['edges'][0]==max(100.*retained[0],coverage[0]) and
          scored['edges'][-1]==min(100.*(retained[-1]+1),coverage[1]))
    check('observed_original_raw_query_join_'+str(coverage),np.array_equal(scored['raw_query_nodes'],np.array(binding['query_raw_node_ids'])))
    check('native_sample_mapping_not_truth_column_identity_'+str(coverage),np.array_equal(scored['query_ids'],np.arange(120)))
wrong={**binding,'query_raw_node_ids':list(reversed(binding['query_raw_node_ids']))}
reject('altered_target_source_column_mapping_rejected',lambda:fit.scoring_parent_v1({'parent_id':'handmade','split':'train'},{},loaded,reader,wrong))
# Gate corruption checks extend author's metadata constructor only; no reader/data IO.
authorpath=ROOT/'research/reviews/census_tsinfer_fit_v1_author/test_consumers_v1.py'
spec=importlib.util.spec_from_file_location('independent_metadata_provider',authorpath)
author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
values=author.synthetic_gate()
check('synthetic32_terminal_metadata_gate',fit.require_terminal_numerical_v1(*values)==values[1])
for fault in ['terminal_evidence_ref','terminal_admission_ref','terminal_job','terminal_status','native_phase','native_fit',
              'freeze_authorizer','freeze_init','freeze_declares_decode','lineage_freeze','lineage_metadata','lineage_ARG',
              'numerical_split','numerical_num_trees','numerical_raw','numerical_oracle_dates','numerical_check_steps']:
    e,n,i,f,t,r,l=deepcopy(values)
    if fault=='terminal_evidence_ref':t['evidence_ref']={'path':'other','sha256':'0'*64}
    elif fault=='terminal_admission_ref':t['admission_ref']={'path':'other','sha256':'0'*64}
    elif fault=='terminal_job':t['job_id']='other'
    elif fault=='terminal_status':t['status']='running'
    elif fault=='native_phase':n['phase']='inspect'
    elif fault=='native_fit':n['fit_authorized']=True
    elif fault=='freeze_authorizer':f['authorizer']='worker'
    elif fault=='freeze_init':f['initial_state_sha256_by_dtype']['float64']='0'*64
    elif fault=='freeze_declares_decode':f['raw_payload_decoded']=True
    elif fault=='lineage_freeze':l['freeze_ref']={'path':'other','sha256':'0'*64}
    elif fault=='lineage_metadata':l['metadata_input_sha256']={}
    elif fault=='lineage_ARG':l['observed_artifact_sha256']={}
    elif fault=='numerical_split':e['cells'][0]['split']='val'
    elif fault=='numerical_num_trees':e['cells'][0]['num_trees']+=1
    elif fault=='numerical_raw':e['cells'][0]['raw_ref']={'path':'other','sha256':'0'*64}
    elif fault=='numerical_oracle_dates':e['cells'][0]['native_oracles']['float64_native_dates']=False
    elif fault=='numerical_check_steps':e['cells'][0]['qualification']['checks'][0]['finite_disposable_AdamW']=False
    reject('independent_metadata_gate_corruption_'+fault,lambda:fit.require_terminal_numerical_v1(e,n,i,f,t,r,l))
for p,h in initial.items():check('consumer_source_snapshot_preserved_'+p,sha(ROOT/p)==h)
for p,h in fixed.items():check('fixed_reference_preserved_'+p,sha(ROOT/p)==h)
report={'schema':'census_tsinfer_consumer_independent_checks_v1','status':'completed_source_and_handmade_CPU',
 'count':len(checks),'checks':checks,'consumer_snapshot_sha256':initial,'fixed_reference_sha256':fixed,
 'test_source_sha256':sha(Path(__file__)),'details':details,
 'reviewer_prior_fit2_authorship_disclosed':True,'review_focus':'new observation/GPU consumer adaptation and exact scientific AST preservation, not independent reendorsement of previously authored fit2 mathematics',
 'actual_arrays_GPU_live_budgets_installs_or_Git':False,'formal_verdict_pending_author_seal':True}
(DIR/'independent_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'checks':len(checks),'status':report['status'],'formal_verdict_pending_author_seal':True},indent=2))
