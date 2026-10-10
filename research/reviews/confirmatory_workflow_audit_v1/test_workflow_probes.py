"""Independent synthetic CPU workflow audit. Positive lock admission is mocked.

No actual DEV selection, fitted temperature, genuine lock, cohort, target or GPU.
The integrated test executes real observed star-tree QA, model factories/checkpoint
loading, separate primary/dense passes, roster admission and joined diagnostics.
"""
import copy
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from arggnn import confirmatory_analysis_runner as analysis
from arggnn import confirmatory_custodian as custody
from arggnn import confirmatory_prediction_runner as pred
from arggnn.secondary_control_bindings import controls_for

ROOT = pred.ROOT
REAL_VALIDATOR = pred.lock_api.validate_confirmatory_lock
torch.set_num_threads(1)


def load_helper(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'tests'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ct = load_helper('independent_custody_fixture_helper', 'test_confirmatory_custodian.py')
pt = load_helper('independent_prediction_settings_helper', 'test_confirmatory_prediction_runner.py')
at = load_helper('independent_analysis_fixture_helper', 'test_confirmatory_analysis_runner.py')


def write(root, name, value, binary=False):
    p = root/name; p.parent.mkdir(parents=True, exist_ok=True)
    body = value if binary else analysis._canonical(value)
    p.write_bytes(body)
    return {'path': name, 'sha256': analysis._sha(body)}


def test_source_schema_interoperability():
    a = analysis.analysis_schema_values()
    p = pred.prediction_schema_values()
    c = custody.custodian_schema_values()
    assert p['observed_npz'] == a['observed_npz']
    assert p['prediction_arrays'] == a['prediction_npz']['arrays']
    assert set(a['admission_ledger']['attempt_fields']) == set(c['attempt_fields'])-{'native_inference_returncode'}
    assert a['policy']['training_seeds'] == c['policy']['training_seeds'] == pred.SEEDS
    assert a['full_protocol_readiness'] is False


def mock_cuda_runtime(root):
    # Only mocked operational metadata. Never initializes or inspects a GPU.
    uid='00000000-0000-4000-8000-000000000002'
    return {'device_kind':'cuda','initialization_stage':'complete','process_title':pred.PROCESS_TITLE,
        'software':{'python':sys.version,'python_executable':sys.executable,'torch':torch.__version__,
                    'torch_cuda':torch.version.cuda or 'SYNTHETIC_MOCK_CUDA','torch_build':torch.__config__.show()},
        'device':{'logical_index':0,'name':'SYNTHETIC_MOCK_GPU_NOT_ACTUAL','compute_capability':[9,0],
            'total_memory_bytes':1000000,'runtime_uuid':'GPU-'+uid,'canonical_runtime_uuid':uid},
        'deterministic_execution_policy':copy.deepcopy(pred.DETERMINISTIC_POLICY),'physical_gpu':2,
        'visible_gpu_uuid':'GPU-'+uid,'canonical_visible_gpu_uuid':uid,
        'lease_path':str(root/'.runtime/locks/gpu_2.lock'),'lease_fd':9,
        'launcher_metadata':{'gpu':2,'uuid':'GPU-'+uid,'job':'synthetic_mock_cuda_metadata_only'}}


def actual_cli_source_pins():
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run([sys.executable,'-c',
        'import json;from arggnn import confirmatory_prediction_runner as p;print(json.dumps(p._IMPORTED_PINS))'],
        cwd=ROOT,env=environment,text=True,capture_output=True,check=True)
    return json.loads(result.stdout)


def make_integrated_study(root, monkeypatch):
    study = ct.study.__wrapped__(root, monkeypatch)
    payload = study['frozen'].payload
    # Deliberately retain the mocked lock object. Consumer-level interoperability,
    # never genuine development replay, fitting, published-lock/cohort admission.
    protocol = json.loads((ROOT/'research/protocols/confirmatory_design_v1.json').read_text())
    payload['bound_references']['protocol_json'] = write(root, 'metadata/protocol.json', protocol)
    registry = json.loads((root/payload['bound_references']['parent_seed_registry']['path']).read_text())
    registry['analysis_seed'] = analysis.ANALYSIS_SEED
    payload['bound_references']['parent_seed_registry'] = write(root, 'metadata/registry.json', registry)
    study['request']['parent_seed_registry_sha256'] = payload['bound_references']['parent_seed_registry']['sha256']
    payload['bound_references']['analysis_contract'] = write(root, 'metadata/contract.json',
        {k: protocol[k] for k in ('confirmation', 'separate_prespecified_analyses')})
    for path in list(analysis.CODE_PATHS.values())+[value['path'] for value in analysis.analysis_schema_values()['terminal_receipt']['source_artifacts'].values()]:
        dst = root/path; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes((ROOT/path).read_bytes())
    methods = {}
    for index, (method, family) in enumerate((('candidate','candidate'), ('strongest_required_baseline','reference_smoother'),
            ('gain_matched_diagonal','gain_matched_diagonal'), ('static_local','static_local'))):
        config = pt.settings('candidate' if family in pred.CONTROLS else family)
        if family in pred.CONTROLS:
            spec = dict(geometry='plain', kernel_depth=1, coupling='full', rate_parameterization='log')
            config.update(input_dim=47, num_classes=4, selected_candidate_spec=spec,
                          model_specs={'unit': controls_for(spec)[family]})
        config_ref = write(root, 'synthetic_development/'+method+'.json', config)
        seeds = []
        for seed in pred.SEEDS:
            torch.manual_seed(seed+index)
            model = pred.make_control('unit', config) if family in pred.CONTROLS else pred.lock_api._primary_model(family, 'unit', config)
            stream = io.BytesIO(); torch.save({'model': model.state_dict()}, stream)
            checkpoint = None if family == 'reference_smoother' else write(root, f'synthetic_development/{method}_{seed}.pt', stream.getvalue(), True)
            devstream = io.BytesIO(); torch.save({'synthetic_only': torch.zeros((1,64,4))}, devstream)
            dev = write(root, f'synthetic_development/{method}_{seed}_dev.pt', devstream.getvalue(), True)
            seeds.append({'seed': seed, 'checkpoint': checkpoint, 'development_prediction': dev})
        methods[method] = dict(family_id=family, configuration_id='unit', protocol=config_ref,
            choice={'parameters':sum(p.numel() for p in model.parameters()), 'synthetic_only':True}, seed_bundles=seeds)
    payload.update(candidate=methods['candidate'], strongest_required_baseline=methods['strongest_required_baseline'],
        secondary_controls={k:methods[k] for k in ('gain_matched_diagonal','static_local')},
        numerical_execution_policy=pred.DETERMINISTIC_POLICY, captured_runtime=mock_cuda_runtime(root),
        binding_implementation_sha256={str(Path(path).relative_to(ROOT)):digest for path,digest in pred.lock_api._IMPORTED_PINS.items()},
        analysis_environment_versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','torch')})
    payload['code_artifacts']['prediction'] = pred.prediction_schema_values()['code_artifacts']['prediction'] | {'entry_point':'main'}
    payload['code_artifacts']['joined_analysis'] = analysis.analysis_schema_values()['code_artifacts']['joined_analysis'] | {'entry_point':'main'}
    temps = dict(schema_version=1, kind='prespecified_validation_temperature_manifest', status='frozen_before_test_preparation',
        training_seeds=pred.SEEDS, final_test_access=False, records=[dict(method_id=m, seed=s['seed'], temperature=1.3,
            checkpoint=s['checkpoint'], development_prediction=s['development_prediction'],
            fit_scope='fresh_development_validation_only') for m,b in methods.items() for s in b['seed_bundles']])
    payload['bound_references']['temperature_manifest'] = write(root, 'metadata/temperature.json', temps)
    ct.rewrite_request(study)
    return study


def test_real_cpu_custodian_prediction_analysis_join(tmp_path, monkeypatch):
    study = make_integrated_study(tmp_path, monkeypatch)
    monkeypatch.setattr(pred, 'require_storage', lambda **kw: 0)  # No global storage scan, fixture below 200MB.
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    events = []; target_paths = study['targets']; original_read = analysis._Capture.read
    def read(capture, ref, **kw):
        path = capture.ref(ref)['path']
        if path in target_paths: events.append(('target',path))
        return original_read(capture,ref,**kw)
    monkeypatch.setattr(analysis._Capture,'read',read)
    release = ct.release(study)
    assert release.receipt['admitted_parents']==2 and release.receipt['attempted_parents']==192
    assert len(release.roster['entries']) == 24
    roster = copy.deepcopy(release.roster)
    initial_ref = release.receipt['initial_prediction_roster']; ledger_ref = release.receipt['admission_ledger']
    admission = analysis.admit_analysis(tmp_path/study['lock_ref']['path'], study['lock_ref']['sha256'],
        tmp_path/initial_ref['path'], initial_ref['sha256'], tmp_path/ledger_ref['path'], ledger_ref['sha256'], project_root=tmp_path)
    events.clear()
    with pytest.raises(ValueError,match='terminal prediction roster'): analysis.analyze_confirmatory(admission)
    assert events == []
    anchor_records = []; forward_records = []; original_anchor = pred.prepare_anchor_kernel_bank; original_forward = pred._forward_grid
    def anchor(bank, *args, **kw):
        value = original_anchor(bank,*args,**kw)
        anchor_records.append((bank.positions.numel(),value.layout.anchors.clone(),kw['num_anchors']))
        return value
    def forward(model, sequence, admitted, device):
        assert torch.is_inference_mode_enabled() is False
        forward_records.append((admitted.method_id,admitted.seed,admitted.parent,len(sequence.positions)))
        return original_forward(model,sequence,admitted,device)
    monkeypatch.setattr(pred,'prepare_anchor_kernel_bank',anchor)
    monkeypatch.setattr(pred,'_forward_grid',forward)
    predictions = []; receipts = []
    for row in roster['entries']:
        cell = pred.admit_prediction(tmp_path/study['lock_ref']['path'], study['lock_ref']['sha256'],
            tmp_path/ledger_ref['path'],ledger_ref['sha256'],row['method_id'],row['seed'],row['parent_id'],project_root=tmp_path)
        result = pred.predict_confirmatory(cell,f"cells/{row['method_id']}_{row['seed']}_{row['parent_id']}",device='cpu')
        assert result['values']['cpu_validation_only'] and not result['values']['cuda_execution_performed']
        assert result['values']['target_artifacts_opened'] is False
        assert result['values']['raw_primary_dense_separate_forward_passes']
        row.update(status='success',prediction=result['prediction'],terminal_receipt=result['receipt'],reason='')
        predictions.append(result['prediction']); receipts.append(result['receipt'])
    assert events==[] and len(forward_records)==48
    assert {r[-1] for r in forward_records}=={128,4096}
    assert len(anchor_records)==24*16 and all(r[2]==32 for r in anchor_records)
    assert all(torch.equal(anchor_records[i][1],anchor_records[i+8][1]) for i in range(0,len(anchor_records),16))
    # Real CPU terminal evidence must reject both direct admission and custody sealing.
    cpu_roster_ref=write(tmp_path,'release/prediction_roster_cpu_validation_only.json',roster)
    with pytest.raises(ValueError,match='CPU-validation prediction'):
        analysis.admit_analysis(tmp_path/study['lock_ref']['path'],study['lock_ref']['sha256'],
            tmp_path/cpu_roster_ref['path'],cpu_roster_ref['sha256'],tmp_path/ledger_ref['path'],ledger_ref['sha256'],project_root=tmp_path)
    seal_args=(tmp_path/study['lock_ref']['path'],study['lock_ref']['sha256'],
        tmp_path/ledger_ref['path'],ledger_ref['sha256'],tmp_path/initial_ref['path'],initial_ref['sha256'])
    with pytest.raises(ValueError,match='CPU-validation prediction'):
        custody.seal_confirmatory_prediction_roster(*seal_args,terminal_receipts=receipts,
            output_directory=tmp_path/'release/reject_actual_cpu',project_root=tmp_path)
    assert events==[]
    # Consumer interoperability positive fixtures deliberately replace CUDA execution
    # metadata only. They are never genuine GPU receipts or lock/cohort evidence.
    cli_pins=actual_cli_source_pins()
    for path,digest in cli_pins.items():
        body=(ROOT/path).read_bytes(); assert analysis._sha(body)==digest
        write(tmp_path,path,body,True)
    mocked_receipts=[]
    for index,ref in enumerate(receipts):
        terminal=json.loads((tmp_path/ref['path']).read_text())
        terminal.update(cpu_validation_only=False,cuda_execution_performed=True,
            runtime=copy.deepcopy(study['frozen'].payload['captured_runtime']),
            imported_source_sha256=cli_pins,cuda_allocator_peak_bytes={'allocated':1,'reserved':1})
        mocked_receipts.append(write(tmp_path,f'synthetic_mock_cuda_terminals/cell{index}.json',terminal))
    sealed=custody.seal_confirmatory_prediction_roster(*seal_args,terminal_receipts=mocked_receipts,
        output_directory=tmp_path/'release/sealed_mock_cuda',project_root=tmp_path)
    assert events==[] and sealed.receipt['matrix_terminal']
    assert sealed.receipt['cell_counts']=={'success':24,'failed':0,'unresolved':0}
    roster_ref=sealed.receipt['roster']
    admitted = analysis.admit_analysis(tmp_path/study['lock_ref']['path'], study['lock_ref']['sha256'],
        tmp_path/roster_ref['path'],roster_ref['sha256'],tmp_path/ledger_ref['path'],ledger_ref['sha256'],project_root=tmp_path)
    assert events==[]
    report = analysis.analyze_confirmatory(admitted,output_path=tmp_path/'release/joined_report.json')['values']
    assert len(events)==4 and len(report['descriptive_parent_seed_diagnostics'])==24
    assert report['all_predictions_successful'] and report['admission_target_shortfall']
    assert report['primary']['report']['independent_parents']==2
    assert not report['primary']['interpretation']['robust_positive_confirmation']
    assert report['conditional_holm_family_size']==2
    assert all(len(d['raw_primary']['calibration'])==15 for d in report['descriptive_parent_seed_diagnostics'])
    assert all(d['validation_temperature']==1.3 for d in report['descriptive_parent_seed_diagnostics'])
    assert not torch.cuda.is_initialized()
    summary = {'mocked_positive_lock_admission':True,'genuine_scientific_evidence':False,
        'handmade_parents':2,'attempts':192,'cells':24,'real_cpu_forward_grid_calls':len(forward_records),
        'primary_grid_points':128,'dense_grid_points':4096,'anchor_preparation_calls':len(anchor_records),
        'fixed_anchor_count':32,'prediction_target_reads':0,'analysis_target_reads':len(events),
        'positive_confirmation':report['primary']['interpretation']['robust_positive_confirmation'],
        'prediction_refs':predictions,'cpu_terminal_receipt_refs':receipts,
        'analysis_accepts_cpu_validation_predictions_without_terminal_receipt_binding':False,
        'actual_cpu_terminal_receipts_rejected':True,'positive_cuda_receipts_explicitly_mocked':True,
        'cross_process_cli_source_pin_count':len(cli_pins),'mock_cuda_terminal_refs':mocked_receipts}
    write(tmp_path,'integrated_summary.json',summary)
    # Preserve small deterministic receipt outside pytest temporary naming as exact evidence reference.
    out = ROOT/'research/reviews/confirmatory_workflow_audit_v1/integrated_summary.json'
    out.write_bytes(analysis._canonical(summary))


def test_actual_validator_rejects_handmade_lock_before_array_read(tmp_path, monkeypatch):
    study = make_integrated_study(tmp_path, monkeypatch)
    monkeypatch.setattr(pred.lock_api,'validate_confirmatory_lock',REAL_VALIDATOR)
    with pytest.raises(ValueError,match='published complete ID'):
        ct.release(study)
    assert not (tmp_path/'release').exists()


def test_cuda_uuid_canonical_forms_are_accepted_without_cuda(monkeypatch):
    runtime={}; import setproctitle
    monkeypatch.setattr(pred,'set_research_title',lambda:None)
    monkeypatch.setattr(setproctitle,'getproctitle',lambda:pred.PROCESS_TITLE)
    monkeypatch.setattr(pred,'configure_deterministic_development',lambda:pred.DETERMINISTIC_POLICY)
    uuid='df11de7f-7a15-ebe2-085d-9c7d907f3786'
    monkeypatch.setattr(pred,'validate_launcher_lease',lambda:{'visible_gpu_uuid':'GPU-'+uuid})
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda i:SimpleNamespace(name='synthetic',total_memory=1,major=9,minor=0,uuid=uuid))
    pred.initialize_execution('cuda',runtime)
    assert runtime['initialization_stage']=='complete'
    assert not torch.cuda.is_initialized()


def test_prediction_publication_rejects_replaced_pending_inode(tmp_path):
    output=tmp_path/'published'; parked=tmp_path/'owned_parked'; foreign=tmp_path/'foreign'
    foreign.write_bytes(b'unrelated replacement bytes')
    def replace_pending():
        pending=next(tmp_path.glob('.prediction_pending_*'))
        pending.rename(parked); foreign.rename(pending)
    with pytest.raises(ValueError,match='inode|ownership|changed'):
        pred._publish(output,b'expected owned bytes',replace_pending)
    assert not output.exists()
    assert next(tmp_path.glob('.prediction_pending_*')).read_bytes()==b'unrelated replacement bytes'


def test_analysis_publication_rejects_replaced_pending_inode(tmp_path):
    output=tmp_path/'published'; parked=tmp_path/'owned_parked'; foreign=tmp_path/'foreign'
    foreign.write_bytes(b'unrelated replacement bytes')
    def replace_pending():
        pending=next(tmp_path.glob('.analysis_pending_*'))
        pending.rename(parked); foreign.rename(pending)
    with pytest.raises(ValueError,match='inode changed'):
        analysis._publish(output,b'expected owned bytes',replace_pending)
    assert not output.exists()
    assert next(tmp_path.glob('.analysis_pending_*')).read_bytes()==b'unrelated replacement bytes'


def test_analysis_exact_fixed_tile_edges_should_reject(tmp_path,monkeypatch):
    study=at.fixture_study(tmp_path,monkeypatch)
    # Deliberately mutate fixed edges independently of helper changes.
    ref=study['identity_refs'][0]
    identity=json.loads((tmp_path/ref['path']).read_text())
    identity['artificial_tile_edges_bp']=[1.,2.,3.,4.,5.,6.,7.]
    study['ledger']['attempts'][0]['observed_identity']=write(tmp_path,ref['path'],identity)
    at.remake(study,'ledger')
    with pytest.raises(ValueError,match='tile|edge|physical'):
        at.admit(study)


def test_secondary_shortfall_retains_adjusted_results_but_blocks_primary_confirmation(tmp_path,monkeypatch):
    study=at.fixture_study(tmp_path,monkeypatch)
    report=analysis.analyze_confirmatory(at.admit(study))['values']
    assert report['admission_target_shortfall']
    assert not report['primary']['interpretation']['robust_positive_confirmation']
    # V1 clearly gates primary precision; no unsupported restriction is imposed
    # on whether secondary adjusted numerical results remain visible.
    assert report['conditional_holm_family_size']==2
    assert all(0<=row['holm_two_sided_pvalue']<=1 for row in report['conditional_secondary_holm'])


def test_precision_and_failure_primary_positive_claim_guards():
    good=analysis.paired_parent_report(np.ones((160,3)), np.full((160,3),1.03),
        parent_ids=list(range(160)),training_seeds=pred.SEEDS)
    assert analysis._interpret(good,complete=True,target=160)['robust_positive_confirmation']
    assert not analysis._interpret(good,complete=False,target=160)['robust_positive_confirmation']
    short=copy.deepcopy(good); short['independent_parents']=159
    assert not analysis._interpret(short,complete=True,target=160)['robust_positive_confirmation']
    wide=copy.deepcopy(good); wide['achieved_t_interval_halfwidth']=.006
    assert not analysis._interpret(wide,complete=True,target=160)['robust_positive_confirmation']


def test_lock_publication_rejects_replaced_pending_inode(tmp_path,monkeypatch):
    lt=load_helper('independent_lock_fixture_helper','test_confirmatory_lock.py')
    root=tmp_path/'lock_fixture'; root.mkdir()
    fixture=lt.evidence.__wrapped__(SimpleNamespace(mktemp=lambda name:root))
    evidence=next(fixture)
    output=root/'unit_audit_lock.json'; parked=root/'owned_lock_parked'; foreign=root/'foreign'
    foreign.write_bytes(b'unrelated lock replacement bytes')
    original=lt.lock._source_stable
    def check_and_replace():
        original()
        pending=list(root.glob('.lock_pending_*'))
        if pending:
            pending[0].rename(parked); foreign.rename(pending[0])
    monkeypatch.setattr(lt.lock,'_source_stable',check_and_replace)
    try:
        with pytest.raises(ValueError,match='inode|ownership|changed'):
            lt.bind(evidence,output=output)
        assert not output.exists()
        assert next(root.glob('.lock_pending_*')).read_bytes()==b'unrelated lock replacement bytes'
    finally:
        next(fixture,None)


@pytest.mark.parametrize('publisher',[pred._publish,analysis._publish])
def test_publication_same_inode_byte_drift_should_reject(tmp_path,publisher):
    output=tmp_path/'published'
    def mutate_bytes():
        pending=next(tmp_path.glob('.*pending_*'))
        inode=pending.stat().st_ino
        pending.write_bytes(b'changed owned-inode output bytes')
        assert pending.stat().st_ino==inode
    with pytest.raises(ValueError,match='drift|hash|bytes|changed'):
        publisher(output,b'expected bytes',mutate_bytes)
    assert not output.exists()


def test_custodian_staged_file_byte_drift_should_reject(tmp_path):
    output=tmp_path/'release'
    def mutate_bytes(**kwargs):
        stage=next(tmp_path.glob('.custodian_pending_*'))
        (stage/'release_receipt.json').write_bytes(b'changed receipt bytes')
    with pytest.raises(ValueError,match='drift|hash|bytes|changed'):
        custody._publish(output,{'admission_ledger.json':b'ledger','prediction_roster_initial.json':b'roster',
            'release_receipt.json':b'expected receipt'},mutate_bytes)
    assert not output.exists()


def test_lock_publication_same_inode_byte_drift_should_reject(tmp_path,monkeypatch):
    lt=load_helper('independent_lock_byte_fixture_helper','test_confirmatory_lock.py')
    root=tmp_path/'lock_fixture'; root.mkdir()
    fixture=lt.evidence.__wrapped__(SimpleNamespace(mktemp=lambda name:root))
    evidence=next(fixture); output=root/'unit_audit_lock.json'; original=lt.lock._source_stable
    def check_and_mutate():
        original()
        pending=list(root.glob('.lock_pending_*'))
        if pending:
            inode=pending[0].stat().st_ino
            pending[0].write_bytes(b'changed same-inode lock bytes')
            assert pending[0].stat().st_ino==inode
    monkeypatch.setattr(lt.lock,'_source_stable',check_and_mutate)
    try:
        with pytest.raises(ValueError,match='drift|hash|bytes|changed'):
            lt.bind(evidence,output=output)
        assert not output.exists()
    finally:
        next(fixture,None)


def test_custodian_cleanup_preserves_replaced_stage_directory(tmp_path):
    output=tmp_path/'release'; parked=tmp_path/'owned_stage_parked'; replacement=[]
    def replace_stage(**kwargs):
        if replacement:return
        stage=next(tmp_path.glob('.custodian_pending_*'))
        stage.rename(parked); stage.mkdir(); (stage/'unrelated').write_bytes(b'unrelated directory bytes')
        replacement.append(stage)
    with pytest.raises((ValueError,OSError)):
        custody._publish(output,{'admission_ledger.json':b'ledger','prediction_roster_initial.json':b'roster',
            'release_receipt.json':b'receipt'},replace_stage)
    assert replacement[0].is_dir()
    assert (replacement[0]/'unrelated').read_bytes()==b'unrelated directory bytes'
    assert not output.exists()
