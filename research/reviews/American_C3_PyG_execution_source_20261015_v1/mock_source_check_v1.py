"""One metadata/mocked wiring check. No observations, labels or weights read."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,inspect,json,sys
from unittest.mock import patch
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('c3_execution_hooks',OUT/'execution_hooks_v1.py')
hooks=importlib.util.module_from_spec(spec);spec.loader.exec_module(hooks)
assert hooks.describe_v1()['Torch_imported'] is False
from arggnn import american_c3_gnn_targets_v1 as targets
from arggnn import american_c3_gnn_train_v1 as train
rows=targets.authoritative_registry_v1()
real_read=hooks.read
mapping={};bindings={}
for parent,row in rows.items():
    stem=str(OUT.relative_to(Path('/workspace-SR003.nfs2/ARG-GNN'))/'synthetic_metadata_only'/parent)
    def rec(name):return dict(path=stem+'/'+name,sha256='a'*64)
    binding=dict(simulation_completion_ref=rec('simulation/completed.json'),
        simulation_config_ref=rec('simulation_config.json'),inferred_observation_ref=rec('inference_manifest.json'))
    cfg=copy.deepcopy(real_read(row['simulation_config_ref']));cfg['execution_enabled']=True
    observed=rec('simulation/observed/manifest.json')
    mapping[binding['simulation_config_ref']['path']]=cfg
    mapping[binding['simulation_completion_ref']['path']]=dict(
        status='completed_C3_simulation_reconstruction',parent_id=parent,class_order=list(hooks.CLASSES),
        sample_unit='haploid',observation_ref=observed,private_truth_ref=rec('simulation/truth/c3_pulse_ancestry.npz'))
    mapping[binding['inferred_observation_ref']['path']]=dict(schema='american_c3_observed_v1',
        parent_id=parent,split='train',sequence_length_bp=2000000,
        counts_haploid={'AFR':200,'EUR':200,'ASIA':200,'ADMIX':1000},class_order=list(hooks.CLASSES),
        sample_unit='haploid',source_observation_ref=observed,query_ancestry_truth_in_inputs=False,
        genealogy_kind='inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG')
    bindings[parent]=binding

def fake_read(record):
    return copy.deepcopy(mapping[record['path']]) if record['path'] in mapping else real_read(record)

with patch.object(hooks,'read',fake_read):
    factory=hooks.make_actual_target_joiner_factory_v1(bindings)
    parent=hooks.TRAIN[0];calls=[]
    expected=mapping[bindings[parent]['simulation_completion_ref']['path']]['private_truth_ref']
    def resolver(pid):calls.append(pid);return expected
    joiner=factory(resolver,max_truth_uncompressed_bytes=32000000)
    assert joiner.rows[parent]['simulation_output_directory'].endswith('/simulation')
    assert joiner.rows[parent]['ancestry_seed']==rows[parent]['ancestry_seed']
    try:joiner.join(hooks.VAL[0],split='val',plan={},observation_seal={})
    except ValueError:pass
    else:raise AssertionError('VAL target join admitted')
    assert not calls
    def inherited_join(self,parent_id,**kwargs):
        assert self.resolve(parent_id)==expected
        return targets.TrainFocalTargetVectorV1(parent_id,{},'synthetic_identity',None,dict(synthetic_only=True))
    with patch.object(targets.TrainTargetJoinV1,'join',inherited_join):
        value=joiner.join(parent,split='train',plan={'observation_ref':bindings[parent]['inferred_observation_ref']},
            observation_seal={'synthetic_only':True})
    assert calls==[parent] and value.receipt['target_equations_unchanged'] is True
    assert value.receipt['actual_simulation_binding_schema']==hooks.BINDING_SCHEMA
    bad=mapping[bindings[parent]['inferred_observation_ref']['path']]
    old=bad['genealogy_kind'];bad['genealogy_kind']='simulated_true_topology_sanitized_ORACLE_not_published_inferred_ARG'
    try:hooks.actual_train_metadata_v1(bindings)
    except ValueError:pass
    else:raise AssertionError('ORACLE metadata accepted')
    bad['genealogy_kind']=old
    captured={}
    def delegated(cfg,*,train_target_joiner_factory,**kwargs):
        captured.update(cfg=cfg,factory=train_target_joiner_factory,kwargs=kwargs)
        return {'path':'synthetic_completion_not_written','sha256':'b'*64}
    boundaries=[]
    with patch.object(train,'run_train_v1',delegated):
        result=hooks.run_train_stage_v1({'synthetic_only':True},parent_plan_refs={},actual_train_bindings=bindings,
            writer=object(),device='cpu',worker_tmp_directory=str(OUT),resolve_train_truth=resolver,
            runtime_guard=boundaries.append)
    assert result['path']=='synthetic_completion_not_written' and len(captured['factory'].execution_binding['parents'])==20
    assert boundaries==['before_C3_TRAIN_scientific_import']

patched=ast.parse((OUT/'_patched_train_for_source_check.py').read_text())
function=next(n for n in patched.body if isinstance(n,ast.FunctionDef) and n.name=='run_train_v1')
assert 'train_target_joiner_factory' in [a.arg for a in function.args.kwonlyargs]
assert 'resolve_val_truth' not in inspect.signature(hooks.predict_one_original_VAL_v1).parameters
assert train.DEFAULTS['epochs']==30 and train.DEFAULTS['total_updates']==600
assert train.DEFAULTS['model_kwargs']['expected_haplotypes']==1600
receipt=dict(status='passed_metadata_and_mocked_wiring_only',actual_scientific_payloads_opened=False,
    actual_checkpoint_or_truth_files_opened=False,model_created=False,GPU=False,optimizer_updates=0,
    all20_actual_path_factory_preserves_scientific_seeds=True,VAL_target_denied_before_resolver=True,
    genuine_metadata_bound_TRAIN_ref_required=True,ORACLE_observation_metadata_rejected=True,
    inherited_truth_parser_and_focal_join_reused=True,full600_update_engine_delegated_without_reimplementation=True,
    patch_adds_explicit_factory_and_protocol_source_binding=True,prediction_API_accepts_no_truth_resolver=True,
    source_sha256=hashlib.sha256((OUT/'execution_hooks_v1.py').read_bytes()).hexdigest())
(OUT/'mock_source_check_receipt_v1.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
