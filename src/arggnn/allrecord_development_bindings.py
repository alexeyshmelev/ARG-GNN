"""Fresh all-record control plans and completion binding; never prepare data."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from . import development_training_bindings as common
from .allrecord_boundary_temporal import INPUT_SCOPE,AGGREGATION,FEATURE_COLUMNS
from .resources import ROOT

EXTRA_CORE=('allrecord_boundary_temporal.py','allrecord_development_bindings.py','train_allrecord_development.py')
FAMILIES=('allrecord_gru','allrecord_transformer')
SEEDS=[101,202,303]
SPLITS={'train':64,'val':16,'test':0}
RAW_NLL='float64 stable log-softmax of uncalibrated raw logits'
SELECTION='minimum mean parent-validation uncalibrated logit NLL'
BINDING_KEYS=('completion_receipt_sha256','source_dataset_manifest_sha256',
    'boundary_dataset_manifest_sha256','boundary_preparation_sha256',
    'age_dataset_manifest_sha256','age_preparation_sha256','preparation_execution_freeze_sha256')
SAFE=re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z')


def safe_id(value):
    if not isinstance(value,str) or SAFE.fullmatch(value) is None:raise RuntimeError('Safe bounded model identifier required')
    return value


def development_path(value):
    if not isinstance(value,(str,Path)) or not str(value) or '\\' in str(value):raise RuntimeError('Invalid development path')
    p=Path(value)
    if '..' in p.parts:raise RuntimeError('Development path traversal is forbidden')
    p=(p if p.is_absolute() else ROOT/p).resolve()
    root=ROOT.resolve()
    if p==root or not p.is_relative_to(root):raise RuntimeError('Development path escapes project')
    for part in p.relative_to(root).parts:
        if part.lower()=='test' or any(token in part.lower() for token in ('confirmatory','final_test','final-test','confirmation')):
            raise RuntimeError('Held-out namespaces are unavailable to all-record development')
    return p


def _unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise RuntimeError('Duplicate JSON key')
        result[key]=value
    return result


@dataclass(frozen=True)
class CapturedControl:
    path: Path
    content: bytes
    sha256: str
    values: dict
    signature: tuple

    def check(self,full=False):
        if common.artifact_signature(self.path)!=self.signature:raise RuntimeError('Captured control changed')
        if full and common.sha(self.path)!=self.sha256:raise RuntimeError('Captured control hash changed')
        if common.artifact_signature(self.path)!=self.signature:raise RuntimeError('Captured control changed during check')


def capture_control(path,expected_sha256):
    path=development_path(path)
    if not common.valid_sha(expected_sha256):raise RuntimeError('Real factual control SHA256 required')
    before=common.artifact_signature(path);content=path.read_bytes()
    if common.artifact_signature(path)!=before or hashlib.sha256(content).hexdigest()!=expected_sha256:
        raise RuntimeError('Control bytes/hash changed or differ from supplied freeze')
    value=json.loads(content,object_pairs_hook=_unique_object)
    if not isinstance(value,dict):raise RuntimeError('Control must be a JSON object')
    return CapturedControl(path,content,expected_sha256,value,before)


def implementation_hashes():return common.snapshot(extra=EXTRA_CORE)


def family_grid(family):
    if family not in FAMILIES:raise RuntimeError('Unknown all-record family')
    models=[];specs={};rates={}
    for width in (32,64):
        for dropout in (0.,.1):
            for lr in (.0003,.001):
                name=f'{family}_h{width}_do{int(dropout*10)}_lr{int(lr*1e6):04d}'
                models.append(name)
                specs[name]={'family':family,'hidden_dim':width,'source_depth':2,
                    'temporal_depth':2,'dropout':dropout,'transformer_heads':4}
                rates[name]=lr
    return {'models':models,'model_specs':specs,'model_learning_rates':rates,
        'input_scope':INPUT_SCOPE,'cross_tree_aggregation':AGGREGATION[family]}


def plan_constants():
    return {'schema_version':1,'kind':'allrecord_fresh_development_search_plan_v1',
        'status':'proposal_only_not_executable_until_completed_preparation_is_bound',
        'development_namespace':'development-v1','splits':dict(SPLITS),'training_seeds':list(SEEDS),
        'input_dim':47,'num_classes':4,'num_queries':64,'reference_haplotypes':24,
        'windows':8,'source_depth':2,'temporal_depth':2,'epochs':48,'patience':8,
        'weight_decay':.0001,'gradient_clip_norm':1.,'max_attention_score_bytes':268435456,
        'nll_policy':RAW_NLL,'selection':SELECTION,'tie_tolerance_nats':.0001,
        'input_scope':INPUT_SCOPE,'source_anchors':'none; observed records are the complete source bank',
        'record_subdivision_invariant':False,'target_created_tokens':False,'confirmatory_access':False,
        'interpretation':'Independent all-record observed-boundary standard temporal controls. Development choices only; no independent confirmation, native reproduction, superiority or novelty claim.',
        'families':{family:family_grid(family) for family in FAMILIES}}


def make_search_plan():return {**plan_constants(),'reviewed_implementation_sha256':implementation_hashes()}


def validate_plan(plan):
    expected=plan_constants()
    if not isinstance(plan,dict) or set(plan)!=set(expected)|{'reviewed_implementation_sha256'} or any(plan.get(k)!=v for k,v in expected.items()):
        raise RuntimeError('Reviewed all-record plan schema/grid/budget/scope changed')
    for key in ('schema_version','input_dim','num_classes','num_queries','reference_haplotypes','windows','source_depth','temporal_depth','epochs','patience','max_attention_score_bytes'):
        if type(plan[key]) is not int:raise RuntimeError('Frozen integer plan dimensions required')
    if plan.get('confirmatory_access') is not False or plan.get('record_subdivision_invariant') is not False or plan.get('target_created_tokens') is not False:
        raise RuntimeError('Boolean development-only source/readout contracts required')
    for family in FAMILIES:
        for spec in plan['families'][family]['model_specs'].values():
            if any(type(spec[key]) is not int for key in ('hidden_dim','source_depth','temporal_depth','transformer_heads')) or isinstance(spec['dropout'],bool):
                raise RuntimeError('Typed reviewed model grid required')
    pins=plan['reviewed_implementation_sha256']
    if pins!=implementation_hashes():raise RuntimeError('Reviewed all-record source/dependency hashes changed')


def _source_settings(completion,completion_path,completion_sha):
    if not common.valid_sha(completion_sha) or not isinstance(completion,dict) or completion.get('inferred_boundary_and_age_tiles')!=640 or completion.get('all_stages_complete') is not True or completion.get('train_parents')!=64 or completion.get('validation_parents')!=16 or completion.get('test_parents')!=0 or completion.get('training_or_confirmatory_selection_frozen') is not False:
        raise RuntimeError('Factual successful fresh64/16/TEST0 completion required')
    facts=completion['stage_manifests']
    value={'completion_receipt':str(development_path(completion_path)),
        'completion_receipt_sha256':completion_sha,
        'source_dataset_manifest_sha256':facts['inferred']['sha256'],
        'boundary_dataset_manifest_sha256':facts['boundary']['sha256'],
        'boundary_preparation_sha256':facts['boundary']['preparation_sha256'],
        'age_dataset_manifest_sha256':facts['age']['sha256'],'age_preparation_sha256':facts['age']['preparation_sha256'],
        'preparation_execution_freeze_sha256':completion['development_execution_freeze_file_sha256'],
        'preparation_execution_freeze':str(development_path(completion.get('development_execution_freeze_path',common.DEFAULT_PREPARATION_FREEZE))),
        'boundary_data':str(development_path(facts['boundary']['path']).parent),
        'age_data':str(development_path(facts['age']['path']).parent)}
    if any(not common.valid_sha(value[k]) for k in BINDING_KEYS):raise RuntimeError('All factual completed-source hashes required')
    return value


def bound_configs(plan,plan_path,plan_sha,completion,completion_path,completion_sha):
    """Pure assembly after external completed-source admission; no files emitted."""
    if not common.valid_sha(plan_sha):raise RuntimeError('Actual reviewed plan hash required')
    validate_plan(plan)
    source=_source_settings(completion,completion_path,completion_sha)
    keys=('schema_version','development_namespace','splits','training_seeds','input_dim','num_classes',
        'num_queries','reference_haplotypes','windows','source_depth','temporal_depth','epochs','patience',
        'weight_decay','gradient_clip_norm','max_attention_score_bytes','nll_policy','selection',
        'record_subdivision_invariant','target_created_tokens','interpretation')
    result={}
    for family in FAMILIES:
        result[family]={**{key:deepcopy(plan[key]) for key in keys},**deepcopy(plan['families'][family]),**source,
            'stage':'fresh_development_allrecord_boundary47_training_v1','family':family,
            'search_plan':str(development_path(plan_path)),'search_plan_sha256':plan_sha,
            'bound_implementation_sha256':dict(plan['reviewed_implementation_sha256']),
            'confirmatory_access':False}
    return result


def validate_settings(settings):
    if not isinstance(settings,dict) or settings.get('family') not in FAMILIES:raise RuntimeError('Supported all-record family required')
    constants=plan_constants()
    keys=('schema_version','development_namespace','splits','training_seeds','input_dim','num_classes',
        'num_queries','reference_haplotypes','windows','source_depth','temporal_depth','epochs','patience',
        'weight_decay','gradient_clip_norm','max_attention_score_bytes','nll_policy','selection',
        'record_subdivision_invariant','target_created_tokens','interpretation')
    if settings.get('record_subdivision_invariant') is not False or settings.get('target_created_tokens') is not False:
        raise RuntimeError('Boolean record/readout contracts required')
    if any(settings.get(key)!=constants[key] for key in keys) or settings.get('stage')!='fresh_development_allrecord_boundary47_training_v1' or settings.get('confirmatory_access') is not False:
        raise RuntimeError('All-record family source/schema/optimization budget changed')
    for key in ('schema_version','input_dim','num_classes','num_queries','reference_haplotypes','windows','source_depth','temporal_depth','epochs','patience','max_attention_score_bytes'):
        if type(settings[key]) is not int:raise RuntimeError('Frozen integer dimensions required')
    grid=family_grid(settings['family'])
    if any(settings.get(key)!=value for key,value in grid.items()):raise RuntimeError('Exact ordered eight-configuration all-record grid/scope required')
    for name in settings['models']:
        safe_id(name)
        spec=settings['model_specs'][name]
        if type(spec['hidden_dim']) is not int or type(spec['source_depth']) is not int or type(spec['temporal_depth']) is not int or type(spec['transformer_heads']) is not int or isinstance(spec['dropout'],bool) or isinstance(settings['model_learning_rates'][name],bool):
            raise RuntimeError('Typed width/depth/dropout/LR grid required')
    for key in (*BINDING_KEYS,'search_plan_sha256'):
        if not common.valid_sha(settings.get(key)):raise RuntimeError('Real completed source/plan bindings required')
    pins=settings.get('bound_implementation_sha256')
    if not isinstance(pins,dict) or pins!=implementation_hashes():raise RuntimeError('Reviewed all-record implementation/dependency binding changed')
    for key in ('search_plan','completion_receipt','preparation_execution_freeze','boundary_data','age_data'):
        development_path(settings[key])
    expected_keys=set(keys)|set(grid)|set(BINDING_KEYS)|{'stage','family','search_plan','search_plan_sha256',
        'bound_implementation_sha256','confirmatory_access','completion_receipt','preparation_execution_freeze','boundary_data','age_data'}
    if set(settings)!=expected_keys:raise RuntimeError('Exact bound config fields required')
    return settings


def capture_settings(path,expected_sha):
    captured=capture_control(path,expected_sha);validate_settings(captured.values);return captured


def require_dataset_contract(dataset):
    if len(dataset.records)!=640:raise RuntimeError('All640 observed source tiles required')
    for split in ('train','val'):
        cases=dataset.cases(split)
        if len(cases)!=SPLITS[split]:raise RuntimeError('Fresh parent count changed')
        for case in cases:
            if len(case.tiles)!=8 or len(case.query_node_ids)!=64 or case.metadata['split']!=split or case.metadata['num_classes']!=4:
                raise RuntimeError('Fresh all-record parent schema changed')
            for tile in case.tiles:
                bank=tile.bank;cache=bank.query_cache
                if tuple(bank.feature_columns)!=FEATURE_COLUMNS or bank.leaf_x.shape[-1]!=47 or cache is None or len(cache.reference_ids)!=24 or not torch_equal(bank.query_node_ids,case.query_node_ids):
                    raise RuntimeError('Sealed boundary47/query/reference contract differs')


def torch_equal(first,second):
    # Delay Torch import until real dataset admission; control-only checks stay cheap.
    import torch
    return torch.equal(first,second)


def bind_completed_development(completion_path,completion_sha,plan_path,plan_sha,*,output_dir=None):
    """Reviewable real configs appear only after full completed-source verification."""
    output=None if output_dir is None else development_path(output_dir)
    if output is not None and output.exists():raise RuntimeError('Fresh all-record config directory required')
    captured=capture_control(plan_path,plan_sha);validate_plan(captured.values)
    pins=implementation_hashes()
    completion,dataset,ages,lock=common.verify_completion(development_path(completion_path),completion_sha,with_snapshot=True)
    del ages
    require_dataset_contract(dataset)
    values=bound_configs(captured.values,captured.path,plan_sha,completion,completion_path,completion_sha)
    for value in values.values():validate_settings(value)
    captured.check(full=True);lock.check(full=True)
    if implementation_hashes()!=pins:raise RuntimeError('All-record dependencies changed during binding')
    outputs={family:(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode() for family,value in values.items()}
    if output_dir is not None:
        output.mkdir(parents=True)
        for family,content in outputs.items():
            with (output/(family+'.json')).open('xb') as handle:handle.write(content)
        captured.check(full=True);lock.check(full=True)
        if implementation_hashes()!=pins:raise RuntimeError('Dependencies changed before config sealing')
    return values,outputs,{family:hashlib.sha256(content).hexdigest() for family,content in outputs.items()}


def admit_captured_run(captured,*,imported_pins):
    """Use the shared AD03/04 admission boundary, including explicit own pins."""
    settings=validate_settings(captured.values)
    captured.check(full=True)
    plan=capture_control(settings['search_plan'],settings['search_plan_sha256']);validate_plan(plan.values)
    completion,dataset,ages,lock=common.admit_run(settings,captured.path,captured.sha256,imported_pins=imported_pins)
    require_dataset_contract(dataset)
    expected=bound_configs(plan.values,plan.path,plan.sha256,completion,settings['completion_receipt'],settings['completion_receipt_sha256'])
    if settings!=expected[settings['family']]:raise RuntimeError('Captured config differs from factual reviewed plan/completion')
    lock.hashes[str(plan.path)]=plan.sha256;lock.signatures[str(plan.path)]=plan.signature
    plan.check(full=True);captured.check(full=True);lock.check(full=True)
    return completion,dataset,ages,lock


def check_live(settings,captured,lock,*,full=False):
    captured.check(full=full)
    return common.check_live(settings,captured.path,captured.sha256,source_snapshot=lock,full_sources=full)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--expected-plan-sha256',required=True)
    parser.add_argument('--completion',required=True);parser.add_argument('--expected-completion-sha256',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args(argv)
    _,_,hashes=bind_completed_development(args.completion,args.expected_completion_sha256,args.plan,args.expected_plan_sha256,output_dir=args.output_dir)
    print(json.dumps({'event':'completed_allrecord_configs_bound','sha256':hashes,'test_access':False}),flush=True)


if __name__=='__main__':main()
