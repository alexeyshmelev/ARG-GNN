"""Completed-source binding for unchanged GST-span/GraphSSM-ZOH adaptations."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass,field
import hashlib
import itertools
import json
from pathlib import Path

from . import argmix_development_bindings as shared
from .resources import ROOT
from .development_execution import DETERMINISTIC_POLICY

FAMILIES=('gst_ca_span','graphssm_zoh')
PLAN='configs/priorart_development_search_plan_v1.json'
AMENDMENT='research/protocols/development_scope_amendment_v2.json'
SEEDS=[101,202,303]
RAW_NLL='float64 stable log-softmax of uncalibrated raw logits'
SELECTION=shared.SELECTION
STATUS='independent_paper_equation_common_input_adaptation'
BINDINGS=('completion_receipt_sha256','source_dataset_manifest_sha256',
    'boundary_dataset_manifest_sha256','boundary_preparation_sha256',
    'age_dataset_manifest_sha256','age_preparation_sha256','preparation_execution_freeze_sha256')
SCOPE={
 'gst_ca_span':'all raw observed marginal-tree nodes/topology and verified F48 within each scored physical tile; every graph token has physical span key mass; no canonicalization, target-created tokens, positional embedding or truth features',
 'graphssm_zoh':'all raw observed marginal-tree nodes/topology and verified F48 within each scored physical tile; persistent query/reference leaf identities align exact-ZOH memory through every hold, anonymous internal memory resets; two local spatial diffusion blocks with reference information also in verified global F40 summaries; no canonicalization or truth features',
}
AGGREGATION={
 'gst_ca_span':'two sequential GST GCN/pooling/set-self-attention/node-cross-attention blocks, span-weighted unordered graph-token keys; no cross-tile aggregation',
 'graphssm_zoh':'two GraphSSM diffusion/exact-ZOH blocks, shared-weight forward/reverse scans in physical genomic order, snapshot-end query readout; no cross-tile aggregation',
}


def implementation_paths():
    return shared.implementation_paths() | {
        'src/arggnn/'+name:Path(__file__).with_name(name) for name in
        ('gst_baseline.py','graphssm_baseline.py','development_execution.py','priorart_development_bindings.py','train_priorart_development.py')
    } | {'scripts/bind_priorart_development.py':ROOT/'scripts/bind_priorart_development.py'}


def implementation_hashes():return {name:shared.sha(path) for name,path in implementation_paths().items()}


_IMPORTED_IMPLEMENTATION_SHA256=implementation_hashes()


def _unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise RuntimeError('Duplicate control JSON key')
        result[key]=value
    return result


@dataclass
class PriorArtSnapshot:
    """Compose the existing source catalog with independently frozen own code."""
    source: shared.IntegritySnapshot=field(default_factory=shared.IntegritySnapshot)
    implementation_sha256: dict=field(default_factory=implementation_hashes)
    captured: dict=field(default_factory=dict)
    control_signatures: dict=field(default_factory=dict)

    def control(self,path,expected):
        path=shared.development_path(path);shared.valid_sha(expected);key=str(path)
        if key in self.captured:
            if self.source.controls[key][0]!=expected:raise RuntimeError('Conflicting prior-art control hash')
            return deepcopy(self.captured[key])
        before=shared.signature(path)
        self.source.control(path,expected)
        if shared.signature(path)!=before:raise RuntimeError('Prior-art control changed during capture')
        value=json.loads(self.source.bytes_for(path),object_pairs_hook=_unique)
        if not isinstance(value,dict):raise RuntimeError('Prior-art control must be a JSON object')
        self.captured[key]=value;self.control_signatures[key]=before
        return deepcopy(value)

    def check(self,full_sources=False):
        if self.implementation_sha256!=_IMPORTED_IMPLEMENTATION_SHA256 or implementation_hashes()!=self.implementation_sha256:raise RuntimeError('Prior-art implementation/dependency drift')
        self.source.check(full_sources=full_sources)
        for path,signature in self.control_signatures.items():
            if shared.signature(path)!=signature:raise RuntimeError('Prior-art control identity drift')

    def bytes_for(self,path):return self.source.bytes_for(path)

    @property
    def sources(self):return self.source.sources


def grid(family):
    if family not in FAMILIES:raise RuntimeError('Unsupported required prior-art family')
    return [{'id':f'{family}_h{w}_do{int(d*10)}_lr{int(lr*1e6):04d}',
        'hidden_dim':w,'dropout':d,'learning_rate':lr}
        for w,d,lr in itertools.product((32,64),(0.,.1),(.0003,.001))]


def verify_grid(family,configurations):
    if not isinstance(configurations,list) or configurations!=grid(family):raise RuntimeError('Exact ordered eight-configuration prior-art grid required')
    for row in configurations:
        if set(row)!={'id','hidden_dim','dropout','learning_rate'} or type(row['hidden_dim']) is not int or isinstance(row['dropout'],bool) or isinstance(row['learning_rate'],bool):
            raise RuntimeError('Typed prior-art width/dropout/LR fields required')
        shared.safe_id(row['id'])


def plan_constants():
    return {'schema_version':1,'kind':'priorart_fresh_development_search_plan_v1',
        'status':'proposal_only_not_executable_until_completed_preparation_is_bound',
        'development_namespace':'development-v1','scientific_status':STATUS,
        'splits':dict(shared.SPLITS),'training_seeds':list(SEEDS),'input_dim':48,'base_feature_count':8,
        'num_classes':4,'num_queries':64,'num_layers':2,'windows':8,'epochs':48,'patience':8,
        'weight_decay':.0001,'gradient_clip_norm':1.,'family_tie_tolerance_nats':.0001,
        'reference_adapter':deepcopy(shared.ADAPTER),'source_paths':dict(shared.PATHS),
        'completion_receipt':shared.COMPLETION,'preparation_freeze':shared.PREPARATION_FREEZE,
        'selection':SELECTION,'nll_policy':RAW_NLL,'deterministic_policy':deepcopy(DETERMINISTIC_POLICY),'confirmatory_access':False,'external_pretraining':False,
        'families':{family:{'configurations':grid(family),'input_scope':SCOPE[family],
            'cross_tree_aggregation':AGGREGATION[family]} for family in FAMILIES}}


def make_search_plan():
    return {**plan_constants(),'scope_amendment':{'path':AMENDMENT,'sha256':shared.sha(shared.development_path(AMENDMENT))},
        'reviewed_implementation_sha256':implementation_hashes()}


def check_plan(plan):
    fixed=plan_constants()
    if not isinstance(plan,dict) or set(plan)!=set(fixed)|{'scope_amendment','reviewed_implementation_sha256'} or any(plan.get(k)!=v for k,v in fixed.items()):raise RuntimeError('Frozen prior-art plan scope/source/budget changed')
    for key in ('schema_version','input_dim','base_feature_count','num_classes','num_queries','num_layers','windows','epochs','patience'):
        if type(plan[key]) is not int:raise RuntimeError('Integer prior-art dimensions required')
    if plan['confirmatory_access'] is not False or plan['external_pretraining'] is not False:raise RuntimeError('Development-only adaptation contract required')
    for family in FAMILIES:verify_grid(family,plan['families'][family]['configurations'])
    amendment=plan['scope_amendment']
    if not isinstance(amendment,dict) or set(amendment)!={'path','sha256'} or shared.development_path(amendment['path'])!=shared.development_path(AMENDMENT):raise RuntimeError('Actual v2 scope-amendment reference required')
    shared.valid_sha(amendment['sha256'])
    if plan['reviewed_implementation_sha256']!=implementation_hashes():raise RuntimeError('Reviewed prior-art source/code pins changed')


def check_amendment(value):
    if value.get('kind')!='development_scope_amendment_proposal' or value.get('development_only') is not True or value.get('authorizes_confirmatory_generation') is not False or value.get('final_test_access') is not False or value.get('counts')!=shared.SPLITS or value.get('training_seeds')!=SEEDS or not set(FAMILIES).issubset(value.get('required_baseline_families',[])) or len(value.get('required_baseline_families',[]))!=9:
        raise RuntimeError('Reviewed nine-family development-only scope amendment required')
    grid_contract=value.get('neural_grid',{})
    expected={'hidden_widths':[32,64],'dropout':[0.,.1],'learning_rates':[.0003,.001],
        'source_layers':2,'temporal_layers':2,'configurations_per_family':8,'epoch_cap':48,
        'patience':8,'weight_decay':.0001,'gradient_clip_norm':1.}
    if grid_contract!=expected or value.get('maximum_prescribed_optimizer_starts')!={'main_neural':216,'secondary_control_reserve':72,'total':288}:raise RuntimeError('Scope amendment family/resource budget differs')


def bound_settings(plan,plan_path,plan_sha,completion,completion_path,completion_sha):
    check_plan(plan);shared.valid_sha(plan_sha);shared.valid_sha(completion_sha)
    facts=completion['stage_manifests']
    keys=('schema_version','development_namespace','scientific_status','splits','training_seeds','input_dim',
        'base_feature_count','num_classes','num_queries','num_layers','windows','epochs','patience','weight_decay',
        'gradient_clip_norm','family_tie_tolerance_nats','reference_adapter','selection','nll_policy',
        'external_pretraining','confirmatory_access','deterministic_policy')
    common={key:deepcopy(plan[key]) for key in keys} | {
        'stage':'fresh_development_supplied_priorart_training_v1','search_plan':str(shared.development_path(plan_path)),
        'search_plan_sha256':plan_sha,'scope_amendment':deepcopy(plan['scope_amendment']),
        'completion_receipt':str(shared.development_path(completion_path)),'completion_receipt_sha256':completion_sha,
        'dataset_manifest':str(shared.development_path(facts['inferred']['path'])),
        'dataset_manifest_sha256':facts['inferred']['sha256'],'source_dataset_manifest_sha256':facts['inferred']['sha256'],
        'dataset_protocol':str(shared.development_path(facts['inferred']['protocol_path'])),
        'dataset_protocol_sha256':facts['inferred']['protocol_sha256'],
        'boundary_dataset_manifest_sha256':facts['boundary']['sha256'],'boundary_preparation_sha256':facts['boundary']['preparation_sha256'],
        'age_dataset_manifest_sha256':facts['age']['sha256'],'age_preparation_sha256':facts['age']['preparation_sha256'],
        'preparation_execution_freeze_sha256':completion['development_execution_freeze_file_sha256'],
        'bound_implementation_sha256':dict(plan['reviewed_implementation_sha256']),
        'interpretation':'Fresh development only. Existing independent paper-equation adaptations, not exact native reproductions, independent confirmation, superiority or novelty proof.'}
    for key in (*BINDINGS,'dataset_protocol_sha256'):shared.valid_sha(common[key])
    return {family:{**deepcopy(common),'family':family,**deepcopy(plan['families'][family])} for family in FAMILIES}


def validate_settings(settings):
    if not isinstance(settings,dict) or settings.get('family') not in FAMILIES:raise RuntimeError('Supported prior-art family required')
    fixed=plan_constants()
    for key in ('schema_version','development_namespace','scientific_status','splits','training_seeds','input_dim',
        'base_feature_count','num_classes','num_queries','num_layers','windows','epochs','patience','weight_decay',
        'gradient_clip_norm','family_tie_tolerance_nats','reference_adapter','selection','nll_policy','external_pretraining','confirmatory_access','deterministic_policy'):
        if settings.get(key)!=fixed[key]:raise RuntimeError('Prior-art source/grid/budget contract changed')
    for key in ('schema_version','input_dim','base_feature_count','num_classes','num_queries','num_layers','windows','epochs','patience'):
        if type(settings[key]) is not int:raise RuntimeError('Integer prior-art dimensions required')
    if settings['confirmatory_access'] is not False or settings['external_pretraining'] is not False:raise RuntimeError('Development-only unpretrained models required')
    family=settings['family'];verify_grid(family,settings.get('configurations'))
    if settings.get('input_scope')!=SCOPE[family] or settings.get('cross_tree_aggregation')!=AGGREGATION[family]:raise RuntimeError('Factual prior-art input/aggregation scope differs')
    for key in (*BINDINGS,'dataset_manifest_sha256','dataset_protocol_sha256','search_plan_sha256'):shared.valid_sha(settings.get(key))
    if settings.get('bound_implementation_sha256')!=implementation_hashes():raise RuntimeError('Prior-art implementation pins differ')
    for key in ('search_plan','completion_receipt','dataset_manifest','dataset_protocol'):shared.development_path(settings[key])
    return settings


def load_admitted_parents(manifest,snapshot):
    # The shared loader requires its original, unchanged implementation pins;
    # composition checks our additional code before/after that admission.
    from .train_argmix_development import load_verified_parents
    snapshot.check(full_sources=True)
    parents=load_verified_parents(manifest,snapshot.source)
    snapshot.check(full_sources=True)
    return parents


def bind_completed_development(completion_path,completion_sha,plan_path=PLAN,plan_sha=None,*,output_dir=None):
    output=None if output_dir is None else shared.development_path(output_dir)
    if output is not None and output.exists():raise RuntimeError('Fresh prior-art bound-config directory required')
    snapshot=PriorArtSnapshot();snapshot.check()
    plan=snapshot.control(plan_path,plan_sha);check_plan(plan)
    check_amendment(snapshot.control(plan['scope_amendment']['path'],plan['scope_amendment']['sha256']))
    snapshot.check()
    completion,manifest,_=shared.verify_completed_development(completion_path,completion_sha,snapshot.source)
    parents=load_admitted_parents(manifest,snapshot);del parents
    settings=bound_settings(plan,plan_path,plan_sha,completion,completion_path,completion_sha)
    snapshot.check(full_sources=True)
    contents={family:(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode() for family,value in settings.items()}
    if output is not None:
        output.mkdir(parents=True)
        for family,content in contents.items():
            with (output/(family+'.json')).open('xb') as handle:handle.write(content)
        snapshot.check(full_sources=True)
    return settings,contents,{family:hashlib.sha256(content).hexdigest() for family,content in contents.items()}


def capture_training_snapshot(config_path,config_sha):
    snapshot=PriorArtSnapshot();snapshot.check()
    settings=snapshot.control(config_path,config_sha);validate_settings(settings)
    plan=snapshot.control(settings['search_plan'],settings['search_plan_sha256']);check_plan(plan)
    check_amendment(snapshot.control(plan['scope_amendment']['path'],plan['scope_amendment']['sha256']))
    snapshot.check()
    completion,manifest,_=shared.verify_completed_development(settings['completion_receipt'],settings['completion_receipt_sha256'],snapshot.source)
    expected=bound_settings(plan,settings['search_plan'],settings['search_plan_sha256'],completion,
        settings['completion_receipt'],settings['completion_receipt_sha256'])
    if settings!=expected[settings['family']]:raise RuntimeError('Captured prior-art config differs from factual plan/completion')
    parents=load_admitted_parents(manifest,snapshot)
    snapshot.check(full_sources=True)
    return settings,parents,snapshot
