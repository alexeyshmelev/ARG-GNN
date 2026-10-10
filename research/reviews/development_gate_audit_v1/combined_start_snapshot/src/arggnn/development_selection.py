"""Hash-bound synthesis of a complete fresh-development family search.

Selection receipts describe developmental choices only. They are not final-test
freezes, novel-mechanism findings, or predictive superiority evidence.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime,timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re

import torch
from .resources import ROOT

SEEDS=(101,202,303)
RAW_NLL='float64 stable log-softmax of uncalibrated raw logits'
PARENT_NLL='float64 stable log-softmax of uncalibrated logits'
PARENT_WEIGHT='uniform fixed coordinates and query haplotypes within one parent'
SELECTION_POLICIES={
    'minimum mean parent-validation NLL',
    'minimum mean parent-validation uncalibrated logit NLL',
    'minimum equal-parent mean validation float64 uncalibrated raw-logit NLL; earliest epoch on exact tie',
}
BINDINGS=('completion_receipt_sha256','source_dataset_manifest_sha256',
    'boundary_dataset_manifest_sha256','boundary_preparation_sha256',
    'age_dataset_manifest_sha256','age_preparation_sha256','preparation_execution_freeze_sha256')
SAFE=re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z')
HEX=re.compile(r'[0-9a-f]{64}\Z')
_IMPORTED_SELECTOR_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _safe(value):
    if not isinstance(value,str) or SAFE.fullmatch(value) is None: raise ValueError('Unsafe frozen identifier')
    return value


def _pin_name(value):
    if not isinstance(value,str) or not value or '\\' in value or value.startswith('/'):
        raise ValueError('Safe relative dependency identifier required')
    parts=value.split('/')
    if any(part in ('','.', '..') for part in parts):
        raise ValueError('Safe relative dependency identifier required')
    for part in parts:_safe(part)
    return value


def _sha(value):
    if not isinstance(value,str) or HEX.fullmatch(value) is None or value=='0'*64: raise ValueError('Real SHA256 required')
    return value


def _number(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
        raise ValueError('Finite nonnegative score required')
    return float(value)


def _unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result: raise ValueError('Duplicate JSON key: '+key)
        result[key]=value
    return result


def _stamp(path):
    stat=path.stat()
    return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)


class _Capture:
    """Capture/hash JSON/config bytes once; reject later identity/stat drift."""
    def __init__(self,root):
        self.root=Path(root).resolve(strict=True)
        self.files={}

    def path(self,value,*,must_exist=True):
        if not isinstance(value,(str,Path)) or not str(value) or '\\' in str(value): raise ValueError('Expected a project file path')
        path=Path(value)
        if not path.is_absolute(): path=self.root/path
        # Resolve existing symlinks and reject escaped/held-out paths before opening.
        path=path.resolve(strict=False)
        if not path.is_relative_to(self.root) or path==self.root:
            raise ValueError('Artifact escapes project root')
        forbidden={'test','final_test','final-test','confirmatory_data','confirmation'}
        parts=path.relative_to(self.root).parts
        phase_parts=parts[:-1] if path.suffix=='.py' else parts
        if any(part.lower() in forbidden or any(token in part.lower() for token in ('final_test','final-test','confirmatory','confirmation')) for part in phase_parts):
            raise ValueError('Final-test paths are unavailable to development selection')
        if must_exist and not path.is_file():raise ValueError('Required artifact is not an existing project file')
        return path

    def read(self,ref,*,json_file=True,max_bytes=536870912,allow_source_reuse=False):
        if not isinstance(ref,dict) or set(ref)!={'path','sha256'}: raise ValueError('Artifact reference requires exact path/hash fields')
        path=self.path(ref['path']); expected=_sha(ref['sha256'])
        key=str(path)
        if key in self.files:
            prior=self.files[key]
            if prior['sha256']!=expected: raise ValueError('Conflicting hashes for one artifact')
            if json_file and 'json' in prior: return prior['json']
            if not json_file and allow_source_reuse and prior.get('source_artifact') is True:return None
            raise ValueError('Binary artifact reused without distinct seed binding')
        before=_stamp(path)
        if before[2]>max_bytes: raise ValueError('Artifact exceeds bounded reader size')
        content=path.read_bytes()
        if _stamp(path)!=before: raise ValueError('Artifact changed during capture')
        actual=hashlib.sha256(content).hexdigest()
        if actual!=expected: raise ValueError('Artifact hash mismatch: '+str(path))
        record={'path':str(path.relative_to(self.root)),'sha256':actual,'bytes':len(content),'stamp':before,
                'source_artifact':not json_file and allow_source_reuse}
        self.files[key]=record
        if json_file:
            value=json.loads(content,object_pairs_hook=_unique_object)
            record['json']=value
            return value
        return content

    def verify_stable(self):
        for path,record in self.files.items():
            if _stamp(Path(path))!=record['stamp']: raise ValueError('Captured input changed before selection sealing')

    def provenance(self):
        return [{key:value for key,value in record.items() if key in ('path','sha256','bytes')}
                for _,record in sorted(self.files.items())]


def tensor_fingerprint(value):
    if not isinstance(value,torch.Tensor) or value.device.type!='cpu': raise ValueError('Fingerprint requires CPU tensor')
    value=value.detach().contiguous()
    header=json.dumps({'dtype':str(value.dtype),'shape':list(value.shape)},sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(header+b'\n'+value.numpy().tobytes()).hexdigest()


@dataclass(frozen=True)
class DevelopmentSelectionReceipt:
    payload: dict
    sha256: str
    path: str | None


@dataclass(frozen=True)
class FrozenDevelopmentSelectionPlan:
    values: dict
    sha256: str
    path: str
    project_root: str


def _normalize_config(config):
    """Normalize one supported list schema without changing captured bytes/hash."""
    if not isinstance(config,dict):raise ValueError('Configuration must be a JSON object')
    if 'configurations' not in config:return config
    if any(key in config for key in ('models','model_specs','model_learning_rates')):
        raise ValueError('Ambiguous list/dictionary configuration schemas')
    rows=config['configurations']
    if not isinstance(rows,list) or any(not isinstance(row,dict) or set(row)!={'id','hidden_dim','dropout','learning_rate'} for row in rows):
        raise ValueError('Exact width/dropout/LR configurations-list schema required')
    result=dict(config)
    result['models']=[_safe(row['id']) for row in rows]
    result['model_specs']={row['id']:{'hidden_dim':row['hidden_dim'],'dropout':row['dropout']} for row in rows}
    result['model_learning_rates']={row['id']:row['learning_rate'] for row in rows}
    return result


def _grid(config,kind):
    names=config.get('models');specs=config.get('model_specs');rates=config.get('model_learning_rates',{})
    if not isinstance(names,list) or len(names)!=8 or any(not isinstance(n,str) or _safe(n)!=n for n in names) or len(set(names))!=8:
        raise ValueError('Require exact eight distinct configuration IDs')
    if not isinstance(specs,dict) or set(specs)!=set(names) or any(not isinstance(specs[n],dict) for n in names): raise ValueError('Exact configuration specifications required')
    if kind=='candidate':
        if any(specs[n].get('coupling')!='full' or specs[n].get('rate_parameterization')!='log' or type(specs[n].get('kernel_depth')) is not int for n in names):
            raise ValueError('Candidate grid requires full/log integer-depth specifications')
        actual=[(specs[n].get('geometry'),specs[n].get('kernel_depth'),rates.get(n)) for n in names]
        expected={(g,d,lr) for g in ('plain','age') for d in (1,2) for lr in (.0003,.001)}
    elif kind=='width_dropout_lr':
        if any(type(specs[n].get('hidden_dim',config.get('hidden_dim'))) is not int for n in names):
            raise ValueError('Integer neural widths required')
        actual=[(specs[n].get('hidden_dim',config.get('hidden_dim')),
                 specs[n].get('dropout',config.get('dropout')),rates.get(n)) for n in names]
        expected={(h,d,lr) for h in (32,64) for d in (0.,.1) for lr in (.0003,.001)}
    elif kind=='bandwidth_pseudocount':
        if any(type(specs[n].get('bandwidth_bp')) is not int for n in names):raise ValueError('Integer bandwidths required')
        actual=[(specs[n].get('bandwidth_bp'),specs[n].get('pseudocount')) for n in names]
        expected={(bw,pc) for bw in (10000,30000,100000,300000) for pc in (.001,.01)}
    else: raise ValueError('Unknown prescribed family grid')
    if any(isinstance(item,bool) for row in actual for item in row):raise ValueError('Boolean grid settings are invalid')
    if len(set(actual))!=8 or set(actual)!=expected: raise ValueError('Family does not cover its exact prescribed grid')
    return names


def _load_plan(capture,path,expected):
    value=capture.read({'path':str(path),'sha256':expected})
    required={'schema_version','kind','development_namespace','training_seeds','splits',
        'nll_policy','tie_tolerance','source_binding','source_artifacts','validation_parents',
        'families','required_baseline_families','implementation_artifacts'}
    if not isinstance(value,dict) or set(value)!=required or value['schema_version']!=1 or value['kind']!='fresh_development_family_selection_plan':
        raise ValueError('Selection plan schema mismatch')
    if value['development_namespace']!='development-v1' or value['training_seeds']!=list(SEEDS) or value['splits']!={'train':64,'val':16,'test':0} or value['nll_policy']!=RAW_NLL or value['tie_tolerance']!=.0001:
        raise ValueError('Fresh64/16/TEST0, three fixed seeds and raw-NLL/tie policy required')
    binding=value['source_binding']
    if set(binding)!=set(BINDINGS): raise ValueError('Incomplete common source/preparation binding')
    for digest in binding.values():_sha(digest)
    artifacts=value['source_artifacts']
    if set(artifacts)!={'completion','source_manifest','boundary_manifest','age_manifest','preparation_execution_freeze'}:
        raise ValueError('Real completed source artifacts required')
    objects={key:capture.read(ref) for key,ref in artifacts.items()}
    expected_artifact={'completion':'completion_receipt_sha256','source_manifest':'source_dataset_manifest_sha256',
        'boundary_manifest':'boundary_dataset_manifest_sha256','age_manifest':'age_dataset_manifest_sha256',
        'preparation_execution_freeze':'preparation_execution_freeze_sha256'}
    for name,key in expected_artifact.items():
        if artifacts[name]['sha256']!=binding[key]:raise ValueError('Source artifact/common binding disagreement')
    completion=objects['completion']
    if completion.get('all_stages_complete') is not True or completion.get('train_parents')!=64 or completion.get('validation_parents')!=16 or completion.get('test_parents')!=0:
        raise ValueError('Completed fresh development receipt required')
    if objects['boundary_manifest'].get('status')!='complete' or objects['age_manifest'].get('status')!='complete':
        raise ValueError('Incomplete observed caches')
    if completion.get('training_or_confirmatory_selection_frozen') is not False or completion.get('inferred_boundary_and_age_tiles')!=640 or completion.get('development_execution_freeze_file_sha256')!=binding['preparation_execution_freeze_sha256']:
        raise ValueError('Fresh preparation receipt phase/freeze mismatch')
    facts=completion.get('stage_manifests',{})
    for name,stage in (('source_manifest','inferred'),('boundary_manifest','boundary'),('age_manifest','age')):
        if facts.get(stage,{}).get('sha256')!=artifacts[name]['sha256']:
            raise ValueError('Completion stage/manifest binding disagreement')
    if objects['boundary_manifest'].get('source',{}).get('manifest_sha256')!=binding['source_dataset_manifest_sha256'] or objects['age_manifest'].get('source_manifest_sha256')!=binding['source_dataset_manifest_sha256'] or objects['age_manifest'].get('boundary_manifest_sha256')!=binding['boundary_dataset_manifest_sha256'] or objects['age_manifest'].get('boundary_preparation_sha256')!=binding['boundary_preparation_sha256']:
        raise ValueError('Observed source/boundary/age chain disagreement')
    if objects['boundary_manifest'].get('preparation_sha256')!=binding['boundary_preparation_sha256'] or objects['age_manifest'].get('preparation_sha256')!=binding['age_preparation_sha256']:
        raise ValueError('Preparation binding disagreement')
    preparation_freeze=objects['preparation_execution_freeze']
    if preparation_freeze.get('kind')!='fresh_development_preparation_execution_freeze' or preparation_freeze.get('confirmatory_access') is not False or preparation_freeze.get('not_confirmatory_test_freeze') is not True:
        raise ValueError('Completed preparation must bind a development-only execution freeze')
    source=objects['source_manifest']
    if source.get('protocol',{}).get('splits')!={'train':64,'val':16,'test':0}: raise ValueError('Observed source must exclude final-test parents')
    records=source.get('replicates',[])
    if not isinstance(records,list) or any(not isinstance(r,dict) or type(r.get('replicate')) is not int or r.get('split') not in ('train','val') for r in records):
        raise ValueError('Typed fresh development parent records required')
    ids=[(r['split'],r['replicate']) for r in records]
    expected_ids={(split,i) for split,count in (('train',64),('val',16)) for i in range(count)}
    if len(ids)!=80 or set(ids)!=expected_ids:raise ValueError('Observed parent identities/counts differ')
    if source['protocol'].get('master_seed')!=20261101 or source['protocol'].get('inference',{}).get('master_seed')!=20261102:
        raise ValueError('Fresh development source seeds differ')
    parents=value['validation_parents']
    if not isinstance(parents,list) or len(parents)!=16 or len({p.get('parent_id') for p in parents})!=16:
        raise ValueError('Ordered16 validation-parent contracts required')
    if {p.get('parent_id') for p in parents}!={binding['source_dataset_manifest_sha256']+':val:'+str(i) for i in range(16)}:
        raise ValueError('Validation parent contracts do not match observed source identities')
    for parent in parents:
        if set(parent)!={'parent_id','positions_sha256','query_ids_sha256','logit_shape'} or not isinstance(parent['parent_id'],str) or not parent['parent_id']:
            raise ValueError('Parent contract schema mismatch')
        _sha(parent['positions_sha256']);_sha(parent['query_ids_sha256'])
        shape=parent['logit_shape']
        if not isinstance(shape,list) or len(shape)!=3 or any(type(n) is not int or n<1 for n in shape) or shape[-1]!=4:
            raise ValueError('Expected positive position/query/four-class shape')
    families=value['families']
    if not isinstance(families,dict) or len(families)<2:raise ValueError('Candidate and explicitly required baseline families required')
    required_baselines=value['required_baseline_families']
    if not isinstance(required_baselines,list) or not required_baselines or any(not isinstance(name,str) for name in required_baselines) or len(set(required_baselines))!=len(required_baselines):
        raise ValueError('Explicit unique required baseline family IDs required')
    for name in required_baselines:_safe(name)
    roles=[]
    pins={}
    configs={}
    for family,spec in families.items():
        _safe(family)
        if not isinstance(spec,dict) or set(spec)!={'role','grid_kind','protocol','expected_implementation_sha256','input_scope','cross_tree_aggregation'}:raise ValueError('Family plan schema mismatch')
        role=_safe(spec['role']);roles.append(role)
        for field in ('input_scope','cross_tree_aggregation'):
            if not isinstance(spec[field],str) or not spec[field].strip():raise ValueError('Factual family input scope and aggregation required')
        config=_normalize_config(capture.read(spec['protocol']))
        if config.get('family')!=family or config.get('development_namespace')!='development-v1' or config.get('training_seeds')!=list(SEEDS) or config.get('splits')!={'train':64,'val':16,'test':0} or config.get('nll_policy')!=RAW_NLL:
            raise ValueError('Family protocol namespace/seed/parent/raw-NLL mismatch')
        if config.get('selection') not in SELECTION_POLICIES:raise ValueError('Checkpoint selection must attest equal-parent raw-logit NLL policy')
        if any(config.get(field)!=spec[field] for field in ('input_scope','cross_tree_aggregation')):
            raise ValueError('Family input-scope/protocol disagreement')
        if config.get('epochs')!=48 or config.get('patience')!=8 or config.get('weight_decay')!=.0001 or type(config.get('epochs')) is not int or type(config.get('patience')) is not int:
            raise ValueError('Every family requires the same48-epoch/patience8 development budget')
        if any(config.get(key)!=digest for key,digest in binding.items()):raise ValueError('Family common source binding mismatch')
        _grid(config,spec['grid_kind'])
        if role=='candidate' and spec['grid_kind']!='candidate' or role=='observed_reference_smoother' and spec['grid_kind']!='bandwidth_pseudocount' or role not in {'candidate','observed_reference_smoother'} and spec['grid_kind']!='width_dropout_lr':
            raise ValueError('Family role/grid mismatch')
        expected=spec['expected_implementation_sha256']
        if not isinstance(expected,dict) or not expected or config.get('bound_implementation_sha256')!=expected:
            raise ValueError('Frozen protocol/dependency pin mismatch')
        for name,digest in expected.items():
            _pin_name(name);_sha(digest)
            if name in pins and pins[name]!=digest:raise ValueError('Shared implementation dependency drift across families')
            pins[name]=digest
        configs[family]=config
    all_ids=[name for config in configs.values() for name in config['models']]
    if len(all_ids)!=len(set(all_ids)):raise ValueError('Configuration IDs must be globally distinct')
    if roles.count('candidate')!=1:raise ValueError('Exactly one candidate family required')
    noncandidate={family for family,spec in families.items() if spec['role']!='candidate'}
    if set(required_baselines)!=noncandidate:raise ValueError('Every planned baseline must be explicitly required; silent omission forbidden')
    implementation=value['implementation_artifacts']
    if set(implementation)!=set(pins):raise ValueError('Frozen dependency artifact set mismatch')
    for name,digest in pins.items():
        if implementation[name]['sha256']!=digest:raise ValueError('Dependency artifact digest differs')
        # Source files are byte artifacts, not executable imports.
        capture.read(implementation[name],json_file=False,allow_source_reuse=True)
    return value,configs


def load_development_selection_plan(path,expected_sha256,*,project_root=ROOT):
    capture=_Capture(project_root);value,_=_load_plan(capture,path,_sha(expected_sha256));capture.verify_stable()
    return FrozenDevelopmentSelectionPlan(value,expected_sha256,str(capture.path(path)),str(capture.root))


def _same(actual,expected,description):
    if actual!=expected:raise ValueError(description+' mismatch')


def _artifact(capture,path,digest):
    return torch.load(io.BytesIO(capture.read({'path':path,'sha256':digest},json_file=False)),map_location='cpu',weights_only=True)


def _cell(capture,plan,configs,family,config_id,seed,body,run_hash):
    spec=plan['families'][family];config=configs[family];binding=plan['source_binding']
    if config_id not in config['models'] or type(seed) is not int or seed not in SEEDS:
        raise ValueError('Unexpected configuration or training seed')
    count=body.get('parameters')
    if type(count) is not int or count<0:raise ValueError('Invalid selected-model parameter count')
    if spec['role']=='observed_reference_smoother' and count!=0:
        raise ValueError('The prescribed observed-reference smoother must be fixed and have zero fitted parameters')
    rows=body.get('per_parent')
    if rows is None and isinstance(body.get('per_replicate'),list):
        rows=[{**row,'parent_id':binding['source_dataset_manifest_sha256']+':val:'+str(row['replicate'])} for row in body['per_replicate']]
    if not isinstance(rows,list) or len(rows)!=16:raise ValueError('Require every ordered parent raw-NLL score')
    expected_ids=[p['parent_id'] for p in plan['validation_parents']]
    _same([r.get('parent_id') for r in rows],expected_ids,'Ordered parent IDs')
    scores=[]
    for row in rows:
        if row.get('nll_policy') not in (RAW_NLL,PARENT_NLL) or row.get('weighting')!=PARENT_WEIGHT:
            raise ValueError('Raw-logit equal-parent policy is mandatory')
        scores.append(_number(row.get('nll')))
    mean=math.fsum(scores)/16
    if not math.isclose(_number(body.get('validation_nll')),mean,rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError('Reported validation NLL is not equal-parent mean')
    _same(body.get('ending_config_sha256'),spec['protocol']['sha256'],'Ending configuration hash')
    _same(body.get('ending_implementation_sha256'),spec['expected_implementation_sha256'],'Ending implementation pins')
    history=body.get('history')
    epoch=body.get('best_epoch')
    checkpoint=body.get('checkpoint')
    if count==0:
        if body.get('smoother_fitting')!='none; seed-independent deterministic predictor' or spec['role']!='observed_reference_smoother' or checkpoint is not None or body.get('checkpoint_sha256') is not None or epoch is not None or history!=[]:
            raise ValueError('Only the fixed no-fit smoother may lack a checkpoint')
    else:
        if not isinstance(history,list) or not history or len(history)>48 or any(not isinstance(item,dict) or type(item.get('epoch')) is not int for item in history) or [item.get('epoch') for item in history]!=list(range(len(history))):
            raise ValueError('Complete ordered training history required')
        values=[_number(item.get('val_nll')) for item in history]
        earliest=min(range(len(values)),key=lambda i:(values[i],i))
        if type(epoch) is not int or epoch!=earliest or not math.isclose(mean,values[earliest],rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('Selected checkpoint is not earliest minimum parent raw-NLL epoch')
        payload=_artifact(capture,checkpoint,_sha(body.get('checkpoint_sha256')))
        if not isinstance(payload,dict) or not isinstance(payload.get('model'),dict) or not payload['model']:
            raise ValueError('Real weights-only selected model checkpoint required')
        for tensor in payload['model'].values():
            if not isinstance(tensor,torch.Tensor) or not bool(torch.isfinite(tensor).all()):
                raise ValueError('Selected model state must contain finite tensors')
        if type(payload.get('epoch')) is not int:raise ValueError('Checkpoint epoch must be an integer')
        _same(payload.get('epoch'),epoch,'Checkpoint epoch')
        _same(payload.get('seed'),seed,'Checkpoint seed')
        _same(payload.get('protocol_sha256'),spec['protocol']['sha256'],'Checkpoint config hash')
        _same(payload.get('completion_receipt_sha256'),binding['completion_receipt_sha256'],'Checkpoint completion binding')
        _same(payload.get('boundary_manifest_sha256',payload.get('boundary_dataset_manifest_sha256')),binding['boundary_dataset_manifest_sha256'],'Checkpoint boundary binding')
        _same(payload.get('implementation_sha256'),spec['expected_implementation_sha256'],'Checkpoint dependency binding')
        if not math.isclose(_number(payload.get('validation_nll')),mean,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Checkpoint score mismatch')
    prediction_path=body.get('validation_logits',body.get('validation_predictions'))
    prediction_hash=_sha(body.get('validation_logits_sha256',body.get('validation_predictions_sha256')))
    prediction=_artifact(capture,prediction_path,prediction_hash)
    if not isinstance(prediction,dict) or any(key in prediction for key in ('labels','truth','populations','baseline_probabilities')):
        raise ValueError('Prediction artifact must exclude targets/truth annotations')
    _same(prediction.get('protocol_sha256'),spec['protocol']['sha256'],'Prediction config binding')
    _same(prediction.get('seed'),seed,'Prediction seed')
    _same(prediction.get('nll_policy'),RAW_NLL,'Prediction raw-logit policy')
    _same(prediction.get('completion_receipt_sha256'),binding['completion_receipt_sha256'],'Prediction completion binding')
    prediction_ids=prediction.get('parent_ids')
    if prediction_ids is None and isinstance(prediction.get('parent_replicates'),list):
        if any(type(rep) is not int for rep in prediction['parent_replicates']):raise ValueError('Integer parent replicate IDs required')
        prediction_ids=[binding['source_dataset_manifest_sha256']+':val:'+str(rep) for rep in prediction['parent_replicates']]
    _same(prediction_ids,expected_ids,'Prediction ordered parent IDs')
    logits,positions,queries=prediction.get('logits'),prediction.get('positions'),prediction.get('query_node_ids')
    if any(not isinstance(items,list) or len(items)!=16 for items in (logits,positions,queries)):
        raise ValueError('Prediction artifact must contain every parent')
    logit_hashes=[]
    for expected,z,x,q in zip(plan['validation_parents'],logits,positions,queries):
        if not isinstance(z,torch.Tensor) or z.device.type!='cpu' or z.dtype!=torch.float64 or list(z.shape)!=expected['logit_shape'] or not bool(torch.isfinite(z).all()):
            raise ValueError('Finite raw float64 logits with expected shape required')
        if not isinstance(x,torch.Tensor) or x.ndim!=1 or len(x)!=z.shape[0] or not x.is_floating_point() or not bool(torch.isfinite(x).all()) or not isinstance(q,torch.Tensor) or q.ndim!=1 or q.dtype!=torch.long or len(q)!=z.shape[1] or torch.unique(q).numel()!=len(q):
            raise ValueError('Prediction coordinate/query contract invalid')
        _same(tensor_fingerprint(x),expected['positions_sha256'],'Prediction coordinate order')
        _same(tensor_fingerprint(q),expected['query_ids_sha256'],'Prediction query order')
        logit_hashes.append(tensor_fingerprint(z))
    return {'status':'success','family_id':family,'configuration_id':config_id,'seed':seed,
        'parameters':count,'validation_nll':mean,'per_parent_nll':scores,'ordered_parent_ids':expected_ids,
        'best_epoch':epoch,'history':history,'checkpoint':checkpoint,'checkpoint_sha256':body.get('checkpoint_sha256'),
        'prediction':prediction_path,'prediction_sha256':prediction_hash,'parent_logit_sha256':logit_hashes,
        'run_receipt_sha256':run_hash,'protocol_sha256':spec['protocol']['sha256'],
        'source_binding':dict(binding),'implementation_sha256':spec['expected_implementation_sha256']}


def _winner(configurations):
    eligible=[c for c in configurations if c['eligible']]
    if not eligible:return None
    best=min(c['mean_three_seed_nll'] for c in eligible)
    tie=[c for c in eligible if c['mean_three_seed_nll']<=best+.0001]
    choice=min(tie,key=lambda c:(c['parameters'],c['configuration_id'],c['family_id']))
    return {'family_id':choice['family_id'],'configuration_id':choice['configuration_id'],
        'mean_three_seed_nll':choice['mean_three_seed_nll'],'parameters':choice['parameters'],
        'global_best_mean_nll':best,'tie_tolerance':.0001,
        'tie_ids':sorted((c['family_id'],c['configuration_id']) for c in tie),
        'rule':'one global best+.0001 window, parameter count, frozen lexicographic ID'}


def synthesize_development_selection(plan_path,expected_plan_sha256,*,run_receipts=(),
        failure_ledgers=(),output_path=None,project_root=ROOT):
    """Validate all evidence and emit a fresh immutable development receipt.

    run_receipts references add family_id to exact path/hash fields. They may
    describe a complete or a partial run, but duplicate successful cells fail.
    A missing cell is explicit/ineligible and blocks its family choice. A
    terminal failure resolves its cell while disqualifying that configuration.
    """
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=_IMPORTED_SELECTOR_SHA256:
        raise ValueError('Selection implementation changed after module import')
    selector_stamp=_stamp(Path(__file__))
    capture=_Capture(project_root);plan,configs=_load_plan(capture,plan_path,_sha(expected_plan_sha256))
    cells={};failures=[];attempts=set()
    expected={(family,name,seed) for family,c in configs.items() for name in c['models'] for seed in SEEDS}
    for ref in run_receipts:
        if set(ref)!={'family_id','path','sha256'}:raise ValueError('Run reference schema mismatch')
        family=ref['family_id']
        if family not in configs:raise ValueError('Unexpected run family')
        run=capture.read({key:ref[key] for key in ('path','sha256')})
        spec=plan['families'][family]
        if run.get('development_namespace')!='development-v1' or run.get('evaluation_split')!='fresh_development_validation_only' or run.get('training_seeds')!=list(SEEDS) or run.get('nll_policy')!=RAW_NLL:
            raise ValueError('Run is not a fresh three-seed raw-NLL development receipt')
        _same(run.get('family'),family,'Run family')
        _same(run.get('protocol_sha256'),spec['protocol']['sha256'],'Run config hash')
        _same(run.get('completion_receipt_sha256'),plan['source_binding']['completion_receipt_sha256'],'Run completion binding')
        _same(run.get('boundary_dataset_manifest_sha256'),plan['source_binding']['boundary_dataset_manifest_sha256'],'Run boundary binding')
        _same(run.get('initial_implementation_sha256'),spec['expected_implementation_sha256'],'Initial implementation pins')
        seed=run.get('training_seed')
        if type(seed) is not int or seed not in SEEDS:raise ValueError('Unexpected run training seed')
        models=run.get('models')
        if not isinstance(models,dict) or not models:raise ValueError('Run has no configuration results')
        for name,body in models.items():
            key=(family,name,seed)
            if key not in expected or key in cells:raise ValueError('Extraneous/duplicate successful configuration-seed cell')
            cells[key]=_cell(capture,plan,configs,family,name,seed,body,ref['sha256'])
    for ref in failure_ledgers:
        ledger=capture.read(ref)
        if set(ledger)!={'schema_version','kind','plan_sha256','entries'} or ledger['schema_version']!=1 or ledger['kind']!='development_failure_ledger' or ledger['plan_sha256']!=expected_plan_sha256:
            raise ValueError('Failure ledger binding/schema mismatch')
        for entry in ledger['entries']:
            if set(entry)!={'family_id','configuration_id','seed','attempt_id','terminal','failure_class','reason','test_access'}:
                raise ValueError('Failure entry schema mismatch')
            key=(entry['family_id'],entry['configuration_id'],entry['seed'])
            if key not in expected or type(entry['seed']) is not int or type(entry['terminal']) is not bool or entry['test_access'] is not False or not isinstance(entry['reason'],str) or not entry['reason']:
                raise ValueError('Invalid failure cell/reason/test-access declaration')
            attempt=(*key,_safe(entry['attempt_id']))
            if attempt in attempts:raise ValueError('Duplicate ledgered attempt')
            attempts.add(attempt)
            if entry['failure_class'] not in ('infrastructure','numerical','data_admission','model_unavailable','other'):
                raise ValueError('Unknown ledger failure class')
            if not entry['terminal'] and entry['failure_class']!='infrastructure':
                raise ValueError('Only logged infrastructure retries are nonterminal')
            failures.append({**entry,'ledger_sha256':ref['sha256']})
            if entry['terminal']:
                if key in cells:raise ValueError('Successful/terminal-failed cell conflict or duplicate terminal failure')
                cells[key]={'status':'failed',**entry,'ledger_sha256':ref['sha256']}
    matrix=[];families={};all_configurations=[]
    for family,config in configs.items():
        configurations=[];family_complete=True
        for name in config['models']:
            row=[]
            for seed in SEEDS:
                key=(family,name,seed)
                value=cells.get(key,{'status':'missing','family_id':family,'configuration_id':name,'seed':seed})
                row.append(value);matrix.append(value)
                if value['status']=='missing':family_complete=False
            successful=[cell for cell in row if cell['status']=='success']
            counts={cell['parameters'] for cell in successful}
            if len(counts)>1:raise ValueError('Parameter count changed across fixed seeds')
            eligible=len(successful)==3
            if eligible and plan['families'][family]['role']=='observed_reference_smoother':
                if any(cell['per_parent_nll']!=successful[0]['per_parent_nll'] or cell['parent_logit_sha256']!=successful[0]['parent_logit_sha256'] for cell in successful[1:]):
                    raise ValueError('Seed-independent no-fit smoother changed scores or logits across seeds')
            configuration={'family_id':family,'configuration_id':name,'eligible':eligible,
                'seed_scores':{str(cell['seed']):cell.get('validation_nll') for cell in row},
                'seed_statuses':{str(cell['seed']):cell['status'] for cell in row},
                'parameters':next(iter(counts)) if counts else None,
                'mean_three_seed_nll':math.fsum(cell['validation_nll'] for cell in successful)/3 if eligible else None,
                'ineligibility_reason':None if eligible else 'All three fixed seed cells must succeed; no successful-seed averaging'}
            configurations.append(configuration);all_configurations.append(configuration)
        family_eligible=family_complete and any(c['eligible'] for c in configurations)
        families[family]={'role':plan['families'][family]['role'],
            'input_scope':plan['families'][family]['input_scope'],
            'cross_tree_aggregation':plan['families'][family]['cross_tree_aggregation'],
            'protocol':plan['families'][family]['protocol'],'matrix_resolved':family_complete,
            'family_selection_eligible':family_eligible,'configurations':configurations,
            'winner':_winner(configurations) if family_eligible else None}
    baseline_families=plan['required_baseline_families']
    baseline_ready=all(families[f]['family_selection_eligible'] for f in baseline_families)
    candidate=next(f for f,spec in plan['families'].items() if spec['role']=='candidate')
    complete=all(v['matrix_resolved'] for v in families.values())
    capture.verify_stable()
    if _stamp(Path(__file__))!=selector_stamp:raise ValueError('Selection implementation changed before sealing')
    receipt={'schema_version':1,'kind':'immutable_fresh_development_selection_receipt',
        'created_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':expected_plan_sha256,
        'development_namespace':'development-v1','training_seeds':list(SEEDS),'nll_policy':RAW_NLL,
        'aggregation':'equal-parent validation NLL within each selected checkpoint, then mean across all three fixed seeds',
        'source_binding':plan['source_binding'],'ordered_parent_ids':[p['parent_id'] for p in plan['validation_parents']],
        'matrix_resolved':complete,'matrix_counts':{status:sum(c['status']==status for c in matrix) for status in ('success','failed','missing')},
        'cells':matrix,'failures_and_retry_ledger':failures,'families':families,
        'candidate_choice':families[candidate]['winner'],
        'strongest_required_baseline':_winner([c for c in all_configurations if c['family_id'] in baseline_families]) if baseline_ready else None,
        'baseline_selection_eligible':baseline_ready,'required_baseline_families':baseline_families,
        'required_baseline_roles':sorted({plan['families'][f]['role'] for f in baseline_families}),
        'scope_interpretation':'Input scopes and cross-tree aggregation are factual protocol attestations. Shared source bindings do not establish equal information access.',
        'source_artifacts':capture.provenance(),'selection_implementation_sha256':_IMPORTED_SELECTOR_SHA256,
        'scientific_scope':'Developmental choices only. Not a final-test freeze, independent confirmation, novelty, mechanism or superiority proof.',
        'final_test_access':False,'promotes_final_test_phase':False}
    serialized=(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    receipt_sha256=hashlib.sha256(serialized).hexdigest()
    target=None
    if output_path is not None:
        target=capture.path(output_path,must_exist=False)
        if not target.parent.exists():raise ValueError('Selection output parent directory must exist inside project')
        with target.open('xb') as handle:handle.write(serialized)
    return DevelopmentSelectionReceipt(receipt,receipt_sha256,None if target is None else str(target))
