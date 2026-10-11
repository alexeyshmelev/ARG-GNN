"""Inactive H021 native biological fit, enabled only by exact root capabilities.

This module never opens truth trees. Qualified observed loader is reused intact.
Ancestry arrays are separate TRAIN-loss / post-seal VAL-score capabilities.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from .census_tree_encoder_v1 import CensusTreeGNNV1
from .census_biological_bank_v1 import forward_certified_census_biological_v1, validate_census_biological_certificate_v1, fingerprint_sha256_v1
from .census_native_qualification_runner_v1 import VerifiedReaderV1, read_manifests_v1, load_parent_v1, prepare_tile_v1, runtime_identity_v1

ROOT=Path(__file__).resolve().parents[2]
CONTROLS=('proposed','eta0','permuted_groups','raw_diagonal','local')
PARENTS=tuple(f'bio_paths_pilot_{s}_{i:03d}' for s in ('train','val') for i in range(2))
REFS={'profile','config','review','runtime','storage','launcher','numerical_evidence','numerical_admission','inspection_evidence','inspection_freeze','numerical_terminal','target_manifest','target_bindings'}


def _utc(value):
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    if result.tzinfo is None:raise ValueError('Timezone-aware root dates required')
    return result.astimezone(timezone.utc)


def load_config_v1():
    return json.loads((ROOT/'configs/census_biological_fit_v2.json').read_text())


def require_terminal_numerical_v1(evidence,numerical_admission,inspection,freeze,terminal,refs):
    """JSON-only gate. Byte verification precedes this function in admit_fit_v1."""
    if terminal != {'schema':'census_numerical_root_terminal_binding_v1','status':'complete','returncode':0,
        'evidence_ref':refs['numerical_evidence'],'admission_ref':refs['numerical_admission'],
        'inspection_ref':refs['inspection_evidence'],'freeze_ref':refs['inspection_freeze'],
        'job_id':numerical_admission['job_id']}:
        raise ValueError('Root terminal exit0 must bind numerical evidence and lineage')
    expected={f'{p}:{i}' for p in PARENTS for i in range(8)}
    if (evidence.get('schema')!='census_native_qualification_evidence_v1' or evidence.get('phase')!='numerical'
        or evidence.get('status')!='numerical_complete_no_fitting_or_predictive_claim'
        or evidence.get('fixture_mode') is not False or evidence.get('expected_cells')!=32
        or evidence.get('biological_targets_opened') is not False or evidence.get('fit_performed') is not False
        or evidence.get('GPU_used') is not False or evidence.get('geometry_adapted') is not False
        or evidence.get('failed_cells')!=[] or len(evidence.get('performed_cell_coverage',[]))!=32
        or set(evidence['performed_cell_coverage'])!=expected or len(evidence.get('cells',[]))!=32):
        raise ValueError('Complete actual32 target-free terminal numerical evidence required')
    if (inspection.get('schema')!='census_native_qualification_evidence_v1' or inspection.get('phase')!='inspect'
        or inspection.get('status')!='inspection_complete_pending_root_freeze' or inspection.get('failed_cells')!=[]
        or inspection.get('fixture_mode') is not False or len(inspection.get('cells',[]))!=32
        or freeze.get('schema')!='root_external_census_inspection_freeze_v1' or freeze.get('expected_cells')!=32
        or freeze.get('inspection_ref')!=refs['inspection_evidence']
        or freeze.get('model_or_label_outcomes_used_for_selection') is not False
        or freeze.get('source_and_original_count_lineage_rechecked') is not True):
        raise ValueError('Independently frozen original observation inspection required')
    if (evidence.get('job_id')!=numerical_admission.get('job_id')
        or evidence.get('admission_ref')!=refs['numerical_admission']
        or numerical_admission.get('phase')!='numerical'
        or numerical_admission.get('inspection_manifest')!=refs['inspection_evidence']
        or numerical_admission.get('expected_certificates')!=freeze.get('expected_certificates')
        or set(freeze.get('expected_certificates',{}))!=expected):raise ValueError('Numerical admission/certificate binding differs')
    for key in ('source_sha256','observed_artifact_sha256'):
        if evidence.get(key)!=numerical_admission.get(key) or inspection.get(key)!=evidence.get(key):
            raise ValueError('Numerical/inspection original source closure differs')
    for key in ('runner_config','native_manifest','prepared_manifest','feature_lineage'):
        if evidence['refs'].get(key)!=numerical_admission['refs'].get(key) or inspection['refs'].get(key)!=evidence['refs'].get(key):
            raise ValueError('Numerical/inspection original reference differs')
    frozen={f"{c['parent_id']}:{c['tile_id']}":c for c in inspection['cells']}
    seen=set();wanted=[[d,c] for d in ('float64','float32') for c in CONTROLS]
    for cell in evidence['cells']:
        identity=f"{cell['parent_id']}:{cell['tile_id']}";qualification=cell.get('qualification',{})
        if identity in seen or identity not in expected:raise ValueError('Duplicate/unknown numerical cell')
        seen.add(identity)
        if (cell.get('status')!='numerically_qualified' or cell.get('certificate_sha256')!=freeze['expected_certificates'][identity]
            or frozen[identity].get('certificate_sha256')!=cell['certificate_sha256']
            or cell.get('split')!=frozen[identity].get('split') or cell.get('num_trees')!=frozen[identity].get('num_trees')
            or cell.get('decoded_lineage')!=frozen[identity].get('decoded_lineage')
            or cell.get('event_collision_counts')!=frozen[identity].get('event_collision_counts')
            or qualification.get('status')!='qualified' or qualification.get('performed_controls')!=wanted
            or len(qualification.get('checks',[]))!=10):raise ValueError('Incomplete or changed per-cell numerical qualification')
        for check,(dtype,control) in zip(qualification['checks'],wanted):
            if (check.get('dtype')!=dtype or check.get('control')!=control
                or check.get('initial_state_sha256')!=numerical_admission['initial_state_sha256_by_dtype'][dtype]
                or check.get('finite_disposable_AdamW') is not True or check.get('forest_output_VJP_parity') is not True
                or check.get('sample_readout_target_column_order') is not True):raise ValueError('Changed numerical check/order/state')
    return numerical_admission


def _jsonable_v1(value):
    if isinstance(value,torch.Tensor):return _jsonable_v1(value.detach().cpu().numpy())
    if isinstance(value,np.ndarray):return _jsonable_v1(value.tolist())
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):_jsonable_v1(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_jsonable_v1(v) for v in value]
    return value


class FitOutputV1:
    """Exclusive bounded receipt, prediction and checkpoint writes, no overwrite."""
    def __init__(self,directory,reservation,reader):
        self.path=(reader.root/reader.relative(directory)).resolve()
        try:self.path.relative_to(reader.root.resolve())
        except ValueError:raise ValueError('Fit output resolves outside repository') from None
        self.reservation=reservation
        if self.path.exists():raise ValueError('Exclusive new fit output required')
        self.path.mkdir(parents=True);self.limit=reservation['reserved_new_bytes'];self.used=0
    def save(self,name,value,kind='json'):
        if Path(name).name!=name:raise ValueError('Named fit files only')
        if self.reservation.get('expires_at_utc') and datetime.now(timezone.utc)>=_utc(self.reservation['expires_at_utc']):raise ValueError('Fit output reservation expired')
        buffer=io.BytesIO()
        if kind=='json':data=(json.dumps(_jsonable_v1(value),sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        else:
            if kind=='npz':np.savez_compressed(buffer,**value)
            elif kind=='torch':torch.save(value,buffer)
            else:raise ValueError('Unknown bounded output type')
            data=buffer.getvalue()
        if self.used+len(data)+8192>self.limit:raise ValueError('Reserved fit output cap exceeded')
        stat=os.statvfs(self.path)
        if len(data)+8192>stat.f_bavail*stat.f_frsize:raise ValueError('Insufficient free bytes')
        with (self.path/name).open('xb') as stream:stream.write(data)
        self.used+=len(data)+8192
        return {'path':str((self.path/name).relative_to(ROOT)),'sha256':hashlib.sha256(data).hexdigest()}


def admit_fit_v1(admission_ref):
    reader=VerifiedReaderV1();admission=reader.decode_json(admission_ref)
    required={'schema','status','fit_authorized','job_id','created_at_utc','expires_at_utc','refs','source_sha256','seed','device','output_directory','expected_certificates','observed_artifact_sha256','group_permutation'}
    if (set(admission)!=required or admission['schema']!='census_biological_fit_admission_v2'
        or admission['status']!='active' or admission['fit_authorized'] is not True or set(admission['refs'])!=REFS):
        raise ValueError('Fresh root biological fitting capability required')
    _check_active_v1(admission)
    cfg=reader.decode_json(admission['refs']['config'])
    if cfg!=load_config_v1() or cfg['execution_status']!='inactive_source_only_root_capabilities_required':raise ValueError('Shipped inactive fitting protocol changed')
    sources=dict(cfg['source_sha256']);sources['src/arggnn/census_biological_fit_v2.py']=admission['source_sha256'].get('src/arggnn/census_biological_fit_v2.py')
    if admission['source_sha256']!=sources:raise ValueError('Complete exact executed fit source closure required')
    for path,sha in sources.items():reader.read({'path':path,'sha256':sha})
    review=reader.decode_json(admission['refs']['review'])
    if (not str(review.get('status','')).startswith('qualified') or review.get('actual_admission') is not False or review.get('fixture_mode') is True
        or review.get('artifact_sha256',{}).get('src/arggnn/census_biological_fit_v2.py')!=sources['src/arggnn/census_biological_fit_v2.py']
        or review.get('artifact_sha256',{}).get('configs/census_biological_fit_v2.json')!=admission['refs']['config']['sha256']):raise ValueError('Fresh independent exact fit source/config review required')
    gate={key:reader.decode_json(admission['refs'][key]) for key in ('numerical_evidence','numerical_admission','inspection_evidence','inspection_freeze','numerical_terminal')}
    native_admission=require_terminal_numerical_v1(gate['numerical_evidence'],gate['numerical_admission'],gate['inspection_evidence'],gate['inspection_freeze'],gate['numerical_terminal'],admission['refs'])
    qcfg=reader.decode_json(native_admission['refs']['runner_config'])
    if qcfg!=json.loads((ROOT/'configs/census_native_qualification_runner_v1.json').read_text()):raise ValueError('Qualified original native runner configuration changed')
    if (admission['job_id']==native_admission['job_id'] or admission['expected_certificates']!=native_admission['expected_certificates']
        or admission['observed_artifact_sha256']!=native_admission['observed_artifact_sha256']
        or admission['group_permutation']!=native_admission['group_permutation']
        or admission['seed'] not in cfg['seeds'] or admission['device'] not in ('cpu','cuda')):raise ValueError('New fit job must preserve qualified cells and permutation')
    if cfg['model']!=qcfg['model']:raise ValueError('Fit cannot weaken qualified full-tree architecture')
    # Execution source bytes used in numerical evidence must still be identical.
    if set(native_admission['source_sha256'])!=set(qcfg['required_source_paths'])|{'src/arggnn/census_native_qualification_runner_v1.py'} or any(native_admission['source_sha256'].get(p)!=h for p,h in qcfg['source_sha256'].items()):raise ValueError('Complete original numerical execution source pins required')
    for path,sha in native_admission['source_sha256'].items():reader.read({'path':path,'sha256':sha})
    runtime=reader.decode_json(admission['refs']['runtime'])
    if (runtime.get('schema')!='census_biological_fit_runtime_v2' or runtime.get('python')!=str(ROOT/'.runtime/env/bin/python')
        or runtime.get('torch')!=str(torch.__version__) or runtime.get('numpy')!=str(np.__version__)
        or set(runtime.get('runtime_file_sha256',{}))!=set(qcfg['runtime_files'])):raise ValueError('Fit runtime identity/files changed')
    import sys
    if Path(sys.executable).resolve()!=(ROOT/'.runtime/env/bin/python').resolve():raise ValueError('Main project interpreter required')
    for path,sha in runtime['runtime_file_sha256'].items():reader.read({'path':path,'sha256':sha},runtime_file=True)
    numerical_runtime=reader.decode_json(native_admission['refs']['runtime'])
    if runtime['runtime_file_sha256']!=numerical_runtime['runtime_file_sha256']:raise ValueError('Numerically qualified runtime bytes changed')
    storage=reader.decode_json(admission['refs']['storage']);snapshot=reader.decode_json(storage['snapshot_ref'])
    now=datetime.now(timezone.utc)
    if (storage.get('schema')!='census_biological_fit_storage_v2' or storage.get('status')!='active'
        or storage.get('job_id')!=admission['job_id'] or storage.get('output_directory')!=admission['output_directory']
        or type(storage.get('reserved_new_bytes')) is not int or not cfg['storage']['reserved_new_bytes_per_seed']<=storage['reserved_new_bytes']<=cfg['storage']['output_cap_bytes_per_seed']
        or type(storage.get('concurrent_reserve_bytes')) is not int or storage['concurrent_reserve_bytes']<cfg['storage']['concurrent_reserve_bytes']
        or storage.get('effective_project_ceiling_bytes')!=cfg['storage']['effective_project_ceiling_bytes']
        or type(storage.get('used_project_bytes')) is not int or storage['used_project_bytes']<0
        or storage['used_project_bytes']+storage['reserved_new_bytes']+storage['concurrent_reserve_bytes']>=storage['effective_project_ceiling_bytes']
        or not _utc(storage['created_at_utc'])<=now<_utc(storage['expires_at_utc'])):raise ValueError('Active root fit storage reservation required')
    if (snapshot.get('status')!='complete' or snapshot.get('returncode')!=0 or snapshot.get('project_root')!=str(ROOT)
        or snapshot.get('used_project_bytes')!=storage['used_project_bytes'] or not 0<=(now-_utc(snapshot['created_at_utc'])).total_seconds()<=3600):raise ValueError('Fresh complete root storage snapshot required')
    launcher=reader.decode_json(admission['refs']['launcher'])
    if (launcher.get('schema')!='census_biological_fit_launcher_v2' or launcher.get('job_id')!=admission['job_id']
        or launcher.get('device')!=admission['device'] or launcher.get('GPU_admitted')!=(admission['device']=='cuda')
        or launcher.get('python')!='.runtime/env/bin/python' or (admission['device']=='cpu' and os.environ.get('CUDA_VISIBLE_DEVICES')!='') or launcher.get('external_watchdog') is not True
        or launcher.get('profile_gate_passed') is not True or launcher.get('PYTHONDONTWRITEBYTECODE')!='1'
        or os.environ.get('PYTHONDONTWRITEBYTECODE')!='1'):raise ValueError('Fresh root launcher/profile/watchdog capability required')
    profile=reader.decode_json(admission['refs']['profile'])
    if (profile.get('schema')!='census_biological_fit_profile_gate_v2' or profile.get('status')!='qualified_for_fixed_protocol'
        or profile.get('fit_config_ref')!=admission['refs']['config'] or profile.get('source_sha256')!=admission['source_sha256']
        or profile.get('numerical_ref')!=admission['refs']['numerical_evidence'] or profile.get('device')!=admission['device']
        or profile.get('controls')!=list(CONTROLS) or profile.get('model')!=cfg['model']
        or (admission['device']=='cuda' and profile.get('cuda_float32_output_VJP_parity_qualified') is not True)
        or profile.get('full_whole_tree_forward_backward_measured') is not True
        or profile.get('fresh_root_resource_reservation') is not True or not profile.get('profile_evidence_ref')):raise ValueError('Bound full-tree backward profile and resource evidence required')
    reader.read(profile['profile_evidence_ref'])
    bindings=reader.decode_json(admission['refs']['target_bindings']);metadata=reader.decode_json(admission['refs']['target_manifest'])
    if (bindings.get('schema')!='census_biological_targets_root_binding_v1' or bindings.get('observed_artifact_sha256')!=admission['observed_artifact_sha256']
        or bindings.get('manifest_ref')!=admission['refs']['target_manifest'] or set(bindings.get('parents',{}))!=set(PARENTS)
        or admission['refs']['target_manifest']!=cfg['external_target_manifest']):raise ValueError('Separate exact root target bindings required')
    byid={row['parent_id']:row for row in metadata['replicates']}
    for parent,record in bindings['parents'].items():
        split='train' if '_train_' in parent else 'val'
        expected={'path':f'data/biological_paths_execution_pilot_v1/{parent}/truth/full_truth_intervals.npz','sha256':byid[parent]['files']['truth/full_truth_intervals.npz']['sha256']}
        if (set(record)!={'split','truth_npz','query_sample_indexes','query_raw_node_ids'} or record['split']!=split
            or byid[parent]['split']!=split or record['truth_npz']!=expected or len(record['query_sample_indexes'])!=120
            or len(record['query_raw_node_ids'])!=120):raise ValueError('Exact target source/identity record required')
    # No target NPZ bytes are read by admission.
    return admission,cfg,native_admission,qcfg,bindings,reader,storage


def _check_active_v1(admission):
    now=datetime.now(timezone.utc)
    if (not _utc(admission['created_at_utc'])<=now<_utc(admission['expires_at_utc'])
        or (_utc(admission['expires_at_utc'])-_utc(admission['created_at_utc'])).total_seconds()>86400):raise ValueError('Fit admission expired or exceeds24 hours')


def _check_resources_v1(admission,storage,lease):
    _check_active_v1(admission)
    if datetime.now(timezone.utc)>=_utc(storage['expires_at_utc']):raise ValueError('Fit storage reservation expired during execution')
    from .biological_candidate import check_fit_lease
    check_fit_lease(lease)


def metric_helpers_v1():
    spec=importlib.util.spec_from_file_location('census_biological_span_helpers_v1',ROOT/'scripts/train_ancestralpaths_biological_v1.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    return helper


def scoring_parent_v1(parent,bank,loaded,reader,root_binding):
    _,roster,positions,_,coverage,refs,_=loaded
    source=json.loads(reader.read(refs['source']));query=np.array([i for i,r in enumerate(source) if r['role']=='query'],dtype=np.int64)
    raw=np.array([source[i]['observed_node_id'] for i in query],dtype=np.int64)
    if query.tolist()!=root_binding['query_sample_indexes'] or raw.tolist()!=root_binding['query_raw_node_ids']:raise ValueError('External target columns must equal original observed query identities')
    retained=np.flatnonzero((positions>=coverage[0])&(positions<coverage[1]));edges=np.r_[100.*retained,100.*(retained[-1]+1)]
    edges[0]=max(edges[0],coverage[0]);edges[-1]=min(edges[-1],coverage[1])
    cohorts=np.array([source[i]['observed_cohort'] for i in query])
    required={'present_day':64,'Bronze_Age':16,'BAA':8,'Neo':16,'Yam':16}
    if {key:int((cohorts==key).sum()) for key in required}!=required or len(query)!=120:raise ValueError('Frozen query strata changed')
    return {'parent_id':parent['parent_id'],'split':parent['split'],'samples':source,'query_slots':query,'raw_query_nodes':raw,
        'cohorts':cohorts,'positions':positions[retained],'edges':edges,'retained_indexes':retained,'coverage_original':coverage,
        'query_ids':np.array([r.actual_id for r in roster if r.role=='query'],dtype=np.int64)}


class ExternalTargetV1:
    """Only C4 interval targets. Target graph/source ancestry identities never load."""
    def __init__(self,parent,binding,purpose,reader,prediction_record=None):
        if purpose not in ('TRAIN_loss','VAL_after_sealed_prediction') or parent['split']!=('train' if purpose=='TRAIN_loss' else 'val'):raise ValueError('Target purpose/split mismatch')
        if purpose=='VAL_after_sealed_prediction':
            if prediction_record is None:raise ValueError('VAL target opening requires sealed prediction')
            reader.read(prediction_record)
        data=reader.read(binding['truth_npz'])
        with np.load(io.BytesIO(data),allow_pickle=False) as archive:
            if set(archive.files)!={'intervals_bp','query_raw_node_ids','native_paths_1based','source_classes_0based'}:raise ValueError('Exact full-interval target schema required')
            self.intervals=archive['intervals_bp'].copy();self.c4=archive['source_classes_0based'].copy();ids=archive['query_raw_node_ids'].copy()
        # Six-path labels are deliberately never decoded or used.
        if (self.intervals.dtype!=np.float64 or self.intervals.ndim!=2 or self.intervals.shape[1]!=2
            or self.intervals[0,0]!=0 or self.intervals[-1,1]!=2000000 or not np.array_equal(self.intervals[:-1,1],self.intervals[1:,0])
            or not np.isfinite(self.intervals).all() or np.any(self.intervals[:,1]<=self.intervals[:,0])
            or self.c4.shape!=(len(self.intervals),120) or self.c4.dtype.kind not in 'iu' or np.any((self.c4<0)|(self.c4>3))
            or not np.array_equal(ids,parent['raw_query_nodes'])):raise ValueError('Full C4 interval/query identity target join invalid')
        self.parent_id=parent['parent_id'];self.split=parent['split'];self.purpose=purpose
        self.provenance={'record':binding['truth_npz'],'purpose':purpose,'prediction_record':prediction_record}
    def target_mass(self,parent,helpers):
        bins,rows,spans=helpers.intersections(parent['edges'],self.intervals)
        mass=np.zeros((len(parent['positions']),120,4),dtype=np.float64)
        np.add.at(mass,(bins[:,None],np.arange(120)[None,:],self.c4[rows]),spans[:,None])
        if not np.allclose(mass.sum(-1),np.diff(parent['edges'])[:,None],rtol=0,atol=1e-8):raise ValueError('Exact class-span mass incomplete')
        return mass


def optimizer_groups_v1(model,cfg):
    rate=[p for name,p in model.named_parameters() if name.endswith(('raw_exchange','raw_damping','raw_eta','mixture_logits'))]
    ids={id(p) for p in rate}
    return [{'params':[p for p in model.parameters() if id(p) not in ids],'weight_decay':cfg['training']['weight_decay']},
            {'params':rate,'weight_decay':0.}]


def matched_states_v1(cfg,seed,device):
    if seed not in cfg['seeds']:raise ValueError('Frozen optimizer seed required')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    initial=CensusTreeGNNV1(**cfg['model']).to(device=device,dtype=torch.float32)
    initial_hash=fingerprint_sha256_v1({k:v.detach().cpu() for k,v in initial.state_dict().items()})
    states={}
    for control in CONTROLS:
        model=deepcopy(initial)
        states[control]={'model':model,'optimizer':torch.optim.AdamW(optimizer_groups_v1(model,cfg),lr=cfg['training']['learning_rate'],betas=(.9,.999),eps=1e-8),
            'initial':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'initial_sha256':initial_hash,'steps':0,'best':float('inf'),'best_epoch':None,'history':[]}
    return states


def train_step_v1(state,cert,expected_hash,mass,control,permutation,cfg,device):
    if cert.provenance['split']!='train':raise ValueError('TRAIN-only backward requires TRAIN observed parent')
    model=state['model'];model.train();state['optimizer'].zero_grad(set_to_none=True)
    kwargs={'group_permutation':permutation} if control=='permuted_groups' else {}
    result=forward_certified_census_biological_v1(model,cert,expected_certificate_sha256=expected_hash,control=control,readout_chunk_size=cfg['readout_chunk_size'],**kwargs)
    target=torch.as_tensor(mass,dtype=torch.float64,device=device)
    if result.logits.shape!=target.shape or not bool(torch.isfinite(target).all() & (target>=0).all()) or not target.sum()>0:raise ValueError('Actual query120 exact TRAIN mass shape/values differ')
    loss=-(torch.log_softmax(result.logits,dim=-1).double()*target).sum()/target.sum()
    if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite exact TRAIN query-bp loss')
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite TRAIN gradient')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training']['gradient_clip_norm'],error_if_nonfinite=True)
    state['optimizer'].step();state['steps']+=1
    if not all(torch.isfinite(p).all() for p in model.parameters()):raise ValueError('Nonfinite fitted parameter')
    validate_census_biological_certificate_v1(cert,expected_hash)
    return {'loss':float(loss.detach()),'unclipped_gradient_norm':float(norm),'steps':state['steps']}


def raw_prediction_v1(state,cert,expected_hash,control,permutation,cfg):
    state['model'].eval();kwargs={'group_permutation':permutation} if control=='permuted_groups' else {}
    with torch.no_grad():
        result=forward_certified_census_biological_v1(state['model'],cert,expected_certificate_sha256=expected_hash,control=control,readout_chunk_size=cfg['readout_chunk_size'],**kwargs)
        logits=result.logits.detach().cpu();p=logits.softmax(-1).numpy()
    if not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(-1),1.,rtol=1e-5,atol=1e-6):raise ValueError('Raw C4 probabilities invalid without floors/renormalization')
    return logits.numpy(),p,result.scored_query_ids.numpy()


def strata_v1(parent):
    groups={'primary_present_day_64':np.flatnonzero(parent['cohorts']=='present_day'),
        'secondary_ancient_56':np.flatnonzero(parent['cohorts']!='present_day'),'all_queries_120':np.arange(120)}
    for cohort in ('Bronze_Age','BAA','Neo','Yam'):groups['age_stratum_'+cohort]=np.flatnonzero(parent['cohorts']==cohort)
    return groups


def score_sealed_v1(prediction_record,parent,binding,reader,helpers,cfg):
    data=reader.read(prediction_record)
    with np.load(io.BytesIO(data),allow_pickle=False) as archive:
        p=archive['C4_probabilities'].copy()
        checks={'positions_bp':parent['positions'],'bin_edges_bp':parent['edges'],'retained_fixed100bp_indexes':parent['retained_indexes'],'query_sample_indexes':parent['query_slots']}
        if any(not np.array_equal(archive[k],v) for k,v in checks.items()) or p.shape!=(len(parent['positions']),120,4):raise ValueError('Sealed prediction output/physical/identity join differs')
    if not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(-1),1,atol=1e-6,rtol=1e-5):raise ValueError('Invalid raw probabilities')
    target=ExternalTargetV1(parent,binding,'VAL_after_sealed_prediction',reader,prediction_record)
    groups={name:helpers.full_span_score(p,parent,target,columns,cfg['metrics']) for name,columns in strata_v1(parent).items()}
    return {'parent_id':parent['parent_id'],'split':'val','prediction_seal':prediction_record,'target_provenance':target.provenance,'groups':groups,'development_only':True,'TEST_parents':0}


def _nll_v1(value):return float('inf') if value['infinite_nll_query_bp'] else value['nll']


def run_fit_v1(admission_ref):
    """Actual use is root-admitted only. Author tests call handmade components."""
    admission,cfg,native_admission,qcfg,bindings,reader,storage=admit_fit_v1(admission_ref)
    helpers=metric_helpers_v1();paired=read_manifests_v1(native_admission,qcfg,reader)
    output=FitOutputV1(admission['output_directory'],storage,reader);tmp=output.path/'tmp';tmp.mkdir()
    from .biological_candidate import configure_fit,check_fit_lease
    device,runtime,lease=configure_fit(admission['device'],cfg,output.path)
    output.used=sum(max(p.stat().st_size,p.stat().st_blocks*512)+8192 for p in output.path.rglob('*') if p.is_file())
    if output.used>output.limit:raise ValueError('Runtime qualification writes exceeded fit reservation')
    states=matched_states_v1(cfg,admission['seed'],device);permutation=torch.tensor(admission['group_permutation'],dtype=torch.long)
    receipt={'schema':'census_biological_fit_evidence_v2','status':'running','hypothesis_id':'H021','job_id':admission['job_id'],'admission_ref':admission_ref,
        'seed':admission['seed'],'source_sha256':admission['source_sha256'],'config_ref':admission['refs']['config'],'numerical_ref':admission['refs']['numerical_evidence'],
        'runtime':runtime,'same_initial_weights_sha256':states['proposed']['initial_sha256'],'models':{},'development_VAL_parents':2,'TEST_parents':0,
        'handmade_fixture':bool(runtime.get('handmade_CPU',False)),'full_whole_tree_architecture':True,'target_inputs':False,'older_reference_proxy_evidence':True,'fitting_claim':False}
    output.save('started.json',receipt);start=time.monotonic()
    try:
        for epoch in range(1,cfg['training']['epochs']+1):
            _check_resources_v1(admission,storage,lease)
            order=list(range(cfg['training']['TRAIN_tiles_per_epoch']));random.Random(admission['seed']+epoch).shuffle(order)
            # Load one qualified full TRAIN parent at a time, but honor frozen global tile order.
            loaded_by_parent={};parent_scores={};targets={};train_tiles=[]
            for parent,bank in paired:
                if parent['split']!='train':continue
                loaded=load_parent_v1(parent,bank,native_admission,qcfg,reader,tmp_directory=tmp)
                loaded_by_parent[parent['parent_id']]=loaded
                score_parent=scoring_parent_v1(parent,bank,loaded,reader,bindings['parents'][parent['parent_id']]);parent_scores[parent['parent_id']]=score_parent
                target=ExternalTargetV1(score_parent,bindings['parents'][parent['parent_id']],'TRAIN_loss',reader)
                targets[parent['parent_id']]=target.target_mass(score_parent,helpers)
                train_tiles.extend((parent,bank,tile) for tile in sorted(bank['tiles'],key=lambda t:t['tile_id']))
            losses={c:[] for c in CONTROLS}
            for index in order:
                _check_resources_v1(admission,storage,lease)
                parent,bank,tile=train_tiles[index];cert,_,_=prepare_tile_v1(parent,bank,tile,loaded_by_parent[parent['parent_id']],native_admission,qcfg,reader)
                identity=f"{parent['parent_id']}:{tile['tile_id']}";expected=admission['expected_certificates'][identity]
                validate_census_biological_certificate_v1(cert,expected)
                indexes=np.asarray(tile['position_indexes'],dtype=int)
                for control in CONTROLS:
                    _check_resources_v1(admission,storage,lease)
                    losses[control].append(train_step_v1(states[control],cert,expected,targets[parent['parent_id']][indexes],control,permutation,cfg,device)['loss'])
                del cert
            del loaded_by_parent,targets
            # All five models' VAL predictions are sealed before any epoch VAL target read.
            seals={c:[] for c in CONTROLS};val_parents={}
            for parent,bank in paired:
                if parent['split']!='val':continue
                loaded=load_parent_v1(parent,bank,native_admission,qcfg,reader,tmp_directory=tmp)
                score_parent=scoring_parent_v1(parent,bank,loaded,reader,bindings['parents'][parent['parent_id']]);val_parents[parent['parent_id']]=score_parent
                logits={c:np.empty((len(score_parent['positions']),120,4),dtype=np.float32) for c in CONTROLS};seen=np.zeros(len(score_parent['positions']),dtype=bool)
                for tile in sorted(bank['tiles'],key=lambda t:t['tile_id']):
                    _check_resources_v1(admission,storage,lease)
                    cert,_,_=prepare_tile_v1(parent,bank,tile,loaded,native_admission,qcfg,reader);expected=admission['expected_certificates'][f"{parent['parent_id']}:{tile['tile_id']}"]
                    indexes=np.asarray(tile['position_indexes'],dtype=int)
                    if seen[indexes].any():raise ValueError('Repeated VAL physical output bins')
                    for control in CONTROLS:
                        _check_resources_v1(admission,storage,lease)
                        values,_,ids=raw_prediction_v1(states[control],cert,expected,control,permutation,cfg)
                        if not np.array_equal(ids,score_parent['query_ids']):raise ValueError('VAL query output identity differs')
                        logits[control][indexes]=values
                    seen[indexes]=True;del cert
                if not seen.all():raise ValueError('Missing VAL bins, no denominator reduction')
                for control in CONTROLS:
                    values=logits[control];p=torch.from_numpy(values).softmax(-1).numpy()
                    seal=output.save(f'{control}.epoch_{epoch:02d}.{parent["parent_id"]}.npz',{'logits':values,'C4_probabilities':p,
                        'positions_bp':score_parent['positions'],'bin_edges_bp':score_parent['edges'],'retained_fixed100bp_indexes':score_parent['retained_indexes'],
                        'query_sample_indexes':score_parent['query_slots'],'normalization_length_bp':np.array(2000000)},'npz')
                    seals[control].append((parent['parent_id'],seal))
                del loaded,logits
            _check_active_v1(admission)
            for control in CONTROLS:
                _check_resources_v1(admission,storage,lease)
                cells=[score_sealed_v1(seal,val_parents[parent_id],bindings['parents'][parent_id],reader,helpers,cfg) for parent_id,seal in seals[control]]
                nll=float(np.mean([_nll_v1(cell['groups']['primary_present_day_64']) for cell in cells]));state=states[control]
                row={'epoch':epoch,'TRAIN_mean_tile_query120_NLL':float(np.mean(losses[control])),'VAL_equal_parent_present64_NLL':None if not np.isfinite(nll) else nll,'VAL_NLL_is_infinite':not np.isfinite(nll),'optimizer_steps':state['steps'],'cells':cells}
                state['history'].append(row);output.save(f'{control}.epoch_{epoch:02d}.metrics.json',row)
                # Every epoch runs. Selection is descriptive development VAL minimum, earliest tie.
                if state['best_epoch'] is None or nll<state['best']:
                    checkpoint=output.save(f'{control}.best_epoch_{epoch:02d}.pt',{'model_state':{k:v.detach().cpu() for k,v in state['model'].state_dict().items()},
                        'optimizer_state':state['optimizer'].state_dict(),'epoch':epoch,'control':control,'seed':admission['seed'],'admission_ref':admission_ref},'torch')
                    state.update(best=nll,best_epoch=epoch,checkpoint=checkpoint)
            print(json.dumps({'stage':'census_fit_epoch','seed':admission['seed'],'epoch':epoch,'all_five_controls_steps':epoch*cfg['training']['TRAIN_tiles_per_epoch']}),flush=True)
        for control,state in states.items():
            if state['steps']!=cfg['training']['optimizer_steps_per_control_seed'] or state['best_epoch'] is None:raise ValueError('Incomplete fixed-budget/finite-selected-checkpoint fit')
            saved=torch.load(io.BytesIO(reader.read(state['checkpoint'])),map_location='cpu',weights_only=True)
            delta=max(float((v-state['initial'][k]).abs().max()) for k,v in saved['model_state'].items() if v.is_floating_point())
            if not delta>0:raise ValueError('Selected checkpoint has no genuine trained weight change')
            selected=state['history'][state['best_epoch']-1]
            means={}
            for group in strata_v1(next(iter(val_parents.values()))):
                means[group]={key:float(np.mean([cell['groups'][group][key] for cell in selected['cells']])) for key in ('accuracy','brier','calibration_ece')}
                cells=[cell['groups'][group] for cell in selected['cells']]
                means[group]['nll']=None if any(c['infinite_nll_query_bp'] for c in cells) else float(np.mean([c['nll'] for c in cells]))
                means[group]['NLL_is_infinite']=any(c['infinite_nll_query_bp'] for c in cells)
            receipt['models'][control]={'status':f"fixed{cfg['training']['epochs']}epochs_{state['steps']}steps_complete",'initial_sha256':state['initial_sha256'],'selected_epoch':state['best_epoch'],
                'checkpoint':state['checkpoint'],'optimizer_steps':state['steps'],'selected_weight_max_delta':delta,'selected_metrics':selected,'equal_parent_metrics':means}
        _check_resources_v1(admission,storage,lease)
        # Recheck all pinned sources and admission bytes at terminal completion.
        reader.read(admission_ref)
        for path,sha in admission['source_sha256'].items():reader.read({'path':path,'sha256':sha})
        receipt.update(status='completed_five_fixed_budget_development_controls_one_seed',fitting_claim=not receipt['handmade_fixture'],optimizer_fit_completed=True,elapsed_seconds=time.monotonic()-start,output_accounted_bytes=output.used)
    except Exception as error:
        receipt.update(status='failed_partial_preserved',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-start,output_accounted_bytes=output.used)
        output.save('failed.json',receipt);raise
    output.save('completed.json',receipt);return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--admission',required=True);parser.add_argument('--expected-admission-sha256',required=True)
    args=parser.parse_args();result=run_fit_v1({'path':args.admission,'sha256':args.expected_admission_sha256});print(json.dumps({'status':result['status']}))

if __name__=='__main__':main()
