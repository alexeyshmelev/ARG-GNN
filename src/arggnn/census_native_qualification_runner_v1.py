"""Two gated observation-only CPU qualification phases, no biological targets.

Root-owned immutable byte pins admit files. Inspection produces evidence, never
its own numerical admission. A fresh admission must freeze that manifest and its
per-tile digests before numerical rebuild/forward. No new geometry/scaler/cache.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import numpy as np
import torch
import tskit
from .census_biological_bank_v1 import (
    ObservedSampleRoleV1, prepare_census_biological_bank_v1,
    forward_certified_census_biological_v1, fingerprint_sha256_v1,
    validate_census_biological_certificate_v1, load_census_biological_config_v1,
    collect_census_biological_support_v1, select_census_numerical_tiles_v1)
from .census_tree_encoder_v1 import CensusTreeGNNV1, encode_whole_tree_contexts_v1
from .census_cut_exchange_v1 import permuted_group_control_v1, pointwise_local_messages_v1

ROOT=Path(__file__).resolve().parents[2]
CONTROLS=('proposed','eta0','permuted_groups','raw_diagonal','local')
ROSTER_FIELDS={'sample_index','sample_id','haplotype','observed_node_id','observed_cohort','age_generations','role','observed_reference_class'}
REF_ROLES={'runner_config','runtime','storage','launcher','native_manifest','prepared_manifest','feature_lineage','runner_review'}


def digest_bytes_v1(value): return hashlib.sha256(value).hexdigest()


def json_bytes_v1(value): return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def _sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _utc(value):
    try: result=datetime.fromisoformat(value.replace('Z','+00:00'))
    except (TypeError,ValueError): raise ValueError('Explicit UTC admission dates required') from None
    if result.tzinfo is None: raise ValueError('Admission date needs timezone')
    return result.astimezone(timezone.utc)


class VerifiedReaderV1:
    """Hash all bytes before giving them to a decoder; injectable for handmade tests."""
    def __init__(self,root=ROOT,read_bytes=None,max_file_bytes=8_000_000_000):
        self.root=Path(root).resolve();self.read_bytes=read_bytes;self.max_file_bytes=max_file_bytes
        self.evidence={}

    def relative(self,value):
        if not isinstance(value,str):raise ValueError('File path must be text')
        candidate=Path(value)
        if '..' in candidate.parts:raise ValueError('Parent traversal forbidden')
        if candidate.is_absolute():
            try: candidate=candidate.relative_to(self.root)
            except ValueError:raise ValueError('All admitted files must be project-local') from None
        if not candidate.parts or str(candidate)=='.':raise ValueError('A file path is required')
        return candidate.as_posix()

    def read(self,record,*,runtime_file=False):
        if set(record)!={'path','sha256'} or not _sha(record['sha256']):raise ValueError('Exact path/SHA256 record required')
        relative=self.relative(record['path']);path=self.root/relative
        if self.read_bytes is None:
            if (path.is_symlink() and not runtime_file) or not path.is_file() or path.stat().st_size>self.max_file_bytes:raise ValueError('Invalid or over-budget admitted file')
            try:path.resolve().relative_to(self.root)
            except ValueError:
                if not runtime_file:raise ValueError('File resolves outside project') from None
            data=path.read_bytes()
        else:data=self.read_bytes(relative)
        if not isinstance(data,bytes) or len(data)>self.max_file_bytes or digest_bytes_v1(data)!=record['sha256']:
            raise ValueError('Admitted source bytes changed: '+relative)
        self.evidence[relative]={'sha256':record['sha256'],'bytes':len(data)}
        return data

    def decode_json(self,record):return json.loads(self.read(record))

    def record(self,record):return {'path':self.relative(record['path']),'sha256':record['sha256']}


class BoundedEvidenceOutputV1:
    """Small JSON evidence only; exclusive new directory, counted output bytes."""
    def __init__(self,path,*,root=ROOT,limit_bytes=1_000_000_000):
        self.root=Path(root).resolve();self.path=Path(path)
        if not self.path.is_absolute():self.path=self.root/self.path
        self.path=self.path.resolve()
        try:self.path.relative_to(self.root)
        except ValueError:raise ValueError('Evidence output must remain project-local') from None
        if self.path.exists():raise ValueError('New exclusive evidence directory required')
        self.path.mkdir(parents=True);self.limit_bytes=limit_bytes;self.written_bytes=0

    def write(self,name,value):
        if Path(name).name!=name or not name.endswith('.json'):raise ValueError('Only named JSON evidence files are supported')
        data=json_bytes_v1(value)
        if self.written_bytes+len(data)>self.limit_bytes:raise ValueError('Evidence output cap exceeded')
        if os.statvfs(self.path).f_bavail*os.statvfs(self.path).f_frsize<len(data):raise ValueError('Insufficient local free bytes')
        with (self.path/name).open('xb') as stream:stream.write(data)
        self.written_bytes+=len(data)
        return {'path':str((self.path/name).relative_to(self.root)),'sha256':digest_bytes_v1(data),'bytes':len(data)}


def runtime_identity_v1():
    return {'python':sys.version.split()[0],'torch':str(torch.__version__),'numpy':str(np.__version__),
            'tskit':str(tskit.__version__),'device':'cpu','threads':1,'cuda_visible_devices':''}


def admit_runner_v1(admission_ref,*,phase,reader=None,now=None):
    reader=reader or VerifiedReaderV1();admission=reader.decode_json(admission_ref)
    allowed={'schema','status','job_id','phase','created_at_utc','expires_at_utc','refs','source_sha256','observed_artifact_sha256','output_directory','expected_certificates','inspection_manifest','initial_state_sha256_by_dtype','group_permutation'}
    if set(admission)!=allowed or admission['schema']!='census_native_qualification_admission_v1' or admission['status']!='active' or admission['phase']!=phase or phase not in {'inspect','numerical'}:
        raise ValueError('Fresh exact root active phase admission required')
    clock=now or datetime.now(timezone.utc)
    if not _utc(admission['created_at_utc'])<=clock<_utc(admission['expires_at_utc']):raise ValueError('Admission is future or expired')
    if not isinstance(admission['job_id'],str) or not admission['job_id']:raise ValueError('Distinct root job identity required')
    if set(admission['refs'])!=REF_ROLES:raise ValueError('All source/runtime/storage/observation/prepared/feature refs required')
    # Complete source pins are checked before any observation decoder is reached.
    config=reader.decode_json(admission['refs']['runner_config'])
    if (clock-_utc(admission['created_at_utc'])).total_seconds()>config['admission_max_age_seconds'] or (_utc(admission['expires_at_utc'])-_utc(admission['created_at_utc'])).total_seconds()>config['admission_max_age_seconds']:raise ValueError('Root active admission is not fresh')
    required=set(config['required_source_paths'])|{'src/arggnn/census_native_qualification_runner_v1.py'}
    if set(admission['source_sha256'])!=required:raise ValueError('Complete exact executed source/review pins required')
    for path,sha in admission['source_sha256'].items():reader.read({'path':path,'sha256':sha})
    for path,sha in config['source_sha256'].items():
        if admission['source_sha256'].get(path)!=sha:raise ValueError('Frozen runner source dependency changed')
    if config['kind']!='census_native_qualification_runner_v1' or config['controls']!=list(CONTROLS) or config['expected_tiles']!=32 or config['output_cap_bytes']!=1_000_000_000:
        raise ValueError('Frozen qualification method/cap changed')
    shipped=json.loads((ROOT/'configs/census_native_qualification_runner_v1.json').read_text())
    allowed_fixture={'fixture_mode','role_counts','references_per_class','model'}
    if config['fixture_mode']:
        if set(config)!=set(shipped) or any(config[k]!=v for k,v in shipped.items() if k not in allowed_fixture):raise ValueError('Fixture changes only counts and small model dimensions')
    elif config!=shipped:raise ValueError('Shipped actual qualification config changed')
    if now is not None and not config['fixture_mode']:raise ValueError('Injected clock belongs only to handmade mode')
    if reader.max_file_bytes!=config['max_file_bytes']:raise ValueError('Frozen per-file byte budget changed')
    if config['fixture_mode'] and reader.read_bytes is None:raise ValueError('Fixture config requires injected bytes; cannot admit actual files')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('PYTHONDONTWRITEBYTECODE')!='1':raise ValueError('CPU-only disabled-bytecode launcher required')
    if Path(sys.executable).resolve()!=(ROOT/'.runtime/env/bin/python').resolve():raise ValueError('Main project Python interpreter required')
    torch.set_num_threads(1)
    runtime=reader.decode_json(admission['refs']['runtime'])
    if set(runtime)!={'schema','identity','runtime_file_sha256'} or runtime['schema']!='census_native_CPU_runtime_v1' or runtime['identity']!=runtime_identity_v1() or set(runtime['runtime_file_sha256'])!=set(config['runtime_files']):raise ValueError('Exact main-runtime identity/files changed')
    for path,sha in runtime['runtime_file_sha256'].items():reader.read({'path':path,'sha256':sha},runtime_file=True)
    review=reader.decode_json(admission['refs']['runner_review'])
    expected_review={'schema':'census_native_qualification_review_binding_v1','status':'qualified_source_handmade_CPU_only','runner_source_sha256':admission['source_sha256']['src/arggnn/census_native_qualification_runner_v1.py'],'runner_config_sha256':admission['refs']['runner_config']['sha256'],'actual_admission':False,'review_receipt_ref':review.get('review_receipt_ref')}
    if review!=expected_review or not review['review_receipt_ref']:raise ValueError('Fresh independent exact runner/config review binding required')
    full_review=reader.decode_json(review['review_receipt_ref'])
    if (not str(full_review.get('status','')).startswith('qualified') or full_review.get('artifact_sha256',{}).get('src/arggnn/census_native_qualification_runner_v1.py')!=expected_review['runner_source_sha256'] or full_review.get('artifact_sha256',{}).get('configs/census_native_qualification_runner_v1.json')!=expected_review['runner_config_sha256'] or (full_review.get('fixture_mode') is True and not config['fixture_mode'])):raise ValueError('Full independent review receipt does not bind executed source/config')
    storage=reader.decode_json(admission['refs']['storage']);launcher=reader.decode_json(admission['refs']['launcher'])
    if (storage.get('schema')!='census_native_storage_reservation_v1' or storage.get('status')!='active' or storage.get('job_id')!=admission['job_id']
        or type(storage.get('used_project_bytes')) is not int or not 0<=storage['used_project_bytes']<450_000_000_000
        or type(storage.get('reserved_new_bytes')) is not int or storage['reserved_new_bytes']<config['output_cap_bytes']
        or storage.get('output_directory')!=admission['output_directory'] or not _utc(storage['created_at_utc'])<=clock<_utc(storage['expires_at_utc'])
        or type(storage.get('concurrent_reserve_bytes')) is not int or storage['concurrent_reserve_bytes']<0
        or storage['used_project_bytes']+storage['reserved_new_bytes']+storage['concurrent_reserve_bytes']>=450_000_000_000
        or storage.get('project_limit_bytes')!=500_000_000_000):raise ValueError('New active bounded storage reservation required')
    snapshot=reader.decode_json(storage['snapshot_ref'])
    if (snapshot.get('schema')!='census_native_root_storage_snapshot_v1' or snapshot.get('status')!='complete' or snapshot.get('returncode')!=0 or snapshot.get('project_root')!=str(ROOT) or snapshot.get('used_project_bytes')!=storage['used_project_bytes'] or not 0<=((clock-_utc(snapshot['created_at_utc'])).total_seconds())<=config['storage_snapshot_max_age_seconds']):raise ValueError('Fresh complete root storage snapshot binding required')
    if launcher!={'schema':'census_native_CPU_launcher_v1','job_id':admission['job_id'],'python':'.runtime/env/bin/python','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','GPU_admitted':False}:
        raise ValueError('Exact new CPU launcher receipt required')
    if phase=='inspect':
        if admission['expected_certificates'] or admission['inspection_manifest'] is not None or admission['initial_state_sha256_by_dtype'] or admission['group_permutation']:
            raise ValueError('Inspection cannot grant itself numerical certificates')
    else:
        if not admission['inspection_manifest'] or len(admission['expected_certificates'])!=32 or set(admission['initial_state_sha256_by_dtype'])!=set(config['dtypes']):raise ValueError('Root frozen inspection/digests/initial weights required')
    return admission,config,reader


def _bound(reader,admission,record):
    ref=reader.record({'path':record['path'],'sha256':record['sha256']})
    if any(any(word in part.lower() for word in ('truth','oracle','target')) for part in Path(ref['path']).parts):raise ValueError('Biological truth/oracle/target paths are not observation capabilities')
    if admission['observed_artifact_sha256'].get(ref['path'])!=ref['sha256']:raise ValueError('Observed file not explicitly admitted by root')
    return ref


def read_manifests_v1(admission,config,reader):
    native=reader.decode_json(admission['refs']['native_manifest']);prepared=reader.decode_json(admission['refs']['prepared_manifest'])
    if (native.get('kind')!='biological_paths_relate_v1' or native.get('status')!='four_development_parents_inferred_pre_and_post_five_iterations'
        or native.get('primary_condition')!='post_five_size_iterations_primary' or native.get('truth_or_oracle_read') is not False
        or prepared.get('kind')!='biological_candidate_prepared_v1' or prepared.get('status')!='complete_observed_only_Q184_R96_actualQ120'
        or prepared.get('truth_opened') is not False or prepared.get('labels_in_features_or_banks') is not False
        or prepared.get('observed_manifest_sha256')!=admission['refs']['native_manifest']['sha256']):raise ValueError('Original completed observed/prepared lineage required')
    parents=native['replicates'];banks=prepared['parents']
    if (len(parents)!=4 or len(banks)!=4 or {p['parent_id'] for p in parents}!=set(config['parents']) or {p['parent_id'] for p in banks}!=set(config['parents'])):raise ValueError('Exactly frozen four development parents required')
    feature=reader.decode_json(admission['refs']['feature_lineage'])
    expected={'schema':'census_original_feature_lineage_v1','prepared_manifest_sha256':admission['refs']['prepared_manifest']['sha256'],
      'producer_source_sha256':config['source_sha256']['src/arggnn/biological_candidate.py'],
      'reference_adapter_sha256':config['source_sha256']['src/arggnn/reference_encoder.py'],
      'feature_columns':list(range(7))+list(range(8,48)),'scaler':'identity_no_fitted_scaler',
      'original_TRAIN_parent_ids':[p for p in config['parents'] if '_train_' in p],'labels_in_input':False}
    if feature!=expected:raise ValueError('Original TRAIN feature/scaler lineage changed')
    if native.get('implementation_sha256')!=config['source_sha256']['scripts/infer_biological_paths_relate_v1.py'] or native.get('config_sha256')!=config['source_sha256']['configs/biological_paths_relate_v1.json'] or prepared.get('config_sha256')!=config['source_sha256']['configs/biological_candidate_pilot_v1.json']:raise ValueError('Original producer/config lineage changed')
    for path in ('src/arggnn/biological_candidate.py','src/arggnn/reference_encoder.py','src/arggnn/genealogy_baselines.py','src/arggnn/data.py'):
        if prepared.get('source_sha256',{}).get(path)!=config['source_sha256'][path]:raise ValueError('Prepared source feature transformation changed')
    paired=[];paths=set()
    for p in sorted(parents,key=lambda p:p['parent_id']):
        bank=next(b for b in banks if b['parent_id']==p['parent_id'])
        split='train' if '_train_' in p['parent_id'] else 'val'
        if p.get('split')!=split or bank.get('split')!=split or p.get('status')!='inferred_both_conditions' or p.get('truth_or_oracle_read') is not False:raise ValueError('Original development split/provenance changed')
        if bank.get('relate_entry')!=p:raise ValueError('Prepared parent observed producer entry changed')
        if p['inferred_tree'].get('condition')!='post_five_size_iterations_primary':raise ValueError('Post-five primary tree required')
        if len(bank['tiles'])!=8 or {t['tile_id'] for t in bank['tiles']}!=set(range(8)):raise ValueError('All32 original tiles required before selection')
        for record in (p['inferred_tree'],p['observed_sample_manifest'],p['original_observed_sample_manifest'],p['completion_provenance'],p['coordinate_contract']['fixed_grid']):paths.add(_bound(reader,admission,record)['path'])
        for tile in bank['tiles']:paths.add(_bound(reader,admission,tile['whole_observed_graph'])['path'])
        paired.append((p,bank))
    if paths!=set(admission['observed_artifact_sha256']):raise ValueError('Exact observed-only admitted file closure required')
    for path,sha in admission['observed_artifact_sha256'].items():reader.read({'path':path,'sha256':sha})
    return paired


def _decode_native_v1(data,tmp_directory):
    # tskit requires a file descriptor. Only verified bytes reach this temporary.
    with tempfile.NamedTemporaryFile(dir=tmp_directory,suffix='.handed_native.trees') as handle:
        handle.write(data);handle.flush();return tskit.load(handle.name)


def _decode_grid_v1(data):
    with np.load(io.BytesIO(data),allow_pickle=False) as grid:
        if set(grid.files)!={'positions_bp','bin_edges_bp'}:raise ValueError('Original grid whitelist required')
        return {k:grid[k].copy() for k in grid.files}


def _decode_observation_v1(data):return torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)


def load_parent_v1(parent,bank,admission,config,reader,*,tmp_directory,native_decoder=None):
    coordinate=parent['coordinate_contract'];derived=parent['observed_sample_manifest'];source=parent['original_observed_sample_manifest']
    refs={'native':_bound(reader,admission,parent['inferred_tree']),'derived':_bound(reader,admission,derived),
      'source':_bound(reader,admission,source),'completion':_bound(reader,admission,parent['completion_provenance']),'grid':_bound(reader,admission,coordinate['fixed_grid'])}
    # Verify every parent byte record before decoding any tree, roster or grid.
    data={k:reader.read(v) for k,v in refs.items()}
    original=json.loads(data['source']);rows=json.loads(data['derived']);completion=json.loads(data['completion'])
    if completion!={k:v for k,v in parent.items() if k!='completion_provenance'} or reader.record(derived['source_manifest'])!=refs['source']:raise ValueError('Derived/completion source lineage differs')
    count=sum(config['role_counts'].values());mapping=parent['sample_index_to_inferred_tree_node_id']
    if derived.get('rows')!=count or len(original)!=count or len(rows)!=count or len(mapping)!=count or [m['sample_index'] for m in mapping]!=list(range(count)):raise ValueError('Complete original ordered roster required')
    roster=[]
    for i,(raw,row,joined) in enumerate(zip(original,rows,mapping)):
        if set(raw)!=ROSTER_FIELDS or raw['sample_index']!=i or type(raw['sample_index']) is not int:raise ValueError('Observation-only source roster fields/order required')
        if set(joined)!={'sample_index','inferred_tree_node_id'} or type(joined['inferred_tree_node_id']) is not int or joined['inferred_tree_node_id']<0 or type(raw['observed_node_id']) is not int or raw['observed_node_id']<0 or type(raw['haplotype']) is not int or raw['role'] not in config['role_counts'] or type(raw['age_generations']) not in (int,float) or not math.isfinite(raw['age_generations']) or raw['age_generations']<0:raise ValueError('Explicit observed/native numeric roles and dates required')
        if raw['role']!='pure_source_reference' and (raw['observed_reference_class'] is not None or raw['age_generations']>=350.):raise ValueError('All source Q strictly younger350 with no class')
        expected=dict(raw);expected['source_observed_node_id']=expected.pop('observed_node_id');expected['inferred_tree_node_id']=joined['inferred_tree_node_id'];expected['observed_node_id']=joined['inferred_tree_node_id']
        if row!=expected:raise ValueError('Source-to-native roster attribute or identity changed')
        cls=raw['observed_reference_class']
        if raw['role']=='pure_source_reference' and (type(cls) is not int or cls not in range(4) or raw['observed_cohort']!=config['classes'][cls]):raise ValueError('Observed proxy class/cohort differs')
        if not isinstance(raw['sample_id'],str) or not raw['sample_id'] or raw['haplotype'] not in (0,1):raise ValueError('Explicit sample/haplotype ordering required')
        roster.append(ObservedSampleRoleV1(i,joined['inferred_tree_node_id'],raw['role'],raw['age_generations'],raw['observed_reference_class']))
    if not config['fixture_mode']:
        for i in range(0,count,2):
            a,b=original[i:i+2]
            if a['sample_id']!=b['sample_id'] or [a['haplotype'],b['haplotype']]!=[0,1] or any(a[k]!=b[k] for k in ('role','observed_cohort','age_generations','observed_reference_class')):raise ValueError('Original diploid subject/role/date pairing changed')
    grid=_decode_grid_v1(data['grid']);positions=grid['positions_bp'];edges=grid['bin_edges_bp']
    if (positions.dtype!=np.float64 or not np.array_equal(positions,np.arange(50.,2000000.,100.)) or not np.array_equal(edges,np.arange(0.,2000001.,100.))):raise ValueError('Original complete100bp physical grid required')
    coverage=coordinate['coverage_original_bp0'];offset=coordinate['local_to_original_offset_bp']
    if (coordinate['normalization_length_bp']!=2000000 or coordinate['original_parent_interval_bp0']!=[0,2000000]
        or coordinate['physical_coordinate_transform']!='physical_original = local_tree_position + coverage_left'
        or coordinate['uncovered_tails']!='excluded; no extrapolation' or len(coverage)!=2 or not 0<=coverage[0]<coverage[1]<=2000000 or offset!=coverage[0]):raise ValueError('Original native physical coverage transform changed')
    retained=np.flatnonzero((positions>=coverage[0])&(positions<coverage[1]))
    if coordinate['fixed_grid'].get('total')!=20000 or coordinate['retained_fixed100bp_indexes']!=retained.tolist():raise ValueError('Original retained readout ordering changed')
    post=parent['post_five_iterations']
    post_checks={'clean_tree':parent['inferred_tree']['path'],'clean_tree_sha256':parent['inferred_tree']['sha256'],'sample_index_to_inferred_tree_node_id':mapping,'observed_sample_manifest':derived,'coverage_original_bp0':coverage,'local_to_original_offset_bp':offset,'sequence_length_local_bp':coverage[1]-coverage[0],'retained_fixed100bp_indexes':retained.tolist()}
    if any(post.get(k)!=v for k,v in post_checks.items()):raise ValueError('Condition-specific native source lineage changed')
    for tile in bank['tiles']:reader.read(_bound(reader,admission,tile['whole_observed_graph']))
    if len(data['native'])>config['output_cap_bytes']//2:raise ValueError('Verified native temporary exceeds reserved evidence/temp budget')
    ts=(native_decoder(data['native']) if native_decoder else _decode_native_v1(data['native'],tmp_directory))
    if type(post.get('trees')) is not int or post['trees']!=ts.num_trees:raise ValueError('Native original tree count changed')
    if ts.sequence_length!=coverage[1]-coverage[0] or set(ts.samples())!={r.native_sample_node for r in roster}:raise ValueError('Native tree sample or physical coverage join differs')
    for r in roster:
        if float(ts.node(r.native_sample_node).time)!=r.age_generations:raise ValueError('Direct native float64 observed sample date changed')
    dynamic=[r for r in roster if r.role!='pure_source_reference'];scored=[r for r in roster if r.role=='query'];references=[r for r in roster if r.role=='pure_source_reference']
    checks={'dynamics_sample_indexes':[r.actual_id for r in dynamic],'actual_query_sample_indexes':[r.actual_id for r in scored],
      'reference_sample_indexes':[r.actual_id for r in references],'actual_query_output_indexes':[dynamic.index(r) for r in scored],
      'query_node_ids':[r.native_sample_node for r in scored],'dynamics_query_node_ids':[r.native_sample_node for r in dynamic],
      'coverage_original_bp0':coverage,'retained_fixed100bp_indexes':retained.tolist(),'positions_bp':positions[retained].tolist()}
    if any(bank.get(k)!=v for k,v in checks.items()):raise ValueError('Prepared original sample/readout/target-column ordering differs')
    lineage={'source_roster':refs['source'],'derived_roster':refs['derived'],'native_tree':refs['native'],'grid':refs['grid'],'completion':refs['completion'],
      'ordered_observed_subjects':[[r['sample_id'],r['haplotype']] for r in original],
      'scored_sample_indexes':checks['actual_query_sample_indexes'],'target_columns':checks['actual_query_output_indexes'],
      'native_float64_sample_dates':[float(ts.node(r.native_sample_node).time) for r in roster]}
    return ts,roster,positions,offset,coverage,refs,lineage


def event_collision_counts_v1(ts,left,right,offset,positions):
    intervals=[[max(float(t.interval.left)+offset,left),min(float(t.interval.right)+offset,right)] for t in ts.trees() if min(float(t.interval.right)+offset,right)>max(float(t.interval.left)+offset,left)]
    boundaries=np.array([left]+[b for _,b in intervals],dtype=np.float64)
    anchors=left+(np.arange(32,dtype=np.float64)+.5)*((right-left)/32.)
    frozen=left/2000000.+(np.arange(32,dtype=np.float64)+.5)*((right/2000000.-left/2000000.)/32.)
    readouts=np.asarray(positions,dtype=np.float64)
    mismatches=0
    for i,a in enumerate(anchors):
        if (a in boundaries or a in readouts) and frozen[i]!=a/2000000.:mismatches+=1
    return {'anchor_boundary':len(np.intersect1d(anchors,boundaries)),'anchor_readout':len(np.intersect1d(anchors,readouts)),
      'boundary_readout':len(np.intersect1d(boundaries,readouts)),'unequal_IEEE_anchor_collision':mismatches,
      'policy':'reject equal physical bp with unequal frozen normalized IEEE; no epsilon'}


def prepare_tile_v1(parent,bank,tile,loaded,admission,config,reader,*,observation_decoder=None):
    ts,roster,positions,offset,coverage,refs,lineage=loaded
    left=max(tile['tile_id']*250000.,coverage[0]);right=min((tile['tile_id']+1)*250000.,coverage[1])
    indexes=np.flatnonzero((positions>=left)&(positions<right))
    original_tree_ids=[int(t.index) for t in ts.trees() if min(float(t.interval.right)+offset,right)>max(float(t.interval.left)+offset,left)]
    dyn=[r.native_sample_node for r in roster if r.role!='pure_source_reference'];reference=[r for r in roster if r.role=='pure_source_reference'];landmarks=[]
    for cls in range(4):
        panel=[r.native_sample_node for r in reference if r.observed_reference_class==cls]
        if not panel:raise ValueError('All observed reference classes required')
        landmarks.extend(panel[i] for i in np.linspace(0,len(panel)-1,4,dtype=int))
    checks={'physical_interval_bp0':[left,right],'normalized_interval':[left/2000000.,right/2000000.],
      'original_grid_indexes':indexes.tolist(),'position_indexes':np.searchsorted(np.flatnonzero((positions>=coverage[0])&(positions<coverage[1])),indexes).tolist(),'original_tree_ids':original_tree_ids,'landmarks_observed_node_ids':landmarks,
      'all_tree_nodes_retained':True,'all280_leaves_retained':True,'mixed_query_flag':0,
      'query_dynamics':len(dyn),'pure_reference_sources':len(reference),'supervised_outputs':sum(r.role=='query' for r in roster),'mixed_latent_outputs':sum(r.role=='mixed_reference_only' for r in roster)}
    if any(tile.get(k)!=v for k,v in checks.items()):raise ValueError('Original prepared tile/tree/grid/landmark metadata changed')
    record=_bound(reader,admission,tile['whole_observed_graph']);data=reader.read(record)
    if observation_decoder and not config['fixture_mode']:raise ValueError('Injected observation decoder belongs only to handmade mode')
    raw=observation_decoder(data) if observation_decoder else _decode_observation_v1(data)
    collision=event_collision_counts_v1(ts,left,right,offset,positions[indexes])
    adapter_config=load_census_biological_config_v1()
    if config['fixture_mode']:adapter_config.update(fixture_mode=True,role_counts=config['role_counts'],references_per_class=config['references_per_class'])
    bindings={'native_observed_ts':refs['native'],'serialized_whole_observed':record,'observed_roster':refs['derived'],'observed_grid':refs['grid'],'producer_receipt':refs['completion']}
    cert=prepare_census_biological_bank_v1(ts,raw,roster,left_bp=left,right_bp=right,offset_bp=offset,
      positions_bp=torch.tensor(positions[indexes],dtype=torch.float64),original_grid_indexes=torch.tensor(indexes,dtype=torch.long),
      source_bindings=bindings,parent_id=parent['parent_id'],split=parent['split'],tile_id=tile['tile_id'],config=adapter_config)
    if cert.stored_sequence.query_node_ids.tolist()!=dyn:raise ValueError('Decoded dynamics order changed')
    lineage=dict(lineage,whole_observed_graph=record,original_grid_indexes=indexes.tolist(),
      original_tree_ids=original_tree_ids,landmarks_native=landmarks,producer_replay='value-exact dtype/shape/stride equality; signed zero bytes may differ',
      feature_columns=list(range(7))+list(range(8,48)),scaler='identity_no_fitted_scaler')
    return cert,collision,lineage


def native_oracle_checks_v1(cert,loaded):
    ts,roster,*_=loaded
    q=[r.native_sample_node for r in roster if r.role!='pure_source_reference'];r=[r.native_sample_node for r in roster if r.role=='pure_source_reference']
    actual=[row.actual_id for row in roster if row.role!='pure_source_reference'];maximum=0.
    for graph,tree_id in enumerate(cert.provenance['original_tree_ids']):
        tree=ts.at_index(tree_id)
        oracle=torch.tensor([[tree.time(tree.mrca(a,b)) for b in r] for a in q],dtype=torch.float64)
        oracle=torch.softmax(-torch.log1p(oracle)[None]/math.log1p(10000.)/torch.tensor([.05,.1,.2],dtype=torch.float64)[:,None,None],-1)
        torch.testing.assert_close(cert.bank.donor_weights[graph],oracle,rtol=1e-10,atol=1e-12)
        maximum=max(maximum,float((cert.bank.donor_weights[graph]-oracle).abs().max()))
        remaining=set(range(len(q)));groups=[]
        while remaining:
            first=min(remaining);members={i for i in remaining if tree.time(tree.mrca(q[first],q[i]))<=350.}
            groups.append(frozenset(actual[i] for i in members));remaining-=members
        if set(groups)!=set(cert.bank.partitions[graph].group_members):raise ValueError('Native census oracle differs')
    return {'direct_native_donor_max_abs_error':maximum,'direct_native_census_groups':True,'float64_native_dates':True}


def _model(config,dtype):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seed']);model=CensusTreeGNNV1(**config['model'])
    return model.to(dtype=getattr(torch,dtype))


def numerical_cell_v1(cert,expected_digest,config,permutation,expected_states):
    """Disposable target-free checks. Same initialized state per dtype/control."""
    validate_census_biological_certificate_v1(cert,expected_digest)
    coverage=[];records=[];baseline_outputs={};baseline_gradients={}
    for dtype in config['dtypes']:
        initial=_model(config,dtype);state=fingerprint_sha256_v1(initial.state_dict())
        if state!=expected_states[dtype]:raise ValueError('Externally frozen initial model state differs')
        tol=config['tolerances'][dtype]
        # Qualify the same retained forest using separate and batched sweeps/VJPs.
        encoder=initial.encoder
        separate=[encoder(item) for item in cert.inputs];batched=encoder(cert.forest)
        for a,b in zip(separate,batched):torch.testing.assert_close(a,b,rtol=tol['output_rtol'],atol=tol['output_atol'])
        gradients=[]
        for values in (separate,batched):gradients.append(torch.autograd.grad(sum((i+1)*v.double().square().mean() for i,v in enumerate(values)),tuple(encoder.parameters())))
        for a,b in zip(*gradients):torch.testing.assert_close(a,b,rtol=tol['vjp_rtol'],atol=tol['vjp_atol'])
        for control in CONTROLS:
            model=deepcopy(initial);kwargs={'group_permutation':permutation} if control=='permuted_groups' else {}
            before=fingerprint_sha256_v1(model.state_dict())
            output=forward_certified_census_biological_v1(model,cert,expected_certificate_sha256=expected_digest,control=control,**kwargs)
            scored=cert.bank.query_ids[cert.bank.partitions[0].scored_query_mask]
            if output.scored_query_ids.tolist()!=scored.tolist() or output.logits.shape!=(len(cert.stored_sequence.positions),len(scored),4):raise ValueError('Actual sample/readout/target-column output order differs')
            bank=permuted_group_control_v1(cert.bank,permutation) if control=='permuted_groups' else cert.bank
            q,r=encode_whole_tree_contexts_v1(model.encoder,cert.inputs,bank.query_ids,bank.reference_ids)
            work=bank.to('cpu',getattr(torch,dtype));anchor=work.layout.anchor_graph
            current_anchor=q[anchor];current_target=q[work.target_graph]
            minimum_mass=math.inf;maximum_source=0.
            for layer in model.layers:
                if control=='local':
                    am,amass=pointwise_local_messages_v1(layer.query_projection(current_anchor).tanh(),layer.reference_projection(r[anchor]).tanh(),work.donor_weights[anchor],layer.exchange,layer.mixture)
                    tm,tmass=pointwise_local_messages_v1(layer.query_projection(current_target).tanh(),layer.reference_projection(r[work.target_graph]).tanh(),work.donor_weights[work.target_graph],layer.exchange,layer.mixture)
                    masses=(amass,tmass)
                else:
                    messages=layer.messages(work,current_anchor,r[anchor],control=control,readout_chunk_size=config['readout_chunk_size'])
                    am,tm=messages.anchors,messages.targets;masses=(messages.mass,)
                for mass in masses:
                    if not bool(torch.isfinite(mass).all() & (mass>0).all()):raise ValueError('Positive mass fails without floors')
                    minimum_mass=min(minimum_mass,float(mass.detach().min()))
                maximum_source=max(maximum_source,float(am.detach().abs().max()),float(tm.detach().abs().max()))
                if maximum_source>1.+tol['output_atol']:raise ValueError('Normalized source bound fails without floors')
                current_anchor=layer.update(current_anchor,am);current_target=layer.update(current_target,tm)
            manual_logits=model.decoder(current_target[:,cert.bank.partitions[0].scored_query_mask])
            torch.testing.assert_close(output.logits,manual_logits,rtol=tol['output_rtol'],atol=tol['output_atol'])
            coefficients=torch.linspace(-.7,1.1,output.logits.numel(),dtype=torch.float64).reshape_as(output.logits)
            objective=(output.logits.double()*coefficients).mean();parameters=tuple(model.parameters())
            grads=torch.autograd.grad(objective,parameters,allow_unused=True)
            active=0
            for (name,parameter),grad in zip(model.named_parameters(),grads):
                inactive=(name.endswith('raw_eta') and (control in {'eta0','local'} or len(cert.native_trees)==1)) or (name.endswith('raw_damping') and control=='local')
                if inactive and grad is not None:raise ValueError('Inactive control parameter has a gradient')
                if not inactive and (grad is None or not torch.isfinite(grad).all()):raise ValueError('Missing/nonfinite active control VJP')
                parameter.grad=grad;active+=grad is not None
            directional=None
            if dtype=='float32':
                torch.testing.assert_close(output.logits.double(),baseline_outputs[control],rtol=tol['output_rtol'],atol=tol['output_atol'])
                for grad,reference in zip(grads,baseline_gradients[control]):
                    if grad is not None:torch.testing.assert_close(grad.double(),reference,rtol=tol['vjp_rtol'],atol=tol['vjp_atol'])
            else:
                baseline_outputs[control]=output.logits.detach().clone()
                baseline_gradients[control]=tuple(None if g is None else g.detach().clone() for g in grads)
            if dtype=='float64':
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(config['seed']+1);directions=[torch.randn_like(p)/math.sqrt(sum(p.numel() for p in parameters)) for p in parameters]
                analytic=sum(float((g*d).sum()) for g,d in zip(grads,directions) if g is not None)
                epsilon=config['directional_epsilon'];original=[p.detach().clone() for p in parameters];values=[]
                for sign in (-1,1):
                    with torch.no_grad():
                        for p,start,direction in zip(parameters,original,directions):p.copy_(start+sign*epsilon*direction)
                    logits=forward_certified_census_biological_v1(model,cert,expected_certificate_sha256=expected_digest,control=control,**kwargs).logits
                    values.append(float((logits.double()*coefficients).mean()))
                with torch.no_grad():
                    for p,start in zip(parameters,original):p.copy_(start)
                finite=(values[1]-values[0])/(2*epsilon)
                if not math.isclose(analytic,finite,rel_tol=tol['vjp_rtol'],abs_tol=tol['vjp_atol']):raise ValueError('Whole-GNN directional derivative differs')
                directional={'analytic':analytic,'central_difference':finite,'absolute_error':abs(analytic-finite),'epsilon':epsilon}
            if fingerprint_sha256_v1(model.state_dict())!=before or before!=state:raise ValueError('Matched initial weights changed before disposable unit step')
            norm=torch.nn.utils.clip_grad_norm_(parameters,config['gradient_clip'],error_if_nonfinite=True)
            torch.optim.AdamW(parameters,lr=config['unit_step_lr'],weight_decay=config['unit_step_weight_decay']).step()
            if not all(torch.isfinite(p).all() for p in parameters):raise ValueError('Disposable clipped AdamW unit step is nonfinite')
            records.append({'dtype':dtype,'control':control,'initial_state_sha256':state,'active_VJP_tensors':active,
              'positive_mass_min':minimum_mass,'bounded_source_max_abs':maximum_source,
              'clipped_gradient_norm':float(norm),'finite_disposable_AdamW':True,'forest_output_VJP_parity':True,'sample_readout_target_column_order':True,
              'directional_float64':directional,'float32_vs_float64_output_VJP_parity':dtype=='float32',
              'synthetic_objective':'fixed coefficients on mean logits, no biological targets'})
            coverage.append([dtype,control])
    return {'status':'qualified','performed_controls':coverage,'checks':records}


def structural_cell_v1(cert,*,hidden_dim):
    """Per-cell structural evidence without numerical or TRAIN selection side effects."""
    spans=[b-a for a,b in cert.provenance['physical_intervals_bp']];total=sum(spans)
    sizes={};rho_weighted={};singleton=0.;per_tree=[];turnover=[]
    from .census_cut_exchange_v1 import clade_turnover_rho_v1
    for graph,(part,span) in enumerate(zip(cert.bank.partitions,spans)):
        values=part.group_size_by_query.tolist();hist={str(int(v)):values.count(v)/len(values) for v in set(values)}
        per_tree.append({'group_count':len(part.group_members),'query_weighted_group_size':hist,'singleton_Q_fraction':hist.get('1',0.)})
        for size,frequency in hist.items():sizes[size]=sizes.get(size,0.)+span/total*frequency
        singleton+=span/total*hist.get('1',0.)
        if graph:
            rho=clade_turnover_rho_v1(cert.bank.partitions[graph-1],part).tolist()
            turnover.append({'right_hold_span_bp':span,'query_weighted_rho':{str(v):rho.count(v)/len(rho) for v in set(rho)}})
            for value in rho:rho_weighted[str(value)]=rho_weighted.get(str(value),0.)+span/len(rho)
    norm=sum(rho_weighted.values())
    if norm:rho_weighted={k:v/norm for k,v in rho_weighted.items()}
    n=len(cert.forest.node);height=len(cert.forest.upward_levels)-1;depth=len(cert.forest.downward_levels)-1
    return {'native_connected_node_counts':[len(t.parent) for t in cert.native_trees],
      'native_edge_counts':[len(t.parent)-1 for t in cert.native_trees],'sample_count':len(cert.native_trees[0].sample_ids),
      'excluded_unrelated_nodes':cert.provenance['excluded_unrelated_nodes_by_tree'],
      'older_R_proxy_count':len(cert.bank.partitions[0].older_reference_ids),'per_tree_query_weighted_distributions':per_tree,
      'physical_spans_bp':spans,'physical_span_weighted_query_group_size':sizes,'physical_span_weighted_singleton_Q_fraction':singleton,
      'cross_query_exchange_possible_fraction':1-singleton,'boundary_turnover':turnover,'right_hold_span_weighted_boundary_rho':rho_weighted,
      'maximum_retained_height':height,'maximum_retained_depth':depth,
      'forest_float32_state_copy_budget_bytes':2*(height+depth+2)*n*hidden_dim*4,
      'budget_scope':'source estimate excludes other activations/allocator/donors/optimizer; no memory measurement',
      'limited_mechanism_opportunity':singleton>=.95 or (bool(rho_weighted) and set(rho_weighted)=={'1.0'}),
      'one_tree_turnover':'unmeasured if no boundaries; no predictive benefit claim'}


def run_qualification_phase_v1(admission_ref,*,phase,reader=None,output=None,native_decoder=None,observation_decoder=None,numerical_probe=None,now=None):
    admission,config,reader=admit_runner_v1(admission_ref,phase=phase,reader=reader,now=now)
    if (native_decoder or observation_decoder or numerical_probe) and not config['fixture_mode']:raise ValueError('Injection is handmade-only')
    output=output or BoundedEvidenceOutputV1(admission['output_directory'],root=reader.root,limit_bytes=config['output_cap_bytes'])
    if str(output.path.relative_to(reader.root))!=admission['output_directory']:raise ValueError('Output differs from reservation')
    tmp=output.path/'tmp';tmp.mkdir()
    result={'schema':'census_native_qualification_evidence_v1','phase':phase,'status':'running','job_id':admission['job_id'],
      'admission_ref':reader.record(admission_ref),'refs':admission['refs'],'source_sha256':admission['source_sha256'],
      'observed_artifact_sha256':admission['observed_artifact_sha256'],'fixture_mode':config['fixture_mode'],
      'biological_targets_opened':False,'fit_performed':False,'GPU_used':False,'geometry_adapted':False,
      'cells':[],'performed_cell_coverage':[],'failed_cells':[],'expected_cells':32,
      'source_replay_semantics':'value-exact original x48; no new scaler; native float64 geometry'}
    try:
        paired=read_manifests_v1(admission,config,reader)
        frozen=None
        if phase=='numerical':
            frozen=reader.decode_json(admission['inspection_manifest'])
            if (frozen.get('schema')!='census_native_qualification_evidence_v1' or frozen.get('phase')!='inspect' or frozen.get('status')!='inspection_complete_pending_root_freeze'
                or frozen.get('failed_cells') or len(frozen.get('cells',[]))!=32 or frozen.get('source_sha256')!=admission['source_sha256']
                or frozen.get('observed_artifact_sha256')!=admission['observed_artifact_sha256']
                or any(frozen['refs'].get(k)!=admission['refs'].get(k) for k in ('runner_config','native_manifest','prepared_manifest','feature_lineage'))):raise ValueError('Fresh frozen complete inspection lineage required')
            expected={f"{c['parent_id']}:{c['tile_id']}":c['certificate_sha256'] for c in frozen['cells']}
            if expected!=admission['expected_certificates']:raise ValueError('Root per-tile expected digest differs from frozen inspection')
            source_cells={f"{p['parent_id']}:{t['tile_id']}":(p['split'],len(t['original_tree_ids'])) for p,b in paired for t in b['tiles']}
            if any(source_cells.get(f"{c['parent_id']}:{c['tile_id']}")!=(c['split'],c['num_trees']) for c in frozen['cells']):raise ValueError('Frozen structural selection counts/splits differ from original prepared metadata')
            frozen_cells={f"{c['parent_id']}:{c['tile_id']}":c for c in frozen['cells']}
            selection=select_census_numerical_tiles_v1([{k:c[k] for k in ('parent_id','split','tile_id','num_trees')} for c in frozen['cells']])
            priority=[(selection[k]['parent_id'],selection[k]['tile_id']) for k in ('minimum','maximum')]
            result['prespecified_TRAIN_numerical_cells']=selection;result['frozen_inspection_ref']=reader.record(admission['inspection_manifest'])
        else:priority=[]
        inventory=[]
        # Parent loading is cached only while that parent's eight cells run. The
        # numeric TRAIN extrema execute first, then all remaining fixed cells.
        tasks=[(p,b,t) for p,b in paired for t in sorted(b['tiles'],key=lambda t:t['tile_id'])]
        if phase=='numerical':tasks.sort(key=lambda task:(0 if (task[0]['parent_id'],task[2]['tile_id']) in priority else 1,priority.index((task[0]['parent_id'],task[2]['tile_id'])) if (task[0]['parent_id'],task[2]['tile_id']) in priority else 0,task[0]['parent_id'],task[2]['tile_id']))
        current=None;loaded=None
        for parent,bank,tile in tasks:
            identity=f"{parent['parent_id']}:{tile['tile_id']}";cell={'parent_id':parent['parent_id'],'split':parent['split'],'tile_id':tile['tile_id']}
            try:
                if not config['fixture_mode'] and datetime.now(timezone.utc)>=_utc(admission['expires_at_utc']):raise ValueError('Active phase admission expired during execution')
                if current!=parent['parent_id']:
                    loaded=load_parent_v1(parent,bank,admission,config,reader,tmp_directory=tmp,native_decoder=native_decoder);current=parent['parent_id']
                cert,collision,lineage=prepare_tile_v1(parent,bank,tile,loaded,admission,config,reader,observation_decoder=observation_decoder)
                cell.update(num_trees=len(cert.native_trees),certificate_sha256=cert.certificate_sha256,event_collision_counts=collision,decoded_lineage=lineage)
                if phase=='inspect':
                    # No forward and no externally expected digest exists here.
                    summary=structural_cell_v1(cert,hidden_dim=config['model']['hidden_dim'])
                    cell.update(status='inspected_pending_root_freeze',structural=summary)
                    inventory.append({k:cell[k] for k in ('parent_id','split','tile_id','num_trees')})
                else:
                    expected=admission['expected_certificates'][identity]
                    validate_census_biological_certificate_v1(cert,expected)
                    previous=frozen_cells[identity]
                    if previous['decoded_lineage']!=lineage or previous['event_collision_counts']!=collision or previous['structural']!=structural_cell_v1(cert,hidden_dim=config['model']['hidden_dim']):raise ValueError('Rebuilt decoded/structural/collision lineage differs from frozen inspection')
                    cell['native_oracles']=native_oracle_checks_v1(cert,loaded)
                    permutation=admission['group_permutation']
                    if any(type(i) is not int for i in permutation) or sorted(permutation)!=list(range(len(cert.bank.query_ids))):raise ValueError('Frozen global Q permutation differs')
                    probe=numerical_probe or numerical_cell_v1
                    qualification=probe(cert,expected,config,torch.tensor(permutation,dtype=torch.long),admission['initial_state_sha256_by_dtype'])
                    wanted=[[dtype,control] for dtype in config['dtypes'] for control in CONTROLS]
                    if qualification.get('status')!='qualified' or qualification.get('performed_controls')!=wanted or len(qualification.get('checks',[]))!=len(wanted):raise ValueError('Incomplete performed dtype/control coverage')
                    cell.update(status='numerically_qualified',qualification=qualification)
                result['performed_cell_coverage'].append(identity)
            except Exception as error:
                cell.update(status='failed',error_type=type(error).__name__,error=str(error));result['failed_cells'].append(identity)
                # Capture collision evidence even when the unchanged adapter rejects.
                if loaded is not None and current==parent['parent_id']:
                    ts,_,positions,offset,coverage,*_=loaded
                    left=max(tile['tile_id']*250000.,coverage[0]);right=min((tile['tile_id']+1)*250000.,coverage[1])
                    cell['event_collision_counts']=event_collision_counts_v1(ts,left,right,offset,positions[(positions>=left)&(positions<right)])
                if phase=='numerical':result['cells'].append(cell);raise
            result['cells'].append(cell)
        if len(result['cells'])!=32 or len(set(result['performed_cell_coverage']))!=32 or result['failed_cells']:
            result['status']='failed_complete_inventory_preserved'
        elif phase=='inspect':
            result['prespecified_TRAIN_numerical_cells']=select_census_numerical_tiles_v1(inventory)
            result['status']='inspection_complete_pending_root_freeze'
        else:result['status']='numerical_complete_no_fitting_or_predictive_claim'
    except Exception as error:
        result.update(status='failed_partial_preserved',error_type=type(error).__name__,error=str(error))
    result['verified_file_bytes']=reader.evidence;result['created_at_utc']=datetime.now(timezone.utc).isoformat()
    result['output_cap_bytes']=config['output_cap_bytes'];result['output_bytes_before_receipt']=output.written_bytes
    record=output.write('evidence.json',result)
    return result,record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('inspect','numerical'),required=True)
    parser.add_argument('--admission',required=True);parser.add_argument('--expected-admission-sha256',required=True)
    args=parser.parse_args()
    result,record=run_qualification_phase_v1({'path':args.admission,'sha256':args.expected_admission_sha256},phase=args.phase)
    print(json.dumps({'status':result['status'],'evidence':record,'performed_cells':len(result['performed_cell_coverage'])}))
    if result['status'].startswith('failed'):raise SystemExit(1)

if __name__=='__main__':main()
