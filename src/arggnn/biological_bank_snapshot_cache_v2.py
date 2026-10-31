"""H020 complete-working-set verified snapshot execution cache.

Only two original bank-loader aliases are scoped. Exact verified bytes and
read-only decoded CPU objects are retained; original caller order is unchanged.
Default is an inactive source report. Actual use needs new root qualification.
"""
from __future__ import annotations
import sys
if __name__=='__main__' and ('--train' in sys.argv or '--cache-profile-only' in sys.argv):
    device=next((x.split('=',1)[1] for x in sys.argv if x.startswith('--device=')),sys.argv[sys.argv.index('--device')+1] if '--device' in sys.argv else 'cuda')
    if device=='cuda':
        from .native_argformer_exec import validate_lease,apply_and_validate_title
        validate_lease();apply_and_validate_title()

import argparse
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import fields,is_dataclass
from datetime import datetime,timezone
import hashlib,io,json,os,time,zipfile
from pathlib import Path
import numpy as np
import torch
from . import biological_candidate as frozen
from . import biological_mechanism_controls_v1 as controls

ROOT=frozen.ROOT
BASE_LOAD_BANK=frozen.load_bank


SOURCE_PATH="src/arggnn/biological_bank_snapshot_cache_v2.py"
CONFIG_PATH="configs/biological_bank_snapshot_cache_v2.json"
CONFIG_SHA256="8a3222afab298105e297e64820f4242d33d32a24e3dc691de17966052db704cf"

def _utc(value):
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    if result.tzinfo is None:raise ValueError('Timezone-aware cache capability dates required')
    return result.astimezone(timezone.utc)


def _identity(path):
    p=frozen.path(str(path));stat=p.stat()
    if p.is_symlink() or not p.is_file():raise ValueError('Regular project-local immutable file required')
    return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)



def verified_bytes_v2(ref,limit):
    """One read and one SHA over the bytes later consumed, never hash/reopen."""
    if type(limit) is not int or limit<0:raise ValueError('Explicit bounded snapshot bytes required')
    p=frozen.path(ref['path'])
    if p.is_symlink() or not p.is_file():raise ValueError('Regular project-local snapshot file required')
    with p.open('rb') as stream:data=stream.read(limit+1)
    if len(data)>limit:raise ValueError('Verified snapshot byte capacity exceeded')
    if hashlib.sha256(data).hexdigest()!=ref['sha256']:raise ValueError('Missing/changed bound snapshot bytes: '+str(p))
    return data


def verified_json_v2(ref):
    return json.loads(verified_bytes_v2(ref,64_000_000))


def archive_storage_bytes_v2(snapshot):
    """Original torch ZIP storage inventory, before allocating decoded tensors."""
    with zipfile.ZipFile(io.BytesIO(snapshot)) as archive:
        entries=archive.infolist()
        # torch.save's storage records are <archive>/data/<storage-key>.
        storage=[x for x in entries if len(x.filename.split('/'))>=3 and x.filename.split('/')[-2]=='data' and x.filename.split('/')[-1].isdigit()]
        if not any(x.filename.endswith('/data.pkl') for x in entries) or not storage:
            raise ValueError('Original torch ZIP tensor archive required')
        return sum(x.file_size for x in storage)


def load_archive_snapshot_v2(snapshot,decode_budget_bytes,stats):
    raw_upper=archive_storage_bytes_v2(snapshot)
    # A BytesIO copy is charged conservatively even when Python shares bytes.
    initial_upper=len(snapshot)+raw_upper
    if initial_upper>decode_budget_bytes:raise ValueError('Cold archive tensor/stream peak capacity exceeded')
    saved=torch.load(io.BytesIO(snapshot),map_location='cpu',weights_only=False)
    actual=resident_bytes_v1(saved)
    if actual>raw_upper:raise ValueError('Decoded tensor storages exceed original archive inventory')
    upper=len(snapshot)+raw_upper+transfer_upper_bytes_v1(saved['fixed_prepared'])
    if upper>decode_budget_bytes:raise ValueError('Cold archive/CPU transform peak capacity exceeded')
    stats['CPU_decode_peak_upper_bytes']=max(stats['CPU_decode_peak_upper_bytes'],upper)
    stats['archive_storage_bytes_loaded']+=actual
    return saved

def _tensors(value):
    if isinstance(value,torch.Tensor):yield value
    elif is_dataclass(value):
        for field in fields(value):yield from _tensors(getattr(value,field.name))
    elif isinstance(value,dict):
        for child in value.values():yield from _tensors(child)
    elif isinstance(value,(tuple,list)):
        for child in value:yield from _tensors(child)


def resident_bytes_v1(value):
    stores={}
    for tensor in _tensors(value):
        storage=tensor.untyped_storage();stores[(str(tensor.device),storage._cdata)]=storage.nbytes()
    return sum(stores.values())


def transfer_upper_bytes_v1(value):
    # Every original .to tensor occurrence can create its own device storage.
    # Expanded zero-stride tensors can become dense, so include logical bytes.
    return sum(max(t.untyped_storage().nbytes(),t.numel()*t.element_size()) for t in _tensors(value))


def _guard(value):
    """Read-only consumer guard, no tensor copies, reductions or RNG actions.

    Reviewed models never mutate banks. Version checks catch ordinary PyTorch
    mutation. Arbitrary .data/external-storage writes are outside this interface.
    """
    if isinstance(value,torch.Tensor):
        if value.requires_grad:raise ValueError('Cached prepared bank must be fixed')
        return ('tensor',str(value.device),str(value.dtype),tuple(value.shape),tuple(value.stride()),value.storage_offset(),value.untyped_storage()._cdata,value.untyped_storage().nbytes(),value._version)
    if is_dataclass(value):return (type(value).__module__,type(value).__name__,tuple((f.name,_guard(getattr(value,f.name))) for f in fields(value)))
    if isinstance(value,dict):return ('dict',tuple((key,_guard(child)) for key,child in value.items()))
    if isinstance(value,(tuple,list)):return (type(value).__name__,tuple(_guard(child) for child in value))
    if isinstance(value,float):return ('float',value.hex())
    if value is None or isinstance(value,(str,int,bool)):return value
    raise ValueError('Unsupported prepared-bank field type')


def _parent_join(parent):
    values={'parent_id':parent['parent_id'],'split':parent['split']}
    for name in ('dynamics_slots','reference_slots','query_slots','positions'):
        values[name]=torch.from_numpy(np.asarray(parent[name]).copy())
    values['inferred_nodes']=torch.from_numpy(np.asarray(parent['observation']['nodes']).copy())
    return frozen.fingerprint_sha(values)



def decode_bank_snapshot_v2(snapshot,tile,cfg,parent,device,decode_budget_bytes,stats):
    saved = load_archive_snapshot_v2(snapshot,decode_budget_bytes,stats)
    if (saved["schema"] != "biological_candidate_full_age_bank_v1"
            or tile["exact_serialization_roundtrip_checked"] is not True
            or tile["all_original_age_cache_fields_retained"] is not True):
        raise ValueError("Exact full bank schema/roundtrip provenance differs")
    # Source precomputation, eigensystem/partition validation, anchor preparation
    # and full-field roundtrip occurred before sealing this hashed artifact.
    # Training reuses the fixed prepared object; no repeated eigensolve or raw
    # matrix validation is introduced into each neural forward.
    prepared = saved["fixed_prepared"].to("cpu", dtype=torch.float32)
    if (prepared.bank.leaf_x.shape[1:] != (280,47)
            or prepared.bank.query_node_ids.numel() != 184
            or prepared.bank.query_cache.reference_ids.numel() != 96
            or prepared.num_scales != 3 or prepared.layout.anchors.numel() != 32):
        raise ValueError("Frozen whole-tree model bank dimensions changed")
    nodes=parent["observation"]["nodes"]
    if (not torch.equal(prepared.bank.query_node_ids, torch.as_tensor(
            nodes[parent["dynamics_slots"]].copy(),dtype=torch.long))
            or not torch.equal(prepared.bank.query_cache.reference_ids,torch.as_tensor(
            nodes[parent["reference_slots"]].copy(),dtype=torch.long))
            or set(prepared.bank.sample_ids.tolist()) != set(nodes.tolist())
            or prepared.bank.feature_columns != tuple(c for c in range(48) if c != 7)):
        raise ValueError("Prepared actual inferred identities/reference roles/features changed")
    indexes=np.asarray(tile["position_indexes"],dtype=int)
    if not np.array_equal(prepared.bank.positions.numpy(),parent["positions"][indexes]/2000000):
        raise ValueError("Prepared original normalized2Mb readouts changed")
    actual=torch.as_tensor(nodes[parent["query_slots"]].copy(),dtype=torch.long)
    expected_flags=torch.isin(prepared.bank.sample_ids,actual).float()
    if not torch.equal(prepared.bank.leaf_x[:,:,4],expected_flags.expand_as(prepared.bank.leaf_x[:,:,4])):
        raise ValueError("Prepared mixed64 flag became an actual-query flag")
    return prepared.to(device,dtype=torch.float32)

class BiologicalBankSnapshotCacheV2:
    """Complete bounded CPU working set and unchanged bounded GPU LRU."""
    def __init__(self,prepared_manifest_ref,config_ref,observed_manifest_ref,*,cpu_budget_bytes,snapshot_budget_bytes,decode_budget_bytes,gpu_budget_bytes,source_sha256=None,active_guard=None):
        budgets=(cpu_budget_bytes,snapshot_budget_bytes,decode_budget_bytes,gpu_budget_bytes)
        if any(type(x) is not int or x<0 for x in budgets):raise ValueError('Explicit nonnegative byte budgets required')
        self.cpu_budget=cpu_budget_bytes;self.snapshot_budget=snapshot_budget_bytes;self.decode_budget=decode_budget_bytes
        self.gpu_budget=gpu_budget_bytes;self.active_guard=active_guard
        self.refs=[dict(r) for r in (prepared_manifest_ref,config_ref,observed_manifest_ref)]+[{'path':p,'sha256':h} for p,h in (source_sha256 or {}).items()]
        self.ref_identity={};metadata={}
        for ref in self.refs:
            p=frozen.path(ref['path']);before=_identity(p)
            data=verified_bytes_v2(ref,64_000_000)
            if _identity(p)!=before:raise ValueError('Pinned metadata/source identity changed during snapshot')
            self.ref_identity[ref['path']]=before;metadata[(ref['path'],ref['sha256'])]=data
        self.cfg=json.loads(metadata[(config_ref['path'],config_ref['sha256'])])
        manifest=json.loads(metadata[(prepared_manifest_ref['path'],prepared_manifest_ref['sha256'])])
        if manifest.get('status')!='complete_observed_only_Q184_R96_actualQ120' or manifest.get('truth_opened') is not False:raise ValueError('Completed target-free original prepared manifest required')
        self.tiles={}
        for parent in manifest['parents']:
            for tile in parent['tiles']:
                identity=(parent['parent_id'],tile['tile_id'])
                if identity in self.tiles:raise ValueError('Duplicate parent/tile manifest identity')
                self.tiles[identity]=(frozen.fingerprint_sha(tile),dict(tile['bank']))
        self.cpu=OrderedDict();self.gpu=OrderedDict();self.joins={};self.snapshots={}
        self.cpu_used=self.snapshot_used=self.gpu_used=0;self.closed=False
        self.stats={'requests':0,'CPU_hits':0,'GPU_hits':0,'cold_snapshot_decodes':0,'CPU_evictions':0,'GPU_evictions':0,'CPU_oversized_bypass':0,'GPU_oversized_bypass':0,'bank_sha_verified_bytes':0,'snapshot_reads':0,'snapshot_peak_bytes':0,'CPU_peak_resident_bytes':0,'CPU_decode_peak_upper_bytes':0,'archive_storage_bytes_loaded':0,'GPU_peak_resident_bytes':0,'request_elapsed_seconds':0.,'complete_working_set_reached':False}

    def _bindings(self):
        if self.closed:raise ValueError('Cache already closed')
        if self.active_guard:self.active_guard()
        for ref in self.refs:
            p=frozen.path(ref['path'])
            if _identity(p)!=self.ref_identity[ref['path']]:
                # Re-hash for diagnosis, but any immutable file-identity change fails.
                frozen.checked(ref);raise ValueError('Pinned source/manifest file identity changed: '+ref['path'])

    def _evict_gpu(self,key):
        entry=self.gpu[key];self._unchanged(entry);self.gpu.pop(key);self.gpu_used-=entry['bytes'];self.stats['GPU_evictions']+=1

    def _unchanged(self,entry):
        if _guard(entry['bank'])!=entry['guard']:raise ValueError('Reviewed fixed bank was mutated in memory')

    def __call__(self,tile,cfg,parent,device,dtype=torch.float32):
        begin=time.perf_counter();self._bindings()
        if dtype!=torch.float32:raise ValueError('Original loader always uses float32 feature/operator dtype')
        if cfg!=self.cfg:raise ValueError('Changed scientific configuration cannot reuse cache')
        identity=(parent['parent_id'],tile['tile_id']);expected=self.tiles.get(identity)
        if expected is None or frozen.fingerprint_sha(tile)!=expected[0] or tile['bank']!=expected[1]:raise ValueError('Parent/tile/file role differs from pinned original manifest')
        join=_parent_join(parent)
        if identity in self.joins and self.joins[identity]!=join:raise ValueError('Parent role, actual identity or physical grid changed')
        self.joins[identity]=join
        self.stats['requests']+=1
        if not isinstance(device,(str,torch.device)):raise ValueError('Explicit original CPU/CUDA device required')
        target=torch.device(device)
        if target.type=='cpu':
            if target.index not in (None,0):raise ValueError('Ambiguous CPU index unsupported')
            target=torch.device('cpu')
        elif target.type=='cuda':
            if target.index not in (None,0):raise ValueError('Original single exposed logical CUDA0 required')
            target=torch.device('cuda:0')
        else:raise ValueError('Original CPU/CUDA devices only')
        if identity in self.cpu:
            entry=self.cpu[identity];self._unchanged(entry);self.cpu.move_to_end(identity);self.stats['CPU_hits']+=1
            cpu_bank=entry['bank']
        else:
            # No path check or reopen follows this verified bytes acquisition.
            snapshot=verified_bytes_v2(expected[1],self.snapshot_budget-self.snapshot_used)
            self.stats['snapshot_reads']+=1;self.stats['bank_sha_verified_bytes']+=len(snapshot)
            cpu_bank=decode_bank_snapshot_v2(snapshot,tile,cfg,parent,torch.device('cpu'),self.decode_budget,self.stats)
            self.stats['cold_snapshot_decodes']+=1
            size=resident_bytes_v1(cpu_bank)
            if self.cpu_used+size>self.cpu_budget:raise ValueError('Complete CPU working-set capacity exceeded; no eviction/bypass')
            entry={'bank':cpu_bank,'bytes':size,'guard':_guard(cpu_bank),'snapshot_ref':dict(expected[1])}
            self.cpu[identity]=entry;self.cpu_used+=size
            self.snapshots[identity]=snapshot;self.snapshot_used+=len(snapshot)
            self.stats['snapshot_peak_bytes']=max(self.stats['snapshot_peak_bytes'],self.snapshot_used)
            self.stats['CPU_peak_resident_bytes']=max(self.stats['CPU_peak_resident_bytes'],self.cpu_used)
            self.stats['complete_working_set_reached']=set(self.cpu)==set(self.tiles) and set(self.snapshots)==set(self.tiles)
        if target.type=='cpu':result=cpu_bank
        else:
            devicekey=identity+(str(target),str(dtype))
            if devicekey in self.gpu:
                entry=self.gpu[devicekey];self._unchanged(entry);self.gpu.move_to_end(devicekey);self.stats['GPU_hits']+=1;result=entry['bank']
            else:
                upper=transfer_upper_bytes_v1(cpu_bank)
                eligible=identity in self.cpu and upper<=self.gpu_budget
                if eligible:
                    while self.gpu and self.gpu_used+upper>self.gpu_budget:self._evict_gpu(next(iter(self.gpu)))
                result=cpu_bank.to(target,dtype=torch.float32)
                size=resident_bytes_v1(result)
                if eligible and size<=upper and self.gpu_used+size<=self.gpu_budget:
                    self.gpu[devicekey]={'bank':result,'bytes':size,'guard':_guard(result)};self.gpu_used+=size
                    self.stats['GPU_peak_resident_bytes']=max(self.stats['GPU_peak_resident_bytes'],self.gpu_used)
                else:self.stats['GPU_oversized_bypass']+=1
        if self.active_guard:self.active_guard()
        self.stats['request_elapsed_seconds']+=time.perf_counter()-begin
        return result

    def report(self):
        return {'schema':'biological_bank_snapshot_cache_execution_v2','CPU_budget_bytes':self.cpu_budget,'CPU_snapshot_budget_bytes':self.snapshot_budget,'CPU_decode_peak_budget_bytes':self.decode_budget,'GPU_LRU_budget_bytes':self.gpu_budget,'CPU_resident_bytes':self.cpu_used,'CPU_snapshot_bytes':self.snapshot_used,'GPU_LRU_resident_bytes':self.gpu_used,'stats':dict(self.stats),'complete_working_set_bound':len(self.tiles),'retained_CPU_entries':len(self.cpu),'retained_snapshot_entries':len(self.snapshots),'all_working_set_retained':set(self.cpu)==set(self.tiles) and set(self.snapshots)==set(self.tiles),'bank_sha_on_every_request':False,'bank_sha_once_per_retained_snapshot':True,'hot_bank_path_IO':False,'CPU_eviction_or_bypass':False,'math_or_science_changed':False,'scope':'exact original verified archive bytes plus unchanged decoded fixed_prepared; caller and GPU LRU unchanged; no speed claim','budget_exclusions':'non-tensor decoded Python/NumPy metadata, temporary deserialization buffers, metadata snapshots, autograd, model/optimizer/runtime, allocator-reserved memory and caller-retained references require independently qualified root headroom'}

    def close(self):
        try:
            for entry in list(self.cpu.values())+list(self.gpu.values()):self._unchanged(entry)
        finally:
            self.cpu.clear();self.gpu.clear();self.snapshots.clear();self.cpu_used=self.snapshot_used=self.gpu_used=0;self.closed=True


@contextmanager
def installed_cache_v2(cache):
    """Patch only TRAIN alias and VAL's global alias, restore even on exception."""
    if frozen.load_bank is not BASE_LOAD_BANK or controls.load_bank is not BASE_LOAD_BANK:raise ValueError('Bank loader already replaced; do not stack wrappers')
    frozen.load_bank=cache;controls.load_bank=cache
    try:yield cache
    finally:
        frozen.load_bank=BASE_LOAD_BANK;controls.load_bank=BASE_LOAD_BANK;cache.close()


def admit_cache_v2(record,*,phase):
    admission=verified_json_v2(record);now=datetime.now(timezone.utc)
    if (admission.get('schema')!='biological_bank_snapshot_cache_admission_v2' or admission.get('status')!='active' or admission.get('phase')!=phase
        or phase not in ('profile','fit') or not admission.get('job_id') or not _utc(admission['created_at_utc'])<=now<_utc(admission['expires_at_utc'])
        or (_utc(admission['expires_at_utc'])-_utc(admission['created_at_utc'])).total_seconds()>86400):raise ValueError('Fresh distinct root cache phase capability required')
    refs=admission['refs'];cfg=verified_json_v2(refs['cache_config'])
    if cfg!=verified_json_v2({'path':CONFIG_PATH,'sha256':CONFIG_SHA256}):raise ValueError('Shipped cache configuration changed')
    if admission['source_sha256']!={**cfg['source_sha256'],'src/arggnn/biological_bank_snapshot_cache_v2.py':admission['source_sha256'].get('src/arggnn/biological_bank_snapshot_cache_v2.py')}:raise ValueError('Complete exact cache source pins required')
    for path,sha in admission['source_sha256'].items():frozen.checked({'path':path,'sha256':sha})
    review=verified_json_v2(refs['review'])
    if (not str(review.get('status','')).startswith('qualified') or review.get('actual_admission') is not False or review.get('fixture_mode') is True
        or review.get('artifact_sha256',{}).get('src/arggnn/biological_bank_snapshot_cache_v2.py')!=admission['source_sha256']['src/arggnn/biological_bank_snapshot_cache_v2.py']
        or review.get('artifact_sha256',{}).get('configs/biological_bank_snapshot_cache_v2.json')!=refs['cache_config']['sha256']):raise ValueError('Fresh independent cache source review required')
    if set(refs)!=({'cache_config','review','original_config','observed_manifest','prepared_manifest','memory_reservation'}|({'profile_gate'} if phase=='fit' else set())):raise ValueError('Exact root cache reference roles required')
    scientific=verified_json_v2(refs['original_config'])
    if refs['original_config']!=cfg['original_config'] or scientific['source_sha256']!=cfg['original_source_sha256']:raise ValueError('Original mechanism source/config changed')
    memory=verified_json_v2(refs['memory_reservation'])
    if (memory.get('schema')!='biological_bank_snapshot_cache_memory_reservation_v2' or memory.get('status')!='active' or memory.get('job_id')!=admission['job_id']
        or memory.get('CPU_cache_bytes')!=cfg['CPU_budget_bytes'] or memory.get('CPU_snapshot_bytes')!=cfg['CPU_snapshot_budget_bytes'] or memory.get('CPU_decode_peak_bytes')!=cfg['CPU_decode_peak_budget_bytes'] or memory.get('GPU_cache_bytes')!=cfg['GPU_budget_bytes']
        or not _utc(memory['created_at_utc'])<=now<_utc(memory['expires_at_utc']) or memory.get('original_model_optimizer_runtime_headroom_preserved') is not True):raise ValueError('Explicit active root CPU/GPU memory reservation required')
    if phase=='fit':
        gate=verified_json_v2(refs['profile_gate'])
        if (gate.get('schema')!='biological_bank_snapshot_cache_profile_gate_v2' or gate.get('status')!='qualified_execution_only'
            or gate.get('cache_config_ref')!=refs['cache_config'] or gate.get('cache_source_sha256')!=admission['source_sha256']['src/arggnn/biological_bank_snapshot_cache_v2.py']
            or gate.get('scientific_refs')!={k:refs[k] for k in ('original_config','observed_manifest','prepared_manifest')}
            or gate.get('all_fields_logits_VJP_optimizer_RNG_parity') is not True or gate.get('measured_profile_acceptable') is not True or gate.get('all32_verified_snapshots_and_decoded_CPU_banks_retained') is not True or gate.get('actual_complete_working_set_RSS_and_cold_decode_headroom_qualified') is not True or not gate.get('profile_evidence_ref') or not gate.get('working_set_memory_evidence_ref')):raise ValueError('Bound measured one-tile execution profile/parity gate required')
        frozen.checked(gate['profile_evidence_ref']);frozen.checked(gate['working_set_memory_evidence_ref'])
    def active():
        clock=datetime.now(timezone.utc)
        if clock>=_utc(admission['expires_at_utc']) or clock>=_utc(memory['expires_at_utc']):raise ValueError('Cache phase or memory reservation expired')
    cache=BiologicalBankSnapshotCacheV2(refs['prepared_manifest'],refs['original_config'],refs['observed_manifest'],cpu_budget_bytes=cfg['CPU_budget_bytes'],snapshot_budget_bytes=cfg['CPU_snapshot_budget_bytes'],decode_budget_bytes=cfg['CPU_decode_peak_budget_bytes'],gpu_budget_bytes=cfg['GPU_budget_bytes'],source_sha256=admission['source_sha256'],active_guard=active)
    expected={(parent,tile) for parent in frozen.PARENTS for tile in range(8)}
    if set(cache.tiles)!=expected:raise ValueError('Actual cache capability must bind all32 original cells')
    return cache,admission,cfg


def run_original_mechanisms_with_cache_v2(record,original_argv):
    cache,admission,cfg=admit_cache_v2(record,phase='fit')
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--expected-config-sha256');parser.add_argument('--expected-observed-manifest-sha256');parser.add_argument('--expected-prepared-manifest-sha256')
    parser.add_argument('--config',default='configs/biological_mechanism_controls_v1.json');parser.add_argument('--observed-manifest',default='data/biological_paths_execution_pilot_v1/relate_inferred_v1/manifest.json');parser.add_argument('--prepared-manifest',default='data/biological_candidate_pilot_v1/manifest.json');parser.add_argument('--output');parser.add_argument('--seed',type=int,default=101);parser.add_argument('--device',default='cuda')
    args,_=parser.parse_known_args(original_argv)
    for role,name,sha_name in [('original_config','config','expected_config_sha256'),('observed_manifest','observed_manifest','expected_observed_manifest_sha256'),('prepared_manifest','prepared_manifest','expected_prepared_manifest_sha256')]:
        expected=admission['refs'][role]
        if frozen.path(getattr(args,name))!=frozen.path(expected['path']) or getattr(args,sha_name)!=expected['sha256']:raise ValueError('Original runner arguments must preserve exact manifest/config bindings')
    if '--train' not in original_argv or args.seed!=admission['seed'] or args.device!=admission['device'] or args.output!=admission['output_directory']:raise ValueError('Same root seed/device/fresh output and unchanged training arguments required')
    old_argv=sys.argv
    try:
        sys.argv=[str(ROOT/'scripts/train_biological_mechanism_controls_v1.py'),*original_argv]
        with installed_cache_v2(cache):controls.main()
    finally:
        sys.argv=old_argv
        output=frozen.path(admission['output_directory'])
        if output.is_dir():
            report=cache.report();report.update(cache_admission_ref=record,original_science_argv=list(original_argv),original_source_sha256=cfg['original_source_sha256'])
            target=output/'bank_snapshot_cache_execution_v2.json'
            with target.open('x') as stream:json.dump(report,stream,sort_keys=True,indent=2);stream.write('\n')


def _rng_v1(device):
    state=np.random.get_state();record={'python':repr(__import__('random').getstate()),'numpy':(state[0],hashlib.sha256(state[1].tobytes()).hexdigest(),state[2],state[3],float(state[4]).hex()),'torch_CPU':frozen.fingerprint_sha(torch.get_rng_state())}
    if torch.device(device).type=='cuda':record['torch_CUDA']=frozen.fingerprint_sha(torch.cuda.get_rng_state())
    return record


def _sync_v1(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize()


def profile_one_tile_v2(record):
    """Inactive root-only target-free one-tile parity and measured loading profile."""
    cache,admission,ccfg=admit_cache_v2(record,phase='profile')
    refs=admission['refs'];cfg,base,helpers=controls.load_protocol(frozen.path(refs['original_config']['path']),refs['original_config']['sha256'],controls.BASE_CONFIG_SHA256)
    parents=frozen.observed_contract(base,frozen.path(refs['observed_manifest']['path']),refs['observed_manifest']['sha256'],helpers)
    prepared=verified_json_v2(refs['prepared_manifest'])
    records=controls.validate_prepared_records(prepared,base,refs['observed_manifest']['sha256'],parents)
    candidates=[(len(tile['original_tree_ids']),parent['parent_id'],tile['tile_id']) for parent in parents if parent['split']=='train' for tile in records[parent['parent_id']]['tiles']]
    maximum=max(x[0] for x in candidates);selection=min(x for x in candidates if x[0]==maximum)
    expected={'parent_id':selection[1],'tile_id':selection[2]}
    if admission.get('profile_cell')!=expected:raise ValueError('Profile uses TRAIN maximum original-tree-count tile, tie parent then tile, before outcomes')
    parent=next(p for p in parents if p['parent_id']==expected['parent_id']);tile=next(t for t in records[parent['parent_id']]['tiles'] if t['tile_id']==expected['tile_id'])
    output=frozen.path(admission['output_directory'])
    if output.exists():raise ValueError('Exclusive new root profile output required')
    output.mkdir(parents=True)
    device,runtime,lease=frozen.configure_fit(admission['device'],cfg,output)
    result={'schema':'biological_bank_snapshot_cache_one_tile_profile_v2','status':'running','cache_admission_ref':record,'cache_config_ref':refs['cache_config'],'cache_source_sha256':admission['source_sha256']['src/arggnn/biological_bank_snapshot_cache_v2.py'],'scientific_refs':{k:refs[k] for k in ('original_config','observed_manifest','prepared_manifest')},'device':str(device),'selection':expected,'original_tree_count':selection[0],'target_arrays_opened':False,'scientific_fit_performed':False,'runtime':runtime,'control_checks':[]}
    try:
        # Include first snapshot SHA/decode and all original cache binding guards.
        times={'original':[],'cache':[]}
        for label,loader in [('original',BASE_LOAD_BANK),('cache',cache)]:
            for repeat in range(4):
                cache._bindings();frozen.check_fit_lease(lease);rng=_rng_v1(device);_sync_v1(device);start=time.perf_counter()
                bank=loader(tile,cfg,parent,device);_sync_v1(device);times[label].append(time.perf_counter()-start)
                if _rng_v1(device)!=rng:raise ValueError('Bank loader changed RNG state')
                del bank
        baseline=BASE_LOAD_BANK(tile,cfg,parent,device);cached=cache(tile,cfg,parent,device)
        if frozen.fingerprint_sha(baseline)!=frozen.fingerprint_sha(cached):raise ValueError('Original/cached full field bytes/dtypes/shapes/strides differ')
        for name in controls.CONTROL_NAMES:
            cache._bindings();frozen.check_fit_lease(lease)
            torch.manual_seed(41);first=controls.model_factory(name,cfg).to(device);second=__import__('copy').deepcopy(first)
            opt1=torch.optim.AdamW(frozen.optimizer_groups(first,cfg),lr=cfg['training']['learning_rate']);opt2=torch.optim.AdamW(frozen.optimizer_groups(second,cfg),lr=cfg['training']['learning_rate'])
            before=frozen.fingerprint_sha(baseline);rng=_rng_v1(device);values=[];gradients=[]
            for model,bank,opt in [(first,baseline,opt1),(second,cached,opt2)]:
                logits=model(bank);coeff=torch.linspace(-.7,1.1,logits.numel(),dtype=logits.dtype,device=device).reshape_as(logits)
                objective=(logits*coeff).mean();objective.backward()
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Missing/nonfinite active model gradient')
                values.append(logits.detach().cpu());gradients.append([p.grad.detach().cpu().clone() for p in model.parameters()])
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            if not torch.equal(*values) or any(not torch.equal(a,b) for a,b in zip(*gradients)):raise ValueError('Cached logits/VJP not exactly equal')
            if frozen.fingerprint_sha(first.state_dict())!=frozen.fingerprint_sha(second.state_dict()) or frozen.fingerprint_sha(opt1.state_dict())!=frozen.fingerprint_sha(opt2.state_dict()):raise ValueError('Optimizer unit step changed')
            if frozen.fingerprint_sha(baseline)!=before or frozen.fingerprint_sha(cached)!=before or _rng_v1(device)!=rng:raise ValueError('Forward/backward changed fixed bank or RNG')
            result['control_checks'].append({'name':name,'fields_logits_VJP_optimizer_RNG_exact':True,'objective':'target-free fixed mean-logit coefficients; disposable unit step only'})
            del first,second,opt1,opt2,values,gradients
        result.update(status='one_tile_parity_and_loading_profile_complete_no_speed_claim',complete_working_set_memory_qualified=False,loading_seconds=times,all_fields_logits_VJP_optimizer_RNG_parity=True,cache=cache.report(),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None)
    except Exception as error:
        result.update(status='failed_partial_preserved',error_type=type(error).__name__,error=str(error));raise
    finally:
        cache.close();target=output/'one_tile_profile.json'
        with target.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2);stream.write('\n')
    return result



def source_report_v2():
    cfg=verified_json_v2({'path':CONFIG_PATH,'sha256':CONFIG_SHA256})
    return {'schema':'biological_bank_snapshot_cache_source_report_v2','status':cfg['status'],'source_ref':{'path':SOURCE_PATH,'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'config_ref':{'path':CONFIG_PATH,'sha256':CONFIG_SHA256},'source_registration':cfg['source_registration'],'prospective_caps':{k:cfg[k] for k in ('CPU_budget_bytes','CPU_snapshot_budget_bytes','CPU_decode_peak_budget_bytes','GPU_budget_bytes')},'actual_inputs_opened':False,'models_instantiated':False,'actual_memory_throughput_qualified':False,'actual_profile_fit_GPU_budget_or_reservation':False}

def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--cache-profile-only',action='store_true');parser.add_argument('--cache-admission',required=True);parser.add_argument('--expected-cache-admission-sha256',required=True)
    if not any(x in sys.argv for x in ('--train','--cache-profile-only')):
        print(json.dumps(source_report_v2(),sort_keys=True,indent=2));return
    args,original=parser.parse_known_args();record={'path':args.cache_admission,'sha256':args.expected_cache_admission_sha256}
    if args.cache_profile_only:profile_one_tile_v2(record)
    else:run_original_mechanisms_with_cache_v2(record,original)

if __name__=='__main__':main()
