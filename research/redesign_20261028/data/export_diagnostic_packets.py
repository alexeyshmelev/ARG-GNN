"""Export only preregistered CPU diagnostic observations and separate labels.

Workers use fork, ordinary imported Python source, and data-only array results.
Only the parent writes the immutable output directory and enforces the byte cap.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import resource
import sys
import time
import numpy as np
from research.redesign_20261028.data.window_data import (
    TrainWindowDataset,uniform_target_draws,uniform_coordinate_context,small_json)

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/redesign_20261028'
OUT=HERE/'diagnostic_packets'
PROTOCOL=HERE/'root/pilot_protocol.json'
JOB=ROOT/'.runtime/benchmarking/workflow_20261022_full_vram_v2/jobs/AP__all_train__treeloom_fit.json'
_WORKER_DATASET=None


def digest(raw):return hashlib.sha256(raw).hexdigest()

def worker_extract(task):
    global _WORKER_DATASET
    started=time.perf_counter()
    # Keep one parent handle per worker. Grouped dispatch avoids repeated opens.
    if _WORKER_DATASET is None or _WORKER_DATASET.parent_id!=task['parent_id']:
        if _WORKER_DATASET is not None and _WORKER_DATASET._vcf is not None:
            _WORKER_DATASET._vcf.close()
        _WORKER_DATASET=TrainWindowDataset(task['parent_id'],task['refs'],halo=2)
    ds=_WORKER_DATASET
    trees,vcf=ds._handles();before=vcf.bytes_read
    example=task['example_index'];row=ds.records[example]
    seed_components=[task['context_seed'],task['split_code'],task['draw_index']]
    rng=np.random.default_rng(np.random.SeedSequence(seed_components))
    spec=uniform_coordinate_context(rng,row['position_local_bp'],task['radius_bp'],
                                    task['context_draws'],trees.sequence_length)
    packet=ds.batch([example],context_specs={example:spec})
    labels=packet.pop('labels')
    if labels.shape!=(1,) or int(packet['focal_example_index'][0])!=example:
        raise ValueError('Exactly one registered target per packet')
    if int(packet['tree_core_mask'].sum())!=1 or len(packet['context_coordinate_local_bp'])!=5:
        raise ValueError('Registered focal/core and context shape')
    if not np.isfinite(packet['context_proposal_density_per_bp']).all():
        raise ValueError('Nonfinite proposal metadata')
    observations=io.BytesIO();np.savez(observations,**packet)
    target=io.BytesIO();np.save(target,np.asarray(int(labels[0]),dtype=np.int64),allow_pickle=False)
    target_metadata=small_json(task['refs']['training_targets_ref'])
    evidence=dict(ds.source_evidence,training_target_arrays_ref=target_metadata['arrays_ref'],
                  inferred_sequence_length_bp=trees.sequence_length,observed_panel_haplotypes=len(ds.panel))
    return dict(split=task['split'],draw_index=task['draw_index'],parent_id=task['parent_id'],
        example_index=example,label=int(labels[0]),observed_cohort=row['cohort'],
        packet_bytes=observations.getvalue(),label_bytes=target.getvalue(),source_binding=evidence,
        context_seed_components=seed_components,
        context_coordinates_local_bp=packet['context_coordinate_local_bp'].tolist(),
        context_density_per_bp=packet['context_proposal_density_per_bp'].tolist(),
        context_weights=packet['context_quadrature_weight'].tolist(),
        context_domain_local_bp=packet['window_context_domain_local_bp'][0].tolist(),
        focal_position_local_bp=float(row['position_local_bp']),
        focal_position_original_bp=float(row['position_original_bp']),
        coordinate_offset_bp=float(ds.offset),
        node_count=len(packet['parent']),tree_count=len(packet['tree_core_mask']),
        observed_site_count=len(packet['site_native_id']),
        elapsed_seconds=time.perf_counter()-started,VCF_logical_bytes_read=vcf.bytes_read-before,
        worker_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main():
    started=time.perf_counter()
    protocol_raw=PROTOCOL.read_bytes();protocol=json.loads(protocol_raw)
    target=protocol['target_sampling'];context=protocol['context']
    expected_fit=['ancestralpaths_european_train_000','ancestralpaths_european_train_001']
    if (protocol['fitting_parent_ids']!=expected_fit or
        protocol['development_parent_id']!='ancestralpaths_european_train_002' or
        target['fitting_draws']!=128 or target['development_draws']!=64 or
        target['fitting_draw_seed']!=20261028 or target['development_draw_seed']!=20261029 or
        target['replacement'] is not True or context['context_seed']!=20261030 or
        context['radius_bp']!=50000 or context['draws_including_forced_focal']!=5 or
        protocol['test_inputs_allowed'] is not False):
        raise ValueError('Exact registered diagnostic protocol required')
    cap=protocol['resource_scope']['max_packet_output_bytes']
    if cap!=2**30:raise ValueError('Registered 1 GiB packet cap required')
    job_raw=JOB.read_bytes();refs=json.loads(job_raw)['call']['spec']['scientific_spec']['train_inputs']
    sizes={'ancestralpaths_european_train_000':16667,'ancestralpaths_european_train_001':16667,
           'ancestralpaths_european_train_002':16666}
    tasks=[]
    supports={
        'fit':({p:sizes[p] for p in expected_fit},128,20261028,0),
        'development':({'ancestralpaths_european_train_002':16666},64,20261029,1)}
    for split,(support,count,seed,code) in supports.items():
        draws=uniform_target_draws(support,count,np.random.default_rng(seed))
        for i,(parent,example) in enumerate(draws):
            tasks.append(dict(split=split,split_code=code,draw_index=i,parent_id=parent,
                example_index=example,target_probability=1/sum(support.values()),refs=refs[parent],
                context_seed=context['context_seed'],radius_bp=context['radius_bp'],context_draws=5))
    # Freeze all target identities before any worker opens labels or scientific arrays.
    plan=dict(schema='diagnostic_target_draw_plan_v1',protocol_sha256=digest(protocol_raw),
              split_codes={'fit':0,'development':1},
              context_seed_derivation='numpy.SeedSequence([20261030, split_code, draw_index])',tasks=tasks)
    OUT.mkdir(exist_ok=False)
    written=0
    def write_new(relative,raw):
        nonlocal written
        if written+len(raw)>cap-2**20:
            raise ValueError('Diagnostic export byte cap; no packet/target truncation')
        path=OUT/relative;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(raw)
        written+=len(raw)
        return digest(raw)
    def write_json(relative,value):
        return write_new(relative,(json.dumps(value,indent=2,sort_keys=True)+'\n').encode())
    source_refs={}
    for source in [Path(__file__).resolve(),HERE/'data/window_data.py',PROTOCOL]:
        raw=source.read_bytes();name='sources/'+source.name
        checksum=write_new(name,raw)
        source_refs[source.name]=dict(original_path=str(source.relative_to(ROOT)),
                                       archived_path=name,sha256=checksum)
    write_json('draw_plan.json',plan)
    manifest=dict(schema='bounded_diagnostic_packets_v1',status='complete',fit=[],development=[],
        protocol_sha256=digest(protocol_raw),
        source_refs=source_refs,fit_job_ref=dict(path=str(JOB.relative_to(ROOT)),sha256=digest(job_raw)),
        source_bindings={},class_order=['Ana_nonBAA','CHG_nonBAA','WHG','EHG','Ana_viaBAA','CHG_viaBAA'],
        target_support_sizes={'fit':33334,'development':16666},
        target_probability={'fit':1/33334,'development':1/16666},
        context_seed_derivation=plan['context_seed_derivation'],split_codes=plan['split_codes'],
        observation_NPZ_contains_labels=False,label_format='scalar int64 NPY',
        paths_relative_to='research/redesign_20261028/diagnostic_packets',
        no_training=True,no_TEST_files=True,original_registered_split_unchanged=True,
        workers=2,numpy_version=np.__version__,python_version=sys.version)
    try:
        # Scheduling only affects compute locality, never preregistered row/context RNGs.
        ordered=sorted(tasks,key=lambda t:(t['parent_id'],t['draw_index']))
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('fork')) as pool:
            pending={pool.submit(worker_extract,task):(task['split'],task['draw_index']) for task in ordered}
            for completed,future in enumerate(as_completed(pending),start=1):
                identity=pending.pop(future);result=future.result()
                if identity!=(result['split'],result['draw_index']):raise ValueError('Worker draw identity')
                split=result.pop('split');i=result['draw_index']
                packet_path=f'{split}/observations/draw_{i:06d}.npz'
                label_path=f'{split}/labels/draw_{i:06d}.npy'
                packet_raw=result.pop('packet_bytes');label_raw=result.pop('label_bytes')
                if written+len(packet_raw)+len(label_raw)>cap-2**20:
                    raise ValueError('Pair would exceed output cap; neither half is written')
                result['packet_sha256']=write_new(packet_path,packet_raw)
                result['label_sha256']=write_new(label_path,label_raw)
                result['packet_path']=packet_path;result['label_path']=label_path
                result['packet_bytes']=len(packet_raw);result['label_bytes']=len(label_raw)
                binding=result.pop('source_binding');parent=result['parent_id']
                if parent in manifest['source_bindings'] and manifest['source_bindings'][parent]!=binding:
                    raise ValueError('Worker source binding mismatch')
                manifest['source_bindings'][parent]=binding
                manifest[split].append(result)
                if completed%16==0:
                    print(json.dumps(dict(completed=completed,total=192,output_bytes=written,
                                          elapsed_seconds=time.perf_counter()-started)),flush=True)
        for split,(support,count,seed,code) in supports.items():
            rows=sorted(manifest[split],key=lambda r:r['draw_index']);manifest[split]=rows
            if [r['draw_index'] for r in rows]!=list(range(count)):raise ValueError('Incomplete draw coverage')
            expected=uniform_target_draws(support,count,np.random.default_rng(seed))
            if [(r['parent_id'],r['example_index']) for r in rows]!=expected:
                raise ValueError('Target selection changed')
            manifest[split+'_summary']=dict(draws=len(rows),
                unique_target_ids=len({(r['parent_id'],r['example_index']) for r in rows}),
                class_counts=np.bincount([r['label'] for r in rows],minlength=6).tolist(),
                cohort_counts={c:sum(r['observed_cohort']==c for r in rows) for c in sorted({r['observed_cohort'] for r in rows})},
                nodes=sum(r['node_count'] for r in rows),trees=sum(r['tree_count'] for r in rows),
                observed_sites_with_occurrence_multiplicity=sum(r['observed_site_count'] for r in rows))
        manifest['elapsed_seconds']=time.perf_counter()-started
        manifest['output_bytes_before_manifest']=written
        manifest['maximum_worker_peak_RSS_KiB']=max(r['worker_peak_RSS_KiB'] for split in supports for r in manifest[split])
        manifest['measurement_limit']='Shared filesystem cache uncontrolled; worker RSS includes accumulated mapped pages'
        manifest_sha=write_json('manifest.json',manifest)
        write_json('completion.json',dict(manifest_sha256=manifest_sha,output_bytes_before_completion=written,
                                       actual_draws=192,complete=True))
        print(json.dumps(dict(status='complete',manifest_sha256=manifest_sha,output_bytes=written,
              fit=manifest['fit_summary'],development=manifest['development_summary'],
              elapsed_seconds=manifest['elapsed_seconds'],maximum_worker_peak_RSS_KiB=manifest['maximum_worker_peak_RSS_KiB'])),flush=True)
    except BaseException as error:
        write_json('failure.json',dict(status='failed',error_type=type(error).__name__,message=str(error),
                                      output_bytes_before_failure=written,partial_files_preserved=True))
        raise

if __name__=='__main__':main()
