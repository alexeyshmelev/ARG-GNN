"""Root-authorized full canonical observation cache and separate target labels.

Ordinary CPU process workers; fixed source/plan, no target subsampling and no
model training. Only the parent writes and enforces the isolated output cap.
"""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import resource
import time
import numpy as np
from .observations import ObservationSource
from .targets import TargetTable

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/joint_model_20261029/data'
_SOURCES={};_TARGETS={}


def digest(raw):return hashlib.sha256(raw).hexdigest()


def extract(task):
    site,binding=task;parent=site['parent_id']
    if parent not in _SOURCES:
        _SOURCES[parent]=ObservationSource(parent,binding)
        _TARGETS[parent]=TargetTable(parent,binding)
    started=time.perf_counter()
    packet,metadata=_SOURCES[parent].extract(site)
    labels=_TARGETS[parent].select(site['example_indices'])
    observations=io.BytesIO();np.savez_compressed(observations,**packet)
    targets=io.BytesIO();np.save(targets,labels,allow_pickle=False)
    metadata.update(window_index=site['window_index'],parent_ordinal=site['parent_ordinal'],
        example_indices=site['example_indices'],sample_indices=site['sample_indices'],
        worker_seconds=time.perf_counter()-started,
        worker_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    return observations.getvalue(),targets.getvalue(),metadata


def main():
    protocol_path=HERE/'DATA_PROTOCOL_v1.json';plan_path=HERE/'site_plan_v1.json';assessment_path=HERE/'SOURCE_ASSESSMENT.json'
    plan_raw=plan_path.read_bytes();protocol_raw=protocol_path.read_bytes()
    assert digest(protocol_raw)=='82d42d8e597ac385bccadfba720350ff689b50be8e37c8ebcec819a88717f387'
    assert digest(plan_raw)=='bfebc589dc0fb5a2f5a658f488d28dda9fd7b3baa2eb6fb03da76ed7f14c3a11'
    plan=json.loads(plan_raw);protocol=json.loads(protocol_raw);assessment=json.loads(assessment_path.read_text())
    authorization=dict(schema='full_site_export_root_authorization_v1',recorded_utc=datetime.now(timezone.utc).isoformat(),
        authority='Root explicit 29Oct message approves all1245 observation windows with4CPU workers and isolated16GiB cap; noTEST',
        protocol_sha256=digest(protocol_raw),plan_sha256=digest(plan_raw),
        original_protocol_retained=True,full_export_authorized=True,cap_bytes=16*2**30)
    output=HERE/'cache_v1';output.mkdir(exist_ok=False)
    for name in ('observations','labels','prepared','sources'):(output/name).mkdir()
    written=0
    def write(relative,raw):
        nonlocal written
        if written+len(raw)>16*2**30-64*2**20:raise ValueError('Isolated16GiB cap; abort whole export without target truncation')
        path=output/relative
        with path.open('xb') as stream:stream.write(raw)
        written+=len(raw)
        return dict(path=str(relative),sha256=digest(raw),bytes=len(raw))
    sources={}
    for path in [Path(__file__).resolve(),HERE/'observations.py',HERE/'targets.py',protocol_path,plan_path,assessment_path]:
        sources[path.name]=write(Path('sources')/path.name,path.read_bytes())
    write('authorization.json',(json.dumps(authorization,indent=2)+'\n').encode())
    manifest=dict(schema='joint_model_site_observations_v1',complete=False,
        protocol_sha256=digest(protocol_raw),plan_sha256=digest(plan_raw),source_refs=sources,
        labels_separate=True,TEST_allowed=False,workers=4,windows=[],source_bindings=assessment['parents'],
        support=protocol['support'],model_config=None,output_cap_bytes=16*2**30)
    started=time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
            tasks=[(site,assessment['parents'][site['parent_id']]) for site in plan['sites']]
            for index,(observations,labels,metadata) in enumerate(pool.map(extract,tasks),start=1):
                stem=f"site_{metadata['window_index']:06d}"
                metadata['observations_ref']=write(Path('observations')/(stem+'.npz'),observations)
                metadata['labels_ref']=write(Path('labels')/(stem+'.npy'),labels)
                manifest['windows'].append(metadata)
                if index%200==0:print(json.dumps(dict(completed_sites=index,total=1245,bytes_written=written,wall_seconds=time.perf_counter()-started)),flush=True)
        assert len(manifest['windows'])==1245 and sum(row['canonical_examples'] for row in manifest['windows'])==50000
        for parent,info in assessment['parents'].items():
            covered=[example for row in manifest['windows'] if row['parent_id']==parent for example in row['example_indices']]
            assert sorted(covered)==list(range(info['canonical_examples']))
        manifest.update(complete=True,wall_seconds=time.perf_counter()-started,bytes_before_manifest=written,
            site_count=1245,target_count=50000,
            max_worker_peak_RSS_bytes=max(row['worker_peak_RSS_bytes'] for row in manifest['windows']))
        ref=write('manifest_observations.json',(json.dumps(manifest,indent=2)+'\n').encode())
        write('observations_complete.json',(json.dumps(dict(manifest_ref=ref,bytes_written_before_receipt=written),indent=2)+'\n').encode())
        print(json.dumps(dict(status='complete',manifest=ref,bytes_written=written,wall_seconds=manifest['wall_seconds'],
                             site_count=1245,target_count=50000,max_worker_peak_RSS_bytes=manifest['max_worker_peak_RSS_bytes']),indent=2))
    except BaseException as error:
        write('failure_observations.json',(json.dumps(dict(error=repr(error),completed=len(manifest['windows']),bytes_written=written,partial_outputs_preserved=True),indent=2)+'\n').encode())
        raise


if __name__=='__main__':main()
