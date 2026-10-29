"""Root-authorized CPU preparation of every registered site; no truncation.

The 16-site size profile precedes full preparation. Cache files contain only
plain Tensor state, loaded with weights_only=True. Failures preserve evidence.
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import io
import json
import multiprocessing
from pathlib import Path
import resource
import time
import numpy as np
import torch
from research.joint_model_20261029.model import joint_model_v1 as model
from .site_dataset import digest,read_ref,require

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
BASE=ROOT/'research/joint_model_20261029'
HERE=BASE/'data'
CACHE=HERE/'cache_v1'
CAP=16*2**30


def scope_bytes():return sum(path.stat().st_size for path in BASE.rglob('*') if path.is_file())


def initialize_worker():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    import setproctitle
    setproctitle.setproctitle('AI Agent Research Data')


def prepare(task):
    row,expected_sha=task
    require(digest(Path(model.__file__).read_bytes())==expected_sha,'Frozen model changed during preparation')
    started=time.perf_counter()
    raw=read_ref(CACHE,row['observations_ref'])
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
        arrays={key:archive[key].copy() for key in archive.files}
    batch=model.prepare_observations(arrays)
    require(batch.window_count==1 and batch.target_count==row['canonical_examples'],'Canonical target cardinality')
    require(torch.equal(batch.focal_sample,torch.tensor(row['sample_indices'],dtype=torch.long)),'Canonical target sample order')
    buffer=io.BytesIO();torch.save(model.prepared_to_state(batch),buffer);value=buffer.getvalue()
    restored=model.prepared_from_state(torch.load(io.BytesIO(value),map_location='cpu',weights_only=True))
    require(restored.target_count==batch.target_count,'Safe state roundtrip target count')
    return value,dict(window_index=row['window_index'],seconds=time.perf_counter()-started,
        serialized_bytes=len(value),statistics=batch.statistics,
        worker_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)


def bounded_results(pool,rows,sha):
    iterator=iter(rows);pending=deque()
    for _ in range(8):
        row=next(iterator,None)
        if row is not None:pending.append((row,pool.submit(prepare,(row,sha))))
    while pending:
        row,future=pending.popleft();yield row,future.result()
        following=next(iterator,None)
        if following is not None:pending.append((following,pool.submit(prepare,(following,sha))))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model-sha256',required=True);args=parser.parse_args()
    require(digest(Path(model.__file__).read_bytes())==args.model_sha256,'Explicit frozen model SHA required')
    require(not (CACHE/'manifest.json').exists(),'Final cache manifest must not already exist')
    raw=(CACHE/'manifest_observations.json').read_bytes();observations=json.loads(raw)
    require(digest(raw)=='43edc132a24c1c57d41e34d82315c4754a2185991c1e443a0b7fe3e4aa95b54b','Frozen complete observation manifest')
    helper=ROOT/'research/redesign_20261028/data/window_data.py'
    require(digest(helper.read_bytes())=='a1788ea0a33335450bfb43c689aa3c598070f07b28c40930be3fb086f42aa309','Frozen transitive observation helper')
    windows=observations['windows'];written=scope_bytes();started=time.perf_counter();measurements={}
    def write(relative,value):
        nonlocal written
        require(written+len(value)<CAP-64*2**20,'Isolated 16 GiB cap; fail whole operation without target truncation')
        with (CACHE/relative).open('xb') as stream:stream.write(value)
        written+=len(value)
        return dict(path=str(relative),sha256=digest(value),bytes=len(value))
    manifest=dict(observations);manifest.update(schema='joint_model_site_cache_v1',complete=False,windows=[])
    manifest['observations_manifest_ref']=dict(path='manifest_observations.json',sha256=digest(raw),bytes=len(raw))
    manifest['model_source_ref']=write(Path('sources')/'joint_model_v1.py',Path(model.__file__).read_bytes())
    for path in (Path(__file__).resolve(),HERE/'site_dataset.py',ROOT/'research/redesign_20261028/data/window_data.py'):
        manifest['source_refs'][path.name]=write(Path('sources')/path.name,path.read_bytes())
    manifest['prepared_state_version']=model.VERSION
    manifest['prepared_authorization']='Root explicitly approved all 1245 CPU prepared caches after model source freeze with four workers and isolated 16 GiB scope.'
    manifest['prepared_started_utc']=datetime.now(timezone.utc).isoformat()
    try:
        indices=set(np.linspace(0,len(windows)-1,16,dtype=int).tolist())
        profile_rows=[row for i,row in enumerate(windows) if i in indices]
        remaining=[row for i,row in enumerate(windows) if i not in indices]
        ready={}
        with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=initialize_worker) as pool:
            profile_started=time.perf_counter()
            for row,(value,measured) in bounded_results(pool,profile_rows,args.model_sha256):
                ref=write(Path('prepared')/f"site_{row['window_index']:06d}.pt",value)
                ready[row['window_index']]=dict(row,prepared_ref=ref,preparation=measured);measurements[row['window_index']]=measured
            profile=dict(schema='prepared_site_profile_v1',model_sha256=args.model_sha256,
                selection='16 evenly spaced registered windows over all three TRAIN parents, including endpoints',
                site_count=16,wall_seconds=time.perf_counter()-profile_started,measurements=list(measurements.values()),
                mean_projected_full_bytes=sum(x['serialized_bytes'] for x in measurements.values())/16*1245,
                max_sample_projected_full_bytes=max(x['serialized_bytes'] for x in measurements.values())*1245,
                projection_is_not_worst_case_bound=True)
            write('profile_prepared.json',(json.dumps(profile,indent=2)+'\n').encode())
            print(json.dumps(dict(event='prepared_profile_complete',mean_projected_full_bytes=profile['mean_projected_full_bytes'],
                max_sample_projected_full_bytes=profile['max_sample_projected_full_bytes'],wall_seconds=profile['wall_seconds'])),flush=True)
            require(written+profile['max_sample_projected_full_bytes']<CAP-64*2**20,'Profile exceeds isolated cap; no subset export')
            for row,(value,measured) in bounded_results(pool,remaining,args.model_sha256):
                ref=write(Path('prepared')/f"site_{row['window_index']:06d}.pt",value)
                ready[row['window_index']]=dict(row,prepared_ref=ref,preparation=measured);measurements[row['window_index']]=measured
                if len(ready)%100==0:
                    written=scope_bytes()
                    print(json.dumps(dict(completed_sites=len(ready),total=1245,scope_bytes=written,wall_seconds=time.perf_counter()-started)),flush=True)
        require(digest(Path(model.__file__).read_bytes())==args.model_sha256,'Frozen model remains unchanged')
        manifest['windows']=[ready[row['window_index']] for row in windows]
        require(len(manifest['windows'])==1245 and sum(row['canonical_examples'] for row in manifest['windows'])==50000,'Full support completeness')
        manifest.update(complete=True,prepared_wall_seconds=time.perf_counter()-started,
            prepared_bytes=sum(x['serialized_bytes'] for x in measurements.values()),
            prepared_max_worker_peak_RSS_bytes=max(x['worker_peak_RSS_bytes'] for x in measurements.values()),
            scope_bytes_before_manifest=scope_bytes())
        ref=write('manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
        receipt=dict(manifest_ref=ref,scope_bytes=scope_bytes(),prepared_bytes=manifest['prepared_bytes'],
            wall_seconds=manifest['prepared_wall_seconds'],site_count=1245,target_count=50000)
        write('prepared_complete.json',(json.dumps(receipt,indent=2)+'\n').encode())
        print(json.dumps(dict(status='complete',**receipt),indent=2),flush=True)
    except BaseException as error:
        write('failure_prepared.json',(json.dumps(dict(error=repr(error),completed=len(measurements),
            scope_bytes=scope_bytes(),partial_outputs_preserved=True),indent=2)+'\n').encode())
        raise


if __name__=='__main__':main()
