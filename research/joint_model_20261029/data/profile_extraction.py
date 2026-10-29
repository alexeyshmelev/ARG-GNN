"""Registered 16-site CPU extraction profile, not bulk data generation."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import resource
import time
import numpy as np
from .observations import ObservationSource

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
HERE=ROOT/'research/joint_model_20261029/data'
_SOURCE=None


def extract(task):
    global _SOURCE
    site,binding=task
    if _SOURCE is None:_SOURCE=ObservationSource(site['parent_id'],binding)
    start=time.perf_counter();packet,metadata=_SOURCE.extract(site);extracted=time.perf_counter()
    buffer=io.BytesIO();np.savez_compressed(buffer,**packet);raw=buffer.getvalue()
    metadata.update(window_index=site['window_index'],example_indices=site['example_indices'],
                    sample_indices=site['sample_indices'],extraction_seconds=extracted-start,
                    serialization_seconds=time.perf_counter()-extracted,
                    uncompressed_array_bytes=sum(value.nbytes for value in packet.values()),
                    worker_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    return raw,metadata


def main():
    plan_path=HERE/'site_plan_v1.json';protocol_path=HERE/'DATA_PROTOCOL_v1.json'
    plan=json.loads(plan_path.read_text());protocol=json.loads(protocol_path.read_text())
    binding=json.loads((HERE/'SOURCE_ASSESSMENT.json').read_text())['parents']
    sites=[site for site in plan['sites'] if site['bounded_profile']]
    assert len(sites)==16 and all(site['parent_ordinal']==0 for site in sites)
    output=HERE/'profile_observations_v1';output.mkdir(exist_ok=False)
    report=dict(schema='joint_model_16site_source_profile_v1',source_plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                data_protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                observations_only=True,labels_read=False,TEST_payloads_opened=False,workers=4,sites=[],
                source_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('observations.py','profile_extraction.py')})
    start=time.perf_counter();written=0
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        for raw,metadata in pool.map(extract,[(site,binding[site['parent_id']]) for site in sites]):
            path=output/f"site_{metadata['window_index']:06d}.npz"
            assert written+len(raw)<256*2**20, 'Bounded diagnostic exceeds256MiB; no targets truncated'
            with path.open('xb') as stream:stream.write(raw)
            written+=len(raw)
            metadata.update(packet_path=str(path.relative_to(HERE)),packet_sha256=hashlib.sha256(raw).hexdigest(),packet_bytes=len(raw))
            report['sites'].append(metadata)
    report.update(wall_seconds=time.perf_counter()-start,observation_bytes=written,
                  canonical_rows=sum(len(site['example_indices']) for site in sites),
                  full1245_mean_size_projection_bytes=written/16*1245,
                  full1245_max_sample_size_projection_bytes=max(site['packet_bytes'] for site in report['sites'])*1245,
                  projection_limit='Observed16-site extrapolation only; prepared graph tensors not yet included and not a worst-case bound')
    (HERE/'EXTRACTION_PROFILE.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key!='sites'},indent=2))


if __name__=='__main__':main()
