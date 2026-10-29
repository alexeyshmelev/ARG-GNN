"""Four spawned CPU workers with bounded in-memory byte transport.

Prepared tensor states are serialized as data in BytesIO, never executable
source. Passing bytes avoids Torch file-descriptor resource sharing on NFS.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
import multiprocessing as mp
from multiprocessing.util import Finalize
import os
import time
import torch
from research.joint_model_20261029.model.joint_model_v1 import (
    prepare_observations, prepared_to_state, prepared_from_state)
from .observations import TestObservationSource

SOURCE = None


def close_worker():
    if SOURCE is not None and SOURCE._vcf is not None:
        SOURCE._vcf.close()


def worker_init(parent_id, binding):
    global SOURCE
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import setproctitle
    setproctitle.setproctitle('AI Agent Research Data')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    SOURCE = TestObservationSource(parent_id, binding)
    Finalize(None, close_worker, exitpriority=10)


def prepare_one(site):
    start = time.monotonic()
    packet, metadata = SOURCE.extract(site)
    prepared = prepare_observations(packet)
    stream = BytesIO()
    torch.save(prepared_to_state(prepared), stream)
    payload = stream.getvalue()
    metadata.update(preparation_seconds=time.monotonic() - start,
                    prepared_transport_bytes=len(payload), worker_pid=os.getpid())
    return site['window_index'], payload, metadata


def prepared_stream(parent_id, binding, sites, *, initializer=worker_init):
    """At most eight pending sites; retain only one inference batch in parent RAM."""
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn'),
                             initializer=initializer, initargs=(parent_id, binding)) as pool:
        pending = deque()
        source = iter(sites)
        for _ in range(min(8, len(sites))):
            pending.append(pool.submit(prepare_one, next(source)))
        while pending:
            result = pending.popleft().result()
            next_site = next(source, None)
            if next_site is not None:
                pending.append(pool.submit(prepare_one, next_site))
            index, payload, metadata = result
            batch = prepared_from_state(torch.load(BytesIO(payload), weights_only=True, map_location='cpu'))
            yield index, batch, metadata
