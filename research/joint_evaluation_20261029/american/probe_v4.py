"""TRAIN-only 128-window lossless compression and exact tensor roundtrip probe."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import time
import numpy as np
import torch
from research.joint_model_20261029.model.joint_model_v1 import prepare_observations, prepared_to_state
from .common import BASE, ROOT, require, now
from .observations import ObservationSource
from .smoke import exact
from .compression_v2 import compress, decompress
from .state_codec_v2 import pack, unpack

PROBE = ROOT / 'experiments/cache/joint_american_20261029/probe_v4'
RESULT = ROOT / 'experiments/results/joint_american_20261029'


WORKER_INITIALIZED = False

def sample_parent(parent, item, count):
    global WORKER_INITIALIZED
    import setproctitle
    setproctitle.setproctitle('AI Agent Research Data')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    if not WORKER_INITIALIZED:
        torch.set_num_interop_threads(1)
        WORKER_INITIALIZED = True
    require(item['split'] == 'train', 'TRAIN public observations only')
    source = ObservationSource(parent, item)
    plan = source.plan()
    sampled = np.linspace(0, len(plan) - 1, count, dtype=np.int64)
    result = []
    out = PROBE / parent
    out.mkdir()
    for index in sampled:
        begin = time.monotonic()
        state = prepared_to_state(prepare_observations(source.extract(plan[index])))
        prepare_seconds = time.monotonic() - begin
        stream = BytesIO()
        torch.save(state, stream)
        raw = stream.getvalue()
        row = dict(parent=parent, window=int(index), raw_bytes=len(raw), prepare_seconds=prepare_seconds, codecs={})
        for codec, level in [('zstd', 1), ('zstd', 3), ('zstd', 9), ('lz4', 0)]:
            name = f'{codec}_{level}'
            begin = time.monotonic()
            encoded = compress(raw, codec, level)
            compression_seconds = time.monotonic() - begin
            before = time.monotonic()
            decoded = decompress(encoded, len(raw), codec)
            decode_seconds = time.monotonic() - before
            require(decoded == raw, 'Bitwise serialized roundtrip')
            begin = time.monotonic()
            restored = torch.load(BytesIO(decoded), map_location='cpu', weights_only=True)
            load_seconds = time.monotonic() - begin
            exact(state, restored)
            if codec == 'zstd' and level == 3:
                with (out / f'{index:05d}.zstd').open('xb') as destination:
                    destination.write(encoded)
            row['codecs'][name] = dict(bytes=len(encoded), compression_seconds=compression_seconds,
                  decode_seconds=decode_seconds, torch_load_seconds=load_seconds, exact_roundtrip=True)
        packed = pack(state)
        for codec, level in [('zstd', 3), ('zstd', 9), ('lz4', 0)]:
            before = time.monotonic()
            encoded = compress(packed, codec, level)
            compression_seconds = time.monotonic() - before
            before = time.monotonic()
            restored = unpack(decompress(encoded, len(packed), codec))
            decode_seconds = time.monotonic() - before
            exact(state, restored)
            row['codecs'][f'packed_{codec}_{level}'] = dict(bytes=len(encoded), compression_seconds=compression_seconds, decode_seconds=decode_seconds, torch_load_seconds=0., exact_roundtrip=True)
        row['worker_peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        result.append(row)
    source.check_stats()
    with (out / 'result.json').open('x') as destination:
        json.dump(result, destination, indent=2, sort_keys=True)
    return result


def main():
    import psutil
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol = json.loads((BASE / 'protocol_v1.json').read_text())
    parents = [(p, item) for p, item in protocol['parents'].items() if item['split'] == 'train']
    PROBE.mkdir(parents=True, exist_ok=False)
    RESULT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    results, peak = [], 0
    with ProcessPoolExecutor(max_workers=16, mp_context=mp.get_context('spawn')) as pool:
        pending = [pool.submit(sample_parent, p, item, 7 if i < 8 else 6) for i, (p, item) in enumerate(parents)]
        for future in as_completed(pending):
            results.extend(future.result())
            processes = [psutil.Process()] + psutil.Process().children(recursive=True)
            rss = sum(p.memory_info().rss for p in processes if p.is_running())
            peak = max(peak, rss)
            require(rss <= 64 * 2**30, '64GiB complete process-family RSS')
            disk = sum(p.stat().st_size for p in PROBE.rglob('*') if p.is_file())
            require(disk <= 2 * 2**30, '2GiB initial probe storage')
            print(json.dumps(dict(windows=len(results), elapsed_seconds=time.monotonic() - start, rss_bytes=rss, disk_bytes=disk)), flush=True)
    require(len(results) == 128, 'Exactly128 TRAIN windows across all20 parents')
    summary = {}
    for codec in results[0]['codecs']:
        summary[codec] = {key: float(np.mean([r['codecs'][codec][key] for r in results])) for key in
                         ('bytes', 'compression_seconds', 'decode_seconds', 'torch_load_seconds')}
        summary[codec]['projected_158476_windows_bytes'] = summary[codec]['bytes'] * 158476
    payload = dict(passed=True, utc=now(), source='128 evenly spaced TRAIN windows across20 parents',
        heldout_opened=False, workers=16, elapsed_seconds=time.monotonic() - start, owned_peak_observed_rss_bytes=peak,
        disk_bytes=disk, mean_raw_bytes=float(np.mean([r['raw_bytes'] for r in results])),
        mean_prepare_seconds=float(np.mean([r['prepare_seconds'] for r in results])), codecs=summary, windows=results)
    with (RESULT / 'compression_probe_v4.json').open('x') as output:
        json.dump(payload, output, indent=2, sort_keys=True)
    print(json.dumps({k: v for k, v in payload.items() if k != 'windows'}), flush=True)


if __name__ == '__main__':
    main()
