"""Compare complete observed American preparation without opening ancestry truth."""
from pathlib import Path
from dataclasses import fields, is_dataclass
import argparse
import hashlib
import json
import os
import resource
import time
import numpy as np
import torch
import setproctitle

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
OUT = ROOT / 'research/preprocessing_optimization_20261026/integration'

def digest_tensor(value):
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else value
    array = np.ascontiguousarray(array)
    raw = memoryview(array).cast('B')
    sha = hashlib.sha256()
    for start in range(0, len(raw), 4 * 2**20):
        sha.update(raw[start:start + 4 * 2**20])
    return dict(dtype=str(array.dtype), shape=list(array.shape), sha256=sha.hexdigest())

def signature(value):
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        return digest_tensor(value)
    if is_dataclass(value):
        return {f.name: signature(getattr(value, f.name)) for f in fields(value)
                if 'stamp' not in f.name}
    if isinstance(value, dict):
        return {str(k): signature(v) for k, v in value.items() if 'stamp' not in str(k)}
    if isinstance(value, (list, tuple)):
        return [signature(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(type(value))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['reference', 'candidate'], required=True)
    args = parser.parse_args()
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from arggnn.american_c3_gnn_dataset_v1 import load_observation
    from research.benchmark_rebuild_20261024.american.treeloom_speed import runtime_v2 as runtime
    plan = json.loads((ROOT / 'configs/benchmarks/paper_campaign_resume_20261026_v1.json').read_text())
    task = next(t for t in plan['tasks'] if t['id'] == 'paper_american_treeloom_fit')
    arguments = task['call']['spec']['arguments']
    registry = runtime.FreshParentRegistry.from_ref(arguments['parent_registry_ref'])
    rt = runtime.FreshTreeLoom(registry)
    if args.variant == 'candidate':
        from research.preprocessing_optimization_20261026.integration.hooks import install_tree
        install_tree()
    parent = 'argformer_american_train_000'
    item = arguments['train_inputs'][parent]
    marks = []
    start = time.monotonic()
    last = [None]
    def check(stage):
        current = int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        if current > 192 * 2**30:
            raise MemoryError('Bounded qualification exceeded 192 GiB RSS')
        if time.monotonic() - start > 7200:
            raise TimeoutError('Two-hour complete-observation qualification limit')
        if stage != last[0]:
            row = dict(stage=stage, elapsed=time.monotonic() - start, rss_bytes=current)
            marks.append(row)
            print(json.dumps(row), flush=True)
            last[0] = stage
    state = load_observation(item['observation_ref'], item['focal_registry_ref'], dict(
        native_preparation_ref=item['native_preparation_ref'], class_order=list(runtime.CLASSES),
        expected_haplotypes=1600, reference_counts=[200,200,200],
        input_reference_class_mask=None, input_reference_sample_indices=None))
    loaded = time.monotonic()
    bundle, metadata = rt.prepare_observation(state, split='train', resource_check=check,
        **{k:item[k] for k in runtime.INPUT_REFS}, max_nodes=400000000,
        max_holds=100000, max_episode_payload_bytes=8*2**30)
    finished = time.monotonic()
    record = dict(variant=args.variant, parent=parent, load_seconds=loaded-start,
        prepare_seconds=finished-loaded, total_seconds=finished-start,
        observed_inputs={k:item[k] for k in runtime.INPUT_REFS},
        bundle=signature(bundle), metadata=signature(metadata), marks=marks,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        truth_opened=False, model_executed=False, full_parent=True)
    target = OUT / ('full_' + args.variant + '.json')
    with target.open('x') as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
    print(json.dumps(dict(completed=str(target), prepare_seconds=finished-loaded)), flush=True)

if __name__ == '__main__':
    main()
