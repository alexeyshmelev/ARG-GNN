"""Read-only recovery from the preserved regular-file Kineto export failure."""
import gc
import hashlib
import json
from pathlib import Path
import resource
import stat
from research.efficiency_20261027.execution.native_activity_ledger_v2 import consume

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
SOURCE = ROOT/'research/efficiency_20261027/root/runs/boundary_activity_gpu5_v2/data/baseline_CPU512_native.fifo'
OUTPUT = ROOT/'research/efficiency_20261027/execution/recovered_boundary_baseline_v2'


def main():
    resource.setrlimit(resource.RLIMIT_AS, (4*2**30, 4*2**30))
    resource.setrlimit(resource.RLIMIT_CPU, (90, 90))
    before = SOURCE.stat()
    assert stat.S_ISREG(before.st_mode) and before.st_size == 111977354
    # Validate the entire JSON, including the final root, before stream recovery.
    with SOURCE.open() as handle:
        native = json.load(handle)
    assert isinstance(native['traceEvents'], list) and native['traceEvents']
    native_event_count = len(native['traceEvents'])
    del native
    gc.collect()
    OUTPUT.mkdir(exist_ok=False)
    with SOURCE.open() as handle:
        summary = consume(handle, OUTPUT/'baseline_CPU512_physical_ledger.json',
                          OUTPUT/'baseline_CPU512_activity_summary.json')
    after = SOURCE.stat()
    assert before == after
    digest = hashlib.sha256()
    with SOURCE.open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            digest.update(block)
    receipt = dict(schema='TreeLoom_baseline_native_recovery_v2', status='passed',
        source_ref=dict(path=str(SOURCE.relative_to(ROOT)), sha256=digest.hexdigest()),
        source_bytes=before.st_size, original_source_untouched=True,
        complete_native_JSON_validated=True, native_event_count=native_event_count,
        learned_calls_executed=0, summary=summary,
        export_failure='Kineto replaced the FIFO with a 111977354-byte regular file; '
                       'the guardian terminated the worker for exceeding its output cap.',
        correction='Do not reuse FIFO export. Recover this complete baseline and profile only the missing fused call.')
    with (OUTPUT/'RECOVERY.json').open('x') as handle:
        json.dump(receipt, handle, indent=2)
    print(json.dumps(dict(status='passed', native_event_count=native_event_count,
                         union=summary['physical_CUDA_union'], ledger_bytes=summary['complete_ledger_bytes'])))


if __name__ == '__main__':
    main()
