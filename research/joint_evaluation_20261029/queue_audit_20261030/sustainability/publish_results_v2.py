"""Reuse score publication and atomically expose owner-aware reports."""
from collections import Counter
import json
import os
from pathlib import Path
import tempfile
from arggnn.benchmarking.paper_campaign import runtime as r
from research.joint_evaluation_20261029.root.reporting_20261030_v1 import publish_results_v1 as base
from research.joint_evaluation_20261029.root.reporting_20261030_v1 import publish_results_recovery_v1 as recovery
from research.joint_evaluation_20261029.queue_audit_20261030.sustainability import effective_state_v2 as effective


def atomic_text(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix='.' + path.name + '.', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(output_directory=None):
    captured = {}
    def current_snapshot(*args, **kwargs):
        value = effective.snapshot(*args, **kwargs)
        captured['value'] = value
        return value
    folder = Path(output_directory) if output_directory else r.path('research/joint_evaluation_20261029/root')
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.producer-report-', dir=folder) as temporary:
        staging = Path(temporary)
        old_snapshot, old_fingerprint = base.snapshot, base.fingerprint
        base.snapshot, base.fingerprint = current_snapshot, effective.fingerprint
        try:
            recovery.main(output_directory=staging)
        finally:
            base.snapshot, base.fingerprint = old_snapshot, old_fingerprint
        value = captured['value']
        status = json.loads((staging / 'CURRENT_STATUS.json').read_text())
        owned = value['producer_owned_tasks']
        history = value['original_main_producer_rows']
        status['native_parallel_producer'] = dict(active_owner_registry=value['producer_active_registry'],
            current_counts=dict(Counter(row['status'] for row in owned.values())), current_tasks=owned,
            monitor_health=value['producer_monitor_health'])
        status['original_main_producer_rows'] = history
        status['historical_main_claim_blocked_rows'] = {name: row for name, row in history.items() if row['status'] == 'blocked'}
        status['queue_observability'] = effective.observability(value)
        status['notes'].append('Parallel native preparation overlays only the registered forty-eight producer IDs; original main claim-blocked rows remain separate historical evidence. Pending blocker and stable monitor causes are publication signals.')
        text = (staging / 'README.md').read_text()
        if owned:
            marker = '<!-- joint-model-20261029:end -->'
            producer_counts = Counter(row['status'] for row in owned.values())
            line = ('- **Native preparation:** the registered parallel producer owns forty-eight stages across six remaining parents. Current stages: ' +
                    ', '.join(str(producer_counts.get(key, 0)) + ' ' + key for key in ('succeeded', 'running', 'pending', 'failed', 'blocked')) +
                    '. Original main claim-blocked rows remain historical and are excluded from current producer counts.\n')
            text = text.replace(marker, line + marker, 1)
        r.save(folder / 'CURRENT_STATUS.json', status)
        readme = folder / 'README.md' if output_directory else r.ROOT / 'README.md'
        atomic_text(readme, text)


if __name__ == '__main__':
    main()
