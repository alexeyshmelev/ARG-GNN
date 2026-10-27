"""Read metadata only, never follow links or cross the project filesystem."""
import concurrent.futures
import json
import os
from pathlib import Path
import stat
import time

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
OUT = ROOT / 'research/storage_cleanup_20261027'
assert Path.cwd() == ROOT and ROOT.resolve() == ROOT
DEVICE = ROOT.stat().st_dev


def inventory(relative):
    start = ROOT / relative
    stack = [start]
    totals = {'path': relative, 'allocated_bytes': 0, 'logical_bytes': 0,
              'files': 0, 'directories': 0, 'symlinks': 0, 'errors': [], 'extensions': {}}
    rows = []
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError as error:
            totals['errors'].append([str(current.relative_to(ROOT)), str(error)])
            continue
        for entry in entries:
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
                if info.st_dev != DEVICE:
                    totals['errors'].append([str(path.relative_to(ROOT)), 'different filesystem'])
                    continue
                if stat.S_ISLNK(info.st_mode):
                    totals['symlinks'] += 1
                    continue
                if stat.S_ISDIR(info.st_mode):
                    totals['directories'] += 1
                    stack.append(path)
                    continue
                if not stat.S_ISREG(info.st_mode):
                    continue
                totals['files'] += 1
                totals['allocated_bytes'] += info.st_blocks * 512
                totals['logical_bytes'] += info.st_size
                suffix = ''.join(path.suffixes[-2:]) or '(none)'
                ext = totals['extensions'].setdefault(suffix, [0, 0])
                ext[0] += 1
                ext[1] += info.st_blocks * 512
                if info.st_size >= 1024 * 1024:
                    rows.append({'path': str(path.relative_to(ROOT)), 'size': info.st_size,
                                 'allocated': info.st_blocks * 512, 'dev': info.st_dev,
                                 'ino': info.st_ino, 'nlink': info.st_nlink,
                                 'mtime_ns': info.st_mtime_ns})
            except FileNotFoundError:
                totals['errors'].append([str(path.relative_to(ROOT)), 'vanished while scanning'])
    return totals, rows


if __name__ == '__main__':
    started = time.time()
    targets = []
    for parent in ('data', 'results', 'checkpoints', 'experiments/results',
                   'experiments/paper_argmix_1611_rebuild_20261024_v1', '.runtime/cache'):
        for child in (ROOT / parent).iterdir():
            if not child.is_symlink() and child.is_dir():
                targets.append(str(child.relative_to(ROOT)))
    summaries = []
    with (OUT / 'large_files.jsonl').open('x') as output:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            jobs = {pool.submit(inventory, target): target for target in targets}
            for future in concurrent.futures.as_completed(jobs):
                summary, rows = future.result()
                summaries.append(summary)
                for row in rows:
                    output.write(json.dumps(row, sort_keys=True) + '\n')
                output.flush()
                if summary['allocated_bytes'] > 1_000_000_000:
                    print(json.dumps({k: summary[k] for k in ('path','allocated_bytes','files','errors')}), flush=True)
    summaries.sort(key=lambda row: row['allocated_bytes'], reverse=True)
    report = {'started_unix': started, 'finished_unix': time.time(),
              'note': 'Per-path totals may count hard links more than once. No symlink targets followed. Directory blocks omitted.',
              'targets': summaries}
    (OUT / 'inventory.json').write_text(json.dumps(report, indent=2) + '\n')
    print('FINISHED', len(summaries), 'targets', round(time.time()-started, 1), 'seconds', flush=True)
