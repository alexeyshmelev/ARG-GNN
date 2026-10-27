"""Delete reviewed regular files under one fixed root; no recursive removal.

Use only audited immutable obsolete directories, absent from live dependency
sets. Live snapshots refresh before deletion and every 256 files or 10 seconds;
these are sampled checks, not atomic exclusion of concurrent file writers.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import time

ROOT = Path('/workspace-SR003.nfs2/ARG-GNN')
AUDIT = ROOT / 'research/storage_cleanup_20261027'
LIVE_BATCH_FILES = 256
LIVE_MAX_AGE_SECONDS = 10.
PROTECTED_SUFFIXES = {'.json', '.jsonl', '.md', '.txt', '.log', '.csv', '.tsv',
                      '.py', '.sh', '.yaml', '.yml', '.toml', '.tex', '.bib',
                      '.pdf', '.png', '.svg', '.xlsx', '.zip'}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def parts(relative):
    need(type(relative) is str, 'Ordinary string path required')
    path = PurePosixPath(relative)
    need(not path.is_absolute() and path.parts and not any(x in ('..', '.', '') for x in path.parts),
         'Invalid relative path: ' + relative)
    need(str(path) == relative, 'Noncanonical relative path: ' + relative)
    return path.parts


def parent_fd(rootfd, relative):
    components = parts(relative)
    handle = os.dup(rootfd)
    try:
        for component in components[:-1]:
            next_handle = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=handle)
            os.close(handle)
            handle = next_handle
        return handle, components[-1]
    except BaseException:
        os.close(handle)
        raise


def process_binding(proc):
    fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    raw = (proc / 'cmdline').read_bytes()
    return dict(pid=int(proc.name), start_ticks=fields[19],
        argv=[value.decode() for value in raw.rstrip(b'\0').split(b'\0')] if raw else [],
        cmdline_sha256=hashlib.sha256(raw).hexdigest())


def validate_exceptions(rows):
    need(type(rows) is list, 'Explicit process exception list required')
    result = {}
    for row in rows:
        need(type(row) is dict and set(row) == {'pid', 'start_ticks', 'argv', 'cmdline_sha256'},
             'Exact PID/start/command exception fields required')
        need(type(row['pid']) is int and row['pid'] > 1 and row['pid'] not in result
             and type(row['start_ticks']) is str and row['start_ticks'].isdigit()
             and type(row['argv']) is list and row['argv'] and all(type(s) is str for s in row['argv'])
             and Path(row['argv'][0]).name == 'ssh-agent'
             and type(row['cmdline_sha256']) is str and len(row['cmdline_sha256']) == 64,
             'Only independently identified exact ssh-agent exceptions are accepted')
        result[row['pid']] = row
    return result


def live_project_paths(exceptions=()):
    allowed = validate_exceptions(list(exceptions))
    paths, inspected, exempted, inaccessible = set(), 0, [], []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        binding = None
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            inspected += 1
            binding = process_binding(proc)
            cmd = ' '.join(binding['argv'])
            if any(token in cmd for token in ('conda install', 'conda create', 'pip install', 'mamba install', 'mamba create')):
                raise RuntimeError('Package manager is active, defer deletion')
            # Every same-UID process is inspected, independent of cwd/title/argv.
            cwd = os.readlink(proc / 'cwd')
            if cwd.startswith(str(ROOT) + '/'):
                paths.add(cwd)
            for entry in (proc / 'fd').iterdir():
                try:
                    value = os.readlink(entry).removesuffix(' (deleted)')
                    if value.startswith(str(ROOT) + '/'):
                        paths.add(value)
                except FileNotFoundError:
                    pass
            for line in (proc / 'maps').read_text().splitlines():
                fields = line.split(None, 5)
                if len(fields) == 6 and fields[5].startswith(str(ROOT) + '/'):
                    paths.add(fields[5].removesuffix(' (deleted)'))
        except (FileNotFoundError, ProcessLookupError):
            pass
        except PermissionError:
            # Recheck identity at the denial, so a changed process is not exempt.
            try:
                current_binding = process_binding(proc)
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError:
                current_binding = None
            if binding is not None and binding == current_binding and allowed.get(int(proc.name)) == binding:
                exempted.append(binding)
            else:
                inaccessible.append(int(proc.name))
    need(not inaccessible, 'Unknown inaccessible processes prevent live-use verification: ' + str(inaccessible))
    # Compute directory membership once. A cwd or FD naming ROOT itself must
    # not block every target; all project processes commonly hold that directory.
    directories = {value for value in paths if value != str(ROOT) and Path(value).is_dir()}
    return dict(paths=paths, directories=directories, inspected=inspected,
                exact_ssh_agent_exceptions=exempted, completed_monotonic=time.monotonic())


def reject_live(relative, snapshot):
    absolute = str(ROOT / relative)
    need(absolute not in snapshot['paths'], 'Candidate has a live descriptor or mapping: ' + relative)
    need(not any(absolute.startswith(directory + '/') for directory in snapshot['directories']),
         'Candidate is inside a live working/open directory: ' + relative)


def verify_regular(info, row, rootdev):
    need(stat.S_ISREG(info.st_mode), 'Regular file required: ' + row['path'])
    need(info.st_dev == rootdev and info.st_nlink == row['nlink'] == 1,
         'Same-device single-link regular file required: ' + row['path'])
    actual = [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_nlink]
    expected = [row['dev'], row['ino'], row['size'], row['mtime_ns'], row['nlink']]
    need(actual == expected, 'Recorded file identity changed: ' + row['path'])


def verify_policy(relative, plan, tracked):
    parts(relative)
    need(any(relative.startswith(scope + '/') for scope in plan['approved_scopes']), 'Outside approved scope')
    need(relative not in tracked, 'Tracked file cannot be deleted: ' + relative)
    need(Path(relative).suffix.lower() not in PROTECTED_SUFFIXES, 'Protected source/report suffix: ' + relative)
    need(all(relative != scope and not relative.startswith(scope + '/') for scope in plan['protected_scopes']),
         'Protected dependency scope: ' + relative)


def verify_retained(rootfd, retained):
    for row in retained:
        fd, name = parent_fd(rootfd, row['path'])
        try:
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            need([info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns] == row['identity'],
                 'Retained file identity changed: ' + row['path'])
        finally:
            os.close(fd)


def write_receipt(path, value):
    need(path.parent == AUDIT, 'Own audit output only')
    # O_EXCL rejects existing files and symlinks, preserving prior evidence.
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    need(ROOT.resolve() == ROOT and Path.cwd() == ROOT, 'Exact ordinary project root required')
    parts(args.manifest)
    manifest = ROOT / args.manifest
    need(manifest.resolve() == manifest and manifest.parent == AUDIT and stat.S_ISREG(manifest.lstat().st_mode),
         'Ordinary manifest must be directly inside audit directory')
    payload = manifest.read_bytes()
    need(hashlib.sha256(payload).hexdigest() == args.sha256, 'Manifest checksum changed')
    plan = json.loads(payload)
    need(plan['root'] == str(ROOT), 'Manifest root changed')
    targets, retained = plan['files'], plan['retained_files']
    need(len({r['path'] for r in targets}) == len(targets), 'Duplicate target paths')
    need(not ({r['path'] for r in targets} & {r['path'] for r in retained}), 'Target overlaps retained file')
    for scope in plan['approved_scopes'] + plan['protected_scopes']:
        parts(scope)
    exceptions = plan.get('live_process_exceptions', [])
    validate_exceptions(exceptions)
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'))
    snapshot = live_project_paths(exceptions)
    rootfd = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    rootdev = os.fstat(rootfd).st_dev
    try:
        verify_retained(rootfd, retained)
        for row in targets:
            verify_policy(row['path'], plan, tracked)
            reject_live(row['path'], snapshot)
            fd, name = parent_fd(rootfd, row['path'])
            try:
                verify_regular(os.stat(name, dir_fd=fd, follow_symlinks=False), row, rootdev)
            finally:
                os.close(fd)
        preflight = dict(schema='cleanup_preflight_v2', unix=time.time(), files=len(targets),
            allocated_bytes=sum(r['allocated'] for r in targets), logical_bytes=sum(r['size'] for r in targets),
            same_uid_processes_inspected=snapshot['inspected'], unknown_inaccessible_pids=[],
            identity_bound_ssh_agent_exceptions=snapshot['exact_ssh_agent_exceptions'],
            manifest_sha256=args.sha256, no_symlinks=True, no_tracked_files=True, preserved_files=len(retained),
            live_refresh_max_files=LIVE_BATCH_FILES, live_refresh_max_seconds=LIVE_MAX_AGE_SECONDS)
        prefix = manifest.stem
        suffix = '.execute.preflight.json' if args.execute else '.preflight.json'
        write_receipt(AUDIT / (prefix + suffix), preflight)
        print(json.dumps(preflight), flush=True)
        if not args.execute:
            return
        # Replace the preflight snapshot before the first unlink; then refresh
        # at bounded batches/age. Timestamps are anchored after completed scans.
        snapshot = live_project_paths(exceptions)
        with (AUDIT / (prefix + '.deletions.jsonl')).open('x') as log:
            for index, row in enumerate(targets):
                if index and (index % LIVE_BATCH_FILES == 0 or time.monotonic() - snapshot['completed_monotonic'] >= LIVE_MAX_AGE_SECONDS):
                    snapshot = live_project_paths(exceptions)
                reject_live(row['path'], snapshot)
                fd, name = parent_fd(rootfd, row['path'])
                try:
                    verify_regular(os.stat(name, dir_fd=fd, follow_symlinks=False), row, rootdev)
                    os.unlink(name, dir_fd=fd)
                finally:
                    os.close(fd)
                log.write(json.dumps(dict(path=row['path'], allocated_bytes=row['allocated'],
                    logical_bytes=row['size'], deleted_unix=time.time())) + '\n')
                log.flush()
        verify_retained(rootfd, retained)
        need(all(not os.path.lexists(ROOT / row['path']) for row in targets), 'A target still exists after deletion')
        preflight.update(completed_unix=time.time(), all_selected_files_absent=True, all_retained_files_unchanged=True)
        write_receipt(AUDIT / (prefix + '.completed.json'), preflight)
        print('COMPLETED', json.dumps(preflight), flush=True)
    finally:
        os.close(rootfd)


if __name__ == '__main__':
    main()
