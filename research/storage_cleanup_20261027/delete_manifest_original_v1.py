"""Delete only reviewed regular files below the fixed repository root.

No recursion, symlink following, directory removal, process control or external
writes. The manifest checksum and recorded inode identity are mandatory.
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
PROTECTED_SUFFIXES = {'.json', '.jsonl', '.md', '.txt', '.log', '.csv', '.tsv',
                      '.py', '.sh', '.yaml', '.yml', '.toml', '.tex', '.bib',
                      '.pdf', '.png', '.svg', '.xlsx', '.zip'}


def parts(relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(x in ('..', '.', '') for x in path.parts):
        raise RuntimeError('Invalid relative path: ' + relative)
    if str(path) != relative:
        raise RuntimeError('Noncanonical relative path: ' + relative)
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


def live_project_paths():
    paths = set()
    inspected = 0
    inaccessible = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            cwd = os.readlink(proc / 'cwd')
            cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            # Detect installers, since cache archives must not be removed mid-install.
            if any(token in cmd for token in ('conda install', 'conda create', 'pip install', 'mamba install', 'mamba create')):
                raise RuntimeError('Package manager is active, defer deletion')
            if cwd == str(ROOT) or cwd.startswith(str(ROOT) + '/') or str(ROOT) in cmd or 'AI Auto Research Pipeline Test' in cmd:
                inspected += 1
                if cwd.startswith(str(ROOT) + '/'):
                    paths.add(cwd)
                for entry in (proc / 'fd').iterdir():
                    try:
                        value = os.readlink(entry)
                        if value.startswith(str(ROOT) + '/'):
                            paths.add(value.removesuffix(' (deleted)'))
                    except FileNotFoundError:
                        pass
                for line in (proc / 'maps').read_text().splitlines():
                    fields = line.split(None, 5)
                    if len(fields) == 6 and fields[5].startswith(str(ROOT) + '/'):
                        paths.add(fields[5].removesuffix(' (deleted)'))
        except (FileNotFoundError, ProcessLookupError):
            pass
        except PermissionError:
            inaccessible.append(int(proc.name))
    return paths, inspected, inaccessible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    assert ROOT.resolve() == ROOT and Path.cwd() == ROOT
    manifest = ROOT / args.manifest
    assert manifest.resolve().parent == AUDIT
    payload = manifest.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == args.sha256
    plan = json.loads(payload)
    assert plan['root'] == str(ROOT)
    targets = plan['files']
    assert len({r['path'] for r in targets}) == len(targets)
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'))
    paths, inspected, inaccessible = live_project_paths()
    rootfd = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    retained = plan['retained_files']
    try:
        for row in retained:
            fd, name = parent_fd(rootfd, row['path'])
            try:
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                assert [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns] == row['identity']
            finally:
                os.close(fd)
        for row in targets:
            relative = row['path']
            parts(relative)
            assert any(relative.startswith(scope + '/') for scope in plan['approved_scopes'])
            assert relative not in tracked and Path(relative).suffix not in PROTECTED_SUFFIXES
            assert all(not relative.startswith(scope + '/') for scope in plan['protected_scopes'])
            assert row['nlink'] == 1
            absolute = str(ROOT / relative)
            assert absolute not in paths
            # Reject a candidate if a live process has its parent working directory.
            assert not any(absolute.startswith(p + '/') for p in paths if p != str(ROOT) and Path(p).is_dir())
            fd, name = parent_fd(rootfd, relative)
            try:
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                assert stat.S_ISREG(info.st_mode)
                assert info.st_dev == os.fstat(rootfd).st_dev
                assert [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_nlink] == [row['dev'], row['ino'], row['size'], row['mtime_ns'], row['nlink']]
            finally:
                os.close(fd)
        preflight = {'schema':'cleanup_preflight_v1', 'unix': time.time(), 'files':len(targets),
                     'allocated_bytes':sum(r['allocated'] for r in targets),
                     'logical_bytes':sum(r['size'] for r in targets), 'project_processes_inspected':inspected,
                     'inaccessible_same_uid_pids':inaccessible, 'manifest_sha256':args.sha256,
                     'no_symlinks':True, 'no_tracked_files':True, 'preserved_files':len(retained)}
        output_prefix = manifest.stem
        (AUDIT / (output_prefix + '.preflight.json')).write_text(json.dumps(preflight, indent=2)+'\n')
        print(json.dumps(preflight), flush=True)
        if not args.execute:
            return
        log_path = AUDIT / (output_prefix + '.deletions.jsonl')
        with log_path.open('x') as log:
            for row in targets:
                fd, name = parent_fd(rootfd, row['path'])
                try:
                    info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                    assert stat.S_ISREG(info.st_mode)
                    assert [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_nlink] == [row['dev'], row['ino'], row['size'], row['mtime_ns'], row['nlink']]
                    os.unlink(name, dir_fd=fd)
                finally:
                    os.close(fd)
                log.write(json.dumps({'path':row['path'], 'allocated_bytes':row['allocated'], 'logical_bytes':row['size'], 'deleted_unix':time.time()})+'\n')
                log.flush()
        for row in retained:
            fd, name = parent_fd(rootfd, row['path'])
            try:
                info = os.stat(name, dir_fd=fd, follow_symlinks=False)
                assert [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns] == row['identity']
            finally:
                os.close(fd)
        assert all(not os.path.lexists(ROOT / r['path']) for r in targets)
        preflight.update({'completed_unix':time.time(), 'all_selected_files_absent':True, 'all_retained_files_unchanged':True})
        (AUDIT / (output_prefix + '.completed.json')).write_text(json.dumps(preflight, indent=2)+'\n')
        print('COMPLETED', json.dumps(preflight), flush=True)
    finally:
        os.close(rootfd)


if __name__ == '__main__':
    main()
