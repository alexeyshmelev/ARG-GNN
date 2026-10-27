"""Focused safety tests; no Git invocation and no file unlink is performed."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cleanup_under_test', HERE / 'delete_manifest.py')
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)
FIXTURES = Path(tempfile.mkdtemp(prefix='safety_test_fixtures_', dir=HERE))
PAYLOAD = FIXTURES / 'payload.bin'
PAYLOAD.write_bytes(b'retained tiny safety fixture\n')
REAL_PATH = Path


def row_for(path):
    info = path.lstat()
    return dict(path=str(path.relative_to(cleanup.ROOT)), dev=info.st_dev, ino=info.st_ino,
                size=info.st_size, mtime_ns=info.st_mtime_ns, nlink=info.st_nlink,
                allocated=info.st_blocks * 512)


class FakeNode:
    def __init__(self, proc, kind):
        self.proc, self.kind = proc, kind
    def read_text(self):
        if self.kind == 'stat':
            return self.proc.name + ' (generic) ' + ' '.join(['S'] + ['0'] * 18 + [self.proc.start])
        if self.kind == 'maps':
            return self.proc.maps
        raise AssertionError(self.kind)
    def read_bytes(self):
        if self.kind == 'cmdline':
            return self.proc.raw
        raise AssertionError(self.kind)
    def iterdir(self):
        if self.kind == 'fd':
            return [FakeNode(self.proc, 'fd/' + str(i)) for i in range(len(self.proc.fds))]
        raise AssertionError(self.kind)


class FakeProc:
    def __init__(self, pid=901, argv=('/usr/bin/python3', 'unrelated.py'), cwd='/tmp', fds=(), maps='', denied=False):
        self.name, self.start = str(pid), '1234'
        self.raw = b'\0'.join(value.encode() for value in argv) + b'\0'
        self.cwd, self.fds, self.maps, self.denied = cwd, list(fds), maps, denied
    def stat(self):
        return SimpleNamespace(st_uid=os.getuid())
    def __truediv__(self, kind):
        return FakeNode(self, kind)


def scan_fake(procs, exceptions=()):
    root = SimpleNamespace(iterdir=lambda: iter(procs))
    def path_factory(value):
        return root if value == '/proc' else REAL_PATH(value)
    def readlink(node):
        if node.kind == 'cwd':
            if node.proc.denied:
                raise PermissionError('fixture denied')
            return node.proc.cwd
        return node.proc.fds[int(node.kind.split('/')[1])]
    with mock.patch.object(cleanup, 'Path', side_effect=path_factory), mock.patch.object(cleanup.os, 'readlink', side_effect=readlink):
        return cleanup.live_project_paths(exceptions)


class SafetyTests(unittest.TestCase):
    def test_paths_reject_traversal_absolute_and_noncanonical(self):
        for value in ('', '.', '..', '../x', 'x/../y', '/tmp/x', './x', 'x//y', 'x/', 17):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                cleanup.parts(value)
        self.assertEqual(cleanup.parts('valid/path.bin'), ('valid', 'path.bin'))

    def test_intermediate_and_leaf_symlink_rejected(self):
        alias = FIXTURES / 'directory_alias'
        alias.symlink_to(FIXTURES, target_is_directory=True)
        leaf = FIXTURES / 'leaf_alias.bin'
        leaf.symlink_to(PAYLOAD)
        rootfd = os.open(cleanup.ROOT, os.O_RDONLY | os.O_DIRECTORY)
        try:
            with self.assertRaises(OSError):
                cleanup.parent_fd(rootfd, str((alias / 'payload.bin').relative_to(cleanup.ROOT)))
            with self.assertRaises(RuntimeError):
                cleanup.verify_regular(leaf.lstat(), row_for(leaf), PAYLOAD.stat().st_dev)
        finally:
            os.close(rootfd)

    def test_exact_identity_and_single_link(self):
        info, row = PAYLOAD.stat(), row_for(PAYLOAD)
        cleanup.verify_regular(info, row, info.st_dev)
        for field in ('dev', 'ino', 'size', 'mtime_ns', 'nlink'):
            changed = dict(row)
            changed[field] += 1
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                cleanup.verify_regular(info, changed, info.st_dev)
        with self.assertRaises(RuntimeError):
            cleanup.verify_regular(info, row, info.st_dev + 1)

    def test_tracked_protected_and_scope_policy(self):
        plan = dict(approved_scopes=['obsolete'], protected_scopes=['obsolete/current.bin', 'obsolete/keep'])
        cleanup.verify_policy('obsolete/old.bin', plan, set())
        for path, tracked in [('elsewhere/old.bin', set()), ('obsolete/source.PY', set()),
                              ('obsolete/current.bin', set()), ('obsolete/keep/old.bin', set()),
                              ('obsolete/old.bin', {'obsolete/old.bin'})]:
            with self.subTest(path=path), self.assertRaises(RuntimeError):
                cleanup.verify_policy(path, plan, tracked)

    def test_unrelated_process_fd_and_maps_detected(self):
        proc = FakeProc(fds=[str(PAYLOAD)], maps='1000-2000 r--p 0 00:00 0 ' + str(PAYLOAD) + '\n')
        result = scan_fake([proc])
        self.assertIn(str(PAYLOAD), result['paths'])
        self.assertEqual(result['inspected'], 1)
        with self.assertRaises(RuntimeError):
            cleanup.reject_live(row_for(PAYLOAD)['path'], result)

    def test_unknown_inaccessible_process_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, 'Unknown inaccessible processes'):
            scan_fake([FakeProc(denied=True)])

    def test_exact_agent_exception_and_binding_changes(self):
        proc = FakeProc(argv=('/usr/bin/ssh-agent', '-s'), denied=True)
        binding = cleanup.process_binding(proc)
        result = scan_fake([proc], [binding])
        self.assertEqual(result['exact_ssh_agent_exceptions'], [binding])
        for field, value in [('pid', 902), ('start_ticks', '999'), ('argv', ['/usr/bin/ssh-agent', '-c']),
                             ('cmdline_sha256', '0' * 64)]:
            changed = dict(binding, **{field: value})
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                scan_fake([proc], [changed])
        with self.assertRaises(RuntimeError):
            cleanup.validate_exceptions([dict(binding, argv=['/usr/bin/python3'])])

    def test_binding_rechecked_at_permission_denial(self):
        proc = FakeProc(argv=('/usr/bin/ssh-agent', '-s'), denied=True)
        binding = cleanup.process_binding(proc)
        changed = dict(binding, start_ticks='5678')
        with mock.patch.object(cleanup, 'process_binding', side_effect=[binding, changed]), self.assertRaises(RuntimeError):
            scan_fake([proc], [binding])

    def test_root_directory_fd_does_not_block_entire_root(self):
        result = scan_fake([FakeProc(cwd=str(cleanup.ROOT), fds=[str(cleanup.ROOT)])])
        self.assertEqual(result['paths'], set())
        cleanup.reject_live(row_for(PAYLOAD)['path'], result)

    def test_directory_set_precomputed_and_scan_completion_timestamp(self):
        with mock.patch.object(cleanup.time, 'monotonic', return_value=123.5):
            result = scan_fake([FakeProc(fds=[str(FIXTURES)])])
        self.assertEqual(result['completed_monotonic'], 123.5)
        self.assertEqual(result['directories'], {str(FIXTURES)})
        with mock.patch.object(REAL_PATH, 'is_dir', side_effect=AssertionError('Must not restat live directories')):
            with self.assertRaisesRegex(RuntimeError, 'inside a live'):
                cleanup.reject_live(row_for(PAYLOAD)['path'], result)

    def test_receipt_never_overwrites_existing_or_symlink(self):
        existing = HERE / (FIXTURES.name + '_existing.json')
        existing.write_text('original evidence\n')
        alias = HERE / (FIXTURES.name + '_alias.json')
        alias.symlink_to(existing)
        for path in (existing, alias):
            with self.subTest(path=path), self.assertRaises(FileExistsError):
                cleanup.write_receipt(path, {})
        self.assertEqual(existing.read_text(), 'original evidence\n')

    def make_manifest(self, label, files, retained=()):
        path = HERE / (FIXTURES.name + '_' + label + '.json')
        plan = dict(root=str(cleanup.ROOT), files=files, retained_files=list(retained),
                    approved_scopes=['research/storage_cleanup_20261027'], protected_scopes=[],
                    live_process_exceptions=[])
        path.write_text(json.dumps(plan))
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_retained_target_overlap_rejected_before_git(self):
        row = row_for(PAYLOAD)
        manifest, digest = self.make_manifest('overlap', [row], [dict(path=row['path'], identity=[])])
        args = ['delete_manifest.py', '--manifest', str(manifest.relative_to(cleanup.ROOT)), '--sha256', digest]
        with mock.patch('sys.argv', args), mock.patch.object(cleanup.subprocess, 'check_output') as git:
            with self.assertRaisesRegex(RuntimeError, 'overlaps retained'):
                cleanup.main()
            git.assert_not_called()

    def test_retained_identity_changed_rejected(self):
        rootfd = os.open(cleanup.ROOT, os.O_RDONLY | os.O_DIRECTORY)
        info = PAYLOAD.stat()
        row = dict(path=str(PAYLOAD.relative_to(cleanup.ROOT)), identity=[info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns])
        try:
            cleanup.verify_retained(rootfd, [row])
            row['identity'][2] += 1
            with self.assertRaises(RuntimeError):
                cleanup.verify_retained(rootfd, [row])
        finally:
            os.close(rootfd)

    def schedule_case(self, label, count, deletion_seconds):
        rows = [dict(row_for(PAYLOAD), path=str((FIXTURES / ('target_' + str(i) + '.bin')).relative_to(cleanup.ROOT))) for i in range(count)]
        manifest, digest = self.make_manifest(label, rows)
        args = ['delete_manifest.py', '--manifest', str(manifest.relative_to(cleanup.ROOT)), '--sha256', digest, '--execute']
        clock, deleted, scans, receipts = [0.], [], [], []
        def snapshot(exceptions):
            scans.append(len(deleted))
            clock[0] += 50.  # Scan latency is not counted as snapshot age.
            return dict(paths=set(), directories=set(), inspected=1, exact_ssh_agent_exceptions=[], completed_monotonic=clock[0])
        def parent(rootfd, relative):
            return os.open(FIXTURES, os.O_RDONLY | os.O_DIRECTORY), PAYLOAD.name
        def pretend_unlink(name, dir_fd):
            deleted.append(name)
            clock[0] += deletion_seconds
        with mock.patch('sys.argv', args), \
             mock.patch.object(cleanup.subprocess, 'check_output', return_value=b'') as git, \
             mock.patch.object(cleanup, 'live_project_paths', side_effect=snapshot), \
             mock.patch.object(cleanup, 'parent_fd', side_effect=parent), \
             mock.patch.object(cleanup, 'verify_regular'), \
             mock.patch.object(cleanup, 'write_receipt', side_effect=lambda path, value: receipts.append(path.name)), \
             mock.patch.object(cleanup.time, 'monotonic', side_effect=lambda: clock[0]), \
             mock.patch.object(cleanup.os, 'unlink', side_effect=pretend_unlink), \
             mock.patch.object(cleanup.os.path, 'lexists', return_value=False), \
             contextlib.redirect_stdout(io.StringIO()):
            cleanup.main()
            self.assertEqual(git.call_count, 1)  # Mock only: no Git process runs.
        self.assertEqual(len(deleted), count)
        self.assertTrue(PAYLOAD.exists())
        self.assertEqual(receipts[0], manifest.stem + '.execute.preflight.json')
        return scans

    def test_refresh_before_unlink_and_every_256_files(self):
        self.assertEqual(self.schedule_case('batch', 514, 0.), [0, 0, 256, 512])

    def test_refresh_after_ten_seconds_since_completed_scan(self):
        self.assertEqual(self.schedule_case('age', 5, 6.), [0, 0, 2, 4])


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SafetyTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    evidence = dict(schema='cleanup_safety_tests_v1', tests_run=result.testsRun,
                    failures=len(result.failures), errors=len(result.errors), success=result.wasSuccessful(),
                    source_sha256=hashlib.sha256((HERE / 'delete_manifest.py').read_bytes()).hexdigest(),
                    test_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    no_actual_unlink=True, no_git_process=True, fixtures=str(FIXTURES),
                    limitations=['Live checks are sampled and do not provide atomic exclusion of concurrent writers.',
                                 'Only audited immutable obsolete directories may be used.'])
    output = HERE / (FIXTURES.name + '_results.json')
    output.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(dict(evidence=str(output), **evidence)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
