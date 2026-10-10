"""Independent contained CPU checks for the shared GPU pool."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import unittest

PROJECT = Path("/workspace-SR003.nfs2/ARG-GNN")
AUDIT = PROJECT / "research/reviews/gpu_pool_v1_independent"
spec = importlib.util.spec_from_file_location("independent_helper_pool_fixture",
    PROJECT / "tests/test_gpu_pool_v1.py")
author = importlib.util.module_from_spec(spec)
spec.loader.exec_module(author)
author.AUDIT = AUDIT
pool = author.pool


class IndependentChecks(unittest.TestCase):
    def setUp(self):
        self.fixture = author.PoolTests()
        self.fixture.setUp()

    def tearDown(self):
        self.fixture.tearDown()

    def test_lease_mutation_blocks_admission_without_metadata_repair(self):
        f = self.fixture
        c = f.controller()
        c.submit(f.spec("held", 65536))
        f.until(lambda: f.observed("held").exists(), c)
        c.submit(f.spec("queued", 16384))
        original = os.pread(c.lease.fileno(), 65536, 0)
        changed = original.replace(b'"job"', b'"bad"')
        self.assertNotEqual(original, changed)
        os.pwrite(c.lease.fileno(), changed, 0)
        try:
            c.tick()
            self.assertIn("Immutable pool lease changed", c.safety_error)
            self.assertEqual(c.jobs["held"]["state"], "running")
            self.assertEqual(c.jobs["queued"]["state"], "queued")
            with self.assertRaisesRegex(RuntimeError, "Immutable pool lease changed"):
                c.submit(f.spec("rejected", 1024))
            self.assertEqual(os.pread(c.lease.fileno(), 65536, 0), changed)
        finally:
            # Restore only our contained handmade lease so fixture cleanup can drain.
            os.pwrite(c.lease.fileno(), original, 0)
            c.tick(launch_queued=False)

    def test_two_pools_concurrently_share_one_storage_reservation_ledger(self):
        f = self.fixture
        c1 = f.controller("poolone", gpu=2, storage_limit_bytes=4_000_000_000,
            external_storage_reserve_bytes=0)
        c2 = f.controller("pooltwo", gpu=3, storage_limit_bytes=4_000_000_000,
            external_storage_reserve_bytes=0)
        jobs = [(c1, f.spec("first", storage_bytes=2_000_000_000)),
                (c2, f.spec("second", storage_bytes=2_000_000_000))]
        barrier = threading.Barrier(2)
        result = []
        def submit(controller, job):
            barrier.wait()
            try:
                result.append(("accepted", controller.submit(job)))
            except ValueError as error:
                result.append(("rejected", str(error)))
        threads = [threading.Thread(target=submit, args=row) for row in jobs]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(sorted(row[0] for row in result), ["accepted", "rejected"])
        self.assertIn("ceiling", next(row[1] for row in result if row[0] == "rejected"))
        ledger = json.loads((f.root / ".runtime/gpup/storage_reservations.json").read_text())
        self.assertEqual(len(ledger["controllers"]), 2)
        self.assertEqual(len(ledger["jobs"]), 1)
        self.assertEqual(sum(row["reserved_bytes"] for row in ledger["jobs"].values()), 2_000_000_000)

    def test_duplicate_canonical_uuid_rejected_even_for_different_physical_indexes(self):
        rows = author.inventory()
        rows[1]["uuid"] = rows[0]["uuid"]
        with self.assertRaisesRegex(ValueError, "Duplicate physical device or UUID"):
            pool.checked_inventory(lambda: rows)
        rows = author.inventory()
        rows[0]["uuid"] = "GPU-ABCDEFAB-0000-0000-0000-000000000002"
        with self.assertRaisesRegex(ValueError, "Canonical lowercase"):
            pool.checked_inventory(lambda: rows)

    def test_real_pidfd_syscall_targets_only_created_cpu_child(self):
        child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"],
            cwd=self.fixture.root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
        descriptor = None
        try:
            before = pool.proc_identity(child.pid)
            descriptor = pool._pidfd_syscall(434, pool.ctypes.c_int(child.pid), pool.ctypes.c_uint(0))
            after = pool.proc_identity(child.pid)
            self.assertEqual((before["uid"], before["start_ticks"]),
                             (after["uid"], after["start_ticks"]))
            pool._pidfd_syscall(424, pool.ctypes.c_int(descriptor), pool.ctypes.c_int(signal.SIGTERM),
                pool.ctypes.c_void_p(), pool.ctypes.c_uint(0))
            self.assertEqual(child.wait(timeout=5), -signal.SIGTERM)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)

    def test_candidate_gst_argmix_mechanism_title_and_exec_guards_are_preserved(self):
        checks = {
            "src/arggnn/biological_candidate.py": "6b3a25d0e59026cab7189032eb41221feb7d1bdf69e15f6c737a672dfb711dd3",
            "src/arggnn/biological_gst_control_v1.py": "66788b41cabcb852c739505200a641bdc93fea3c9fbb642c291d52cefc77350f",
            "scripts/train_biological_gst_control_v1.py": "e8496eb6a92cc2f9edb5f107a89d23cdbc945a82ae0e00bf2a4e0f00ed9ab72b",
            "src/arggnn/official_argmix_biological.py": "4a6c44dfd81ba3f75e74c5287a4b55776bb9fe1612732c457748643cd75d7b1f",
            "src/arggnn/native_argformer_exec.py": "84ed405379ac56511c2f4722c130e8b884226018bbf8e2949ea2bb4dceeb99d0",
            "src/arggnn/biological_mechanism_controls_v1.py": "28f3e7d3f28a48eeee0944b2e4ad55e3ea6ebb314d8b0342bd785facb08dcbeb",
            "scripts/train_biological_mechanism_controls_v1.py": "f19bdead17a135a42ad70513f1522cef2a9fd58dc159d04f189adbe08de67dec",
        }
        for name, expected in checks.items():
            self.assertEqual(hashlib.sha256((PROJECT / name).read_bytes()).hexdigest(), expected)
        candidate = ast.parse((PROJECT / "src/arggnn/biological_candidate.py").read_text())
        fit = next(n for n in candidate.body if isinstance(n, ast.FunctionDef) and n.name == "configure_fit")
        title_line = min(n.lineno for n in ast.walk(fit) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == "apply_and_validate_title")
        cuda_lines = [n.lineno for n in ast.walk(fit) if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Attribute) and isinstance(n.value.value, ast.Name)
            and n.value.value.id == "torch" and n.value.attr == "cuda"]
        self.assertTrue(all(line > title_line for line in cuda_lines))
        for name in ("scripts/train_biological_gst_control_v1.py",
                     "scripts/train_biological_mechanism_controls_v1.py"):
            main = next(n for n in ast.parse((PROJECT / name).read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
            title = min(n.lineno for n in ast.walk(main) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "apply_and_validate_title")
            science = [n.lineno for n in ast.walk(main) if isinstance(n, ast.ImportFrom)
                and n.module and n.module.startswith("arggnn")
                and n.module != "arggnn.native_argformer_exec"]
            self.assertTrue(all(line > title for line in science))
        argmix = (PROJECT / "src/arggnn/official_argmix_biological.py").read_text()
        functions = {n.name: n for n in ast.parse(argmix).body if isinstance(n, ast.FunctionDef)}
        gpu = ast.get_source_segment(argmix, functions["gpu_runtime"])
        self.assertLess(gpu.index("apply_and_validate_title"), gpu.index("source_imports()"))
        bootstrap = ast.get_source_segment(argmix, functions["bootstrap"])
        self.assertLess(bootstrap.index("apply_and_validate_title"), bootstrap.index("os.execve"))
        self.assertIn('os.set_inheritable(lease["lease_fd"], True)', bootstrap)
        self.assertIn("arggnn.biological_mechanism_controls_v1", pool.MODULES)



    def test_fresh_exit_observation_retains_new_inherited_fd_descendant(self):
        f = self.fixture
        worker = f.root / "src/arggnn/pool_fake_worker.py"
        source = worker.read_text()
        assert 'if a.mode in ("fork","escaped","fork-grandchild"):' in source
        source = source.replace('if a.mode in ("fork","escaped","fork-grandchild"):', 
            'if a.mode=="deferfork":\n while not Path(a.gate+".fork").exists():\n  time.sleep(.01)\nif a.mode in ("fork","escaped","deferfork"):')
        worker.write_text(source)
        c = f.controller()
        job = f.spec("race", 65536, mode="deferfork")
        c.submit(job)
        f.until(lambda: f.observed("race").exists(), c)
        c.submit(f.spec("waiting", 16384))
        original_reader = c.proc_reader
        leader = c.children["race"]
        fork_gate = Path(job["args"][job["args"].index("--gate") + 1] + ".fork")
        ready = f.root / "outputs/race/child-ready.json"
        triggered = [False]

        def mixed_snapshot(lease, controller_pid, known=None):
            table = original_reader(lease, controller_pid, known)
            if not triggered[0]:
                triggered[0] = True
                fork_gate.touch()
                deadline = pool.time.monotonic() + 5
                while not ready.exists() or leader.poll() is None:
                    if pool.time.monotonic() > deadline:
                        raise RuntimeError("Contained fork race fixture did not become ready")
                    pool.time.sleep(.01)
                # Reproduce a possible mixed /proc observation: the new child
                # was absent from enumeration but the exited leader is now Z.
                if leader.pid in table:
                    table[leader.pid]["state"] = "Z"
            return table

        c.proc_reader = mixed_snapshot
        c.tick()
        self.assertTrue(triggered[0])
        self.assertEqual(c.jobs["race"]["state"], "draining")
        self.assertEqual(c.jobs["waiting"]["state"], "queued")
        self.assertIsNone(c.jobs["waiting"]["pid"])
        self.assertEqual(c._memory()["active_peak_mib"], 65536)
        self.assertTrue(ready.exists())
        f.gates[0].touch()
        f.until(lambda: c.jobs["race"]["state"] == "succeeded"
                and c.jobs["waiting"]["state"] == "running", c)



    def test_descendant_to_grandchild_during_final_kernel_identity_merge(self):
        f = self.fixture
        worker = f.root / "src/arggnn/pool_fake_worker.py"
        source = worker.read_text()
        assert 'if a.mode in ("fork","escaped","fork-grandchild"):' in source
        source = source.replace('if a.mode in ("fork","escaped","fork-grandchild"):', 
            'if a.mode in ("fork","escaped","granddefer"):')
        block = 'if a.mode=="granddefer":\n while not Path(a.gate+".grandfork").exists():\n  time.sleep(.01)\n next_pid=os.fork()\n if next_pid:\n  os._exit(0)\n write("grandchild-ready.json",{"pid":os.getpid()})\nif a.spam:'
        source = source.replace('if a.spam:', block)
        worker.write_text(source)
        c = f.controller()
        job = f.spec("grandrace", 65536, mode="granddefer")
        c.submit(job)
        ready = f.root / "outputs/grandrace/child-ready.json"
        f.until(lambda: ready.exists())
        descendant = json.loads(ready.read_text())["pid"]
        leader = c.children["grandrace"]
        f.until(lambda: pool.proc_identity(leader.pid)["state"] == "Z")
        c.submit(f.spec("aftergrand", 16384))
        fork_gate = Path(job["args"][job["args"].index("--gate") + 1] + ".grandfork")
        grand_ready = f.root / "outputs/grandrace/grandchild-ready.json"
        original_identity = pool.proc_identity
        observations = [0]
        triggered = [False]
        def move_during_identity(pid):
            if pid == descendant:
                observations[0] += 1
                if observations[0] == 6:
                    triggered[0] = True
                    fork_gate.touch()
                    deadline = pool.time.monotonic() + 5
                    while not grand_ready.exists() or original_identity(descendant)["state"] != "Z":
                        if pool.time.monotonic() > deadline:
                            raise RuntimeError("Contained grandchild fixture did not become ready")
                        pool.time.sleep(.01)
            return original_identity(pid)
        with unittest.mock.patch.object(pool, "proc_identity", side_effect=move_during_identity):
            c.tick()
        self.assertTrue(triggered[0], observations[0])
        self.assertEqual(c.jobs["grandrace"]["state"], "draining")
        self.assertEqual(c._memory()["active_peak_mib"], 65536)
        self.assertEqual(c.jobs["aftergrand"]["state"], "queued")
        f.gates[0].touch()
        f.until(lambda: c.jobs["grandrace"]["state"] == "succeeded"
                and c.jobs["aftergrand"]["state"] == "running", c)



    def test_reused_leader_pid_cannot_adopt_unrelated_same_uid_session(self):
        f = self.fixture
        c = f.controller()
        c.submit(f.spec("identity", 1024))
        f.until(lambda: f.observed("identity").exists(), c)
        leader = c.children["identity"]
        real_reader = c.proc_reader
        def reused(lease, controller_pid, known=None):
            table = real_reader(lease, controller_pid, known)
            row = dict(table[leader.pid])
            row.update(start_ticks=row["start_ticks"] + 100000, ppid=1,
                sid=leader.pid, pgid=leader.pid, lease_fd_present=False,
                controller_descendant=False)
            table[leader.pid] = row
            return table
        try:
            c.proc_reader = reused
            c._process_scan()
            self.assertNotIn(leader.pid, c.pid_jobs)
        finally:
            c.proc_reader = real_reader
            c.tick(launch_queued=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
