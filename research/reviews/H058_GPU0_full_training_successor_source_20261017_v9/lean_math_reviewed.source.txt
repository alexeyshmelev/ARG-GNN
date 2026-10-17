"""INACTIVE prototype for a fresh reviewed H058 successor only.

Trusted pinned eager Torch/PyG computation, not a syscall sandbox. BOTH math
boundaries retain the original charged-scope census and cap detection.
No constructor/function is executed by the source audit.
"""
from contextlib import contextmanager
import fcntl
import os
import stat
import sys
import threading

REQUIRED_ENV = {
    "CUDA_CACHE_DISABLE": "1", "USE_PYTORCH_KERNEL_CACHE": "0",
    "PYTHONDONTWRITEBYTECODE": "1", "PYTORCH_JIT": "0",
    "TORCH_COMPILE_DISABLE": "1", "TORCHDYNAMO_DISABLE": "1",
}
MUTATION_EVENTS = frozenset({
    "os.remove", "os.rmdir", "os.rename", "os.mkdir", "os.link", "os.symlink",
    "os.truncate", "os.chmod", "os.chown", "os.utime", "os.setxattr",
    "os.removexattr", "shutil.copyfile", "shutil.copymode", "shutil.copystat",
    "subprocess.Popen", "os.system", "os.fork", "os.forkpty", "os.posix_spawn",
    "os.exec", "sys.setprofile", "sys.settrace", "ctypes.dlopen",
    "compile", "exec",
})
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC


def require(condition, message):
    if not condition:
        raise ValueError(message)


class MathEffectGateV1:
    """Gate only train_one; never replace the context's full guard globally.

    One process-lifetime audit hook is inactive outside a lean call. Original
    full checks remain at startup, readers, transitions, writers, supervisor and
    both genuine first scheduled steps. No file or receipt is written here.
    """
    def __init__(self, full_guard, math_guard, torch_module, arms):
        require(callable(full_guard) and callable(math_guard), "Bound exact guards")
        self.full_guard, self.math_guard = full_guard, math_guard
        self.torch, self.arms = torch_module, tuple(arms)
        require(len(self.arms) == 2 and len(set(self.arms)) == 2, "Original paired arms")
        self.warmed = set()
        self.active, self.owner = False, None
        self.local = threading.local()
        self.lease_fds, self.escalations = {}, 0
        for name in ("ARGGNN_GPU_LEASE_FD", "H058_LOGICAL_LOCK_FD"):
            fd = int(os.environ[name]); opened = os.fstat(fd)
            require(stat.S_ISREG(opened.st_mode), "Original regular lease/lock FD")
            self.lease_fds[fd] = (opened.st_dev, opened.st_ino)
        require(len(self.lease_fds) == 2, "Original distinct physical/logical FDs")
        self.installed = False
        sys.addaudithook(self._audit)
        sys.audit("arggnn.H058.math_gate_installed", id(self))
        require(self.installed, "Audit hook installation must be observed")

    @contextmanager
    def _trusted_guard(self):
        # Original lease checks open a writable exclusion probe; the full guard
        # opens its mutex. Keep these exact authenticated operations unchanged.
        previous = getattr(self.local, "guard", False)
        self.local.guard = True
        try:
            yield
        finally:
            self.local.guard = previous

    def _full(self):
        with self._trusted_guard():
            self.full_guard()

    def _unexpected(self, event):
        self.escalations += 1
        self._full()
        # A boundary=False replay does not authorize an unbounded new writer.
        raise RuntimeError("Unexpected effect in pinned eager math: " + event)

    def _audit(self, event, args):
        # Global active flag covers audited Python events in other threads;
        # reentrancy bypass applies only to the thread running an exact guard.
        if event == "arggnn.H058.math_gate_installed" and args == (id(self),):
            self.installed = True
            return
        if not self.active or getattr(self.local, "guard", False):
            return
        if event == "open":
            _path, mode, flags = args
            if ((isinstance(mode, str) and any(c in mode for c in "wax+"))
                    or (isinstance(flags, int) and flags & WRITE_FLAGS)):
                self._unexpected(event)
        elif event == "mmap.__new__":
            if args[0] != -1:
                self._unexpected(event)
        elif event in MUTATION_EVENTS:
            self._unexpected(event)
        elif event == "import":
            # First genuine steps warm lazy imports. A later cold import gets
            # full validation; subsequent compile/exec/write still fails here.
            self._full()

    def _runtime(self):
        require(sys.dont_write_bytecode and all(os.environ.get(k) == v
            for k, v in REQUIRED_ENV.items()), "Pre-science cache/JIT/compile controls")
        require(sys.gettrace() is None and sys.getprofile() is None,
                "No Python trace/profiler writer")
        t = self.torch
        require(not t.compiler.is_compiling() and not t.jit.is_scripting()
                and not t.jit.is_tracing(), "Pinned eager execution only")
        require(not t.autograd.profiler._is_profiler_enabled,
                "No active profiler/exporter; inactive record_function remains")
        require(not t.cuda.is_current_stream_capturing(), "No CUDA graph capture")

    def _fds(self):
        # Existing stdout/stderr are supervisor pipes. Only the original two
        # inherited regular lock FDs are permitted; no FD is closed/reopened.
        with os.scandir("/proc/self/fd") as entries:
            for entry in entries:
                fd = int(entry.name); opened = os.fstat(fd)
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                if stat.S_ISREG(opened.st_mode) and flags & os.O_ACCMODE != os.O_RDONLY:
                    if self.lease_fds.get(fd) != (opened.st_dev, opened.st_ino):
                        self._unexpected("unreviewed open regular writer FD")
        for fd, identity in self.lease_fds.items():
            opened = os.fstat(fd)
            require((opened.st_dev, opened.st_ino) == identity, "Original inherited lock FD identity")

    def _math_boundary(self):
        require(self.active and self.owner == threading.get_ident(), "Exact math owner")
        with self._trusted_guard():
            # Same active/expiry/work/lease/title/UUID/RSS/CUDA, 24h timer and
            # charged_census_v1. Only recursive historical accounting is absent.
            self.math_guard()
        self._runtime()
        self._fds()

    def run_step(self, arm, first, operation):
        require(arm in self.arms and type(first) is bool and callable(operation),
                "Original scheduled arm and callable")
        require(not self.active, "No nested math or background scheduled step")
        if first:
            require(arm not in self.warmed, "First genuine scheduled update once per arm")
            result = operation(self.full_guard)
            self.warmed.add(arm)
            return result
        require(self.warmed == set(self.arms), "Both genuine first scheduled steps completed")
        self._runtime(); self._fds()
        self.owner, self.active = threading.get_ident(), True
        try:
            return operation(self._math_boundary)
        finally:
            self.active, self.owner = False, None
