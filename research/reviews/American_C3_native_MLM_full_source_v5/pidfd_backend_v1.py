"""Inert stdlib-only pidfd backend; ROOT alone runs --probe-self-root.

Linux x86_64 pidfd_open434/pidfd_send_signal424 are used through libc.syscall
only when the corresponding native Python function is unavailable. No named
libc pidfd symbols, positive numeric PID/group signal or raw-signal fallback.
"""
import argparse
import ctypes
import errno
import json
import os
import signal
import sys

SOURCE = 'research/reviews/American_C3_native_MLM_full_source_v5/pidfd_backend_v1.py'
SYS_PIDFD_OPEN_X86_64 = 434
SYS_PIDFD_SEND_SIGNAL_X86_64 = 424
_BACKEND = None


class PidfdBackendV1:
    def __init__(self, *, os_api=os, signal_api=signal, platform_name=sys.platform,
                 machine=None, load_libc=None, ctypes_api=ctypes):
        self.os_api = os_api
        self.ctypes_api = ctypes_api
        self.native_open = getattr(os_api, 'pidfd_open', None)
        self.native_send = getattr(signal_api, 'pidfd_send_signal', None)
        self.syscall = None
        self.libc = None
        if not callable(self.native_open) or not callable(self.native_send):
            machine = os_api.uname().machine if machine is None else machine
            if platform_name != 'linux' or machine != 'x86_64' or ctypes_api.sizeof(ctypes_api.c_long) != 8:
                raise RuntimeError('Syscall pidfd fallback requires exact Linux x86_64 64-bit long ABI')
            loader = ctypes_api.CDLL if load_libc is None else load_libc
            self.libc = loader(None, use_errno=True)
            self.syscall = self.libc.syscall
            self.syscall.restype = ctypes_api.c_long
            # syscall is variadic; every supplied argument is explicitly c_long.
            self.syscall.argtypes = [ctypes_api.c_long]

    def _call(self, number, *args):
        c = self.ctypes_api
        c.set_errno(0)
        result = int(self.syscall(*(c.c_long(x) for x in (number, *args))))
        if result == -1:
            error = c.get_errno() or errno.EIO
            raise OSError(error, os.strerror(error))
        return result

    def open(self, pid, flags=0):
        if type(pid) is not int or pid <= 0 or type(flags) is not int or flags != 0:
            raise ValueError('Positive PID and zero pidfd_open flags required')
        if callable(self.native_open):
            return self.native_open(pid, flags)
        # Linux pidfd_open returns an FD_CLOEXEC descriptor by definition.
        return self._call(SYS_PIDFD_OPEN_X86_64, pid, flags)

    def send(self, fd, signum, siginfo=None, flags=0):
        if type(fd) is not int or fd < 0 or not isinstance(signum, int) or signum < 0 or siginfo is not None or flags != 0:
            raise ValueError('Valid pidfd, signal, null siginfo and zero flags required')
        if callable(self.native_send):
            return self.native_send(fd, signum, siginfo, flags)
        result = self._call(SYS_PIDFD_SEND_SIGNAL_X86_64, fd, int(signum), 0, 0)
        if result != 0:
            raise RuntimeError('Unexpected pidfd_send_signal syscall return value')
        return None

    def description(self):
        return dict(open_backend='native_Python' if callable(self.native_open) else 'Linux_x86_64_syscall434',
                    send_backend='native_Python' if callable(self.native_send) else 'Linux_x86_64_syscall424',
                    positive_numeric_PID_or_group_fallback=False)


def backend_v1():
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = PidfdBackendV1()
    return _BACKEND


def pidfd_open_v1(pid, flags=0):
    return backend_v1().open(pid, flags)


def pidfd_send_signal_v1(fd, signum, siginfo=None, flags=0):
    return backend_v1().send(fd, signum, siginfo, flags)


def self_probe_v1(*, backend=None, os_api=os):
    """ROOT-only real capability probe; source tests inject fake backend/os.

    Open own PID, verify CLOEXEC, send signal0 through that pidfd, always close.
    This does not signal any positive signal or claim a new storage admission.
    """
    current = backend_v1() if backend is None else backend
    fd = current.open(os_api.getpid(), 0)
    try:
        if os_api.get_inheritable(fd):
            raise RuntimeError('pidfd descriptor unexpectedly lacks CLOEXEC')
        current.send(fd, 0, None, 0)
    finally:
        os_api.close(fd)
    return dict(schema='native_MLM_pidfd_self_probe_v1', status='passed_own_pidfd_signal0_and_CLOEXEC',
                source=SOURCE, backend=current.description(), descriptor_closed=True,
                positive_signal_sent=False, scientific_or_accounting_execution=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-self-root', action='store_true')
    args = parser.parse_args(argv)
    result = self_probe_v1() if args.probe_self_root else dict(status='source_only_inactive', source=SOURCE, self_probe_executed=False)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
