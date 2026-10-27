"""Inspect only, or signal exact members of a newly created task session."""
from __future__ import annotations
import os
import ctypes
import errno
import platform
from pathlib import Path
import signal
import subprocess
import time
from . import runtime as r


def session_members(leader):
    output = subprocess.check_output(['ps', '-eo', 'pid=,pgid=,sid=,stat='], text=True, timeout=15)
    members = []
    for line in output.splitlines():
        pid, pgid, sid, status = line.split()
        if int(sid) != leader:
            continue
        if int(pgid) != leader:
            raise RuntimeError('Task escaped original process group within its session')
        if status.startswith('Z'):
            continue
        p = Path('/proc') / pid
        try:
            fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            cwd = (p / 'cwd').resolve(strict=True)
        except FileNotFoundError:
            continue
        if not cwd.is_relative_to(r.ROOT):
            raise RuntimeError('Unexpected task session member outside project')
        members.append(dict(pid=int(pid), start_ticks=fields[19], sid=int(sid), pgid=int(pgid)))
    return members


def _pidfd_open(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    if platform.machine() not in ('x86_64', 'aarch64'):
        raise RuntimeError('Unsupported pidfd syscall architecture')
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(434, pid, 0)
    if fd < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return fd


def _pidfd_send(fd, sig):
    if hasattr(signal, 'pidfd_send_signal'):
        return signal.pidfd_send_signal(fd, sig)
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(424, fd, int(sig), 0, 0)
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def signal_member(member, sig):
    pid = member['pid']
    try:
        fd = _pidfd_open(pid)
    except ProcessLookupError:
        return
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[19] != member['start_ticks'] or int(fields[3]) != member['sid'] or int(fields[2]) != member['pgid']:
            raise RuntimeError('Task session member identity changed')
        _pidfd_send(fd, sig)
    except (FileNotFoundError, ProcessLookupError):
        pass
    finally:
        os.close(fd)


def drain(process, *, terminate=False):
    members = session_members(process.pid)
    if members and not terminate:
        return False
    for sig, seconds in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
        if not members:
            break
        for member in members:
            signal_member(member, sig)
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            process.poll()
            members = session_members(process.pid)
            if not members:
                break
            time.sleep(0.25)
    if members:
        raise RuntimeError('Owned session failed to drain')
    if process.poll() is None:
        process.wait(timeout=5)
    return True


def pss(leader):
    import psutil
    total = 0
    for member in session_members(leader):
        try:
            p = psutil.Process(member['pid'])
            total += getattr(p.memory_full_info(), 'pss', p.memory_info().rss)
        except psutil.NoSuchProcess:
            pass
    return total
