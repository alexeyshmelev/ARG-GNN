"""Finite generation-aware PSS sampling for the isolated V8 continuation.

This module changes supervision only. It does not change task caps or scientific
callables. Samples are finite /proc observations, not atomic kernel group limits.
"""
from __future__ import annotations

from collections import OrderedDict, deque
import ctypes
import errno
import os
from pathlib import Path
import platform
import signal


HISTORY_LIMIT = 256
EVENT_LIMIT = 64


def _generation(row):
    return int(row["pid"]), int(row["start_ticks"])


def _exact(row):
    return tuple(row[key] for key in ("pid", "start_ticks", "pgid", "session"))


def _pidfd_open(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid, 0)
    # The project Python has no pidfd wrappers. These two syscall numbers are
    # identical in the Linux x86_64 and aarch64 syscall ABIs.
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("Generation-bound signaling needs Linux pidfd support")
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(result)


def _pidfd_signal(fd, sig):
    if hasattr(signal, "pidfd_send_signal"):
        return signal.pidfd_send_signal(fd, sig, None, 0)
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("Generation-bound signaling needs Linux pidfd support")
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(ctypes.c_long(424), ctypes.c_int(fd), ctypes.c_int(sig),
                          ctypes.c_void_p(0), ctypes.c_uint(0))
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def repaired_owned_group_class(core):
    """Return an API-compatible replacement for the new private V4 core only.

    measure() returns validated PSS bytes, never an RSS fallback upper bound.
    latest explicitly labels incomplete samples. Callers retain the full task
    reservation and must use latest["headroom_credit_bytes"] for admission.
    """
    inherited = core.OwnedGroup

    class FinitePSSOwnedGroup(inherited):
        def __init__(self, pid):
            super().__init__(pid)
            self.root = dict(self.root)
            self._original_root = _exact(self.root)
            self.history = OrderedDict([(_generation(self.root), dict(self.root))])
            self._last_capture = {_generation(self.root): dict(self.root)}
            self.events = deque(maxlen=EVENT_LIMIT)
            self.generation_admissions = 1
            self.retired_generations = 0
            self.history_evictions = 0
            self.incomplete_samples = 0
            self.pidfd_signals = 0
            self._root_identity_invalidated = False

        def _event(self, kind, **values):
            self.events.append(dict(kind=kind, **values))

        def _remember(self, row):
            key = _generation(row)
            if key not in self.history:
                self.generation_admissions += 1
                self.history[key] = dict(row)
                self._event("generation_admitted", identity=dict(row))
            else:
                self.history.move_to_end(key)
            while len(self.history) > HISTORY_LIMIT:
                oldest = next(iter(self.history))
                if oldest == _generation(self.root):
                    self.history.move_to_end(oldest)
                    oldest = next(iter(self.history))
                self.history.pop(oldest)
                self.history_evictions += 1

        def _owned(self, row):
            core.need(_exact(self.root) == self._original_root,
                      "Original root identity must remain exact and immutable")
            if row["pid"] == self.root["pid"] and _exact(row) != self._original_root:
                self._root_identity_invalidated = True
                self._event("reused_root_numeric_pid_excluded", identity=dict(row))
                return False
            if row["pgid"] != self.root["pgid"] or row["session"] != self.root["session"]:
                old = self.history.get(_generation(row))
                if old is not None:
                    self._event("owned_generation_left_session_or_group", identity=dict(row))
                    raise RuntimeError("Known owned generation left its session/group")
                self._event("stale_foreign_enumeration_excluded", identity=dict(row))
                return False
            if row["start_ticks"] < self.root["start_ticks"]:
                self._event("predating_generation_excluded", identity=dict(row))
                return False
            return True

        def capture(self):
            core.need(_exact(self.root) == self._original_root,
                      "Original root identity must remain exact and immutable")
            try:
                current_root = core.identity(self.root["pid"])
            except (FileNotFoundError, ProcessLookupError):
                current_root = None  # Original orphan descendants stay owned.
            if current_root is not None and _exact(current_root) != self._original_root:
                self._root_identity_invalidated = True
                self._event("original_root_identity_replaced_group_excluded",
                            identity=dict(current_root))
            if self._root_identity_invalidated:
                # A recycled PGID/SID can name an unrelated fresh session. Its
                # children must not be admitted just because numeric IDs match.
                self._last_capture = {}
                return {}
            members = {}
            for pid in core.group_pids(self.root["pgid"]):
                try:
                    row = core.identity(pid)
                except (FileNotFoundError, ProcessLookupError):
                    continue
                if not self._owned(row):
                    continue
                members[pid] = dict(row)
            # One bounded end probe closes the whole-group lifetime race between
            # initial leader validation and the separate member identity reads.
            try:
                end_root = core.identity(self.root["pid"])
            except (FileNotFoundError, ProcessLookupError):
                end_root = None
            if end_root is not None and _exact(end_root) != self._original_root:
                self._root_identity_invalidated = True
                self._event("original_root_replaced_during_capture_group_excluded",
                            identity=dict(end_root))
            if self._root_identity_invalidated:
                self._last_capture = {}
                return {}
            for row in members.values():
                self._remember(row)
            current = {_generation(row): row for row in members.values()}
            for key, row in self._last_capture.items():
                if key not in current:
                    self.retired_generations += 1
                    self._event("generation_retired_from_cohort", identity=dict(row))
            self._last_capture = current
            return members

        def _pss(self, pid):
            memory = {}
            for line in core._read(Path("/proc") / str(pid) / "smaps_rollup").splitlines():
                fields = line.split()
                if fields and fields[0] in ("Pss:", "Rss:"):
                    core.need(len(fields) == 3 and fields[2] == "kB",
                              "Unknown proportional memory unit")
                    memory[fields[0][:-1]] = int(fields[1]) * 1024
            core.need(set(memory) == {"Pss", "Rss"}
                      and 0 <= memory["Pss"] <= memory["Rss"],
                      "Incomplete proportional memory receipt")
            return dict(accounted_bytes=memory["Pss"], rss_bytes=memory["Rss"],
                        method="smaps_rollup_PSS")

        def _measure_one(self, before):
            pid = before["pid"]
            # A single bounded retry confirms PSS when the first read fails.
            # Persistent RSS-only readings are diagnostics, never hard-kill input.
            first_error = None
            if before["state"] == "Z":
                measured = dict(accounted_bytes=0, rss_bytes=0,
                                method="zombie_zero_resident")
            else:
                measured = None
                for attempt in range(2):
                    try:
                        current = core.identity(pid)
                        if not core.same(current, before):
                            return None, "generation_changed_before_memory_read"
                        measured = self._pss(pid)
                        if attempt:
                            measured["PSS_confirmation_after_retry"] = True
                        break
                    except (FileNotFoundError, ProcessLookupError):
                        return None, "generation_exited_during_memory_read"
                    except (OSError, ValueError, RuntimeError) as exc:
                        first_error = type(exc).__name__
                if measured is None:
                    try:
                        rss = int(core._read(Path("/proc") / str(pid) / "statm").split()[1])
                        rss *= os.sysconf("SC_PAGE_SIZE")
                        core.need(rss >= 0, "Negative RSS receipt")
                        measured = dict(accounted_bytes=0, rss_bytes=rss,
                            method="RSS_diagnostic_unconfirmed_PSS",
                            unconfirmed_RSS_upper_bound_bytes=rss,
                            fallback_reason=first_error)
                    except (FileNotFoundError, ProcessLookupError):
                        return None, "generation_exited_during_RSS_diagnostic"
                    except (OSError, ValueError, RuntimeError, IndexError) as exc:
                        measured = dict(accounted_bytes=0, rss_bytes=0,
                            method="unreadable_memory_no_credit",
                            fallback_reason=type(exc).__name__)
            try:
                after = core.identity(pid)
            except (FileNotFoundError, ProcessLookupError):
                return None, "generation_exited_after_memory_read"
            if not core.same(after, before):
                return None, "generation_changed_after_memory_read"
            if after["state"] == "Z":
                measured = dict(accounted_bytes=0, rss_bytes=0,
                                method="zombie_zero_resident")
            return dict(before, **measured, state_after=after["state"]), None

        def measure(self):
            initial = self.capture()
            rows = {}
            invalidated = []
            for before in initial.values():
                row, reason = self._measure_one(before)
                if row is None:
                    invalidated.append(dict(identity=dict(before), reason=reason))
                else:
                    rows[_generation(before)] = row
            final = self.capture()
            first_keys = {_generation(row) for row in initial.values()}
            final_keys = {_generation(row) for row in final.values()}
            # Drop retired generations even when their earlier read succeeded.
            rows = {key: row for key, row in rows.items() if key in final_keys}
            missing = final_keys - set(rows)
            new = final_keys - first_keys
            retired = first_keys - final_keys
            uncertain_rows = [row for row in rows.values() if row["method"] in
                              ("RSS_diagnostic_unconfirmed_PSS", "unreadable_memory_no_credit")]
            complete = not (invalidated or missing or new or retired or uncertain_rows)
            total = sum(row["accounted_bytes"] for row in rows.values())
            rss = sum(row["rss_bytes"] for row in rows.values())
            fallback = bool(uncertain_rows)
            self.samples += 1
            self.fallback_samples += int(fallback)
            self.incomplete_samples += int(not complete)
            self.peak_accounted_bytes = max(self.peak_accounted_bytes, total)
            self.peak_summed_RSS_bytes = max(self.peak_summed_RSS_bytes, rss)
            self.latest = dict(
                accounted_bytes=total, enforcement_PSS_bytes=total,
                measured_PSS_bytes=total, summed_RSS_bytes=rss,
                conservative_RSS_fallback=fallback,
                complete=complete, headroom_credit_bytes=total if complete else 0,
                accounted_bytes_are_partial_PSS_observation=not complete,
                reservation_policy="retain_full_task_cap",
                RSS_only_excess_requires_PSS_confirmation=True,
                sampling_policy="one_finite_initial_cohort_one_end_snapshot",
                members=list(rows.values()),
                invalidated_generations=invalidated,
                unmeasured_new_generations=[dict(final[pid]) for pid, start in sorted(missing)
                                            if final[pid]["start_ticks"] == start],
                retired_generation_count=len(retired),
                new_generation_count=len(new))
            return total

        def _signal(self, sig):
            # Never killpg: each pidfd binds the signal to one verified process
            # generation, including across numeric PID reuse after this check.
            for row in self.capture().values():
                descriptor = None
                try:
                    descriptor = _pidfd_open(row["pid"])
                    current = core.identity(row["pid"])
                    if not core.same(current, row) or not self._owned(current):
                        self._event("signal_skipped_changed_generation", identity=dict(row))
                        continue
                    _pidfd_signal(descriptor, sig)
                    self.pidfd_signals += 1
                except (FileNotFoundError, ProcessLookupError):
                    continue
                except OSError as exc:
                    if exc.errno == errno.ESRCH:
                        continue
                    raise
                finally:
                    if descriptor is not None:
                        os.close(descriptor)

        def _reap_adopted(self, direct_pid):
            for pid, row in self.capture().items():
                if pid == direct_pid or row["ppid"] != os.getpid():
                    continue
                try:
                    current = core.identity(pid)
                    if not core.same(current, row):
                        continue
                    waited, status = os.waitpid(pid, os.WNOHANG)
                except (FileNotFoundError, ProcessLookupError, ChildProcessError):
                    continue
                if waited:
                    self.adopted_waits.append(dict(identity=row, actual_waitpid=True,
                        raw_wait_status=status, actual_returncode=os.waitstatus_to_exitcode(status)))

        def receipt(self):
            return dict(
                schema="workflow_v8_finite_generation_PSS_supervision_v1",
                root_identity=dict(self.root),
                memory_metric="validated_owned_smaps_rollup_PSS_RSS_diagnostic_only",
                memory_limit_field="max_rss_bytes",
                peak_accounted_bytes=self.peak_accounted_bytes,
                peak_summed_RSS_bytes=self.peak_summed_RSS_bytes,
                memory_samples=self.samples, fallback_samples=self.fallback_samples,
                incomplete_samples=self.incomplete_samples, last_memory_sample=self.latest,
                owned_identity_history=list(self.history.values()),
                identity_history_limit=HISTORY_LIMIT, event_limit=EVENT_LIMIT,
                generation_admissions=self.generation_admissions,
                retired_generations=self.retired_generations,
                history_evictions=self.history_evictions,
                recent_identity_events=list(self.events),
                direct_Popen_wait=self.direct_wait,
                actual_adopted_child_waits=self.adopted_waits,
                cleanup_errors=self.cleanup_errors,
                signal_method="verified_generation_pidfd",
                pidfd_signals=self.pidfd_signals,
                original_root_identity_invalidated=self._root_identity_invalidated,
                periodic_proc_sampling_is_not_atomic_kernel_group_limit=True)

    FinitePSSOwnedGroup.__name__ = "FinitePSSOwnedGroup"
    return FinitePSSOwnedGroup
