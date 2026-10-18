"""UNPINNED review candidate. No live integration, signaling, or GPU code.

The caller supplies the exact known group and trusted member PIDs. Kernel PGID
lookups check completeness without reading unrelated command/environment/RAM.
PID/start/session identity is retained and checked around every measurement.
"""
from __future__ import annotations
import copy
import os
from pathlib import Path

LIMIT=1024*1024


class AccountingError(RuntimeError):
    pass


def _need(ok,message):
    if not ok:raise AccountingError(message)


def _read(path):
    with Path(path).open("rb") as stream:raw=stream.read(LIMIT+1)
    _need(len(raw)<=LIMIT,"Owned process metadata exceeded its bound")
    return raw.decode()


def _identity(pid):
    fields=_read(Path("/proc")/str(pid)/"stat").rsplit(")",1)[1].split()
    return dict(pid=pid,start_ticks=int(fields[19]),pgid=int(fields[2]),
                session=int(fields[3]),state=fields[0])


def _same(actual,expected):
    return all(actual[key]==expected[key] for key in ("pid","start_ticks","pgid","session"))


def _group_pids(pgid):
    # No unrelated cmdline, environment, status, mappings or memory is opened.
    members=set()
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():continue
        pid=int(path.name)
        try:observed=os.getpgid(pid)
        except ProcessLookupError:continue
        except PermissionError as exc:raise AccountingError("Cannot prove complete kernel group membership") from exc
        if observed==pgid:members.add(pid)
    return members


def capture_owned_group(pgid,trusted_pids):
    """Capture only an explicitly trusted, complete, fresh-session group."""
    _need(type(pgid) is int and pgid>0,"Exact owned PGID required")
    pids=set(trusted_pids)
    _need(all(type(pid) is int and pid>0 for pid in pids) and pgid in pids,
          "Trusted leader/member PID inventory required")
    _need(_group_pids(pgid)==pids,"Trusted owned-group inventory is incomplete or stale")
    identities={str(pid):_identity(pid) for pid in sorted(pids)}
    _need(all(row["pgid"]==pgid and row["session"]==pgid for row in identities.values()),
          "Every trusted member must retain the exact newly owned session/group")
    _need(identities[str(pgid)]["state"]!="Z","Capture requires a live known group leader")
    _need(_group_pids(pgid)==pids,"Group membership changed during capture")
    for row in identities.values():
        _need(_same(_identity(row["pid"]),row),"Owned PID identity changed during capture")
    return dict(schema="explicit_owned_PID_start_group_v1",pgid=pgid,members=identities)


def _pss_bytes(pid):
    values={}
    for line in _read(Path("/proc")/str(pid)/"smaps_rollup").splitlines():
        fields=line.split()
        if fields and fields[0] in ("Pss:","Rss:"):
            _need(len(fields)==3 and fields[2]=="kB","Known smaps_rollup memory units required")
            values[fields[0][:-1]]=int(fields[1])*1024
    _need(set(values)=={"Pss","Rss"} and 0<=values["Pss"]<=values["Rss"],
          "Complete proportional/resident memory fields required")
    return values


def _rss_bytes(pid):
    return int(_read(Path("/proc")/str(pid)/"statm").split()[1])*os.sysconf("SC_PAGE_SIZE")


def measure_owned_group(snapshot,*,unreadable="fail"):
    """PSS sum; unreadable live members either conservatively use RSS or fail.

    A retained map never silently absorbs a new PID or reused identity. The
    caller must authenticate a new complete capture when workers change.
    Zombies are explicitly retained as zombies with zero resident memory.
    This receipt never asserts kernel group absence or actual process waits.
    """
    _need(unreadable in ("fail","rss"),"Explicit unreadable-PSS policy required")
    _need(snapshot.get("schema")=="explicit_owned_PID_start_group_v1","Exact owned inventory required")
    pgid=snapshot["pgid"];members=snapshot["members"]
    expected={int(pid) for pid in members}
    _need(_group_pids(pgid)==expected,"Group membership changed; authenticate a fresh inventory")
    rows=[]
    for pid in sorted(expected):
        identity=members[str(pid)];before=_identity(pid)
        _need(_same(before,identity),"Owned PID/start/session changed before memory read")
        if before["state"]=="Z":
            row=dict(pid=pid,accounted_bytes=0,rss_bytes=0,method="zombie_zero_resident")
        else:
            try:
                memory=_pss_bytes(pid)
                row=dict(pid=pid,accounted_bytes=memory["Pss"],rss_bytes=memory["Rss"],method="smaps_rollup_PSS")
            except (OSError,ValueError,AccountingError) as exc:
                if unreadable=="fail":raise AccountingError("Owned live PSS unreadable; accounting fails closed") from exc
                rss=_rss_bytes(pid)  # Failure here also propagates; never zero-fill.
                row=dict(pid=pid,accounted_bytes=rss,rss_bytes=rss,method="conservative_RSS_fallback",
                         unreadable_reason=type(exc).__name__)
        after=_identity(pid)
        _need(_same(after,identity),"Owned PID/start/session changed after memory read")
        row.update(start_ticks=identity["start_ticks"],state_before=before["state"],state_after=after["state"])
        rows.append(row)
    _need(_group_pids(pgid)==expected,"Group changed during memory measurement")
    for pid in expected:
        _need(_same(_identity(pid),members[str(pid)]),"Owned identity changed after group measurement")
    return dict(schema="owned_group_proportional_memory_candidate_v1",owned_inventory=copy.deepcopy(snapshot),
        accounted_bytes=sum(row["accounted_bytes"] for row in rows),
        summed_RSS_bytes=sum(row["rss_bytes"] for row in rows),members=rows,
        conservative_fallback_used=any(row["method"]=="conservative_RSS_fallback" for row in rows),
        group_absence_asserted=False,scientific_wait_asserted=False)
