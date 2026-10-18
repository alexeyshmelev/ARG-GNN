"""V7-only capture repair for never-admitted process enumeration candidates."""
from __future__ import annotations


def capture_members(tracker, *, group_pids, identity, same, need):
    """Commit only current exact members; retain all known-identity failures.

    getpgid enumeration and /proc identity reads are separate kernel snapshots.
    A never-tracked candidate that no longer belongs is not an owned process.
    Previously tracked PID/start/group/session identities remain fail-closed.
    """
    members = {}
    for pid in group_pids(tracker.root["pgid"]):
        try:
            row = identity(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        known = tracker.history.get(pid)
        if known is not None:
            need(same(row, known), "Owned PID/start/session/group identity changed or was reused")
        elif row["pgid"] != tracker.root["pgid"] or row["session"] != tracker.root["session"]:
            # Stale enumeration only: do not remember, measure or signal this row.
            continue
        need(row["pgid"] == tracker.root["pgid"] and row["session"] == tracker.root["session"],
             "Owned session/group changed")
        need(row["start_ticks"] >= tracker.root["start_ticks"],
             "Group member predates the actual owned root")
        if known is None:
            tracker.history[pid] = row
        members[pid] = row
    return members


def repaired_owned_group_class(core):
    """ROOT binds the returned subclass only inside the new v7 private core."""
    class MembershipCheckedOwnedGroup(core.OwnedGroup):
        def capture(self):
            return capture_members(self, group_pids=core.group_pids, identity=core.identity,
                                   same=core.same, need=core.need)
    return MembershipCheckedOwnedGroup
