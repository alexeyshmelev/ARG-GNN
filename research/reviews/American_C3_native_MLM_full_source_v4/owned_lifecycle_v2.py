
"""Targeted owned-process tracking with pidfd-only signaling; no numeric group signals."""
from pathlib import Path
import os,signal
def _stat(pid):
    try:
        row=Path("/proc/%d/stat"%pid).read_text().rsplit(")",1)[1].split()
        return dict(pid=pid,state=row[0],ppid=int(row[1]),group=int(row[2]),session=int(row[3]),start=int(row[19]))
    except FileNotFoundError:return None
def _children(pid):
    result=set()
    try:
        for task in Path("/proc/%d/task"%pid).iterdir():
            try:result.update(int(p) for p in (task/"children").read_text().split())
            except FileNotFoundError:pass
    except FileNotFoundError:pass
    return sorted(result)
def _same(actual,expected):
    return actual is not None and expected is not None and (actual["pid"],actual["start"])==(expected.get("pid"),expected.get("start"))
def signal_identity(expected,signum,*,read_stat=_stat,open_fd=None,send_fd=None,close_fd=None):
    """An opened pidfd never retargets a reused PID, including after the second stat."""
    open_fd=os.pidfd_open if open_fd is None else open_fd
    send_fd=signal.pidfd_send_signal if send_fd is None else send_fd
    close_fd=os.close if close_fd is None else close_fd
    if not _same(read_stat(expected["pid"]),expected):return False
    try:fd=open_fd(expected["pid"],0)
    except ProcessLookupError:return False
    try:
        if not _same(read_stat(expected["pid"]),expected):return False
        try:send_fd(fd,signum,None,0)
        except ProcessLookupError:return False
        return True
    finally:close_fd(fd)
class OwnedTree:
    def __init__(self,leader,*,owner=None,read_stat=_stat,read_children=_children):
        self.owner=os.getpid() if owner is None else owner;self.read_stat=read_stat;self.read_children=read_children
        first=read_stat(leader)
        if first is None or first["ppid"]!=self.owner or first["group"]!=leader or first["session"]!=leader:
            raise ValueError("Exact direct new-session leader identity required")
        self.leader=leader;self.known={leader:first};self.reused=set()
    def refresh(self):
        # The caller starts with no unrelated children and is a subreaper.
        # Unknown PIDs enter only through a currently verified owned parent,
        # or as actual direct adopted children of this supervisor.
        queue=[(p,self.owner) for p in self.read_children(self.owner)]
        queue.extend((p,None) for p in self.known);visited=set()
        while queue:
            pid,parent=queue.pop()
            if pid in visited:continue
            visited.add(pid);row=self.read_stat(pid);old=self.known.get(pid)
            if row is None:continue
            if old is not None and not _same(row,old):
                self.reused.add(pid);continue
            if old is None and (parent is None or row["ppid"]!=parent):continue
            if old is None and parent!=self.owner and not _same(self.read_stat(parent),self.known.get(parent,{})):continue
            self.known[pid]=row
            # Recheck parent identity around its targeted child enumeration.
            children=self.read_children(pid)
            if not _same(self.read_stat(pid),row):continue
            queue.extend((child,pid) for child in children)
        return self
    def signal_all(self,signum):
        self.refresh()
        for row in list(self.known.values()):signal_identity(row,signum,read_stat=self.read_stat)
    def live(self):
        self.refresh()
        result=[]
        for p,row in self.known.items():
            actual=self.read_stat(p)
            if _same(actual,row) and actual["state"] not in ("Z","X"):result.append(p)
        return result
    def identities_absent(self):
        self.refresh()
        return all(not _same(self.read_stat(p),row) for p,row in self.known.items())
    def evidence(self):
        return dict(owned_identities=[dict(pid=p,start=row["start"]) for p,row in sorted(self.known.items())],
            reused_numeric_PIDs_excluded=sorted(self.reused),signals_use_pidfd_only=True,no_numeric_group_signals=True)
