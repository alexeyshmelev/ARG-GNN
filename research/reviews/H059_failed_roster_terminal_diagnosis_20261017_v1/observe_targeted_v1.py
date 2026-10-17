"""ROOT-requested targeted read-only observation; no signals, scientific IO or guards."""
import os,json,time,hashlib
from pathlib import Path
BASE=Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=BASE/"research/reviews/H059_failed_roster_terminal_diagnosis_20261017_v1"
INPUT="research/reviews/European_outer000_ROOT_actual_Relate64_20261017_v1/input.json"
INPUT_SHA="3561b5898cb6abc553b2a8bff4528baf02f6a2a02c416d9c696f7355376b7010"
LAUNCH="research/reviews/European_outer000_ROOT_actual_Relate64_20261017_v1/worker_aux/launch.json"
LAUNCH_SHA="a63b32e86b796355adcf2234f0bc90748cb8ec186ebd2383c58c53633e935c04"
UID=1000
ROOT_PID=3998053
ROOT_START=488962157
OBSERVED_ANCHORS={3998053:488962157,4036548:489505214}
KNOWN_GROUPS={3998053,4036548}

def read(pid,detail=False):
    try:
        s=Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()
        r=dict(pid=pid,state=s[0],ppid=int(s[1]),group=int(s[2]),session=int(s[3]),start=int(s[19]),uid=Path(f"/proc/{pid}").stat().st_uid)
        if detail:
            r["cmdline"]=Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0",b" ").decode(errors="replace")[:4096]
            uid_line=next(line for line in Path(f"/proc/{pid}/status").read_text().splitlines() if line.startswith("Uid:"))
            r["uids"]=list(map(int,uid_line.split()[1:]))
        return r
    except (FileNotFoundError,ProcessLookupError):return None

def same(a,b):
    return a is not None and b is not None and all(a[k]==b[k] for k in ("pid","start","uid"))

def observe():
    refs=[]
    for p,expected in ((INPUT,INPUT_SHA),(LAUNCH,LAUNCH_SHA)):
        got=hashlib.sha256((BASE/p).read_bytes()).hexdigest()
        if got!=expected:raise ValueError("Changed exact pinned operational input or launch")
        refs.append(dict(path=p,sha256=got))
    anchor_rows={p:read(p,True) for p in OBSERVED_ANCHORS}
    known={};queue=[];blocked=[];edges=[]
    for p,r in anchor_rows.items():
        if r is None:continue
        if r["start"]!=OBSERVED_ANCHORS[p] or r["uid"]!=UID or r["uids"]!=[UID]*4:
            blocked.append(dict(pid=p,reason="anchor_identity_or_uid_mismatch",actual=r));continue
        if p==ROOT_PID and (r["group"]!=p or r["session"]!=p or INPUT not in r["cmdline"] or INPUT_SHA not in r["cmdline"]):
            blocked.append(dict(pid=p,reason="root_scope_or_group_mismatch",actual=r));continue
        known[p]=r;queue.append(p)
    while queue:
        p=queue.pop();r=known[p]
        if not same(read(p),r):blocked.append(dict(parent=p,reason="identity_changed_before_children"));continue
        try:
            children=set()
            for task in Path(f"/proc/{p}/task").iterdir():
                try:children.update(map(int,(task/"children").read_text().split()))
                except (FileNotFoundError,ProcessLookupError):pass
        except (FileNotFoundError,ProcessLookupError):children=set()
        if not same(read(p),r):blocked.append(dict(parent=p,reason="identity_changed_after_children"));continue
        for c in sorted(children):
            cr=read(c,True)
            if cr is None:continue
            if cr["ppid"]!=p or cr["uid"]!=UID or cr["uids"]!=[UID]*4 or not same(read(p),r):
                blocked.append(dict(parent=p,child=c,reason="ancestry_or_uid_unstable"));continue
            edges.append(dict(parent=dict(pid=p,start=r["start"],uid=r["uid"]),child=dict(pid=c,start=cr["start"],uid=cr["uid"]),verified_current_direct_child=True))
            if c not in known:
                if len(known)>=4096:raise ValueError("Read-only observational bound exceeded")
                known[c]=cr;queue.append(c)
    groups=[];scan_errors=[]
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():continue
        try:r=read(int(p.name))
        except Exception as exc:
            scan_errors.append(dict(pid=int(p.name),error=type(exc).__name__+": "+str(exc)));continue
        if r and (r["group"] in KNOWN_GROUPS or r["session"] in KNOWN_GROUPS):
            groups.append(dict(**r,identity_in_authenticated_targeted_tree=any(same(r,k) for k in known.values())))
    return dict(schema="H059_failed_roster_targeted_readonly_observation_v1",observed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),observer_uid=os.getuid(),operational_refs=refs,anchor_rows=anchor_rows,targeted_rows=list(known.values()),verified_edges=edges,blocked_edges=blocked,exact_known_group_session_members=groups,stat_scan_errors=scan_errors,original_direct_wait="UNKNOWN",original_owned_drain="UNKNOWN",signals_sent=False,unrelated_fd_or_cmdline_scan=False,scientific_files_opened=False,original_guards_or_science_executed=False)

if __name__=="__main__":
    result=observe()
    target=OUT/"observation.current_readonly_v1.json"
    if target.exists():raise ValueError("Do not replace observation receipt")
    target.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(dict(observation_path=str(target.relative_to(BASE)),sha256=hashlib.sha256(target.read_bytes()).hexdigest(),targeted_identity_count=len(result["targeted_rows"]),known_group_session_member_count=len(result["exact_known_group_session_members"]),blocked_edges=result["blocked_edges"],stat_scan_errors=result["stat_scan_errors"],observed_utc=result["observed_utc"],signals_sent=False),sort_keys=True))
