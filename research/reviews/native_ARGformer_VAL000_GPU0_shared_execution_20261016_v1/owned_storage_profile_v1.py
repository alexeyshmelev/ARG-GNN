"""Periodic own-scope metadata failstop; no payload reads or import-time I/O.
No OS quota or atomic ceiling: writes may overshoot between bounded polls.
Existing outer supervisor owns any resulting verified-identity stop and drain.
"""
from pathlib import Path
import errno,os,stat,time
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
_NEXT_SCAN=0.
MAX_NODES=16384
MAX_DEPTH=24
def require(ok,message):
    if not ok:raise ValueError(message)
def _sizes(root,*,allow_checkpoint_alias=False):
    """Descriptor-relative no-follow metadata traversal; no file is opened."""
    root=Path(root)
    require(root.is_relative_to(ROOT) and root.resolve()==root and ".." not in root.parts,"Own canonical project scope")
    records=[];count=0
    try:fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    except FileNotFoundError:return records
    def visit(directory,parts):
        nonlocal count
        require(len(parts)<=MAX_DEPTH,"Bounded owned scope depth")
        with os.scandir(directory) as entries:
            names=[e.name for e in entries]
        require(len(names)<=MAX_NODES,"Bounded owned directory entries")
        for name in names:
            count+=1;require(count<=MAX_NODES,"Bounded own metadata census")
            try:s=os.stat(name,dir_fd=directory,follow_symlinks=False)
            except FileNotFoundError:continue  # Normal Composer retention race.
            rel=parts+(name,)
            if stat.S_ISDIR(s.st_mode):
                records.append((rel,4096,"directory"))
                try:child=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=directory)
                except FileNotFoundError:continue
                try:visit(child,rel)
                finally:os.close(child)
            elif stat.S_ISREG(s.st_mode):
                require(s.st_nlink==1,"Owned scope hardlink forbidden")
                records.append((rel,max(s.st_size,s.st_blocks*512)+4096,"regular"))
            elif stat.S_ISLNK(s.st_mode):
                require(allow_checkpoint_alias and "checkpoints" in parts and name=="latest-rank0.pt"
                    and s.st_size<=256,"Only original own Composer alias is allowed; never followed")
                records.append((rel,s.st_size+4096,"alias"))
            else:raise ValueError("Unexpected nonregular owned scope entry")
    try:visit(fd,())
    finally:os.close(fd)
    return records
def validate_snapshot_v1(snapshot,caps):
    require(caps["reservation_cap_bytes"]==7*2**30 and caps["max_checkpoint_bytes"]==3*2**30
        and caps["max_checkpoint_files"]==2 and caps["cache_allowance_bytes"]==512*2**20
        and caps["temp_allowance_bytes"]+caps["child_artifact_allowance_bytes"]==256*2**20
        and caps["outer_auxiliary_cap_bytes"]==128*2**20
        and caps["root_packet_overhead_cap_bytes"]==4*2**20,"Exact explicitly reviewed smaller operational profile")
    for key in ("checkpoint","cache","misc","auxiliary","packet"):
        require(type(snapshot[key]) is int and snapshot[key]>=0,"Nonnegative actual owned metadata bytes")
    require(snapshot["checkpoint_files"]<=2
        and snapshot["largest_checkpoint_file_bytes"]<=3*2**30
        and snapshot["checkpoint"]<=caps["checkpoint_live_allowance_bytes"],"Owned checkpoint transient/per-file cap exceeded")
    require(snapshot["cache"]<=512*2**20,"Combined own cache cap exceeded")
    require(snapshot["misc"]<=256*2**20,"Combined own TMP/artifact cap exceeded")
    require(snapshot["auxiliary"]<=128*2**20 and snapshot["packet"]<=4*2**20,"Own outer metadata/auxiliary cap exceeded")
    require(sum(snapshot[k] for k in ("checkpoint","cache","misc","auxiliary","packet"))<=7*2**30,
        "Own whole-phase7GiB metadata cap exceeded")
    return snapshot
def check_profile_v1(packet,output,caps):
    global _NEXT_SCAN
    now=time.monotonic()
    if now<_NEXT_SCAN:return None
    _NEXT_SCAN=now+1.
    packet=Path(packet);output=Path(output);output=output if output.is_absolute() else ROOT/output
    phase=caps["phase"]
    tmp=ROOT/(".runtime/tmp/c3val0shm/m" if phase=="mlm" else ".runtime/tmp/c3val0shpost/"+("f" if phase=="fit" else "r"))
    snapshot=dict(checkpoint=0,cache=0,misc=0,auxiliary=0,packet=0,checkpoint_files=0,largest_checkpoint_file_bytes=0)
    for parts,n,kind in _sizes(output,allow_checkpoint_alias=True):
        key="checkpoint" if "checkpoints" in parts else ("cache" if "write_scopes" in parts else "misc")
        snapshot[key]+=n
        if key=="checkpoint" and kind=="regular" and parts[-1].endswith(".pt") and not parts[-1].startswith("latest"):
            snapshot["checkpoint_files"]+=1
            snapshot["largest_checkpoint_file_bytes"]=max(snapshot["largest_checkpoint_file_bytes"],n-4096)
    snapshot["misc"]+=sum(n for _,n,_ in _sizes(tmp))
    for parts,n,_ in _sizes(packet):
        snapshot["auxiliary" if parts[0]=="worker_aux" else "packet"]+=n
    return validate_snapshot_v1(snapshot,caps)
