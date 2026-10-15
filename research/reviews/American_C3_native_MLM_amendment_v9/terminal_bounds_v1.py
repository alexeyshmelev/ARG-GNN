
"""Terminal-only operational subcategory verification; no quota or live scan."""
from pathlib import Path
import os,stat
def measure(path):
    if not path.exists():return 0
    total=0;stack=[path]
    while stack:
        p=stack.pop();s=p.lstat()
        if stat.S_ISLNK(s.st_mode):total+=max(4096,s.st_size);continue
        if stat.S_ISDIR(s.st_mode):
            total+=4096;stack.extend(p.iterdir());continue
        if stat.S_ISREG(s.st_mode) or stat.S_ISSOCK(s.st_mode):total+=max(4096,s.st_size);continue
        raise ValueError("Unexpected owned terminal file type")
    return total
def verify(directory,tmp,caps):
    # Caller reaches this only after actual direct wait, adopted reaping/groupESRCH.
    parts=dict(checkpoints=measure(directory/"checkpoints"),caches=measure(directory/"write_scopes"),
        temp=measure(tmp))
    parts["other_child_artifacts"]=max(0,measure(directory)-parts["checkpoints"]-parts["caches"])
    limits=dict(checkpoints=caps["checkpoint_live_allowance_bytes"],caches=caps["cache_allowance_bytes"],
        temp=caps["temp_allowance_bytes"],other_child_artifacts=caps["child_artifact_allowance_bytes"])
    if any(parts[k]>limits[k] for k in parts):raise ValueError("Terminal operational category allowance exceeded; preserve all artifacts")
    return dict(bytes=parts,limits=limits,terminal_only=True,aggregate_OS_quota_claimed=False,
        transient_cache_workspace_bound_proven=False)
