# Bounded read-only observations

Run from /workspace-SR003.nfs2/ARG-GNN on anogena. These commands use only stdlib filesystem metadata and bounded existing log-prefix reads; they do not import project or guard modules.

```sh
.runtime/env/bin/python -B - <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import json, os, stat
root = Path("/workspace-SR003.nfs2/ARG-GNN")
run = root / "experiments/results/American_C3_native_MLM_full30mb_mlm_fit_v8/amended_v12_attempt1"
cap = root / "research/reviews/American_C3_native_MLM_full_fit_v8/admitted_mlm_v1/worker_aux/amended_v12_attempt1/terminal_capture"
result = {"observed_UTC": datetime.now(timezone.utc).isoformat(),
          "parent_completed_exists": (run / "completed.json").exists(),
          "child_completed_exists": (run / "mlm/completed.json").exists(),
          "checkpoint_metadata_only": []}
entries = sorted((run / "mlm/checkpoints").iterdir())
if len(entries) > 16:
    raise RuntimeError("Exact checkpoint directory unexpectedly exceeds16entries")
for p in entries:
    s = p.lstat()
    result["checkpoint_metadata_only"].append(
        {"name": p.name, "mode": stat.S_IFMT(s.st_mode),
         "bytes": s.st_size, "mtime_ns": s.st_mtime_ns,
         "symlink": stat.S_ISLNK(s.st_mode)})
result["terminal_prefixes"] = {}
for name in ("native.wait", "launcher.wait", "outer.wait", "native.tail", "outer.tail"):
    p = cap / name
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        s = os.fstat(fd)
        prefix = os.pread(fd, 256, 0)
    finally:
        os.close(fd)
    result["terminal_prefixes"][name] = {
        "bytes": s.st_size, "mtime_ns": s.st_mtime_ns,
        "first256_all_zero": not prefix.strip(b"\0"),
        "prefix_text": prefix.strip(b"\0").decode("utf-8", errors="replace")}
print(json.dumps(result, indent=2))
PY
```

The preallocated tail sizes are not current output byte counts. A zero first256bytes is a bounded prefix observation, not proof that every byte is zero. Framed wait records become authoritative only under the existing checksummed terminal reader after the actual wait. Do not use a timestamp/filename alone as verified optimizer state.

For live console evidence, root should poll its already-owned exec session48073, with a bounded output budget. The relay is deliberately sparse after its first eight banners, so unchanged console output alone does not diagnose a hang.
