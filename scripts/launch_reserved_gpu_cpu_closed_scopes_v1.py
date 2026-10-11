#!/usr/bin/env python3
"""Use the existing GPU lease launcher with certified CPU/GPU storage credits."""
from pathlib import Path
import hashlib
from arggnn import project_storage_cpu_closed_scopes_v1 as storage

SOURCE = "scripts/launch_reserved_gpu_cpu_closed_scopes_v1.py"
ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
ORIGINAL = ROOT / "scripts/launch_reserved_gpu_full_storage_v1.py"
ORIGINAL_SHA256 = "d7e5b2bcb464b9d3863b4f644e69aaa1c73c18aaa2550e44ecd021e2199f6dff"

def main():
    if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("Pinned original GPU lease launcher changed")
    # Only accounting labels change in the pinned launch implementation.
    # GPU selection, lease locking, process propagation and wait logic are intact.
    text = ORIGINAL.read_text().replace(
        '"full_reserved_no_closed_scope_credit"', '"journal_cpu_completed_scopes_credit_v1"')
    namespace = {"__name__": "_arggnn_original_gpu_lease_launcher", "__file__": str(ORIGINAL)}
    exec(compile(text, str(ORIGINAL), "exec"), namespace)
    def credited_check(record, source_sha, launcher_sha, *, required_write_bytes=0):
        if (record["budget_components"]["implementation"]["sha256"] != source_sha
                or record["budget_components"]["launcher"]["sha256"] != launcher_sha):
            raise ValueError("Original core and lease dependencies must match CLI pins")
        actual = namespace["budget"].v4.reference(ROOT, SOURCE)
        if actual not in record["reservation"]["consumer_bindings"]:
            raise ValueError("Actual credited GPU launcher must be reserved")
        account = storage.check_storage_locked_v1(record, boundary=True,
                                                  required_write_bytes=required_write_bytes)
        return dict(account, actual_launcher_ref=actual)
    namespace["SOURCE"] = SOURCE
    namespace["full_storage_check_locked"] = credited_check
    return namespace["main"]()

if __name__ == "__main__":
    raise SystemExit(main())
