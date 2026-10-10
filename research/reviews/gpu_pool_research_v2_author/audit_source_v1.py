"""Reproduce source-only audit and dependency preservation checks; no live state."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
sha = lambda path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
baseline = json.loads((ROOT / "research/reviews/gpu_pool_research_v2_author/preserved_source_baseline.json").read_text())
changed = [path for path, digest in baseline.items() if sha(path) != digest]
assert not changed, changed
sources = [
    "scripts/gpu_pool_research_v2.py", "configs/gpu_pool_research_v2.json",
    "src/arggnn/anchor_compact_cuda_bootstrap_v1.py", "configs/anchor_compact_cuda_bootstrap_v1.json",
    "tests/test_gpu_pool_research_v2.py", "tests/test_anchor_compact_cuda_bootstrap_v1.py",
]
for path in sources:
    if path.endswith(".py"):
        compile((ROOT / path).read_text(), str(ROOT / path), "exec")
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("source_only_extension_audit", ROOT / sources[0])
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)
config = extension.verify_bindings()
assert "torch" not in sys.modules
assert config["bootstrap_worker"]["source"]["sha256"] == sha(sources[2])
bootstrap = json.loads((ROOT / sources[3]).read_text())
assert all(sha(path) == digest for path, digest in bootstrap["source_sha256"].items())
template = json.loads((ROOT / "research/reviews/anchor_compact_cuda_bootstrap_v1_author/root_permit_template.json").read_text())
assert template["bootstrap_source"] == config["bootstrap_worker"]["source"]
assert template["pool_extension"]["source"]["sha256"] == sha(sources[0])
assert template["pool_extension"]["config"]["sha256"] == sha(sources[1])
print(json.dumps({
    "preserved_source_count": len(baseline),
    "changed_frozen_sources": changed,
    "all_pins_match": True,
    "torch_imported": False,
    "source_sha256": {path: sha(path) for path in sources},
}, indent=2, sort_keys=True))
