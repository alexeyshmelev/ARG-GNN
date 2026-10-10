"""Read-only source/diff audit. Does not open metadata, arrays or live state."""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
sha = lambda path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
BASELINE = ROOT / "research/reviews/gpu_pool_research_v3_author/preserved_source_baseline.json"
baseline = json.loads(BASELINE.read_text())
changed = {path: {"before": digest, "after": sha(path)}
           for path, digest in baseline.items() if sha(path) != digest}
assert not changed, changed
BOOT_OLD = "src/arggnn/anchor_compact_cuda_bootstrap_v1.py"
BOOT_NEW = "src/arggnn/anchor_compact_cuda_bootstrap_v2.py"
POOL_OLD = "scripts/gpu_pool_research_v2.py"
POOL_NEW = "scripts/gpu_pool_research_v3.py"
old_config = json.loads((ROOT / "configs/anchor_compact_cuda_bootstrap_v1.json").read_text())
new_config = json.loads((ROOT / "configs/anchor_compact_cuda_bootstrap_v2.json").read_text())
adapter = json.loads((ROOT / "configs/anchor_compact_actual_bank_v1.json").read_text())
expected_metadata = {adapter[key]["path"]: adapter[key]["sha256"]
                     for key in ("prepared_manifest", "observed_manifest")}
assert new_config["metadata_manifest_sha256"] == expected_metadata
assert new_config["default_bound_file_limit_bytes"] == 1048576
assert new_config["metadata_manifest_limit_bytes"] == 67108864
config_without_repair = dict(new_config)
for key in ("metadata_manifest_sha256", "default_bound_file_limit_bytes", "metadata_manifest_limit_bytes"):
    config_without_repair.pop(key)
config_without_repair["kind"] = old_config["kind"]
assert config_without_repair == old_config
hash_identities = {
    sha("configs/anchor_compact_cuda_bootstrap_v2.json"): sha("configs/anchor_compact_cuda_bootstrap_v1.json"),
    sha("configs/gpu_pool_research_v3.json"): sha("configs/gpu_pool_research_v2.json"),
    sha(BOOT_NEW): sha(BOOT_OLD),
}
def normalize(value):
    if isinstance(value, str):
        value = value.replace("anchor_compact_cuda_bootstrap_v2", "anchor_compact_cuda_bootstrap_v1")
        value = value.replace("anchor_compact_cuda_bootstrap_protocol_v2", "anchor_compact_cuda_bootstrap_protocol_v1")
        value = value.replace("gpu_pool_research_v3", "gpu_pool_research_v2")
        return hash_identities.get(value, value)
    if isinstance(value, dict):
        return {normalize(key): normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value
assert normalize(json.loads((ROOT / "configs/gpu_pool_research_v3.json").read_text())) == json.loads(
    (ROOT / "configs/gpu_pool_research_v2.json").read_text())
class Normalize(ast.NodeTransformer):
    def visit_Constant(self, node):
        node.value = normalize(node.value)
        return node
def tree(path):
    source = (ROOT / path).read_text()
    compile(source, str(ROOT / path), "exec")
    return ast.parse(source)
old_tree, new_tree = tree(BOOT_OLD), tree(BOOT_NEW)
old_helpers = {node.name: ast.dump(node, include_attributes=False)
               for node in old_tree.body if isinstance(node, ast.FunctionDef)}
new_helpers = {node.name: ast.dump(Normalize().visit(node), include_attributes=False)
               for node in new_tree.body if isinstance(node, ast.FunctionDef)}
changed_helpers = [name for name in old_helpers if old_helpers[name] != new_helpers[name]]
assert set(old_helpers) == set(new_helpers) and changed_helpers == ["bound_bytes"]
added_globals = {"DEFAULT_BOUND_FILE_LIMIT_BYTES", "METADATA_MANIFEST_LIMIT_BYTES", "METADATA_MANIFEST_SHA256"}
def without_repair(module, remove_globals=False):
    module.body = [node for node in module.body
                   if not (isinstance(node, ast.FunctionDef) and node.name == "bound_bytes")
                   and not (remove_globals and isinstance(node, ast.Assign)
                            and any(isinstance(target, ast.Name) and target.id in added_globals
                                    for target in node.targets))]
    return ast.dump(Normalize().visit(module), include_attributes=False)
assert without_repair(tree(BOOT_OLD)) == without_repair(tree(BOOT_NEW), remove_globals=True)
assert ast.dump(tree(POOL_OLD), include_attributes=False) == ast.dump(Normalize().visit(tree(POOL_NEW)), include_attributes=False)
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("source_only_pool_v3_audit", ROOT / POOL_NEW)
pool_extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool_extension)
configuration = pool_extension.verify_bindings()
assert "torch" not in sys.modules
template = json.loads((ROOT / "research/reviews/anchor_compact_cuda_bootstrap_v2_author/root_permit_template.json").read_text())
assert template["bootstrap_source"] == configuration["bootstrap_worker"]["source"]
assert template["bootstrap_config"] == configuration["bootstrap_worker"]["config"]
assert template["pool_extension"]["source"]["sha256"] == sha(POOL_NEW)
assert template["pool_extension"]["config"]["sha256"] == sha("configs/gpu_pool_research_v3.json")
assert template["status"] == "inactive_template_root_must_issue_fresh_receipt"
print(json.dumps({
    "kind": "bootstrap_v2_pool_v3_source_fidelity_audit",
    "preserved_source_and_review_files": len(baseline),
    "changed_frozen_files": changed,
    "changed_bootstrap_helpers_after_identity_normalization": changed_helpers,
    "unchanged_bootstrap_helpers": sorted(set(old_helpers) - {"bound_bytes"}),
    "bootstrap_module_AST_identical_except_bound_bytes_added_caps_and_version_identities": True,
    "pool_module_AST_identical_except_version_identities": True,
    "bootstrap_config_identical_except_version_identity_and_exact_metadata_caps": True,
    "pool_config_identical_except_exact_version_source_config_pins": True,
    "scientific_source_config_math_and_tolerances_unchanged": True,
    "actual_metadata_or_arrays_or_live_state_opened": False,
    "torch_imported": False,
    "exact_original_metadata_pins": expected_metadata,
    "source_sha256": {path: sha(path) for path in [
        BOOT_NEW, "configs/anchor_compact_cuda_bootstrap_v2.json",
        POOL_NEW, "configs/gpu_pool_research_v3.json"]},
}, indent=2, sort_keys=True))
