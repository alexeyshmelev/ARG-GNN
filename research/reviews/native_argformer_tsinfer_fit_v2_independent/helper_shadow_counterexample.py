"""Negative helper gate fixture. No real helper installation or module import."""
import hashlib, importlib.util, importlib.machinery, json, marshal, struct, sys
from pathlib import Path
from unittest.mock import patch
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REVIEW=ROOT/"research/reviews/native_argformer_tsinfer_fit_v2_independent"
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location("_helper_gate_independent",ROOT/"src/arggnn/native_argformer_tsinfer_fit_v2.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
fixture=REVIEW/"helper_fixture_root";fixture.mkdir()
directory=fixture/".runtime/isolated_helpers/threadpoolctl_case";directory.mkdir(parents=True)
original=ROOT/".runtime/env/lib/python3.11/site-packages/threadpoolctl.py"
(directory/"threadpoolctl.py").write_bytes(original.read_bytes())
(directory/"numpy.pyc").write_bytes(importlib.util.MAGIC_NUMBER+struct.pack("<III",0,0,0)+marshal.dumps(compile('raise RuntimeError("independent shadow fixture deliberately rejects import")',"<shadow-fixture>","exec")))
wheel=fixture/"metadata/threadpoolctl.whl";wheel.parent.mkdir();wheel.write_bytes(b"handmade wheel-reference bytes only")
inventory=fixture/"metadata/inventory.json"
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
inventory.write_text(json.dumps({"directory":str(directory),"files_sha256":{p.name:digest(p) for p in directory.iterdir()}},sort_keys=True))
def record(p):return {"path":str(p),"sha256":digest(p)}
helper={"module":"threadpoolctl","version":"3.6.0","implementation":record(directory/"threadpoolctl.py"),"wheel":record(wheel),"directory_inventory":record(inventory)}
with patch.object(m,"ROOT",fixture):
    accepted=m.additive_helper_paths({"additive_helpers":[helper]})
resolved=importlib.machinery.PathFinder.find_spec("numpy",accepted)
assert accepted==[str(directory)] and resolved.origin==str(directory/"numpy.pyc") and isinstance(resolved.loader,importlib.machinery.SourcelessFileLoader)
result={"kind":"native_argformer_tsinfer_fit_v2_independent_helper_shadow_counterexample","status":"reproduced","id":"SOURCLESS_BYTECODE_SHADOW_BYPASSES_HELPER_ALLOWLIST","source":"src/arggnn/native_argformer_tsinfer_fit_v2.py","source_lines":[367,368,369,370,371,372],"fixture_root_relocated_only":True,"production_runtime_or_isolated_helper_mutated":False,"scientific_model_or_shadow_module_executed":False,"accepted_helper_paths":accepted,"resolver_origin":resolved.origin,"resolver_loader":type(resolved.loader).__name__,"observation":"Complete helper inventory with exact module/wheel/hash references accepts numpy.pyc because only .py/.so filenames are forbidden. CPython resolves numpy to that sourceless bytecode before native site-packages when the admitted helper directory is placed on PYTHONPATH.","fix_requirement":"Enforce an exact allowed helper-directory inventory including executable import suffixes, or load threadpoolctl by its bound file without adding its directory to general scientific import search.","implementation":m.reference(m.SCRIPT),"config":m.reference(m.CONFIG),"test_source":{"path":str(Path(__file__).relative_to(ROOT)),"sha256":digest(Path(__file__))}}
with (REVIEW/"helper_shadow_counterexample.json").open("x") as f:json.dump(result,f,sort_keys=True,indent=2)
print(json.dumps(result,indent=2))
