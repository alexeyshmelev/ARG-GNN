"""Build explicit inactive source successors in this packet only."""
from __future__ import annotations
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=ROOT/"research/reviews/C3_isolated_boundary_forest_repair_20261015_v1"
TREE=OUT/"staged_tree"
OLD_REL="research/reviews/C3_GNN_prefetch_current980_root_20261015_v1"
NEW_REL="research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2"
OLD_PREF="research/reviews/C3_GNN_prefetch_provider_20261015_v1/prefetch_provider_v1.py"
NEW_PREF=NEW_REL+"/prefetch_provider_isolated_v2.py"
MODULES={name+"_v1":name+"_isolated_v2" for name in (
    "american_c3_gnn_observations","american_c3_gnn_dataset","american_c3_gnn_stream_train",
    "american_c3_gnn_run","american_c3_gnn_predict","american_c3_gnn_shared30mb","american_c3_gnn_selection")}
OLD_DISPATCH="src/arggnn/american_c3_gnn_prefetch_current980_exec_v1.py"
NEW_DISPATCH="src/arggnn/american_c3_gnn_prefetch_isolated980_exec_v2.py"
inventory=[];diffs=[]

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def file_ref(path):
    p=Path(path);p=p if p.is_absolute() else ROOT/p
    return dict(path=str(p.relative_to(ROOT)),sha256=sha_bytes(p.read_bytes()))
def intended_ref(target):
    return dict(path=target,sha256=sha_bytes((TREE/target).read_bytes()))
def transform(s):
    s=s.replace(OLD_REL,NEW_REL).replace(OLD_REL.replace("/","."),NEW_REL.replace("/","."))
    s=s.replace(OLD_PREF,NEW_PREF).replace(OLD_PREF[:-3].replace("/","."),NEW_PREF[:-3].replace("/","."))
    s=s.replace("american_c3_gnn_prefetch_current980_exec_v1","american_c3_gnn_prefetch_isolated980_exec_v2")
    s=s.replace("C3_GNN_prefetch_current980","C3_GNN_prefetch_isolated980")
    s=s.replace("prefetch-current980","prefetch-isolated980")
    s=s.replace("prefetch_current980","prefetch_isolated980")
    s=s.replace("c3gnnprefetch980","c3gnnprefetchisolated980")
    for old,new in MODULES.items():s=s.replace(old,new)
    return s
def stage(old,target,s,kind):
    ast.parse(s,filename=target) if target.endswith(".py") else json.loads(s)
    p=TREE/target;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
    original=(ROOT/old).read_text() if old else ""
    inventory.append(dict(original_ref=file_ref(old) if old else None,intended_ref=intended_ref(target),staged_ref=file_ref(p),change_kind=kind))
    diffs.extend(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile=old or "/dev/null",tofile=target))
def set_assignment(s,name,value):
    nodes=[n for n in ast.parse(s).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
    assert len(nodes)==1,name
    node=nodes[0];seg=ast.get_source_segment(s,node)
    assert s.count(seg)==1
    return s.replace(seg,name+"="+repr(value),1)

# Canonical scientific successors: one encoder change; all remaining changes are imports/source provenance.
for old,new in MODULES.items():
    src="src/arggnn/"+old+".py";target="src/arggnn/"+new+".py"
    if old=="american_c3_gnn_observations_v1":
        s=(OUT/"american_c3_gnn_observations_isolated_v2.staged.py").read_text();kind="isolated_sample_forest_encoding"
    else:s=transform((ROOT/src).read_text());kind="import_and_source_identity_only"
    stage(src,target,s,kind)

# Prefetch successor reuses exact original lifecycle/IPC helpers; only source bindings change.
s=transform((ROOT/OLD_PREF).read_text())
s=s.replace("from .abstract_ipc_bootstrap_v1 import ","from research.reviews.C3_GNN_prefetch_provider_20261015_v1.abstract_ipc_bootstrap_v1 import ")
s=s.replace("from .bounded_workers_v1 import ","from research.reviews.C3_GNN_prefetch_provider_20261015_v1.bounded_workers_v1 import ")
pins={p:intended_ref(p)["sha256"] for p in (
    "src/arggnn/american_c3_gnn_stream_train_isolated_v2.py",
    "src/arggnn/american_c3_gnn_dataset_isolated_v2.py",
    "src/arggnn/american_c3_gnn_observations_isolated_v2.py")}
s=set_assignment(s,"PINS",pins)
stage(OLD_PREF,NEW_PREF,s,"import_source_pin_and_package_location_only")

op_names=("metadata_contract_v1.py","persistent_workers_v1.py","persistent_provider_v1.py",
    "tracking_factory_v1.py","trainer_prefetch_v1.py","execution_hooks_prefetch_v1.py",
    "worker_entry_v1.py","root_driver_v1.py","canonical_dispatch.staged.py")
for name in op_names:
    old=OLD_REL+"/"+name;new=NEW_REL+"/"+name
    stage(old,new,transform((ROOT/old).read_text()),"fresh_namespace_and_import_bindings_only")

# Fresh source contract pins every new successor and retains historical immutable observation-plan routes.
old=OLD_REL+"/root_contract_v1.py";s=transform((ROOT/old).read_text())
s=set_assignment(s,"HOOK_REF",intended_ref(NEW_REL+"/execution_hooks_prefetch_v1.py"))
s=set_assignment(s,"TRAIN_SOURCE_REF",intended_ref(NEW_REL+"/trainer_prefetch_v1.py"))
old_contract=ast.parse((ROOT/old).read_text())
old_pins=next(ast.literal_eval(n.value) for n in old_contract.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="PREFETCH_SOURCE_PINS" for t in n.targets))
source_pins={NEW_PREF if p==OLD_PREF else p:(intended_ref(NEW_PREF)["sha256"] if p==OLD_PREF else h) for p,h in old_pins.items()}
source_pins.update(pins)
source_pins[OLD_PREF]=old_pins[OLD_PREF]
s=set_assignment(s,"PREFETCH_SOURCE_PINS",source_pins)
# Metadata-only validators imported from original trainer retain their complete transitive imports.
needle='def source_paths():\n'
extra='ORIGINAL_METADATA_IMPORT_SOURCE_PATHS='+repr(tuple("src/arggnn/"+name+".py" for name in ("american_c3_gnn_dataset_v1","american_c3_gnn_stream_train_v1","american_c3_gnn_observations_v1","american_c3_gnn_run_v1","american_c3_gnn_train_v1")))+'\n'
extra+='PRIOR_PREFETCH_REL='+repr(OLD_REL)+'\nPRIOR_PREFETCH_DRIVER_REF='+repr(file_ref(OLD_REL+"/root_driver_v1.py"))+'\n'
assert s.count(needle)==1;s=s.replace(needle,extra+needle,1)
needle='REL+"/metadata_contract_v1.py",REL+"/tracking_factory_v1.py",REL+"/persistent_provider_v1.py",REL+"/persistent_workers_v1.py",*PREFETCH_SOURCE_PINS,LEGACY_DRIVER_REF["path"],FROZEN_TRAIN_CONFIG_REF["path"],'
replacement=needle+'*ORIGINAL_METADATA_IMPORT_SOURCE_PATHS,PRIOR_PREFETCH_DRIVER_REF["path"],'
assert s.count(needle)==1;s=s.replace(needle,replacement,1)
needle='require(evidence["route"] in ("current980_zero_worker","prefetch_isolated980"),"Known immutable plan route")'
replacement='require(evidence["route"] in ("current980_zero_worker","prefetch_current980","prefetch_isolated980"),"Known immutable plan route")'
assert s.count(needle)==1;s=s.replace(needle,replacement,1)
needle='legacy=evidence["route"]=="current980_zero_worker"\n rel=LEGACY_REL if legacy else REL\n prefix="C3_GNN_current980_" if legacy else "C3_GNN_prefetch_isolated980_"\n driver=LEGACY_DRIVER_REF if legacy else ref(DRIVER)'
replacement='legacy=evidence["route"]=="current980_zero_worker"\n prior_prefetch=evidence["route"]=="prefetch_current980"\n rel=LEGACY_REL if legacy else (PRIOR_PREFETCH_REL if prior_prefetch else REL)\n prefix="C3_GNN_current980_" if legacy else ("C3_GNN_prefetch_current980_" if prior_prefetch else "C3_GNN_prefetch_isolated980_")\n driver=LEGACY_DRIVER_REF if legacy else (PRIOR_PREFETCH_DRIVER_REF if prior_prefetch else ref(DRIVER))'
assert s.count(needle)==1;s=s.replace(needle,replacement,1)
stage(old,NEW_REL+"/root_contract_v1.py",s,"new_pins_namespace_and_explicit_immutable_plan_route")

# Canonical executable is a fresh additive entry, with exact staged bytes.
stage(OLD_DISPATCH,NEW_DISPATCH,transform((ROOT/OLD_DISPATCH).read_text()),"fresh_additive_canonical_dispatcher")
assert (TREE/NEW_DISPATCH).read_bytes()==(TREE/NEW_REL/"canonical_dispatch.staged.py").read_bytes()

# Scientific configuration is byte-for-byte frozen. Caps/request stay inactive.
stage(OLD_REL+"/training_config.inactive.json",NEW_REL+"/training_config.inactive.json",(ROOT/OLD_REL/"training_config.inactive.json").read_text(),"byte_identical_scientific_config")
caps=json.loads((ROOT/OLD_REL/"caps_train.root_active_v1.json").read_text())
caps.update(active=False,schema="C3_GNN_prefetch_isolated980_operational_caps_v1")
caps_target=NEW_REL+"/caps_train.proposed.inactive.json"
stage(OLD_REL+"/caps_train.root_active_v1.json",caps_target,json.dumps(caps,indent=2,sort_keys=True)+"\n","inactive_proposed_prior_resource_caps")
req=copy.deepcopy(json.loads((ROOT/OLD_REL/"request_train.root_active_v1.json").read_text()))
req.pop("caps_ref",None);req.update(active=False,schema="C3_GNN_prefetch_isolated980_request_v1",
    caps_template_ref=intended_ref(caps_target),hook_ref=intended_ref(NEW_REL+"/execution_hooks_prefetch_v1.py"),
    scientific_train_source_ref=intended_ref(NEW_REL+"/trainer_prefetch_v1.py"),
    scientific_training_config_ref=intended_ref(NEW_REL+"/training_config.inactive.json"))
stage(OLD_REL+"/request_train.root_active_v1.json",NEW_REL+"/request_train.proposed.inactive.json",json.dumps(req,indent=2,sort_keys=True)+"\n","inactive_fresh_root_request_original20_observation_plan_bindings")
(OUT/"integration_sources.diff").write_text("".join(diffs))
(OUT/"staged_source_install_manifest.v1.json").write_text(json.dumps(dict(schema="C3_isolated_v2_inactive_install_manifest_v1",canonical_install_performed=False,future_root_packet=NEW_REL,future_admission_path=NEW_REL+"/admitted_train_v1/",future_output="experiments/results/C3_GNN_prefetch_isolated980_train_v1",files=inventory),indent=2,sort_keys=True)+"\n")
print(json.dumps(dict(staged_files=len(inventory),install_manifest_ref=file_ref(OUT/"staged_source_install_manifest.v1.json"),canonical_dispatch=intended_ref(NEW_DISPATCH),root_contract=intended_ref(NEW_REL+"/root_contract_v1.py"),request=intended_ref(NEW_REL+"/request_train.proposed.inactive.json"),scientific_config=intended_ref(NEW_REL+"/training_config.inactive.json")),indent=2))
