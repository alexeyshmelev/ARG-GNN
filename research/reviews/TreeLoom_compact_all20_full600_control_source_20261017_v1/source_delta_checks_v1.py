"""Stdlib AST/source checks only. No science, payload, runtime or resource import."""
from pathlib import Path
import ast
import hashlib
import json

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/TreeLoom_compact_all20_full600_control_source_20261017_v1"
BASE="research/reviews/frozen_TreeLoom_bounded_training_source_20261017_v1/full600_ROOT_continuation_owner4_v1.py"
BASE_SHA="31a893670cf041798564d96caad20f58bedc12ee08b6e8bc818bd1cb12249021"


def need(ok,message):
    if not ok:raise AssertionError(message)


def functions(source):
    return {n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}


def dump(node):
    return ast.dump(node,include_attributes=False)


def run_v1():
    base=(ROOT/BASE).read_text()
    need(hashlib.sha256(base.encode()).hexdigest()==BASE_SHA,"Exact independently passed owner4 control base")
    source=(ROOT/HERE/"full600_ROOT_compact_continuation_v1.py").read_text()
    before,after=functions(base),functions(source)
    unchanged=[name for name in before if dump(before[name])==dump(after[name])]
    changed=[name for name in before if dump(before[name])!=dump(after[name])]
    need(changed==["_request","_all20","_admit","continue_v1"],"Only declared control adaptation functions change")
    for name in ("_qualification","_owner4","_launch","_wait","_controls","_remaining_fits","main"):
        need(name in unchanged,"Original qualification/owner/window/no-retry/launch gate: "+name)
    def assignment(function,name):
        return next(n for n in ast.walk(function)if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name)and t.id==name for t in n.targets))
    need(dump(assignment(before["_admit"],"direct"))==dump(assignment(after["_admit"],"direct")),
        "Original full64 direct role expression preserved; source/manifest replaced one-for-one")
    need(dump(assignment(before["_admit"],"argv"))==dump(assignment(after["_admit"],"argv")),
        "Launch exact original frozen scientific parent with fresh input")
    # Remove only the three declared metadata consolidation keys.
    admit=after["_admit"]
    extra={"compact_control_source_review_ref","all20_metadata_builder_parameters_ref","all20_compact_control_manifest_ref"}
    removed=[]
    for n in ast.walk(admit):
        if isinstance(n,ast.Call):
            kept=[]
            for keyword in n.keywords:
                if keyword.arg in extra:removed.append(keyword.arg)
                else:kept.append(keyword)
            n.keywords=kept
    need(set(removed)==extra and dump(admit)==dump(before["_admit"]),
        "Admission body unchanged except three fields inside existing real control bundle")
    own_namespace=None
    for n in ast.parse(source).body:
        if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=="HERE"for t in n.targets):
            own_namespace=ast.unparse(n.value)
    need(own_namespace=="c.HERE","Scientific template/model namespace remains original")
    helper=(ROOT/HERE/"readiness_ROOT_v1.py").read_text()
    ast.parse(helper)
    need(not any(s in helper for s in ("np.load(","numpy.load(","torch.load(","tskit.load(","os.exec","apply_reserve_v1","apply_close")),
        "Readiness metadata source has no scientific load, launch or accounting operation")
    need("verify_values_v1" in source and "qualification_weights_optimizer_or_rng_reused" in source,
        "Compact-aware all20 authentication retains original fresh-state barrier")
    return dict(schema="TreeLoom_compact_all20_full600_control_AST_delta_check_v1",status="PASS",
        changed_inherited_functions=changed,unchanged_inherited_functions=unchanged,
        original_direct_binding_expression_unchanged=True,
        original_sole_scientific_parent_argv_unchanged=True,
        original_admission_body_unchanged_except_existing_bundle_metadata=True,
        scientific_tests_executed=False,scientific_payloads_read=False,
        runtime_resource_accounting_operations=False)


if __name__=="__main__":
    print(json.dumps(run_v1(),sort_keys=True,indent=2))
