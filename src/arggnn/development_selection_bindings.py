"""Complete factual fresh-development selection-plan binding, never selection."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import tempfile

import torch
from .resources import ROOT
from . import development_selection as selection
from . import development_training_bindings as preparation
from .development_execution import DETERMINISTIC_POLICY

DEFAULT_COMPLETION="experiments/results/fresh_development_preparation_v1/completion.json"
SCOPE_AMENDMENT="research/protocols/development_scope_amendment_v2.json"
FAMILIES=("candidate","gru","transformer","reference_smoother","canonical_gru",
          "argmix_full_tree","allrecord_gru","allrecord_transformer","gst_ca_span","graphssm_zoh")
ALIASES={"same_anchor_gru":"gru","same_anchor_transformer":"transformer",
         "allrecord_boundary_gru":"allrecord_gru","allrecord_boundary_transformer":"allrecord_transformer"}
ROLES={"candidate":"candidate","gru":"same_source_gru","transformer":"same_source_transformer",
       "reference_smoother":"observed_reference_smoother","canonical_gru":"canonical_tree_gru",
       "argmix_full_tree":"argmix_adaptation","allrecord_gru":"allrecord_boundary_gru",
       "allrecord_transformer":"allrecord_boundary_transformer","gst_ca_span":"gst_ca_span_adaptation",
       "graphssm_zoh":"graphssm_zoh_adaptation"}
BUDGET={"main_neural":216,"secondary_control_reserve":72,"total":288}
POLICY_FIELDS={family:("deterministic_policy" if family in ("canonical_gru","gst_ca_span","graphssm_zoh") else "deterministic_execution_policy") for family in FAMILIES}
CHECKPOINT_RULE="minimum equal-parent validation float64 uncalibrated raw-logit NLL,earliest epoch on exact tie"
FAMILY_RULE="mean selected-checkpoint NLL over all3 prescribed seeds. One global best+0.0001 tie window,then parameter count and frozen lexical ID"
SECONDARY_POLICY="Preserve original selected-geometry/depth gain-matched diagonal and static-local contrasts,plus selected-age scalar-age or selected-plain-depth2 noncontextual contrast. Match information/source/learning rules and freeze before final generation."
FAILURE_POLICY="Preserve every start/failure. No missing or wholly ineligible required baseline family may be silently removed from strongest-baseline choice."
_PRODUCER_FILES=("development_selection_bindings.py","development_selection.py",
    "development_training_bindings.py","development_execution.py","resources.py",
    "argmix_development_bindings.py","canonical_development_bindings.py",
    "allrecord_development_bindings.py","allrecord_boundary_temporal.py","priorart_development_bindings.py")
_IMPORTED_SOURCES={str(ROOT/"src/arggnn"/name):preparation.sha(ROOT/"src/arggnn"/name) for name in _PRODUCER_FILES}
_IMPORTED_SOURCES[str(ROOT/"scripts/bind_development_selection.py")]=preparation.sha(ROOT/"scripts/bind_development_selection.py")


@dataclass(frozen=True)
class DevelopmentSelectionBinding:
    plan: dict
    content: bytes
    sha256: str
    receipt: dict
    receipt_content: bytes
    receipt_sha256: str
    path: str | None
    receipt_path: str | None


def guard_development_path(value,*,project_root=ROOT):
    """Guard lexical aliases and resolved destinations before opening bytes."""
    if not isinstance(value,(str,Path)) or not str(value) or "\\" in str(value):
        raise ValueError("A project development path is required")
    path=Path(value)
    if ".." in path.parts:raise ValueError("Development path traversal is forbidden")
    parts=path.parts[:-1] if path.suffix==".py" else path.parts
    if any(part.lower()=="test" or any(token in part.lower() for token in
           ("confirmatory","confirmation","final_test","final-test")) for part in parts):
        raise ValueError("Held-out paths are unavailable before reads")
    return selection._Capture(project_root).path(path,must_exist=False)


def _ref(capture,path,digest):
    guarded=guard_development_path(path,project_root=capture.root)
    return {"path":str(capture.path(guarded).relative_to(capture.root)),"sha256":selection._sha(digest)}


def _normalize_family(value):
    if not isinstance(value,str):raise ValueError("A factual family ID is required")
    return ALIASES.get(value,value)


def _amendment(value):
    if not isinstance(value,dict) or type(value.get("schema_version")) is not int or value.get("schema_version")!=1 or value.get("kind")!="development_scope_amendment_proposal":
        raise ValueError("Reviewed v2 scope amendment is required")
    fixed={"development_only":True,"authorizes_confirmatory_generation":False,"final_test_access":False,
           "no_fresh_development_scores_inspected":True,"counts":{"train":64,"val":16,"test":0},
           "training_seeds":[101,202,303],"candidate_grid_unchanged":True,
           "checkpoint_rule":CHECKPOINT_RULE,"family_rule":FAMILY_RULE,
           "secondary_control_policy":SECONDARY_POLICY,"failure_policy":FAILURE_POLICY}
    if any(value.get(key)!=expected for key,expected in fixed.items()):
        raise ValueError("Scope amendment phase/selection/control/failure policy differs")
    for flag in ("development_only","authorizes_confirmatory_generation","final_test_access",
                 "no_fresh_development_scores_inspected","candidate_grid_unchanged"):
        if type(value.get(flag)) is not bool:raise ValueError("Typed scope flags required")
    grid={"hidden_widths":[32,64],"dropout":[0.,.1],"learning_rates":[.0003,.001],
          "source_layers":2,"temporal_layers":2,"configurations_per_family":8,"epoch_cap":48,
          "patience":8,"weight_decay":.0001,"gradient_clip_norm":1.}
    if value.get("neural_grid")!=grid or any(isinstance(v,bool) for v in value.get("neural_grid",{}).values()):
        raise ValueError("Exact8-grid/48-epoch/patience8 scope budget required")
    budget=value.get("maximum_prescribed_optimizer_starts")
    if budget!=BUDGET or any(type(v) is not int for v in budget.values()):
        raise ValueError("216 main plus72 control reserve, maximum288 starts required")
    smoother={"configuration_seed_cells":24,"distinct_predictors":8,"optimizer_starts":0,"independent_seed_evidence":False}
    if value.get("deterministic_smoother")!=smoother or value["deterministic_smoother"].get("independent_seed_evidence") is not False:
        raise ValueError("Fixed no-fit smoother accounting required")
    rows=value.get("families")
    if not isinstance(rows,list) or len(rows)!=10 or any(not isinstance(row,dict) for row in rows):
        raise ValueError("Ten explicit scope families required")
    ids=[_normalize_family(row.get("id")) for row in rows]
    if len(set(ids))!=10 or set(ids)!=set(FAMILIES):raise ValueError("Required closer-prior-art/all-record families missing")
    for row,family in zip(rows,ids):
        if row.get("role")!=("candidate" if family=="candidate" else "baseline") or not isinstance(row.get("scope"),str) or not row["scope"].strip():
            raise ValueError("Factual candidate/baseline scope declarations required")
    baselines=value.get("required_baseline_families")
    if not isinstance(baselines,list):raise ValueError("Nine mandatory baseline families required")
    normalized=[_normalize_family(name) for name in baselines]
    if len(normalized)!=9 or len(set(normalized))!=9 or set(normalized)!=set(FAMILIES)-{"candidate"}:
        raise ValueError("No mandatory baseline family may be omitted")
    aliases={row["id"]:family for row,family in zip(rows,ids) if row["id"]!=family}
    return aliases


def _family_contract(family):
    """Reviewed producers supply exact scope and full dependency contracts."""
    if family in ("candidate","gru","transformer","reference_smoother"):
        return {**preparation.family_scope(family),"pins":preparation.snapshot()}
    if family=="canonical_gru":
        from . import canonical_development_bindings as canonical
        if canonical.FAMILY!=family:raise ValueError("Canonical producer must freeze actual canonical_gru ID")
        return {"input_scope":canonical.INPUT_SCOPE,"cross_tree_aggregation":canonical.AGGREGATION,
                "pins":canonical.implementation_hashes()}
    if family=="argmix_full_tree":
        from . import argmix_development_bindings as raw
        return {"input_scope":"all raw observed marginal-tree nodes/topology and F48, original F8 prefix with verified full-parent reference adapter; no canonicalization or truth features",
                "cross_tree_aggregation":"none","pins":raw.implementation_hashes()}
    if family in ("allrecord_gru","allrecord_transformer"):
        from . import allrecord_development_bindings as allrecord
        return {"input_scope":allrecord.INPUT_SCOPE,"cross_tree_aggregation":allrecord.AGGREGATION[family],
                "pins":allrecord.implementation_hashes()}
    if family in ("gst_ca_span","graphssm_zoh"):
        from . import priorart_development_bindings as priorart
        return {"input_scope":priorart.SCOPE[family],"cross_tree_aggregation":priorart.AGGREGATION[family],
                "pins":priorart.implementation_hashes()}
    raise ValueError("Unsupported factual family")


def validate_producer_settings(family,settings,*,completion,completion_path,
        completion_sha256,capture,source_snapshot):
    """Replay complete typed producer config assembly without loading tensors.

    The caller must first perform mandatory completed-source admission. Captured
    plans and manifest bytes are incorporated into the caller's final seal.
    """
    if family not in FAMILIES or not isinstance(settings,dict) or settings.get("family")!=family:
        raise ValueError("Exact factual producer family required")
    guarded_completion=guard_development_path(completion_path,project_root=capture.root)
    selection._sha(completion_sha256)
    for key in ("completion_receipt","boundary_data","age_data","preparation_execution_freeze",
                "search_plan","dataset_manifest","dataset_protocol"):
        if key in settings:guard_development_path(settings[key],project_root=capture.root)
    if "scope_amendment" in settings:
        amendment=settings["scope_amendment"]
        if not isinstance(amendment,dict) or set(amendment)!={"path","sha256"}:
            raise ValueError("Exact producer scope amendment reference required")
        guard_development_path(amendment["path"],project_root=capture.root)
    if "source_artifact_sha256" in settings:
        catalog=settings["source_artifact_sha256"]
        if not isinstance(catalog,dict):raise ValueError("Producer source catalog required")
        for path,digest in catalog.items():
            guard_development_path(path,project_root=capture.root);selection._sha(digest)
    if family in ("candidate","gru","transformer","reference_smoother"):
        expected=preparation.make_configs(completion,guarded_completion,completion_sha256)[family]
        preparation.check_completion_settings(settings,completion)
    else:
        if "search_plan" not in settings or "search_plan_sha256" not in settings:
            raise ValueError("Captured reviewed producer search plan required")
        plan_path=guard_development_path(settings["search_plan"],project_root=capture.root)
        plan_sha=selection._sha(settings["search_plan_sha256"])
        plan=capture.read(_ref(capture,plan_path,plan_sha))
        args=(plan,plan_path,plan_sha,completion,guarded_completion,completion_sha256)
        if family=="argmix_full_tree":
            from . import argmix_development_bindings as raw
            raw.check_plan(plan);expected=raw.bound_settings(*args)
        elif family=="canonical_gru":
            from . import canonical_development_bindings as canonical
            canonical.check_plan(plan)
            fact=completion["stage_manifests"]["inferred"]
            manifest=capture.read(_ref(capture,fact["path"],fact["sha256"]))
            catalog={}
            records=manifest.get("replicates")
            if not isinstance(records,list) or len(records)!=80:
                raise ValueError("Complete canonical raw-source catalog requires80 parents")
            for record in records:
                for field,hash_field in (("metadata","metadata_sha256"),("tensor_file","tensor_sha256"),
                                        ("prepared_tensor_file","prepared_tensor_sha256")):
                    path=str(guard_development_path(record[field],project_root=capture.root))
                    digest=selection._sha(record[hash_field])
                    if path in catalog:raise ValueError("Canonical raw-source catalog aliases different records")
                    if source_snapshot.hashes.get(path)!=digest:
                        raise ValueError("Canonical raw source is absent from verified completion catalog")
                    catalog[path]=digest
            expected=canonical.bound_settings(*args,catalog)
        elif family in ("allrecord_gru","allrecord_transformer"):
            from . import allrecord_development_bindings as allrecord
            allrecord.validate_plan(plan);allrecord.validate_settings(settings)
            expected=allrecord.bound_configs(*args)[family]
            preparation.check_completion_settings(settings,completion)
        else:
            from . import priorart_development_bindings as priorart
            priorart.check_plan(plan);priorart.validate_settings(settings)
            amendment=plan["scope_amendment"]
            captured_amendment=capture.read(_ref(capture,amendment["path"],amendment["sha256"]))
            priorart.check_amendment(captured_amendment)
            expected=priorart.bound_settings(*args)[family]
    # Canonical JSON equality also rejects int/float and bool/int substitutions.
    canonical_json=lambda value:json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)
    if canonical_json(settings)!=canonical_json(expected):
        raise ValueError("Complete typed producer settings differ from reviewed assembly/completion")
    return deepcopy(expected)


def _dependency_ref(capture,name,digest):
    selection._pin_name(name)
    path=Path(name)
    if len(path.parts)==1:
        if path.suffix!=".py":raise ValueError("Bare dependency keys must name source .py files")
        path=Path("src/arggnn")/path
    path=guard_development_path(path,project_root=capture.root)
    ref=_ref(capture,path,digest)
    capture.read(ref,json_file=False,allow_source_reuse=True)
    return ref


def _producer_refs(capture):
    result={}
    for name,digest in _IMPORTED_SOURCES.items():
        path=guard_development_path(name,project_root=capture.root)
        ref=_ref(capture,path,digest)
        capture.read(ref,json_file=False,allow_source_reuse=True)
        result[str(path.relative_to(capture.root))]=ref
    script=capture.root/"scripts/bind_development_selection.py"
    ref=_ref(capture,script,preparation.sha(script))
    capture.read(ref,json_file=False,allow_source_reuse=True)
    result[str(script.relative_to(capture.root))]=ref
    return result


def _check_all(capture,source_lock):
    source_lock.check(full=True)
    capture.verify_stable()
    for name,record in capture.files.items():
        before=preparation.artifact_signature(name)
        if before!=record["stamp"] or preparation.sha(name)!=record["sha256"] or preparation.artifact_signature(name)!=before:
            raise ValueError("Captured binding source changed before sealing")
    source_lock.check(full=True)


def _parent_contracts(dataset,source_sha):
    cases=dataset.cases("val")
    by_id={case.parent_id:case for case in cases}
    ids=[source_sha+":val:"+str(i) for i in range(16)]
    if len(cases)!=16 or len(by_id)!=16 or set(by_id)!=set(ids):
        raise ValueError("Every verified observed validation parent is required")
    result=[]
    for parent_id in ids:
        case=by_id[parent_id]
        positions,queries=case.positions,case.query_node_ids
        if not isinstance(positions,torch.Tensor) or positions.device.type!="cpu" or positions.ndim!=1 or not positions.is_floating_point() or not len(positions) or not bool(torch.isfinite(positions).all()) or not bool((positions[1:]>positions[:-1]).all()):
            raise ValueError("Verified ordered finite observed coordinates required")
        if not isinstance(queries,torch.Tensor) or queries.device.type!="cpu" or queries.dtype!=torch.long or queries.ndim!=1 or len(queries)!=64 or torch.unique(queries).numel()!=64:
            raise ValueError("Verified64 unique observed query identities required")
        result.append({"parent_id":parent_id,"positions_sha256":selection.tensor_fingerprint(positions),
            "query_ids_sha256":selection.tensor_fingerprint(queries),"logit_shape":[len(positions),len(queries),4]})
    return result


def bind_development_selection(completion_path,expected_completion_sha256,*,
        scope_amendment_path=SCOPE_AMENDMENT,expected_scope_amendment_sha256,
        family_configs,output_directory=None,project_root=ROOT):
    """Validate only by default; bind a complete plan after genuine preparation."""
    capture=selection._Capture(project_root)
    completion_path=guard_development_path(completion_path,project_root=capture.root)
    amendment_path=guard_development_path(scope_amendment_path,project_root=capture.root)
    if amendment_path!=capture.root/SCOPE_AMENDMENT:raise ValueError("Actual v2 scope-amendment path required")
    output=None if output_directory is None else guard_development_path(output_directory,project_root=capture.root)
    if output is not None and (output.exists() or not output.parent.is_dir()):
        raise ValueError("A fresh output directory with existing project parent is required")
    if not isinstance(family_configs,(list,tuple)) or len(family_configs)!=10:
        raise ValueError("Ten factual family config references required")
    refs={};input_aliases={}
    for raw in family_configs:
        if not isinstance(raw,dict) or set(raw)!={"family_id","path","sha256"}:
            raise ValueError("Family reference requires exact family_id/path/sha256 fields")
        family=_normalize_family(raw["family_id"])
        if family not in FAMILIES or family in refs:raise ValueError("Missing, duplicate or unexpected factual family")
        path=guard_development_path(raw["path"],project_root=capture.root)
        selection._sha(raw["sha256"])
        refs[family]={"path":path,"sha256":raw["sha256"]}
        if raw["family_id"]!=family:input_aliases[raw["family_id"]]=family
    if set(refs)!=set(FAMILIES):raise ValueError("All nine baseline families and one candidate are mandatory")
    selection._sha(expected_completion_sha256);selection._sha(expected_scope_amendment_sha256)
    if not completion_path.is_file():raise RuntimeError("No successful preparation completion exists; no plan written")
    completion_ref=_ref(capture,completion_path,expected_completion_sha256)
    captured_completion=capture.read(completion_ref)
    if not isinstance(captured_completion,dict) or not isinstance(captured_completion.get("stage_manifests"),dict):
        raise ValueError("Factual completed stage references required")
    # Guard declared aliases before the mandatory verifier opens source bytes.
    for fact in captured_completion["stage_manifests"].values():
        if not isinstance(fact,dict):raise ValueError("Factual completed stage reference required")
        for key in ("path","protocol_path"):
            if key in fact:guard_development_path(fact[key],project_root=capture.root)
    if "development_execution_freeze_path" in captured_completion:
        guard_development_path(captured_completion["development_execution_freeze_path"],project_root=capture.root)
    completion,dataset,ages,source_lock=preparation.verify_completion(
        completion_path,expected_completion_sha256,with_snapshot=True)
    del ages
    if completion!=captured_completion:raise ValueError("Completed preparation changed during admission")
    amendment_ref=_ref(capture,amendment_path,expected_scope_amendment_sha256)
    aliases=_amendment(capture.read(amendment_ref))
    aliases.update(input_aliases)
    facts=completion["stage_manifests"]
    binding={"completion_receipt_sha256":expected_completion_sha256,
        "source_dataset_manifest_sha256":facts["inferred"]["sha256"],
        "boundary_dataset_manifest_sha256":facts["boundary"]["sha256"],
        "boundary_preparation_sha256":facts["boundary"]["preparation_sha256"],
        "age_dataset_manifest_sha256":facts["age"]["sha256"],
        "age_preparation_sha256":facts["age"]["preparation_sha256"],
        "preparation_execution_freeze_sha256":completion["development_execution_freeze_file_sha256"]}
    for value in binding.values():selection._sha(value)
    freeze_path=completion.get("development_execution_freeze_path",str(preparation.DEFAULT_PREPARATION_FREEZE))
    source_refs={"completion":completion_ref,
        "source_manifest":_ref(capture,facts["inferred"]["path"],binding["source_dataset_manifest_sha256"]),
        "boundary_manifest":_ref(capture,facts["boundary"]["path"],binding["boundary_dataset_manifest_sha256"]),
        "age_manifest":_ref(capture,facts["age"]["path"],binding["age_dataset_manifest_sha256"]),
        "preparation_execution_freeze":_ref(capture,guard_development_path(freeze_path,project_root=capture.root),binding["preparation_execution_freeze_sha256"])}
    for ref in source_refs.values():capture.read(ref)
    families={};pins={};implementation={};resolved_pins={}
    for family in FAMILIES:
        ref=_ref(capture,refs[family]["path"],refs[family]["sha256"])
        settings=capture.read(ref)
        if not isinstance(settings,dict) or settings.get("family")!=family:
            raise ValueError("Config family must use its actual producer/run ID")
        contract=_family_contract(family)
        policy_field=POLICY_FIELDS[family]
        if any(other in settings for other in {"deterministic_execution_policy","deterministic_policy"}-{policy_field}):
            raise ValueError("Ambiguous producer deterministic policy fields")
        policy=settings.get(policy_field)
        if not isinstance(policy,dict) or set(policy)!=set(DETERMINISTIC_POLICY) or any(type(policy[key]) is not type(value) or policy[key]!=value for key,value in DETERMINISTIC_POLICY.items()):
            raise ValueError("Exact strict deterministic execution policy required")
        for field in ("input_scope","cross_tree_aggregation"):
            if settings.get(field)!=contract[field] or not isinstance(settings[field],str) or not settings[field].strip():
                raise ValueError("Reviewed factual protocol scope/aggregation differs")
        if settings.get("bound_implementation_sha256")!=contract["pins"]:
            raise ValueError("Complete reviewed family dependency pins differ")
        if any(settings.get(key)!=digest for key,digest in binding.items()):
            raise ValueError("Family completed source binding differs")
        validate_producer_settings(family,settings,completion=completion,completion_path=completion_path,
            completion_sha256=expected_completion_sha256,capture=capture,source_snapshot=source_lock)
        config=selection._normalize_config(settings)
        kind="candidate" if family=="candidate" else "bandwidth_pseudocount" if family=="reference_smoother" else "width_dropout_lr"
        selection._grid(config,kind)
        for name,digest in contract["pins"].items():
            if name in pins and pins[name]!=digest:raise ValueError("Shared dependency hash conflict")
            dependency=_dependency_ref(capture,name,digest)
            path=dependency["path"]
            if path in resolved_pins and resolved_pins[path]!=digest:raise ValueError("Resolved-file dependency alias conflict")
            pins[name]=digest;resolved_pins[path]=digest;implementation[name]=dependency
        families[family]={"role":ROLES[family],"grid_kind":kind,"protocol":ref,
            "expected_implementation_sha256":deepcopy(contract["pins"]),
            "input_scope":settings["input_scope"],"cross_tree_aggregation":settings["cross_tree_aggregation"]}
    producers=_producer_refs(capture)
    plan={"schema_version":1,"kind":"fresh_development_family_selection_plan",
        "development_namespace":"development-v1","training_seeds":[101,202,303],
        "splits":{"train":64,"val":16,"test":0},"nll_policy":selection.RAW_NLL,"tie_tolerance":.0001,
        "source_binding":binding,"source_artifacts":source_refs,
        "validation_parents":_parent_contracts(dataset,binding["source_dataset_manifest_sha256"]),
        "families":families,"required_baseline_families":[family for family in FAMILIES if family!="candidate"],
        "implementation_artifacts":implementation}
    content=(json.dumps(plan,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    digest=hashlib.sha256(content).hexdigest()
    _check_all(capture,source_lock)
    scratch=capture.root/".runtime/tmp"
    scratch.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="development_selection_binding_",dir=scratch) as directory:
        temporary=Path(directory)/"plan.json"
        with temporary.open("xb") as handle:handle.write(content)
        frozen=selection.load_development_selection_plan(temporary,digest,project_root=capture.root)
        if frozen.values!=plan:raise ValueError("Temporary strict plan validation changed captured values")
    _check_all(capture,source_lock)
    target=None if output is None else output/"plan.json"
    receipt_target=None if output is None else output/"binding_receipt.json"
    receipt={"schema_version":1,"kind":"fresh_development_selection_plan_binding_receipt",
        "created_utc":datetime.now(timezone.utc).isoformat(),"plan_sha256":digest,
        "plan_path":None if target is None else str(target.relative_to(capture.root)),
        "scope_amendment":amendment_ref,"proposal_to_actual_family_aliases":dict(sorted(aliases.items())),
        "maximum_prescribed_optimizer_starts":dict(BUDGET),"prescribed_main_optimizer_starts":216,
        "configuration_seed_cells":240,"smoother_configuration_seed_cells":24,"smoother_optimizer_starts":0,
        "deterministic_execution_policy":deepcopy(DETERMINISTIC_POLICY),"family_deterministic_policy_fields":dict(POLICY_FIELDS),"checkpoint_rule":CHECKPOINT_RULE,
        "family_rule":FAMILY_RULE,"secondary_control_policy":SECONDARY_POLICY,"failure_policy":FAILURE_POLICY,
        "family_config_references":{family:families[family]["protocol"] for family in FAMILIES},
        "source_binding":dict(binding),"source_artifacts":source_refs,
        "producer_source_artifacts":producers,"captured_input_artifacts":capture.provenance(),
        "verified_completed_source_snapshot":source_lock.manifest(),"plans_written":output is not None,
        "validation_only":output is None,"scores_inspected":False,"selection_performed":False,
        "final_test_access":False,"authorizes_confirmatory_generation":False,
        "interpretation":"Factual development search-plan binding only; no selected model, strongest baseline, training result or confirmation."}
    receipt_content=(json.dumps(receipt,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
    receipt_sha=hashlib.sha256(receipt_content).hexdigest()
    _check_all(capture,source_lock)
    if output is not None:
        output.mkdir()
        created=[]
        try:
            for path,payload in ((target,content),(receipt_target,receipt_content)):
                with path.open("xb") as handle:
                    created.append(path)
                    handle.write(payload)
            _check_all(capture,source_lock)
        except BaseException:
            for path in created:path.unlink()
            output.rmdir()
            raise
    return DevelopmentSelectionBinding(plan,content,digest,receipt,receipt_content,receipt_sha,
        None if target is None else str(target),None if receipt_target is None else str(receipt_target))
