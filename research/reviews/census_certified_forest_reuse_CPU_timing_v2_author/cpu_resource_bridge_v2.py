"""Narrow storage-only successor of the sealed original CPU admission.

All non-storage original admission statements remain identical. No fake reader
is used for actual admission. v4 is read-only and measures only own scopes.
"""
from __future__ import annotations
from functools import wraps
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from arggnn import census_native_qualification_runner_v1 as native
from arggnn.census_native_qualification_runner_v1 import (
    VerifiedReaderV1, ROOT, CONTROLS, REF_ROLES, _utc, runtime_identity_v1)
torch = native.torch

CONFIG_PATH = "research/reviews/census_certified_forest_reuse_CPU_timing_v2_author/protocol.json"
CONFIG_SHA256 = "c7a72c73b601f8a4bdbd4d5d4f8a9583a4a0d5d0069e5468c22e9964497ad50e"

def load_bridge_config_v2(reader):
    cfg = reader.decode_json({"path": CONFIG_PATH, "sha256": CONFIG_SHA256})
    registration = reader.decode_json(cfg["root_source_registration_ref"])
    if (registration.get("authorizer") != "root_agent"
            or registration.get("hypothesis_id") != "H021"
            or registration.get("actual_execution_authorized") is not False
            or registration.get("preserve_original_prospective_v1") is not True
            or registration.get("resource_authorization") != cfg["human_authorization_ref"]):
        raise ValueError("Exact source-only resource bridge registration required")
    human = reader.decode_json(cfg["human_authorization_ref"])
    if (human.get("kind") != "human_project_storage_authorization"
            or human.get("project_root") != str(ROOT)
            or human.get("authorized_hard_limit_bytes") != 1_000_000_000_000
            or human.get("future_scheduling_ceiling_bytes") != 950_000_000_000):
        raise ValueError("Exact user 1TB authorization and 950GB ceiling required")
    for key in ("budget_source_ref", "budget_config_ref", "original_CPU_runner_ref",
                "preserved_v1_caller_ref", "preserved_v1_pins_ref",
                "accepted_consumer_pattern_source_ref", "accepted_consumer_pattern_review_ref"):
        reader.read(cfg[key])
    return cfg

def validate_account_v2(account, storage, required_write_bytes=0):
    """Validate returned genuine v4 accounting; no alternate concurrency input."""
    if (type(required_write_bytes) is not int or required_write_bytes < 0
            or account.get("legacy_ledger_mutated") is not False
            or account.get("supplemental_ledger_mutated") is not False
            or account.get("projection_is_historical_accounting_not_global_remeasurement") is not True
            or account.get("limit_bytes") != 950_000_000_000
            or account.get("hard_limit_bytes") != 1_000_000_000_000
            or account.get("reservation") != storage["reservation"]):
        raise ValueError("Exact read-only v4 combined accounting required")
    keys = ("baseline_bytes", "external_reserve_bytes", "legacy_reserved_bytes",
            "supplemental_reserved_bytes", "projected_bytes")
    if any(type(account.get(key)) is not int or account[key] < 0 for key in keys):
        raise ValueError("All genuine combined accounting components required")
    if (sum(account[key] for key in keys[:-1]) != account["projected_bytes"]
            or account["projected_bytes"] >= 950_000_000_000
            or account["projected_bytes"] >= 1_000_000_000_000):
        raise ValueError("Combined forecast exceeds admitted project ceiling")
    scope = account.get("scope_measurement", {}).get("conservative_scope_bytes")
    if (type(scope) is not int or scope < 0
            or scope + required_write_bytes > storage["reservation"]["reserved_bytes"]):
        raise ValueError("Planned boundary write exceeds exact conservative scope reservation")
    return account

def validate_storage_metadata_v2(storage, admission, original_config, cfg, reader, clock):
    fields = {"schema", "status", "job_id", "created_at_utc", "expires_at_utc",
              "output_directory", "reserved_new_bytes", "mode", "reservation",
              "initial_ledger_pins", "bridge_config_ref", "review_ref"}
    reservation = storage.get("reservation", {})
    config_ref = {"path": CONFIG_PATH, "sha256": CONFIG_SHA256}
    sources = {path: hashlib.sha256(reader.read(
        {"path": path, "sha256": hashlib.sha256((ROOT/path).read_bytes()).hexdigest()})).hexdigest()
        for path in cfg["executed_bridge_source_paths"]}
    needed = [{"path": path, "sha256": value} for path, value in sources.items()]
    needed += [config_ref, admission["refs"]["runtime"],
               cfg["budget_source_ref"], cfg["budget_config_ref"], cfg["human_authorization_ref"]]
    if (set(storage) != fields or storage["schema"] != "census_CPU_timing_storage_v2"
            or storage["status"] != "active" or storage["job_id"] != admission["job_id"]
            or storage["output_directory"] != admission["output_directory"]
            or storage["mode"] != "project_budget_v4"
            or storage["reserved_new_bytes"] != 1_000_000_000
            or original_config["output_cap_bytes"] != 1_000_000_000
            or storage["bridge_config_ref"] != config_ref
            or reservation.get("key") != admission["job_id"]
            or reservation.get("output") != str(ROOT/reader.relative(admission["output_directory"]))
            or reservation.get("auxiliary_scopes") != []
            or reservation.get("reserved_bytes") != 1_000_000_000
            or any(value not in reservation.get("consumer_bindings", []) for value in needed)
            or set(storage["initial_ledger_pins"]) != {"legacy_ledger_sha256", "supplemental_ledger_sha256"}
            or any(not native._sha(value) for value in storage["initial_ledger_pins"].values())
            or not _utc(storage["created_at_utc"]) <= clock < _utc(storage["expires_at_utc"])
            or _utc(storage["created_at_utc"]) < _utc(admission["created_at_utc"])
            or _utc(storage["expires_at_utc"]) > _utc(admission["expires_at_utc"])):
        raise ValueError("Fresh exact reviewed v4 CPU timing reservation required")
    review = reader.decode_json(storage["review_ref"])
    expected = dict(sources, **{CONFIG_PATH: CONFIG_SHA256})
    if (review.get("status") != "PASS" or review.get("review_role") != "independent_reviewer"
            or review.get("actual_admission") is not False or review.get("fixture_mode") is True
            or any(review.get("artifact_sha256", {}).get(path) != value for path, value in expected.items())):
        raise ValueError("Exact independent CPU caller/resource bridge PASS required")
    return storage

def require_storage_v2(storage, admission, original_config, reader, clock,
                       *, first, required_write_bytes=0):
    cfg = load_bridge_config_v2(reader)
    validate_storage_metadata_v2(storage, admission, original_config, cfg, reader, clock)
    from arggnn import project_budget_v4 as budget
    account = budget.check_reservation(
        ROOT, storage["reservation"],
        expected_source_sha256=cfg["budget_source_ref"]["sha256"],
        expected_config_sha256=cfg["budget_config_ref"]["sha256"],
        initial_ledger_pins=storage["initial_ledger_pins"] if first else None,
        boundary=True, required_write_bytes=required_write_bytes)
    return validate_account_v2(account, storage, required_write_bytes)

def sticky_authority_v2(callback):
    """Once any authority check denies, that caller may never write later."""
    denied = False
    @wraps(callback)
    def guarded(*args, **kwargs):
        nonlocal denied
        if denied:
            raise ValueError("Sticky CPU timing authority denial")
        try:
            return callback(*args, **kwargs)
        except Exception:
            denied = True
            raise
    return guarded


def admit_cpu_timing_v2(admission_ref,*,phase,reader=None,now=None,first_storage_read=True,required_write_bytes=0):
    reader=reader or VerifiedReaderV1();admission=reader.decode_json(admission_ref)
    allowed={'schema','status','job_id','phase','created_at_utc','expires_at_utc','refs','source_sha256','observed_artifact_sha256','output_directory','expected_certificates','inspection_manifest','initial_state_sha256_by_dtype','group_permutation'}
    if set(admission)!=allowed or admission['schema']!='census_native_qualification_admission_v1' or admission['status']!='active' or admission['phase']!=phase or phase not in {'inspect','numerical'}:
        raise ValueError('Fresh exact root active phase admission required')
    clock=now or datetime.now(timezone.utc)
    if not _utc(admission['created_at_utc'])<=clock<_utc(admission['expires_at_utc']):raise ValueError('Admission is future or expired')
    if not isinstance(admission['job_id'],str) or not admission['job_id']:raise ValueError('Distinct root job identity required')
    if set(admission['refs'])!=REF_ROLES:raise ValueError('All source/runtime/storage/observation/prepared/feature refs required')
    # Complete source pins are checked before any observation decoder is reached.
    config=reader.decode_json(admission['refs']['runner_config'])
    if (clock-_utc(admission['created_at_utc'])).total_seconds()>config['admission_max_age_seconds'] or (_utc(admission['expires_at_utc'])-_utc(admission['created_at_utc'])).total_seconds()>config['admission_max_age_seconds']:raise ValueError('Root active admission is not fresh')
    required=set(config['required_source_paths'])|{'src/arggnn/census_native_qualification_runner_v1.py'}
    if set(admission['source_sha256'])!=required:raise ValueError('Complete exact executed source/review pins required')
    for path,sha in admission['source_sha256'].items():reader.read({'path':path,'sha256':sha})
    for path,sha in config['source_sha256'].items():
        if admission['source_sha256'].get(path)!=sha:raise ValueError('Frozen runner source dependency changed')
    if config['kind']!='census_native_qualification_runner_v1' or config['controls']!=list(CONTROLS) or config['expected_tiles']!=32 or config['output_cap_bytes']!=1_000_000_000:
        raise ValueError('Frozen qualification method/cap changed')
    shipped=json.loads((ROOT/'configs/census_native_qualification_runner_v1.json').read_text())
    allowed_fixture={'fixture_mode','role_counts','references_per_class','model'}
    if config['fixture_mode']:
        if set(config)!=set(shipped) or any(config[k]!=v for k,v in shipped.items() if k not in allowed_fixture):raise ValueError('Fixture changes only counts and small model dimensions')
    elif config!=shipped:raise ValueError('Shipped actual qualification config changed')
    if now is not None and not config['fixture_mode']:raise ValueError('Injected clock belongs only to handmade mode')
    if reader.max_file_bytes!=config['max_file_bytes']:raise ValueError('Frozen per-file byte budget changed')
    if config['fixture_mode'] and reader.read_bytes is None:raise ValueError('Fixture config requires injected bytes; cannot admit actual files')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('PYTHONDONTWRITEBYTECODE')!='1':raise ValueError('CPU-only disabled-bytecode launcher required')
    if Path(sys.executable).resolve()!=(ROOT/'.runtime/env/bin/python').resolve():raise ValueError('Main project Python interpreter required')
    torch.set_num_threads(1)
    runtime=reader.decode_json(admission['refs']['runtime'])
    if set(runtime)!={'schema','identity','runtime_file_sha256'} or runtime['schema']!='census_native_CPU_runtime_v1' or runtime['identity']!=runtime_identity_v1() or set(runtime['runtime_file_sha256'])!=set(config['runtime_files']):raise ValueError('Exact main-runtime identity/files changed')
    for path,sha in runtime['runtime_file_sha256'].items():reader.read({'path':path,'sha256':sha},runtime_file=True)
    review=reader.decode_json(admission['refs']['runner_review'])
    expected_review={'schema':'census_native_qualification_review_binding_v1','status':'qualified_source_handmade_CPU_only','runner_source_sha256':admission['source_sha256']['src/arggnn/census_native_qualification_runner_v1.py'],'runner_config_sha256':admission['refs']['runner_config']['sha256'],'actual_admission':False,'review_receipt_ref':review.get('review_receipt_ref')}
    if review!=expected_review or not review['review_receipt_ref']:raise ValueError('Fresh independent exact runner/config review binding required')
    full_review=reader.decode_json(review['review_receipt_ref'])
    if (not str(full_review.get('status','')).startswith('qualified') or full_review.get('artifact_sha256',{}).get('src/arggnn/census_native_qualification_runner_v1.py')!=expected_review['runner_source_sha256'] or full_review.get('artifact_sha256',{}).get('configs/census_native_qualification_runner_v1.json')!=expected_review['runner_config_sha256'] or (full_review.get('fixture_mode') is True and not config['fixture_mode'])):raise ValueError('Full independent review receipt does not bind executed source/config')
    storage=reader.decode_json(admission['refs']['storage'])
    require_storage_v2(storage,admission,config,reader,clock,
                       first=first_storage_read,required_write_bytes=required_write_bytes)
    launcher=reader.decode_json(admission['refs']['launcher'])
    if launcher!={'schema':'census_native_CPU_launcher_v1','job_id':admission['job_id'],'python':'.runtime/env/bin/python','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','GPU_admitted':False}:
        raise ValueError('Exact new CPU launcher receipt required')
    if phase=='inspect':
        if admission['expected_certificates'] or admission['inspection_manifest'] is not None or admission['initial_state_sha256_by_dtype'] or admission['group_permutation']:
            raise ValueError('Inspection cannot grant itself numerical certificates')
    else:
        if not admission['inspection_manifest'] or len(admission['expected_certificates'])!=32 or set(admission['initial_state_sha256_by_dtype'])!=set(config['dtypes']):raise ValueError('Root frozen inspection/digests/initial weights required')
    return admission,config,reader
