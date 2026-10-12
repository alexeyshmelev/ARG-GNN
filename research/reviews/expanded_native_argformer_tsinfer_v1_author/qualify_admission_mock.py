"""Mock root admission only: no storage helper, worker, arrays or native runtime."""
import copy
from datetime import datetime, timedelta, timezone
import types
from unittest import mock
from arggnn import expanded_native_argformer_tsinfer_v1 as native
from arggnn import prepare_expanded_native_argformer_tsinfer_v1 as supervisor

config = native.default_config_v1()
report = supervisor.source_report_v1()
spec = supervisor.admission_draft_v1()
spec.update(status="active", execution_authorized=True,
    expires_at_utc=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat())
input_ref = dict(path="research/reviews/expanded_native_argformer_tsinfer_v1_author/MOCK_ONLY_ACTIVE.json",
                 sha256="0" * 64)
spec["preparation_runtime_ref"] = dict(
    path="research/reviews/expanded_native_argformer_tsinfer_v1_author/MOCK_ONLY_RUNTIME.json",
    sha256="1" * 64)
runtime = dict(schema="expanded_argformer_tsinfer_CPU_runtime_v1", stage="prepare",
    source_ref=config["source_ref"], prefix=str(native._path(native.PREP_PREFIX)),
    python=[3, 11, 9], packages={"numpy": "1.26.4", "tskit": "0.6.4", "torch": "2.4.0"},
    CUDA_VISIBLE_DEVICES="", GPU_used=False)
storage = dict(accounting_mode=supervisor.MODE, output_directory=spec["output_directory"],
    reserved_new_bytes=spec["output_cap_bytes"], reservation=dict(consumer_bindings=[]))
bindings = [spec[k] for k in ("source_ref", "config_ref", "adapter_ref", "storage_adapter_ref",
                              "observation_runtime_ref", "preparation_runtime_ref")]
bindings += [dict(path=p, sha256=sha) for p, sha in report["source_sha256"].items()]
bindings += [config["observation"][k] for k in ("generation_config_ref", "registration_ref", "recipe_ref", "pipeline_ref")]
bindings += [spec["completion_refs"][p] for p in spec["execution_parent_ids"]]
storage["reservation"]["consumer_bindings"] = bindings
spec["storage"] = storage
actual_read = native._read
current_spec = [spec]


def read(ref):
    if ref == input_ref:
        return current_spec[0]
    if ref == spec["preparation_runtime_ref"]:
        return runtime
    return actual_read(ref)


def fail_expected(candidate):
    current_spec[0] = candidate
    try:
        supervisor._admit(input_ref)
    except (ValueError, KeyError):
        return
    raise AssertionError("Malformed root admission accepted")


with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""}), \
     mock.patch.object(native, "_read", side_effect=read), \
     mock.patch.object(supervisor, "importlib", types.SimpleNamespace(import_module=lambda name: types.SimpleNamespace(MODE=supervisor.MODE))):
    admitted = supervisor._admit(input_ref)
    assert admitted[0]["execution_parent_ids"] == ["ap_expanded_val_000"]
    wrong = copy.deepcopy(spec); wrong["truth_authorized"] = True; fail_expected(wrong)
    wrong = copy.deepcopy(spec); wrong["GPU_authorized"] = True; fail_expected(wrong)
    wrong = copy.deepcopy(spec); wrong["execution_parent_ids"] = ["ap_expanded_train_000"]; fail_expected(wrong)
    wrong = copy.deepcopy(spec); wrong["resource_caps"]["maximum_native_nodes"] = 0; fail_expected(wrong)
    wrong = copy.deepcopy(spec); wrong["source_sha256"].pop(native.FROZEN); fail_expected(wrong)
    wrong = copy.deepcopy(spec); wrong["storage"]["reservation"]["consumer_bindings"] = []; fail_expected(wrong)
    wrong = copy.deepcopy(spec); wrong["execution_scope"] = "primary_VAL"
    wrong["execution_parent_ids"] = list(native.VAL_PARENTS); wrong["native_canary_ref"] = None; fail_expected(wrong)

# Rejection precedes any namespace fit or GPU authorizer call.
for fn, args in (
    (native.fit_parent_v1, ({"path": "MOCK.json", "sha256": "2" * 64}, "MOCK")),
    (native.infer_parent_v1, ({"path": "MOCK.json", "sha256": "2" * 64},
                             {"path": "MOCK.json", "sha256": "2" * 64}, "MOCK"))):
    with mock.patch.object(native, "_read", return_value=dict(parent_id="ap_expanded_train_000", split="train")), \
         mock.patch.object(native, "_native_namespace", side_effect=AssertionError("Native fit must not be reached")):
        try:
            fn(*args, config=config, guard=lambda planned=0: True, gpu_authorizer=lambda value: {})
        except ValueError:
            pass
        else:
            raise AssertionError("TRAIN fit/prediction accepted")
print("Passed mocked canary admission, rejected truth/GPU/scope/cap/source/closure/missing-canary, and TRAIN-fit rejection.")
