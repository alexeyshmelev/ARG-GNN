"""Tiny source/fake tests. No subprocess, GPU, data or storage probe."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from arggnn.benchmarking import execution as e
from arggnn.benchmarking.contracts import ContractError


def spec():
    source = e.c.ROOT / "src/arggnn/benchmarking/execution.py"
    return dict(schema=e.SCHEMA, job_id="RAM_worker", resource="gpu0", physical_gpu=0,
                gpu_uuid=e.GPU0_UUID, python=".runtime/env/bin/python",
                output_directory="experiments/results/RAM_future_worker",
                max_output_bytes=16*2**30, max_seconds=86400, max_rss_bytes=128*2**30,
                minimum_free_gpu_bytes=2**30, max_cuda_reserved_bytes=2**30,
                callable_module="arggnn.benchmarking.argformer_execution",
                callable_name="run_contrastive", scientific_spec={"output_directory":"experiments/results/RAM_future_worker/native"},
                source_refs=[dict(path=str(source.relative_to(e.c.ROOT)),
                                  sha256=hashlib.sha256(source.read_bytes()).hexdigest())])


class ExecutionTests(unittest.TestCase):
    def test_default_cli_is_inert(self):
        with patch.object(e,"validate_spec",return_value={"job_id":"RAM"}),              patch.object(e,"execute",side_effect=AssertionError("launch forbidden")),              contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(e.main(["--job-ref",'{"path":"RAM","sha256":"fake"}']),0)

    def test_current_source_spec_is_validated_without_runtime_import(self):
        value=spec()
        with patch.object(e.c,"read_ref",return_value=value):
            self.assertEqual(e.validate_spec({}),value)

    def test_only_physical_gpu0_is_valid(self):
        for gpu in (False,1,4,5):
            value=spec()
            value["physical_gpu"]=gpu
            with patch.object(e.c,"read_ref",return_value=value),self.assertRaises(ContractError):
                e.validate_spec({})

    def test_caps_are_positive_finite_integers(self):
        for key in ("max_output_bytes","max_seconds","max_rss_bytes","minimum_free_gpu_bytes","max_cuda_reserved_bytes"):
            for invalid in (0,-1,False,1.5):
                value=spec()
                value[key]=invalid
                with patch.object(e.c,"read_ref",return_value=value),self.assertRaises(ContractError):
                    e.validate_spec({})

    def test_output_cap_has_initial_16GiB_ceiling(self):
        value=spec()
        value["max_output_bytes"]=e.MAX_OUTPUT+1
        with patch.object(e.c,"read_ref",return_value=value),self.assertRaises(ContractError):
            e.validate_spec({})

    def test_native_output_must_be_nested_in_assigned_scope(self):
        value=spec()
        value["scientific_spec"]["output_directory"]="experiments/results/other/native"
        with patch.object(e.c,"read_ref",return_value=value),self.assertRaises(ContractError):
            e.validate_spec({})

    def test_policy_rejection_precedes_any_storage_or_gpu_operation(self):
        value=spec()
        with patch.object(e,"validate_spec",return_value=value),              patch.object(e,"_approved",side_effect=ContractError("held")),              patch.object(e,"gpu0_lease",side_effect=AssertionError("GPU probe forbidden")),              self.assertRaises(ContractError):
            e.execute({})

    def test_gpu0_preflight_checks_uuid_and_free_memory_without_cuda(self):
        from types import SimpleNamespace
        good=SimpleNamespace(stdout=f"0, {e.GPU0_UUID}, 80000\n")
        with patch.object(e.subprocess,"run",return_value=good):
            e._inventory_check(2**30)
        wrong=SimpleNamespace(stdout="0, GPU-wrong, 80000\n")
        with patch.object(e.subprocess,"run",return_value=wrong),self.assertRaises(RuntimeError):
            e._inventory_check(2**30)

    def test_unknown_numerical_callable_rejected(self):
        value=spec()
        value["callable_name"]="arbitrary"
        with patch.object(e.c,"read_ref",return_value=value),self.assertRaises(ContractError):
            e.validate_spec({})

    def test_worker_command_has_no_policy_receipt_self_hash_cycle(self):
        value=spec()
        ref={"path":"RAM.json","sha256":"a"*64}
        command=e.worker_argv(ref,value)
        self.assertEqual(command[-2],"--job-ref")
        self.assertEqual(json.loads(command[-1]),ref)
        self.assertNotIn("--policy-ref",command)

    def test_output_escape_rejected(self):
        value=spec()
        value["output_directory"]="."
        with patch.object(e.c,"read_ref",return_value=value),self.assertRaises(ContractError):
            e.validate_spec({})


if __name__=="__main__":
    unittest.main()
