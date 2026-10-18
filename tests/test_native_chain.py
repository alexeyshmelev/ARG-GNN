"""Metadata/mock tests for the finite native chain; never start native science."""
from contextlib import nullcontext
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from arggnn.benchmarking import native_chain as chain
from arggnn.benchmarking import contracts as c


class NativeChainTests(unittest.TestCase):
    def setUp(self):
        self.ref = {"path": "mock/chain.json", "sha256": "a"*64}
        self.parent_ref = {"path": "mock/parent.json", "sha256": "b"*64}
        self.spec = dict(parent_job_id="actual_parent", parent_input_ref=self.parent_ref,
                         stages=copy.deepcopy(chain.STAGES),
                         output_root="experiments/results/benchmark_chain_mock",
                         records_directory=".runtime/tmp/benchmark_chain_mock_records",
                         max_cuda_reserved_bytes=74*chain.GiB,
                         minimum_free_gpu_bytes=78*chain.GiB,
                         native_python=chain.native.NATIVE_PYTHON,
                         source_refs=[c.file_ref(chain.execution.__file__)],
                         worker_source_ref=c.file_ref(chain.execution.__file__))
        self.parent = self.entry("actual_parent", "succeeded", self.parent_ref)

    def entry(self, name, status, input_ref=None):
        return dict(admission=dict(job_id=name, input_ref=input_ref or self.parent_ref,
                                   admission_receipt_ref=self.ref), status=status,
                    actual_returncode=0 if status == "succeeded" else
                    1 if status == "failed" else None, error=None)

    def test_default_plan_is_read_only(self):
        class Queue:
            def jobs(q): return [self.parent]
        with patch.object(chain, "_load", return_value=(self.spec, 100000)), \
             patch.object(chain, "_scheduler", return_value=Queue()), \
             patch.object(chain, "_prepare") as prepare, \
             patch.object(chain, "_write_policy") as policy:
            result = chain.run(self.ref)
        self.assertEqual(result["status"], "materialize_next")
        self.assertEqual(result["stage"], "query_train")
        prepare.assert_not_called()
        policy.assert_not_called()

    def test_failed_or_unknown_parent_never_replays(self):
        for state in ("failed", "unknown", "rejected"):
            result = chain.inspect(self.spec, [self.entry("actual_parent", state)])
            self.assertEqual(result["status"], "blocked")
        with self.assertRaises(c.ContractError):
            chain.inspect(self.spec, [self.entry("actual_parent", "succeeded", self.ref)])

    def test_fixed_complete_stage_order_and_caps(self):
        jobs = [self.parent]
        for stage in chain.STAGES:
            result = chain.inspect(self.spec, jobs)
            self.assertEqual(result["stage"], stage["id"])
            jobs.append(self.entry("actual_parent/"+stage["id"], "succeeded"))
        self.assertEqual(chain.inspect(self.spec, jobs)["status"], "complete")
        self.assertEqual(sum(s["disk"] for s in chain.STAGES), 76*chain.GiB)
        self.assertEqual(chain.STAGES[3]["disk"], 64*chain.GiB)

    def test_only_exact_released_chain_can_add_job(self):
        policy = dict(approved_native_chain_ref=self.ref, approved_jobs=[])
        result = chain._released_policy(policy, self.ref, self.parent_ref, self.ref,
                                        "actual_parent/query_train", 76*chain.GiB)
        self.assertEqual(policy["approved_jobs"], [])
        self.assertEqual(result["approved_new_storage_cap_bytes"], 76*chain.GiB+chain.METADATA_CAP)
        self.assertEqual(len(result["approved_jobs"]), 1)
        with self.assertRaises(c.ContractError):
            chain._released_policy(policy, self.parent_ref, self.parent_ref, self.ref, "x", 1)
        result["approved_jobs"][0]["input_ref"] = self.ref
        with self.assertRaises(c.ContractError):
            chain._released_policy(result, self.ref, self.parent_ref, self.ref,
                                   "actual_parent/query_train", 1)

    def test_independent_cpu_caps_remain_fully_reserved(self):
        policy = dict(approved_native_chain_ref=self.ref, approved_jobs=[],
                      independent_cpu_output_cap_bytes=64*chain.GiB)
        result = chain._released_policy(policy, self.ref, self.parent_ref, self.ref,
                                        "actual_parent/query_train", 76*chain.GiB)
        self.assertEqual(result["approved_new_storage_cap_bytes"],
                         (76+64)*chain.GiB+chain.METADATA_CAP)
        self.assertEqual(result["independent_cpu_output_cap_bytes"], 64*chain.GiB)
        self.assertEqual(policy["independent_cpu_output_cap_bytes"], 64*chain.GiB)
        for invalid in (-1, True, 1.5, "64GiB"):
            policy["independent_cpu_output_cap_bytes"] = invalid
            with self.assertRaises(c.ContractError):
                chain._released_policy(policy, self.ref, self.parent_ref, self.ref,
                                       "actual_parent/query_train", 76*chain.GiB)

    def test_live_parent_waits_without_policy_mutation(self):
        self.parent["status"] = "started"
        class Queue:
            def jobs(q): return [self.parent]
        times = iter([0, 100001])
        with patch.object(chain, "_load", return_value=(self.spec, 100000)), \
             patch.object(chain, "_scheduler", return_value=Queue()), \
             patch.object(chain.c, "file_ref", return_value=self.ref), \
             patch.object(chain.c, "check_execution_release",
                          return_value={"approved_native_chain_ref": self.ref}), \
             patch.object(chain, "_chain_lock", return_value=nullcontext()), \
             patch.object(chain, "_write_policy") as policy, \
             patch.object(chain, "_prepare") as prepare:
            result = chain.run(self.ref, execute=True, clock=lambda: next(times), sleep=lambda _: None)
        self.assertEqual(result["status"], "blocked")
        prepare.assert_not_called()
        policy.assert_not_called()

    def test_actual_failed_child_halts_after_one_queue_dispatch(self):
        test = self
        class Queue:
            controller_lock_path = "mock"
            def __init__(q): q.entries=[test.parent]; q.enqueues=0; q.runs=0
            def jobs(q): return q.entries
            def _lock(q, _): return nullcontext()
            def enqueue(q, _):
                q.enqueues+=1
                q.entries.append(test.entry("actual_parent/query_train", "queued"))
            def run(q, **kwargs):
                q.runs+=1
                q.entries[-1].update(status="failed", actual_returncode=1)
        queue = Queue()
        policy = dict(approved_native_chain_ref=self.ref, approved_jobs=[])
        with patch.object(chain, "_load", return_value=(self.spec, 100000)), \
             patch.object(chain, "_scheduler", return_value=queue), \
             patch.object(chain.c, "file_ref", return_value=self.ref), \
             patch.object(chain.c, "check_execution_release", return_value=policy), \
             patch.object(chain, "_chain_lock", return_value=nullcontext()), \
             patch.object(chain, "_write_policy", return_value=self.ref), \
             patch.object(chain, "_prepare", return_value=(self.parent_ref, self.ref)):
            result = chain.run(self.ref, execute=True, clock=lambda: 0, sleep=lambda _: None)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(queue.enqueues, 1)
        self.assertEqual(queue.runs, 1)

    def test_stage_materialization_writes_only_metadata(self):
        with tempfile.TemporaryDirectory(dir=c.ROOT / ".runtime/tmp") as temporary:
            spec = dict(self.spec, records_directory=str(Path(temporary).relative_to(c.ROOT)))
            def scientific(*args, **kwargs):
                return dict(output_directory=str(Path(kwargs["output_directory"]).relative_to(c.ROOT)))
            with patch.object(chain, "_producer_bundle", return_value={}), \
                 patch.object(chain.prediction, "build_embedding_spec", side_effect=scientific):
                job_ref, receipt_ref = chain._prepare(spec, chain.STAGES[0], [self.parent])
            job, receipt = c.read_ref(job_ref), c.read_ref(receipt_ref)
            self.assertEqual(job["callable_name"], "run_embedding")
            self.assertEqual(job["max_output_bytes"], chain.GiB)
            self.assertEqual(receipt["input_ref"], job_ref)
            self.assertFalse(receipt["historical_wait_claimed"])
            self.assertFalse(c.contained_path(job["output_directory"]).exists())

    def test_real_planner_has_no_scientific_imports(self):
        import sys
        parent_ref = c.file_ref("configs/benchmarks/jobs/native_argformer_val000_contrastive_ordered_state.json")
        spec = chain.build_spec(parent_input_ref=parent_ref,
                                output_root="experiments/results/benchmark_chain_mock",
                                records_directory=".runtime/tmp/benchmark_chain_mock_records")
        self.assertEqual(spec["total_prediction_scope_caps_bytes"], 76*chain.GiB)
        self.assertFalse(any(n in sys.modules for n in ("torch", "numpy", "composer", "transformers")))


if __name__ == "__main__":
    unittest.main()
