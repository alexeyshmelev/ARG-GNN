"""Bounded CPU tests; every archive mutation uses synthetic fixture payloads."""
import copy
import gzip
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.repairs.benchmark_recovery_20261020.storage import closed_archives_v1 as recovery

HERE = Path(__file__).resolve().parent
REAL_ROOT = HERE.parents[3]

def save(root, relative, data):
    item = root / relative
    item.parent.mkdir(parents=True, exist_ok=True)
    item.write_text(json.dumps(data, sort_keys=True))
    return recovery.reference(root, item)

class Fixture:
    def __init__(self, root):
        self.root = root
        self.scope = "experiments/results/benchmark_campaign_20261018_workflow/release_fixture/european/parent/ap_infer"
        owner = root / self.scope
        (owner / "native/pre_size").mkdir(parents=True)
        raw = owner / "native/pre_size/props.bin"
        with raw.open("wb") as handle:
            handle.write(b"synthetic closed ancestry intermediate\n" * 20000)
            handle.flush()
            os.fsync(handle.fileno())
        evidence = owner / "native/evidence.log"
        evidence.write_text("Synthetic original failure evidence stays unchanged.\n")
        identifier = "AP__synthetic__ap_infer"
        task = dict(id=identifier, output_directory=self.scope, resource="cpu", max_output_bytes=2**30)
        source_ref = save(root, "source_plan.json", dict(records_directory="source_records", tasks=[task]))
        job_ref = save(root, "original_job.json", dict(plan_ref=source_ref, task=task))
        wait_ref = save(root, self.scope + "/actual_wait.json",
                        dict(job_ref=job_ref, actual_direct_wait=True,
                             actual_direct_returncode=-15, owned_process_group_absent=True))
        before = recovery.scope_bytes(owner)
        self.entry = dict(status="failed", actual_controller_wait=1, owned_process_group_absent=True,
                          job_ref=job_ref, retained_bytes=before)
        save(root, "source_records/state.json", dict(plan_ref=source_ref, tasks={identifier:self.entry}))
        candidates = [dict(path=str(raw.relative_to(root)), **recovery.record(raw))]
        protected = [dict(path=str(item.relative_to(root)), **recovery.record(item))
                     for item in (evidence, owner / "actual_wait.json")]
        bound = math.ceil((2*candidates[0]["charge_bytes"]+4*recovery.METADATA)/2**30)*2**30
        self.target = dict(key="synthetic", scope=self.scope,
                           producer=dict(wait_ref=wait_ref, job_ref=job_ref, source_plan_ref=source_ref,
                                         task_id=identifier, entry=self.entry),
                           before_scope_bytes=before, candidates=candidates, protected_files=protected,
                           max_output_bytes=bound, future_old_scope_writes_require_root_revocation=True,
                           accounting=dict(kind="external_storage", group_scopes=[self.scope], group_cap_bytes=before))
        codec_path = root / recovery.CODEC
        codec_path.parent.mkdir(parents=True)
        codec_path.write_text((REAL_ROOT / recovery.CODEC).read_text())
        inventory = dict(schema=recovery.SCHEMA+"_inventory", targets=[self.target],
                         initial_credit_bytes=0, historical_unknown_scopes_untouched=True,
                         codec_ref=recovery.reference(root, codec_path))
        self.inventory_ref = save(root, recovery.INVENTORY, inventory)
        self.plan = dict(storage_ceiling_bytes=980_000_000_000,
                         external_storage=[dict(scopes=[self.scope], cap_bytes=before)],
                         workflow_external_predecessors=[], tasks=[])
        self.plan, _ = recovery.add_recovery_tasks(self.plan, "new_outputs", repo=root)
        self.task = self.plan["tasks"][0]
        self.spec = dict(schema=recovery.SCHEMA, inventory_ref=self.inventory_ref,
                         target_key="synthetic",
                         output_directory=str(root/self.task["output_directory"]/"native"))

    def run(self):
        original_loader = recovery.load_target
        with patch.object(recovery, "ROOT", self.root), patch.object(recovery, "load_target",
             side_effect=lambda ref,key: original_loader(ref,key,repo=self.root)), patch.dict(os.environ, CUDA_VISIBLE_DEVICES=""):
            manifest_ref = recovery.run(self.spec, resource_check=lambda event: None, execution_check=lambda: None)
        job_ref = save(self.root, "new_job.json", dict(task=self.task, call=dict(module=recovery.MODULE,
                       function="run", spec=self.spec)))
        output = self.task["output_directory"]
        save(self.root, output+"/actual_wait.json", dict(job_ref=job_ref, actual_direct_wait=True,
             actual_direct_returncode=0, owned_process_group_absent=True))
        result_ref = save(self.root, output+"/result.json", dict(job_ref=job_ref,value=manifest_ref))
        entry = dict(status="succeeded", actual_controller_wait=0, owned_process_group_absent=True,
                     job_ref=job_ref, result_ref=result_ref,
                     retained_bytes=recovery.scope_bytes(self.root/output))
        return manifest_ref, {"tasks":{self.task["id"]:entry}}

class ClosedRecoveryTests(unittest.TestCase):
    def test_real_inventory_has_only_exact_closed_targets_and_zero_credit(self):
        plan = json.loads((REAL_ROOT/"configs/benchmarks/executable_workflow_20261019_v9.json").read_text())
        before = copy.deepcopy(plan)
        changed, evidence = recovery.add_recovery_tasks(plan, "experiments/results/source_only_recovery")
        self.assertEqual(plan, before)
        self.assertEqual(evidence["tasks_added"], 12)
        self.assertEqual(evidence["initial_credit_bytes"], 0)
        tasks = changed["tasks"][:12]
        self.assertEqual(changed["tasks"][12:], plan["tasks"])
        state = {"tasks":{task["id"]:{"status":"pending"} for task in tasks}}
        self.assertEqual(recovery.completed_storage_reduction(changed,state),0)

    def test_synthetic_lossless_roundtrip_and_exact_external_adjustment(self):
        with tempfile.TemporaryDirectory(dir=HERE) as value:
            fixture = Fixture(Path(value))
            manifest_ref,state = fixture.run()
            manifest = recovery.read(fixture.root,manifest_ref)
            item = manifest["entries"][0]
            self.assertEqual(gzip.open(item["archive_path"],"rb").read(),
                             b"synthetic closed ancestry intermediate\n"*20000)
            self.assertFalse((fixture.root/item["original_path"]).exists())
            delta = recovery.completed_storage_reduction(fixture.plan,state,repo=fixture.root)
            current = recovery.scope_bytes(fixture.root/fixture.scope)
            self.assertEqual(delta,fixture.target["before_scope_bytes"]-current)
            archive_retained = state["tasks"][fixture.task["id"]]["retained_bytes"]
            self.assertEqual(fixture.target["before_scope_bytes"]+archive_retained-delta,
                             current+archive_retained)

    def test_missing_actual_archive_wait_cannot_release_credit(self):
        with tempfile.TemporaryDirectory(dir=HERE) as value:
            fixture = Fixture(Path(value))
            _,state = fixture.run()
            save(fixture.root,fixture.task["output_directory"]+"/actual_wait.json",
                 dict(job_ref=state["tasks"][fixture.task["id"]]["job_ref"],
                      actual_direct_wait=True,actual_direct_returncode=1,owned_process_group_absent=True))
            with self.assertRaises(ValueError):
                recovery.completed_storage_reduction(fixture.plan,state,repo=fixture.root)

    def test_duplicate_recovery_never_double_credits(self):
        with tempfile.TemporaryDirectory(dir=HERE) as value:
            fixture = Fixture(Path(value))
            _,state = fixture.run()
            duplicate = copy.deepcopy(fixture.task)
            duplicate["id"] += "_duplicate"
            fixture.plan["tasks"].append(duplicate)
            state["tasks"][duplicate["id"]] = copy.deepcopy(state["tasks"][fixture.task["id"]])
            with self.assertRaises(ValueError):
                recovery.completed_storage_reduction(fixture.plan,state,repo=fixture.root)

    def test_live_original_or_unknown_drain_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as value:
            fixture = Fixture(Path(value))
            changed = copy.deepcopy(fixture.entry)
            changed["status"] = "running"
            source_ref = fixture.target["producer"]["source_plan_ref"]
            save(fixture.root,"source_records/state.json",
                 dict(plan_ref=source_ref,tasks={"AP__synthetic__ap_infer":changed}))
            with self.assertRaises(ValueError):
                recovery.load_target(fixture.inventory_ref,"synthetic",repo=fixture.root)

    def test_protected_evidence_tamper_is_detected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as value:
            fixture = Fixture(Path(value))
            _,state = fixture.run()
            (fixture.root/fixture.scope/"native/evidence.log").write_text("changed")
            with self.assertRaises(ValueError):
                recovery.completed_storage_reduction(fixture.plan,state,repo=fixture.root)

    def test_external_cap_change_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as value:
            fixture = Fixture(Path(value))
            _,state = fixture.run()
            fixture.plan["external_storage"][0]["cap_bytes"] -= 1
            with self.assertRaises(ValueError):
                recovery.completed_storage_reduction(fixture.plan,state,repo=fixture.root)

    def test_candidate_policy_preserves_observed_tree_truth_and_metadata(self):
        for relative in ("native/post_size.observed.trees","native/inference.json",
                         "native/commands.json","truth/targets.npz","actual_wait.json"):
            self.assertFalse(recovery.candidate(relative,True))
        self.assertTrue(recovery.candidate("native/pre_size/props.bin",True))
        self.assertFalse(recovery.candidate("native/pre_size/props.bin",False))
        self.assertTrue(recovery.candidate("native/inputs/input.vcf",False))

if __name__ == "__main__":
    unittest.main()
