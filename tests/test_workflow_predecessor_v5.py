"""V5 metadata and real tiny CPU producer regressions, without CUDA or science."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
from arggnn.benchmarking import contracts as c
from arggnn.benchmarking import workflow_predecessor_v5 as helper

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,c.ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
builder=load("v5_builder","scripts/build_benchmark_workflow_v5.py")
old_tests=load("v4_fixtures","tests/test_workflow_predecessor_v4.py")

class V5Tests(unittest.TestCase):
    def setUp(self):
        self.f=old_tests.ContinuationTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)

    def materialization_only(self):
        f=self.f
        path=f.root/f.retry["output_directory"]
        (path/"actual_wait.json").unlink();path.rmdir()
        f.entries[f.retry["id"]]=dict(status="failed",external=False,
            error="Materialization: synthetic public-lineage rejection")
        f.state_ref=f.save_state()
        identity=helper.proc_identity(f.process.pid)
        f.write(f.live["output_directory"]+"/supervision_identity.json",
            dict(job_ref=f.entries["live"]["job_ref"],child_identity=identity,
                 schema="workflow_v4_Popen_identity_v1"))

    def prepare(self):
        f=self.f
        return builder.prepare_continuation(f.plan_ref,f.state_ref,state_path="oldrecords/state.json",
            records="newrecords",prior_output=f.old,output=f.new,expected_successes=1)

    def test_real_inherited_CPU_producer_stays_external_and_fully_reserved(self):
        f=self.f
        self.assertIsNone(helper.external_result(f.external))
        self.assertEqual(helper.external_resources(f.external_plan),(1,100000000))
        self.assertEqual(helper.storage_reservation_bytes(f.external_plan),100000)
        self.assertEqual(helper.live_group_bindings(f.external_plan)[0]["pid"],f.process.pid)

    def test_GPU_binding_reserves_RAM_without_inventing_a_CPU_lane(self):
        # Synthetic resource arithmetic only. The fixture process itself is CPU.
        binding=dict(self.f.binding,resource="gpu0")
        with patch.object(helper,"_source",return_value=({},{"status":"running"})), \
             patch.object(helper,"_drained",return_value=None):
            self.assertEqual(helper.external_resources({"workflow_external_predecessors":[binding]}),
                             (0,100000000))
            self.assertEqual(helper.storage_reservation_bytes({"workflow_external_predecessors":[binding]}),
                             100000)

    def test_genuine_CPU_wait_unwraps_large_payload_reference_without_opening_it(self):
        f=self.f;value={"path":"giant.json","sha256":"b"*64}
        with (f.root/"giant.json").open("wb") as stream:stream.truncate(c.MAX_METADATA_BYTES+1)
        f.finish_real_producer(value)
        self.assertEqual(helper.external_result(f.external)["value"],value)
        self.assertEqual(helper.external_resources(f.external_plan),(0,0))
        self.assertEqual(helper.storage_reservation_bytes(f.external_plan),1000)
        with self.assertRaises(c.ContractError):c.read_ref(value)

    def test_current_live_group_uses_durable_original_identity_and_no_duplicate_cap(self):
        self.materialization_only();f=self.f
        plan,entries,evidence=self.prepare()
        self.assertEqual(len(plan["workflow_external_predecessors"]),1)
        binding=plan["workflow_external_predecessors"][0]
        self.assertIn("original_supervision_identity_ref",binding)
        self.assertEqual(binding["owned_process_group_id"],f.process.pid)
        self.assertEqual(binding["resource"],"cpu")
        self.assertNotIn("reservation_bytes",entries["live"])
        self.assertEqual(plan["external_storage"],[])
        self.assertEqual(evidence["total_failed_charge_bytes"],0)
        tasks={t["id"]:t for t in plan["tasks"]}
        self.assertEqual(tasks["complete"],f.done)
        self.assertEqual(tasks[f.retry["id"]]["output_directory"],f.new+"/"+f.retry["id"])
        self.assertFalse((f.root/f.new).exists())

    def test_inherited_external_binding_is_flattened_once_across_another_generation(self):
        self.materialization_only();f=self.f
        evidence=f.write("mid_evidence.json",dict(adopted_succeeded={
            "complete":dict(result_ref=f.entries["complete"]["result_ref"],
                            actual_wait_ref=c.file_ref(f.root/f.done["output_directory"]/"actual_wait.json"))}))
        mid=copy.deepcopy(f.plan);mid["records_directory"]="midrecords"
        mid["continuation_evidence_ref"]=evidence
        mid["tasks"]=[copy.deepcopy(f.done),copy.deepcopy(f.external),copy.deepcopy(f.retry)]
        midref=f.write("mid_plan.json",mid)
        entries=copy.deepcopy(f.entries);entries["live"]=dict(status="pending",external=True)
        state=dict(plan_ref=midref,tasks=entries)
        f.write("midrecords/state.json",state);snapshot=f.write("mid_snapshot.json",state)
        plan,new,evidence=builder.prepare_continuation(midref,snapshot,state_path="midrecords/state.json",
            records="newrecords",prior_output=f.old,output=f.new,expected_successes=1)
        self.assertEqual(plan["workflow_external_predecessors"],[f.binding])
        self.assertEqual(new["live"]["preserved_original_job_ref"],f.binding["original_job_ref"])
        self.assertNotIn("reservation_bytes",new["live"])
        self.assertEqual(plan["external_storage"],[])

    def test_durable_identity_tampering_is_rejected(self):
        self.materialization_only();f=self.f
        binding=builder.bind_live(f.live,f.entries["live"],f.plan_ref,f.state_ref,"oldrecords/state.json")
        task=dict(id="live",resource="external",workflow_predecessor=binding)
        path=f.root/binding["original_supervision_identity_ref"]["path"]
        path.write_text("{}")
        with self.assertRaises(c.ContractError):helper.external_result(task)

    def test_old_unknown_cannot_be_silently_relabelled_as_new_drained_failure(self):
        # Original fixture's attempted failure has explicit groupabsence=false.
        with self.assertRaisesRegex(ValueError,"separate ROOT acceptance"):
            self.prepare()

if __name__=="__main__":unittest.main()
