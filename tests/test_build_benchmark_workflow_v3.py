"""Tiny metadata-only v3 adoption tests; no scientific factory or payload."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from arggnn.benchmarking import contracts as c

spec=importlib.util.spec_from_file_location("v3_builder",c.ROOT/"scripts/build_benchmark_workflow_v3.py")
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


class ContinuationTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=c.ROOT/"tests")
        self.root=Path(self.temp.name)
        self.patcher=patch.object(c,"ROOT",self.root);self.patcher.start()
        self.addCleanup(self.patcher.stop);self.addCleanup(self.temp.cleanup)
        self.old="experiments/results/old"
        self.new="experiments/results/new"
        def task(identity,stage="fit",deps=()):
            return dict(id=identity,resource="gpu0",dependencies=list(deps),
                factory_module="arggnn.benchmarking.synthetic",factory_function="factory",
                factory_arguments=dict(stage=stage),output_directory=self.old+"/"+identity,
                max_output_bytes=10000,max_seconds=100,max_rss_bytes=10000)
        completed=task("AP__complete","prepare");completed["resource"]="cpu"
        completed["factory_arguments"]["max_metadata_bytes"]=128*2**20
        self.tasks=[completed,task(builder.HOLDS[0]),task(builder.HOLDS[1]),task(builder.HOLDS[2]),
                    task(builder.FIRST_GPU,"mlm",["AP__complete"]),task("retire","retire",[builder.FIRST_GPU])]
        self.tasks[2]["factory_arguments"]["nested"]={"max_metadata_bytes":128*2**20}
        self.tasks[-1]["factory_arguments"]["owner_scopes"]=[completed["output_directory"],self.tasks[-2]["output_directory"]]
        self.plan=dict(schema="arggnn_executable_workflow_v1",physical_gpu=0,tasks=self.tasks,
                       sources=[],external_storage=[],records_directory="oldrecords")
        self.plan_ref=self.write("old_plan.json",self.plan)
        self.entries={t["id"]:dict(status="pending",external=False) for t in self.tasks}
        self.success(completed)
        self.failure(self.tasks[-2],retained=200)
        self.failure(self.tasks[1],retained=None)
        self.entries[self.tasks[2]["id"]]["status"]="blocked"
        self.entries[self.tasks[3]["id"]]["status"]="blocked"
        self.state_ref=self.save_state()

    def write(self,path,value):
        p=self.root/path;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(value));return c.file_ref(p)

    def job(self,task):
        return self.write("jobs/"+task["id"]+".json",dict(plan_ref=self.plan_ref,task=task,
                          task_fingerprint=c.fingerprint(task)))

    def success(self,task):
        job=self.job(task);out=task["output_directory"]
        completion=self.write(out+"/native/completion.json",dict(status="completed",
            prediction_payload_ref=dict(path="MUST_NOT_READ.huge",sha256="0"*64)))
        result=self.write(out+"/result.json",dict(job_ref=job,value=completion))
        self.write(out+"/actual_wait.json",dict(job_ref=job,actual_direct_wait=True,
            actual_direct_returncode=0,owned_process_group_absent=True,supervisor_error=None,output_bytes=10))
        self.entries[task["id"]]=dict(status="succeeded",external=False,job_ref=job,result_ref=result,
            actual_controller_wait=0,owned_process_group_absent=True,reservation_bytes=10000,retained_bytes=100)

    def failure(self,task,retained):
        job=self.job(task);out=task["output_directory"]
        self.write(out+"/actual_wait.json",dict(job_ref=job,actual_direct_wait=True,
            actual_direct_returncode=-15,owned_process_group_absent=True,supervisor_error="cancelled",output_bytes=100))
        entry=dict(status="failed",external=False,job_ref=job,actual_controller_wait=1,
                   owned_process_group_absent=True,reservation_bytes=10000)
        if retained is not None:entry["retained_bytes"]=retained
        self.entries[task["id"]]=entry

    def save_state(self):
        return self.write("old_state.json",dict(plan_ref=self.plan_ref,tasks=self.entries))

    def prepare(self):
        return builder.prepare_continuation(self.plan_ref,self.state_ref,records="newrecords",
                                             prior_output=self.old,output=self.new)

    def test_adopts_exact_success_resets_fresh_and_charges_failures(self):
        original=copy.deepcopy(self.plan)
        plan,entries,evidence=self.prepare()
        byid={t["id"]:t for t in plan["tasks"]}
        self.assertEqual(byid["AP__complete"],original["tasks"][0])
        self.assertEqual(entries["AP__complete"]["result_ref"],self.entries["AP__complete"]["result_ref"])
        self.assertEqual(entries["AP__complete"]["retained_bytes"],100)
        self.assertEqual(entries[builder.FIRST_GPU]["status"],"pending")
        self.assertNotIn("reservation_bytes",entries[builder.FIRST_GPU])
        self.assertEqual(byid[builder.FIRST_GPU]["output_directory"],self.new+"/"+builder.FIRST_GPU)
        self.assertEqual(byid["retire"]["factory_arguments"]["owner_scopes"],
            [self.old+"/AP__complete",self.new+"/"+builder.FIRST_GPU])
        self.assertEqual(evidence["adopted_succeeded"],1)
        self.assertEqual(evidence["total_failed_charge_bytes"],10200)
        self.assertEqual(evidence["first_ready_gpu_task_id"],builder.FIRST_GPU)
        self.assertEqual(plan["tasks"][0]["id"],builder.FIRST_GPU)
        self.assertEqual(plan["dispatch_hold_task_ids"],list(builder.HOLDS))
        self.assertEqual(byid[builder.HOLDS[1]]["factory_arguments"]["nested"]["max_metadata_bytes"],512*2**20)
        for task in plan["tasks"]:self.assertEqual(task["dependencies"],next(t for t in original["tasks"] if t["id"]==task["id"])["dependencies"])
        self.assertEqual(c.read_ref(self.plan_ref),original)
        self.assertFalse((self.root/self.new).exists())

    def test_succeeded_European_descriptor_remains_exactly_unchanged(self):
        original=copy.deepcopy(self.tasks[0])
        plan,entries,_=self.prepare()
        actual=next(task for task in plan["tasks"] if task["id"]=="AP__complete")
        self.assertEqual(actual,original)
        self.assertEqual(c.fingerprint(actual),c.fingerprint(original))
        self.assertEqual(actual["factory_arguments"]["max_metadata_bytes"],128*2**20)
        self.assertEqual(entries["AP__complete"]["job_ref"],self.entries["AP__complete"]["job_ref"])
        pending=next(task for task in plan["tasks"] if task["id"]==builder.HOLDS[1])
        self.assertEqual(pending["factory_arguments"]["nested"]["max_metadata_bytes"],512*2**20)

    def test_running_unknown_rejected_before_adoption(self):
        for status in ("running","unknown"):
            self.entries[builder.FIRST_GPU]["status"]=status;self.state_ref=self.save_state()
            with self.assertRaisesRegex(ValueError,"no running/unknown"):self.prepare()

    def test_missing_failed_wait_and_unconfirmed_group_rejected(self):
        path=self.root/self.tasks[-2]["output_directory"]/"actual_wait.json"
        path.unlink()
        with self.assertRaises(OSError):self.prepare()
        self.write(str(path.relative_to(self.root)),dict(job_ref=self.entries[builder.FIRST_GPU]["job_ref"],
            actual_direct_wait=True,actual_direct_returncode=1,owned_process_group_absent=False,output_bytes=100))
        with self.assertRaisesRegex(ValueError,"absent owned group"):self.prepare()

    def test_tampered_result_metadata_rejected(self):
        result=self.root/self.entries["AP__complete"]["result_ref"]["path"]
        result.write_text('{}')
        with self.assertRaisesRegex(c.ContractError,"Metadata changed"):self.prepare()

    def test_successful_result_cannot_hide_unsuccessful_direct_wait(self):
        task=self.tasks[0];self.write(task["output_directory"]+"/actual_wait.json",dict(
            job_ref=self.entries["AP__complete"]["job_ref"],actual_direct_wait=True,
            actual_direct_returncode=1,owned_process_group_absent=True,supervisor_error=None))
        with self.assertRaisesRegex(ValueError,"genuinely successful"):self.prepare()

    def test_existing_new_release_and_new_state_are_never_overwritten(self):
        (self.root/self.new).mkdir(parents=True)
        with self.assertRaisesRegex(ValueError,"release root must be fresh"):self.prepare()
        marker=self.root/builder.RECORDS/"state.json";marker.parent.mkdir(parents=True);marker.write_text("preserve")
        with self.assertRaisesRegex(ValueError,"Never overwrite"):builder.build()
        self.assertEqual(marker.read_text(),"preserve")


if __name__=="__main__":unittest.main()
