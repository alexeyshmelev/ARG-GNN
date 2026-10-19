"""Metadata preservation/retry/source pin and isolated reporter checks; no freeze."""
from pathlib import Path
import copy
import importlib
import sys
import unittest
from unittest import mock
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
import build_benchmark_workflow_v8 as builder
from arggnn.benchmarking import contracts as c


def ref(name):
    return dict(path="mock/"+name+".json",sha256=("a" if name!="new" else "b")*64)


def fixture():
    old="experiments/results/NEVER_CREATED_v8_test_prior"
    new="experiments/results/NEVER_CREATED_v8_test_successor"
    history="experiments/results/NEVER_CREATED_v8_test_historical_qual"
    def task(key,deps=(),resource="cpu"):
        return dict(id=key,resource=resource,dependencies=list(deps),factory_module="mock.factory",
                    factory_function="make_call",factory_arguments={},output_directory=old+"/"+key,
                    max_output_bytes=1000,max_rss_bytes=2000,max_seconds=3000)
    done=task("done");dependent=task("done_dependent",("done",))
    healthy=task("healthy",("done",))
    inherited_live=dict(id="inherited_live",resource="external",dependencies=["done"],
                        workflow_predecessor=dict(original_job_ref=ref("live_original"),marker="live"))
    inherited_fail=dict(id=builder.QUALIFICATION,resource="external",dependencies=["done_dependent"],
                        workflow_predecessor=dict(original_job_ref=ref("qual_original"),marker="failed",
                          output_directory=history,max_output_bytes=1000,max_rss_bytes=2000))
    original_qual=task(builder.QUALIFICATION,("done",))
    original_qual["output_directory"]=history
    original_qual["factory_module"]="arggnn.benchmarking.treeloom_preparation_v6"
    original_qual["factory_arguments"]=dict(
        output_directory=str(c.ROOT/history/"native"),
        nested=[dict(path=history+"/native/log.json"),dict(path=history+"_other/keep.json")],
        preparation_ref={"$task":"done_dependent"},seed=77)
    failed=task("failed",("done",))
    pending=task("pending",("done_dependent",))
    pending["factory_arguments"]=dict(kept_input=old+"/done/native/value.json",
                                      retried_input=old+"/failed/native/data.json",seed=88)
    blocked=task("blocked",("failed",))
    fit=task("AM_treeloom_fit",(builder.QUALIFICATION,))
    fit["factory_module"]="arggnn.benchmarking.treeloom_preparation_v6"
    tasks=[done,dependent,healthy,inherited_live,inherited_fail,failed,pending,blocked,fit]
    entries={t["id"]:dict(status="pending") for t in tasks}
    for key in ("done","done_dependent"):
        entries[key]=dict(status="succeeded",result_ref=ref(key),retained_bytes=10,reservation_bytes=100)
    entries["healthy"]=dict(status="running",pid=123,job_ref=ref("healthy_job"),reservation_bytes=1000)
    entries[builder.QUALIFICATION]=dict(status="failed",external=True,error="Original workflow predecessor failed")
    entries["inherited_live"]=dict(status="pending",external=True)
    entries["failed"]=dict(status="failed",actual_controller_wait=-15,owned_process_group_absent=True,
                           job_ref=ref("failed_job"),reservation_bytes=1000,retained_bytes=300)
    entries["blocked"]=dict(status="blocked")
    prior_ref=ref("prior");snapshot_ref=ref("snapshot")
    plan=dict(schema="arggnn_executable_workflow_v1",physical_gpu=0,tasks=tasks,sources=[ref("old_source")],
              external_storage=[dict(cap_bytes=111,scopes=["kept/prior"],completion_evidence=[])],
              prior_unknown_rss_bytes=12345,dispatch_hold_task_ids=["AP_fit","C4_fit"])
    state=dict(plan_ref=prior_ref,tasks=entries)
    live_binding=dict(original_job_ref=ref("healthy_job"),source_task_id="healthy",
                      resource="cpu",max_output_bytes=1000,output_directory=old+"/healthy")
    old_live_task=task("inherited_live");old_live_entry=dict(status="running")
    old_fail_entry=dict(status="failed",retained_bytes=101)
    def source(binding):
        return (old_live_task,old_live_entry) if binding["marker"]=="live" else (original_qual,old_fail_entry)
    def read(value):
        if value==prior_ref:return copy.deepcopy(plan)
        if value==snapshot_ref:return copy.deepcopy(state)
        raise AssertionError("Unexpected metadata read "+str(value))
    patches=[
      mock.patch.object(builder.c,"read_ref",side_effect=read),
      mock.patch.object(builder,"verify_success",side_effect=lambda t,e,p:dict(result_ref=e["result_ref"],actual_wait_ref=ref(t["id"]+"_wait"))),
      mock.patch.object(builder,"bind_live",return_value=live_binding),
      mock.patch.object(builder.predecessor,"_source",side_effect=source),
      mock.patch.object(builder.predecessor,"_drained",return_value=(ref("inherited_wait"),dict(output_bytes=100))),
      mock.patch.object(builder.prior_builder,"_job",return_value={}),
      mock.patch.object(builder.prior_builder,"_wait",return_value=(ref("failed_wait"),dict(
          output_bytes=250,owned_process_group_absent=True,actual_direct_wait=True,actual_direct_returncode=-15)))]
    return locals()


class ContinuationPreservationTest(unittest.TestCase):
    def run_fixture(self,f):
        from contextlib import ExitStack
        with ExitStack() as stack:
            for patch in f["patches"]:stack.enter_context(patch)
            return builder.prepare_continuation(f["prior_ref"],f["snapshot_ref"],
                records="mock/v8_records",state_path="mock/v7_live_state.json",
                prior_output=f["old"],output=f["new"],expected_successes=2)

    def test_completed_and_live_work_preserved_failed_original_factory_is_mapped_and_routed(self):
        f=fixture();source=copy.deepcopy(f["plan"]);state=copy.deepcopy(f["state"])
        plan,entries,evidence=self.run_fixture(f)
        byid={t["id"]:t for t in plan["tasks"]}
        for key in ("done","done_dependent"):
            self.assertEqual(byid[key],next(t for t in source["tasks"] if t["id"]==key))
            for field,value in state["tasks"][key].items():self.assertEqual(entries[key][field],value)
            self.assertIn("adopted_from_v7",entries[key])
        self.assertEqual(len(evidence["adopted_succeeded"]),2)
        self.assertEqual(len(evidence["live_predecessors"]),2)
        for key in ("healthy","inherited_live"):
            self.assertEqual(byid[key]["resource"],"external")
            self.assertTrue(entries[key]["external"])
            self.assertNotIn("reservation_bytes",entries[key])
        retry=byid[builder.QUALIFICATION]
        target=c.ROOT/f["new"]/"inherited"/builder.QUALIFICATION
        self.assertEqual(c.contained_path(retry["output_directory"]),target)
        self.assertEqual(retry["factory_arguments"]["output_directory"],str(target/"native"))
        self.assertEqual(retry["factory_arguments"]["nested"][0]["path"],
                         str(target.relative_to(c.ROOT))+"/native/log.json")
        self.assertEqual(retry["factory_arguments"]["nested"][1]["path"],f["history"]+"_other/keep.json")
        self.assertEqual(retry["dependencies"],["done_dependent"])
        self.assertEqual(retry["factory_module"],builder.FACTORY_MODULE)
        self.assertEqual(byid["AM_treeloom_fit"]["factory_module"],builder.FACTORY_MODULE)
        self.assertEqual(byid["pending"]["factory_arguments"]["kept_input"],f["old"]+"/done/native/value.json")
        self.assertEqual(byid["pending"]["factory_arguments"]["retried_input"],f["new"]+"/failed/native/data.json")
        self.assertEqual(byid["pending"]["factory_arguments"]["seed"],88)
        self.assertEqual(evidence["total_failed_charge_bytes"],401)
        self.assertEqual(plan["external_storage"][0],source["external_storage"][0])
        self.assertEqual(plan["external_storage"][-1]["cap_bytes"],401)
        self.assertEqual(plan["prior_unknown_rss_bytes"],12345)
        self.assertEqual(plan["dispatch_hold_task_ids"],source["dispatch_hold_task_ids"])
        self.assertEqual(plan["continuation_approval_key"],"approved_cohort_continuation_ref")
        self.assertEqual(f["plan"],source);self.assertEqual(f["state"],state)

    def test_missing_wait_or_undrained_failed_attempt_is_rejected(self):
        f=fixture();f["state"]["tasks"]["failed"]["owned_process_group_absent"]=False
        with self.assertRaisesRegex(ValueError,"unknown failed producer"):
            self.run_fixture(f)
        f=fixture();f["state"]["tasks"]["failed"].pop("actual_controller_wait")
        with self.assertRaisesRegex(ValueError,"actual controller wait"):
            self.run_fixture(f)

    def test_source_pins_retained_verbatim_conflicts_rejected(self):
        inherited=[ref("old_source")];old=copy.deepcopy(inherited)
        with mock.patch.object(builder.c,"file_ref",return_value=ref("new")) as read:
            result=builder.merge_source_refs(inherited,["mock/new.json"])
        read.assert_called_once_with("mock/new.json")
        self.assertIn(ref("old_source"),result);self.assertIn(ref("new"),result)
        self.assertEqual(inherited,old)
        with mock.patch.object(builder.c,"file_ref",return_value=dict(ref("old_source"),sha256="f"*64)):
            with self.assertRaisesRegex(ValueError,"Never overwrite"):
                builder.merge_source_refs(inherited,[inherited[0]["path"]])

    def test_new_source_and_scope_constants(self):
        self.assertEqual(builder.PRIOR_PLAN,"configs/benchmarks/executable_workflow_20261018_v7.json")
        self.assertEqual(builder.PLAN,"configs/benchmarks/executable_workflow_20261019_v8.json")
        self.assertEqual(builder.RECORDS,".runtime/benchmarking/workflow_20261019_v8")
        self.assertTrue(builder.PRIOR_OUTPUT.endswith("release_v7"))
        self.assertTrue(builder.OUTPUT.endswith("release_v8"))
        required=("workflow_v8.py","workflow_reporting_v8.py","treeloom_preparation_v8.py",
                  "finite_pss_v1.py","cut_hoist_v1.py","integration_v1.py")
        for suffix in required:self.assertTrue(any(p.endswith(suffix) for p in builder.NEW_SOURCES))


class IsolatedReporterTest(unittest.TestCase):
    def test_private_publisher_and_reporter_class_use_new_paths_and_preserve_v7_formatter(self):
        from arggnn.benchmarking import workflow_reporting_v8 as current
        from arggnn.benchmarking import workflow_reporting_v6 as older
        self.assertNotEqual(current.BEGIN,older.BEGIN)
        self.assertEqual(current.STATUS_PATH,"research/benchmark_active_current.json")
        self.assertEqual(current.SCORE_PATH,"research/active_matched_benchmark_results.json")
        self.assertIs(current.publish.__globals__["_core"],current._core)
        self.assertIs(current.ProgressReporter.submit.__globals__["publish"],current.publish)
        self.assertEqual(current._core.REPORT_PATHS,("README.md",current.STATUS_PATH,current.SCORE_PATH))
        self.assertNotEqual(current._core.RUNTIME_PATH,older._core.RUNTIME_PATH)
        self.assertEqual(older.STATUS_PATH,"research/benchmark_repaired_current.json")
        report=dict(counts={s:0 for s in current._core.STATUSES},stages={},run_status="test",
            observed_utc="metadata_test",running_job_names=[],external_queue_jobs=[],
            reporting_errors=[],preserved_workflow_jobs=[],complete=False)
        block=current._markdown(report,{"results":{}})
        self.assertIn(current.BEGIN,block);self.assertIn(current.END,block)
        self.assertIn(current.STATUS_PATH,block);self.assertNotIn(older.STATUS_PATH,block)

    def test_cli_uses_cohort_approval_and_stops_before_obsolete_publish(self):
        from arggnn.benchmarking import workflow_reporting_v8 as reporter
        plan_ref=ref("new")
        with mock.patch.object(reporter.c,"file_ref",return_value=plan_ref), \
             mock.patch.object(reporter.c,"read_json",side_effect=[
               dict(approved_cohort_continuation_ref=plan_ref),dict(complete=False),
               dict(approved_cohort_continuation_ref=ref("future"))]), \
             mock.patch.object(reporter,"publish",return_value=dict(status="updated")) as publish, \
             mock.patch.object(reporter.time,"sleep") as sleep:
            self.assertEqual(reporter.main(["--plan","mock/plan","--state","mock/state","--loop","--exit-on-complete"]),0)
        self.assertEqual(publish.call_count,1);sleep.assert_called_once_with(60)


if __name__=="__main__":unittest.main()
