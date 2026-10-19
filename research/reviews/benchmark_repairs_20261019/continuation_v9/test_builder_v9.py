"""Metadata-only V9 preservation, qualification and private reporter tests."""
from pathlib import Path
import copy
import unittest
from contextlib import ExitStack
from unittest import mock
from arggnn.benchmarking import contracts as c
import build_benchmark_workflow_v9 as builder


def ref(name):
    return dict(path="mock/"+name+".json", sha256=("a" if name!="new" else "b")*64)


def fixture():
    old="experiments/results/NEVER_CREATED_v9_test_prior"
    new="experiments/results/NEVER_CREATED_v9_test_successor"
    history="experiments/results/NEVER_CREATED_v9_test_AP_test004_v7"
    def task(key,deps=(),resource="cpu"):
        value=dict(id=key,resource=resource,dependencies=list(deps),factory_module="mock.factory",
            factory_function="make_call",factory_arguments={},output_directory=old+"/"+key,
            max_output_bytes=1000,max_rss_bytes=2000,max_seconds=3000)
        if resource=="gpu0":
            value.update(minimum_free_gpu_bytes=74*2**30,max_cuda_reserved_bytes=74*2**30)
        return value
    done=task("done")
    dependent=task("done_dependent",("done",))
    registry_ap=task("registry_ap",("done_dependent",))
    registry_c4=task("registry_c4",("done_dependent",))
    healthy_am=task("AM_treeloom_fit",("done",),"gpu0")
    healthy_c4=task("C4_validation_infer",("done",))
    inherited_id="AP__ancestralpaths_european_test_004__ap_infer"
    inherited_fail=dict(id=inherited_id,resource="external",dependencies=["done_dependent"],
        workflow_predecessor=dict(original_job_ref=ref("APtest004_original"),marker="failed",
            output_directory=history,max_output_bytes=96*2**30,max_rss_bytes=128*2**30))
    original_ap=task(inherited_id,("done",))
    original_ap.update(output_directory=history,max_output_bytes=96*2**30,max_rss_bytes=128*2**30,
        factory_module="arggnn.benchmarking.european_workflow",
        factory_arguments=dict(stage="ap_infer",arguments=dict(seed=77,
            log_path=history+"/native/log.json",unchanged_simulation_ref=ref("recipe"))))
    failed=task("failed",("done",))
    pending=task("pending",("done_dependent",))
    pending["factory_arguments"]=dict(kept_input=old+"/done/native/value.json",
        retried_input=old+"/failed/native/data.json",seed=88)
    blocked=task("blocked",("failed",))
    completed_external=dict(id="completed_external",resource="external",dependencies=["done"],
        workflow_predecessor=dict(original_job_ref=ref("completed_original"),
            source_task_id="completed_external",resource="cpu",max_output_bytes=1000,
            max_rss_bytes=2000,output_directory=history+"_completed"))
    tasks=[done,dependent,registry_ap,registry_c4,healthy_am,healthy_c4,inherited_fail,
           failed,pending,blocked,completed_external]
    for prefix,fit_id,predict_id,module in builder.EUROPEAN_ROUTES:
        registry="registry_ap" if prefix=="AP" else "registry_c4"
        protocol="ancestralpaths_paper200mb_native6" if prefix=="AP" else "ancestralpaths_paper200mb_c4"
        observed=dict(inference_ref={"$task":"done_dependent"},
            observed_vcf_ref={"$task":"done","path":["observed_input_refs","genotypes.vcf"]},
            sampled_registry_ref={"$task":registry,"path":["registry_ref"]})
        common=dict(protocol_id=protocol,sampling_plan_ref=ref("sampled_plan"),
            max_sites=5000000,max_metadata_bytes=512*2**20,
            preparation_limits=dict(max_nodes=10**9,max_holds=500000,
                                    max_episode_payload_bytes=64*2**30),
            max_rss_bytes=512*2**30,observation_cache_bytes=128*2**30,
            preparation_headroom_bytes=128*2**30,max_output_bytes=64*2**20)
        fit=task(fit_id,(registry,),"gpu0")
        fit.update(factory_module=module,factory_function="fit_factory")
        fit["factory_arguments"]=dict(common,training_headroom_bytes=128*2**30,
            train_inputs={builder.EUROPEAN_PARENT:dict(observed,training_targets_ref=ref(prefix+"_truth"))})
        pred=task(predict_id,(fit_id,registry),"gpu0")
        pred.update(factory_module=module,factory_function="prediction_factory")
        pred["factory_arguments"]=dict(common,heldout_inputs={"test":observed},
            fit_completion_ref={"$task":fit_id,"path":["completion_ref"]},
            prediction_headroom_bytes=64*2**30)
        tasks.extend((fit,pred))
    entries={t["id"]:dict(status="pending") for t in tasks}
    for key in ("done","done_dependent","registry_ap","registry_c4"):
        entries[key]=dict(status="succeeded",result_ref=ref(key),retained_bytes=10,reservation_bytes=100)
    for key in ("AM_treeloom_fit","C4_validation_infer"):
        entries[key]=dict(status="running",pid=123 if key=="AM_treeloom_fit" else 124,
                          job_ref=ref(key+"_job"),reservation_bytes=1000)
    entries["completed_external"]=dict(status="succeeded",external=True,result_ref=ref("completed_result"))
    entries[inherited_id]=dict(status="failed",external=True,error="Original workflow predecessor failed")
    entries["failed"]=dict(status="failed",actual_controller_wait=-15,owned_process_group_absent=True,
        job_ref=ref("failed_job"),reservation_bytes=1000,retained_bytes=300)
    entries["blocked"]=dict(status="blocked")
    prior_ref,snapshot_ref=ref("prior"),ref("snapshot")
    plan=dict(schema="arggnn_executable_workflow_v1",physical_gpu=0,tasks=tasks,sources=[ref("old_source")],
        external_storage=[dict(cap_bytes=111,scopes=["kept/prior"],completion_evidence=[])],
        prior_unknown_rss_bytes=12345,external_native_rss_bytes=12345,
        dispatch_hold_task_ids=[route[1] for route in builder.EUROPEAN_ROUTES]+["pending"],
        dispatch_hold_reason="old holds")
    state=dict(plan_ref=prior_ref,tasks=entries)
    charge=16673226804
    old_fail_entry=dict(status="failed",retained_bytes=charge,actual_controller_wait=-15,
                       owned_process_group_absent=True)
    def read(value):
        if value==prior_ref:return copy.deepcopy(plan)
        if value==snapshot_ref:return copy.deepcopy(state)
        raise AssertionError("Unexpected metadata read "+str(value))
    def live(t,e,*_args):
        return dict(original_job_ref=e["job_ref"],source_task_id=t["id"],resource=t["resource"],
                    max_output_bytes=t["max_output_bytes"],max_rss_bytes=t["max_rss_bytes"],
                    output_directory=t["output_directory"],scientific_pid=e["pid"]+100)
    patches=[
        mock.patch.object(builder.c,"read_ref",side_effect=read),
        mock.patch.object(builder,"verify_success",side_effect=lambda t,e,p:
            dict(result_ref=e["result_ref"],actual_wait_ref=ref(t["id"]+"_wait"))),
        mock.patch.object(builder,"bind_live",side_effect=live),
        mock.patch.object(builder.predecessor,"_source",return_value=(original_ap,old_fail_entry)),
        mock.patch.object(builder.predecessor,"_drained",return_value=(ref("APtest004_wait"),
            dict(output_bytes=charge,actual_direct_wait=True,actual_direct_returncode=-15,
                 owned_process_group_absent=True))),
        mock.patch.object(builder.prior_builder,"_job",return_value={}),
        mock.patch.object(builder.prior_builder,"_wait",return_value=(ref("failed_wait"),
            dict(output_bytes=250,owned_process_group_absent=True,
                 actual_direct_wait=True,actual_direct_returncode=-15))),
        mock.patch.object(builder,"resource_corrections",side_effect=lambda p:
            (copy.deepcopy(p),dict(mocked_metadata_only=True))),
        mock.patch.object(builder,"archive_routes",side_effect=lambda p,root:
            (copy.deepcopy(p),dict(archive_tasks_added=[],prospective_savings_credited_bytes=0)))]
    return locals()


class V9BuilderTest(unittest.TestCase):
    def run_fixture(self,f):
        with ExitStack() as stack:
            for patch in f["patches"]:stack.enter_context(patch)
            return builder.prepare_continuation(f["prior_ref"],f["snapshot_ref"],
                records="mock/v9_records",state_path="mock/v8_live_state.json",
                prior_output=f["old"],output=f["new"],expected_successes=4)

    def test_successes_live_jobs_full_original_graph_and_failed_charge_preserved(self):
        f=fixture();source=copy.deepcopy(f["plan"]);state=copy.deepcopy(f["state"])
        plan,entries,evidence=self.run_fixture(f)
        byid={t["id"]:t for t in plan["tasks"]}
        self.assertEqual(set(byid),set(f["entries"]) | {
            prefix+"__train_000__treeloom_preparation_qualification"
            for prefix,*rest in builder.EUROPEAN_ROUTES})
        for key in ("done","done_dependent","registry_ap","registry_c4"):
            self.assertEqual(byid[key],next(t for t in source["tasks"] if t["id"]==key))
            for name,value in state["tasks"][key].items():self.assertEqual(entries[key][name],value)
            self.assertIn("adopted_from_v8",entries[key])
        for key in ("AM_treeloom_fit","C4_validation_infer"):
            self.assertEqual(byid[key]["resource"],"external")
            self.assertTrue(entries[key]["external"])
            self.assertNotIn("reservation_bytes",entries[key])
            original=next(t for t in source["tasks"] if t["id"]==key)
            self.assertEqual(byid[key]["workflow_predecessor"]["original_job_ref"],
                             state["tasks"][key]["job_ref"])
            self.assertEqual(byid[key]["workflow_predecessor"]["resource"],original["resource"])
            self.assertNotIn("deferred_gpu0",byid[key])
        retry=byid[f["inherited_id"]]
        target=c.ROOT/f["new"]/"inherited"/f["inherited_id"]
        self.assertEqual(c.contained_path(retry["output_directory"]),target)
        self.assertEqual(retry["factory_arguments"]["arguments"]["log_path"],str(target.relative_to(c.ROOT))+"/native/log.json")
        self.assertEqual(retry["factory_arguments"]["arguments"]["seed"],77)
        self.assertEqual(retry["dependencies"],["done_dependent"])
        self.assertEqual(evidence["total_failed_charge_bytes"],16673226804+300)
        self.assertEqual(plan["external_storage"][-1]["cap_bytes"],16673226804+300)
        self.assertEqual(plan["external_storage"][0],source["external_storage"][0])
        self.assertEqual(byid["pending"]["factory_arguments"]["kept_input"],f["old"]+"/done/native/value.json")
        self.assertEqual(byid["pending"]["factory_arguments"]["retried_input"],f["new"]+"/failed/native/data.json")
        self.assertEqual(plan["continuation_approval_key"],"approved_resource_continuation_ref")
        completed_binding=f["completed_external"]["workflow_predecessor"]
        self.assertIn(completed_binding,plan["workflow_external_predecessors"])
        self.assertEqual(evidence["retained_completed_predecessors"],[completed_binding])
        self.assertEqual(len(evidence["live_predecessors"]),2)
        self.assertEqual(byid["completed_external"],f["completed_external"])
        self.assertEqual(f["plan"],source);self.assertEqual(f["state"],state)

    def test_observation_only_full_parent_qualification_and_original_science_arguments(self):
        f=fixture();plan,entries,evidence=self.run_fixture(f)
        byid={t["id"]:t for t in plan["tasks"]}
        for prefix,fit_id,predict_id,oldmodule in builder.EUROPEAN_ROUTES:
            qual_id=prefix+"__train_000__treeloom_preparation_qualification"
            qual,fit,predict=(byid[key] for key in (qual_id,fit_id,predict_id))
            self.assertEqual(qual["resource"],"cpu")
            self.assertEqual(qual["max_rss_bytes"],128*2**30)
            self.assertEqual(qual["max_seconds"],7200)
            self.assertEqual(set(qual["factory_arguments"]["observed_inputs"]),
                             {"inference_ref","observed_vcf_ref","sampled_registry_ref"})
            self.assertNotIn("training_targets_ref",str(qual))
            self.assertEqual(len(qual["dependencies"]),3)
            for modified in (fit,predict):
                original=next(t for t in f["tasks"] if t["id"]==modified["id"])
                self.assertEqual(modified["factory_module"],builder.EUROPEAN_FACTORY)
                self.assertEqual(modified["dependencies"],original["dependencies"]+[qual_id])
                expected=dict(original["factory_arguments"],qualification_ref={"$task":qual_id})
                self.assertEqual(modified["factory_arguments"],expected)
                for cap in ("minimum_free_gpu_bytes","max_cuda_reserved_bytes"):
                    self.assertEqual(modified[cap],original[cap])
            self.assertEqual(fit["resource"],"cpu")
            self.assertTrue(fit["deferred_gpu0"])
            self.assertEqual(predict["resource"],"gpu0")
            self.assertNotIn("deferred_gpu0",predict)
        self.assertEqual(plan["dispatch_hold_task_ids"],["pending"])

    def test_missing_real_wait_and_undrained_failure_rejected(self):
        f=fixture();f["state"]["tasks"]["failed"]["owned_process_group_absent"]=False
        with self.assertRaisesRegex(ValueError,"unknown failed producer"):self.run_fixture(f)
        f=fixture();f["state"]["tasks"]["failed"].pop("actual_controller_wait")
        with self.assertRaisesRegex(ValueError,"actual controller wait"):self.run_fixture(f)

    def test_old_source_pins_verbatim_conflicts_rejected(self):
        inherited=[ref("old_source")]
        with mock.patch.object(builder.c,"file_ref",return_value=ref("new")):
            result=builder.merge_source_refs(inherited,["mock/new.json"])
        self.assertIn(ref("old_source"),result)
        with mock.patch.object(builder.c,"file_ref",return_value=dict(ref("old_source"),sha256="f"*64)):
            with self.assertRaisesRegex(ValueError,"Never overwrite"):
                builder.merge_source_refs(inherited,[inherited[0]["path"]])

    def test_new_sources_and_nonreusing_namespaces(self):
        self.assertEqual(builder.PRIOR_PLAN,"configs/benchmarks/executable_workflow_20261019_v8.json")
        self.assertEqual(builder.PLAN,"configs/benchmarks/executable_workflow_20261019_v9.json")
        self.assertEqual(builder.RECORDS,".runtime/benchmarking/workflow_20261019_v9")
        self.assertTrue(builder.PRIOR_OUTPUT.endswith("release_v8"))
        self.assertTrue(builder.OUTPUT.endswith("release_v9"))
        for suffix in ("workflow_v9.py","workflow_reporting_v9.py","european_preparation_v9.py",
                       "resource_accounting_v1.py","native_chain_settlement_v1.json"):
            self.assertTrue(any(path.endswith(suffix) for path in builder.NEW_SOURCES))


class V9ReporterTest(unittest.TestCase):
    def test_private_reporter_endpoints_and_shared_lock_remain_isolated(self):
        from arggnn.benchmarking import workflow_reporting_v9 as current
        from arggnn.benchmarking import workflow_reporting_v8 as previous
        self.assertNotEqual(current.BEGIN,previous.BEGIN)
        self.assertIn("CURRENT RESOURCE BENCHMARK",current.BEGIN)
        self.assertEqual(current.STATUS_PATH,"research/benchmark_resource_current.json")
        self.assertEqual(current.SCORE_PATH,"research/resource_matched_benchmark_results.json")
        self.assertIs(current.publish.__globals__["_core"],current._core)
        self.assertIs(current.ProgressReporter.submit.__globals__["publish"],current.publish)
        self.assertEqual(current._core.REPORT_PATHS,("README.md",current.STATUS_PATH,current.SCORE_PATH))
        self.assertEqual(previous.STATUS_PATH,"research/benchmark_active_current.json")
        # The outer publisher inherited from V4 still uses the same shared lock.
        self.assertIn(".runtime/benchmarking/workflow_reporting/reporting.lock",
                      current.publish.__code__.co_consts)
        report=dict(counts={s:0 for s in current._core.STATUSES},stages={},run_status="test",
            observed_utc="metadata_test",running_job_names=[],external_queue_jobs=[],
            reporting_errors=[],preserved_workflow_jobs=[],complete=False)
        block=current._markdown(report,{"results":{}})
        self.assertIn(current.BEGIN,block)
        self.assertIn(current.STATUS_PATH,block)

    def test_reporter_stops_when_resource_approval_is_superseded(self):
        from arggnn.benchmarking import workflow_reporting_v9 as reporter
        plan_ref=ref("new")
        with mock.patch.object(reporter.c,"file_ref",return_value=plan_ref), \
             mock.patch.object(reporter.c,"read_json",side_effect=[
                 dict(approved_resource_continuation_ref=plan_ref),dict(complete=False),
                 dict(approved_resource_continuation_ref=ref("future"))]), \
             mock.patch.object(reporter,"publish",return_value=dict(status="updated")) as publish, \
             mock.patch.object(reporter.time,"sleep") as sleep:
            self.assertEqual(reporter.main(["--plan","mock/plan","--state","mock/state","--loop","--exit-on-complete"]),0)
        self.assertEqual(publish.call_count,1)
        sleep.assert_called_once_with(60)


if __name__=="__main__":
    unittest.main()
