"""Source-only tests. No guards/models/NVML/process/data payload calls."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
from unittest.mock import patch
import ast,copy,ctypes,hashlib,importlib.util,json,math,sys,types,unittest
HERE=Path(__file__).resolve().parent
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
def load(name):
    ms=importlib.util.spec_from_file_location(name,HERE/(name+".py"))
    m=importlib.util.module_from_spec(ms);ms.loader.exec_module(m);return m
def extracted(filename,names,namespace):
    tree=ast.parse((HERE/filename).read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),filename,"exec"),namespace)
    return namespace
class Checks(unittest.TestCase):
    def test_observed_optimizer_counters_not_timestamp_alias(self):
        ns=extracted("native_child_v1.py",{"optimizer_counter_evidence"}, {})
        f=ns["optimizer_counter_evidence"]
        def state(step):return {"DecoupledAdamW":{"state":{0:{"step":step,"exp_avg":object(),"exp_avg_sq":object()}}}}
        self.assertEqual(f(state(224775))["unique_step_values"],[224775])
        self.assertEqual(f(state(SimpleNamespace(item=lambda:224775.)))["maximum_step"],224775)
        for bad in [state(3),state(224774)]:
            self.assertNotEqual(f(bad)["unique_step_values"],[224775])
        for bad in [{},state(False),state(1.5),{"DecoupledAdamW":{"state":{}}}]:
            with self.assertRaises((ValueError,TypeError)):f(bad)
    def test_final_checkpoint_genuine_Adam_and_Trainer_counts(self):
        values={"state":{"timestamp":{"epoch":5,"batch":224775,"batch_in_epoch":0},
            "model":{"weight":object()},"optimizers":{"DecoupledAdamW":{"state":{0:{"step":224775,"exp_avg":1,"exp_avg_sq":2}}}}}}
        class P:
            name="ep5.pt";suffix=".pt"
            def is_file(self):return True
            def is_symlink(self):return False
            def stat(self):return SimpleNamespace(st_size=100)
        path=P()
        ns=extracted("native_child_v1.py",{"_int_time","optimizer_counter_evidence","_final_checkpoint"},{})
        kwargs=dict(epochs=5,updates=224775,max_files=2,max_bytes=1000,
            torch=SimpleNamespace(load=lambda *a,**k:values),reference=lambda p:{"path":"own.pt","sha256":"observed"})
        result=ns["_final_checkpoint"](SimpleNamespace(iterdir=lambda:[path]),**kwargs)
        self.assertEqual(result["actual_optimizer_updates"],224775)
        values["state"]["optimizers"]["DecoupledAdamW"]["state"][0]["step"]=3
        with self.assertRaises(ValueError):ns["_final_checkpoint"](SimpleNamespace(iterdir=lambda:[path]),**kwargs)
    def test_fresh72h_expiry_only_new_context(self):
        now=datetime(2026,10,15,12,tzinfo=timezone.utc)
        class D:
            @staticmethod
            def now(tz):return now
            fromisoformat=staticmethod(datetime.fromisoformat)
        ns=extracted("root_native_contract_v1.py",{"active"},{"datetime":D,"timezone":timezone,
            "require":lambda v,m:None if v else (_ for _ in ()).throw(ValueError(m))})
        spec=dict(created_at_utc=(now-timedelta(seconds=1)).isoformat(),expires_at_utc=(now+timedelta(seconds=259199)).isoformat())
        ns["active"](spec)
        spec["expires_at_utc"]=(now+timedelta(seconds=259200)).isoformat()
        with self.assertRaises(ValueError):ns["active"](spec)
        self.assertNotIn("_check_active_v1",(HERE/"context_factory_v1.py").read_text())
        driver=(HERE/"root_native_driver_v1.py").read_text()
        self.assertIn('job="American-C3-native-MLM-full30mb-mlm-fit-v8"',driver)
        self.assertNotIn("min(86400",driver)
    def test_progress_actual_first_batch_and_epoch_bounded(self):
        p=load("progress_v8");messages=[];clock=[100.]
        stream=p.Progress(clock=lambda:clock[0],emit=messages.append)
        clock[0]=120.;stream.feed(b"[epoch=1][bat")
        stream.feed(b"ch=1/44955]:\n\t Train throughput/batches_per_sec: 0.5\n")
        self.assertEqual(len(messages),2)
        value=json.loads(messages[0][len(p.PREFIX):])
        self.assertEqual(value["batch_in_epoch"],1)
        self.assertEqual(value["seconds_since_native_spawn_observation"],20.)
        for i in range(10000):stream.feed(b"[epoch=1][batch=100/44955]\n")
        self.assertLessEqual(len(messages),48)
        stream.feed(b"[epoch=2][batch=1/44955]\n")
        self.assertTrue(any(b'"epoch_one_based": 2' in m for m in messages))
    def test_NVML_layout_and_dynamic_size_without_library(self):
        g=load("gpu_monitor_v8")
        self.assertEqual(ctypes.sizeof(g.ProcessInfo),24)
        self.assertEqual([x[0] for x in g.ProcessInfo._fields_],["pid","usedGpuMemory","gpuInstanceId","computeInstanceId"])
        monitor=g.OwnedGpuMemory.__new__(g.OwnedGpuMemory);monitor.owner_pid=123;monitor.handle=None
        def query(handle,count,array):
            if array is None:count._obj.value=1;return 7
            count._obj.value=1;array[0].pid=99;array[0].usedGpuMemory=1024;return 0
        monitor.processes=query
        with patch.object(g.os,"getpid",return_value=123):
            self.assertEqual(monitor.query(),[{"pid":99,"used_memory_bytes":1024}])
        with patch.object(g.os,"getpid",return_value=124):
            with self.assertRaises(ValueError):monitor.query()
    def test_no_recurring_probe_children_or_writer_omission(self):
        context=ast.parse((HERE/"context_factory_v1.py").read_text())
        f=next(n for n in context.body if isinstance(n,ast.FunctionDef) and n.name=="_fast_guard")
        self.assertNotIn("subprocess",ast.unparse(f))
        self.assertIn("gpu.owned_bytes(roster)",ast.unparse(f))
        old=(ROOT/"research/reviews/American_C3_native_MLM_full_source_v5/root_native_driver_v1.py").read_text()
        new=(HERE/"root_native_driver_v1.py").read_text()
        for name in ["_tree","_signal_owned"]:
            def fn(t):return ast.dump(next(n for n in ast.parse(t).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
            self.assertEqual(fn(old),fn(new))
    def test_author_science_rules_and_fresh_optimizer_preserved(self):
        old=ROOT/"research/reviews/American_C3_native_MLM_full_source_v5/native_child_v1.py"
        def function(path,name):return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
        self.assertEqual(function(old,"_io_config"),function(HERE/"native_child_v1.py","_io_config"))
        child=(HERE/"native_child_v1.py").read_text()
        self.assertIn("trainer=module.main(config,return_trainer=True)",child)
        self.assertNotIn("qualify(module",child)
        self.assertIn('and packet["stage"]=="mlm"',child)
        supervisor=(HERE/"supervisor_v1.py").read_text()
        self.assertIn('actual_optimizer_updates=receipt["actual_optimizer_updates"]',supervisor)
        self.assertNotIn("actual_optimizer_updates=224775",supervisor)
    def test_syntax_and_no_old_qualifier_or_lifetime_dispatch(self):
        names=["root_native_contract_v1.py","root_native_driver_v1.py","context_factory_v1.py","consumer_contract_v1.py",
            "native_child_v1.py","supervisor_v1.py","native_runtime_v1.py","owned_lifecycle_v2.py",
            "terminal_bounds_v1.py","qualifier_evidence_v8.py","progress_v8.py","gpu_monitor_v8.py"]
        for name in names:
            text=(HERE/name).read_text();ast.parse(text,filename=name)
            self.assertNotIn("amendment.check_lifetime",text)
        self.assertIn('GPU_PHASES=("mlm",)',(HERE/"consumer_contract_v1.py").read_text())
    def test_exact_caps_and_selected_fresh_deadline(self):
        caps=json.loads((HERE/"mlm_caps.inactive.json").read_text());caps["active"]=True
        request=json.loads((HERE/"mlm_request.inactive.json").read_text())
        def require(value,message):
            if not value:raise ValueError(message)
        ns=extracted("root_native_contract_v1.py",{"validate_caps"},{"require":require,
            "PARENT":"american_c3_train_000","DRIVER":"driver.py","PREP_REF":caps["preparation_ref"],
            "ref":lambda p:caps["driver_source_ref"]})
        ns["validate_caps"](caps,request)
        for key,value in [("operational_deadline_seconds",259201),("operational_deadline_seconds",True),
            ("reservation_cap_bytes",14*2**30+1),("phase_write_budget_bytes",12*2**30-1),
            ("max_cuda_reserved_bytes",50000000001),("aggregate_cache_OS_quota_claimed",True)]:
            bad=copy.deepcopy(caps);bad[key]=value
            with self.assertRaises(ValueError):ns["validate_caps"](bad,request)
    def nvml_fixture(self):
        helper=load("qualifier_evidence_v8")
        def reference(path):return {"path":path,"sha256":hashlib.sha256((ROOT/path).read_bytes()).hexdigest()}
        gpu4_path=HERE/"NVML_GPU4_actual_root_preflight_v1.json"
        gpu4=json.loads(gpu4_path.read_text());gpu5=copy.deepcopy(gpu4)
        raw=json.loads(gpu5["stdout"]);raw.update(physical_gpu=5,GPU_UUID="GPU-unit-test-only",
            own_process_memory=[{"pid":123,"nvml_bytes":1048577,"smi_bytes":1048576,"difference_bytes":1}])
        gpu5.update(physical_gpu=5,stdout=json.dumps(raw))
        refs=[reference(str(gpu4_path.relative_to(ROOT))),{"path":"mock_gpu5_receipt.json","sha256":"test-only"}]
        acceptance={"schema":"native_MLM_NVML_ABI_root_acceptance_v8","root_authorizer":"root_agent","active":True,
            "actual_probe_refs":refs,"root_accepts_actual_UUID_memory_comparison":True,
            "root_accepts_observed_live_memory_differences":True,"root_confirms_probe_PIDs_are_owned":True}
        data={refs[0]["path"]:gpu4,refs[1]["path"]:gpu5}
        runtime=json.loads((ROOT/"research/reviews/American_C3_native_MLM_full_source_v4/runtime_root_GPU4_v1.json").read_text())
        return helper,acceptance,data,runtime,reference
    def test_actual_NVML_envelope_and_owned_memory_acceptance(self):
        helper,acceptance,data,runtime,reference=self.nvml_fixture()
        actual=json.loads((HERE/"nvml_abi_acceptance.inactive.json").read_text())
        actual.update(active=True,root_accepts_actual_UUID_memory_comparison=True,
            root_accepts_observed_live_memory_differences=True,root_confirms_probe_PIDs_are_owned=True)
        def actual_read(ref):
            path=ROOT/ref["path"]
            self.assertLessEqual(path.stat().st_size,1048576)
            self.assertEqual(reference(ref["path"]),ref)
            return json.loads(path.read_text())
        helper.validate_nvml_acceptance(actual,runtime,read=actual_read,reference=reference)
        read=lambda ref:copy.deepcopy(data[ref["path"]])
        helper.validate_nvml_acceptance(acceptance,runtime,read=read,reference=reference)
        for key in ["active","root_confirms_probe_PIDs_are_owned","root_accepts_observed_live_memory_differences"]:
            bad=copy.deepcopy(acceptance);bad[key]=False
            with self.assertRaises(ValueError):helper.validate_nvml_acceptance(bad,runtime,read=read,reference=reference)
        original=copy.deepcopy(data)
        for kind in ["empty","missing_smi","bad_source","exit1","target_uuid"]:
            data.clear();data.update(copy.deepcopy(original))
            ref=acceptance["actual_probe_refs"][1 if kind!="target_uuid" else 0]["path"]
            envelope=data[ref];raw=json.loads(envelope["stdout"])
            if kind=="empty":raw["own_process_memory"]=[]
            if kind=="missing_smi":raw["own_process_memory"][0]["smi_bytes"]=None
            if kind=="bad_source":envelope["source_sha256"]="wrong"
            if kind=="exit1":envelope["returncode"]=1
            if kind=="target_uuid":raw["GPU_UUID"]="GPU-wrong"
            envelope["stdout"]=json.dumps(raw)
            with self.assertRaises(ValueError):helper.validate_nvml_acceptance(acceptance,runtime,read=read,reference=reference)
    def test_genuine_closed_v7_three_step_evidence_and_negative_cases(self):
        helper,acceptance,data,runtime,reference=self.nvml_fixture()
        request=json.loads((HERE/"mlm_request.inactive.json").read_text())
        caps=json.loads((HERE/"mlm_caps.inactive.json").read_text())
        timing=json.loads((HERE/"timing_review.inactive.json").read_text())
        timing.update(active=True,root_accepts_unproven_full_budget_completion=True)
        for key,ref in request["refs"].items():
            if key not in ("timing_review","nvml_abi_acceptance"):data[ref["path"]]=json.loads((ROOT/ref["path"]).read_text())
        for ref in request["upstream_wait_refs"].values():data[ref["path"]]=json.loads((ROOT/ref["path"]).read_text())
        for name in ["qualifier_input","qualifier_source_closure"]:
            ref=helper.FIXED[name];data[ref["path"]]=json.loads((ROOT/ref["path"]).read_text())
        q=data[request["refs"]["qualification"]["path"]]
        child_ref=q["child_completion_ref"];data[child_ref["path"]]=json.loads((ROOT/child_ref["path"]).read_text())
        data[request["refs"]["timing_review"]["path"]]=timing
        data[request["refs"]["nvml_abi_acceptance"]["path"]]=acceptance
        read=lambda ref:copy.deepcopy(data[ref["path"]])
        proof=helper.validate_qualifier_and_timing(request,caps,runtime,read=read,reference=reference)
        self.assertEqual(proof["optimizer_steps"],3)
        original=copy.deepcopy(data)
        for kind in ["512_updates","corpus_RAM","wait","outer","close","timing_extrapolation"]:
            data.clear();data.update(copy.deepcopy(original))
            if kind=="512_updates":data[child_ref["path"]]["qualification"]["optimizer_steps"]=512
            if kind=="corpus_RAM":data[child_ref["path"]]["qualification"]["dataloader_corpus_RAM_qualified"]=True
            if kind=="wait":data[request["upstream_wait_refs"]["qualification"]["path"]]["returncode"]=1
            if kind=="outer":data[request["upstream_wait_refs"]["qualification_outer"]["path"]]["returncode"]=1
            if kind=="close":data[request["refs"]["qualification_close"]["path"]]["value"]["future_writes_and_relaunch_revoked"]=False
            if kind=="timing_extrapolation":data[request["refs"]["timing_review"]["path"]]["synthetic_step_rate_is_full_corpus_rate"]=True
            with self.assertRaises(ValueError):helper.validate_qualifier_and_timing(request,caps,runtime,read=read,reference=reference)
    def test_author_final_checkpoint_scheduler_source_and_behavior(self):
        proof=json.loads((HERE/"final_checkpoint_source_proof.json").read_text())
        for name in ["scheduler_source_ref","saver_source_ref","state_source_ref"]:
            ref=proof[name];self.assertEqual(hashlib.sha256((ROOT/ref["path"]).read_bytes()).hexdigest(),ref["sha256"])
        source=(ROOT/proof["scheduler_source_ref"]["path"]).read_text()
        fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="create_interval_scheduler")
        self.assertEqual(fn.args.args[1].arg,"include_end_of_training");self.assertTrue(fn.args.defaults[0].value)
        saver=(ROOT/proof["saver_source_ref"]["path"]).read_text()
        self.assertIn("save_interval = create_interval_scheduler(save_interval)",saver)
        Event=SimpleNamespace(**{k:k for k in ["BATCH_CHECKPOINT","EPOCH_CHECKPOINT","ITERATION_CHECKPOINT","BATCH_END","EPOCH_END","ITERATION_END"]})
        TimeUnit=SimpleNamespace(**{k:k for k in ["ITERATION","EPOCH","BATCH","TOKEN","SAMPLE","DURATION","SECOND"]})
        class Time:
            unit=TimeUnit.BATCH;value=500
            def __int__(self):return self.value
            @staticmethod
            def from_input(*args):return Time()
        fake=types.ModuleType("composer.core");fake.Event=Event;fake.Time=Time;fake.TimeUnit=TimeUnit;fake.State=object
        ns={"math":math}
        exec(compile("from __future__ import annotations\n"+ast.get_source_segment(source,fn),"<source-only-scheduler>","exec"),ns)
        with patch.dict(sys.modules,{"composer":types.ModuleType("composer"),"composer.core":fake}):
            schedule=ns["create_interval_scheduler"]("500ba")
        state=SimpleNamespace(previous_timestamp=SimpleNamespace(get=lambda unit:224774),
            timestamp=SimpleNamespace(batch=224775,get=lambda unit:224775),get_elapsed_duration=lambda:1.)
        self.assertTrue(schedule(state,Event.EPOCH_CHECKPOINT))
        state.get_elapsed_duration=lambda:.8
        self.assertFalse(schedule(state,Event.BATCH_CHECKPOINT))
    def test_source_closure_and_canonical_original_launcher_bindings(self):
        ns={"__file__":str(HERE/"root_native_contract_v1.py")}
        exec(compile((HERE/"root_native_contract_v1.py").read_text(),"<inert-root-contract>","exec"),ns)
        paths=ns["source_paths"]()
        for rel in ["qualifier_evidence_v8.py","progress_v8.py","gpu_monitor_v8.py","nvml_abi_provenance.json"]:
            self.assertIn(str((HERE/rel).relative_to(ROOT)),paths)
        self.assertIn("research/reviews/American_C3_native_MLM_full_source_v5/pidfd_backend_v1.py",paths)
        self.assertIn("src/arggnn/american_c3_native_mlm_full_fit_v8.py",paths)
        driver=(HERE/"root_native_driver_v1.py").read_text()
        self.assertIn("(*native.PROJECT_SOURCES,c.SOURCE,c.DRIVER,c.CONTEXT,c.RUNTIME_HELPER)",driver)
        self.assertIn("consumer_bindings=consumers",driver)
        self.assertEqual(ns["LAUNCHER_REF"]["sha256"],"d4cc990e430625e9cc7d02739e7e8012f5c8b3bdc78f048f801febacea64583d")

if __name__=="__main__":unittest.main()
