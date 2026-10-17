"""Concrete H058 context: original reader/output/configure_fit/episode checks."""
from pathlib import Path
import ast,hashlib,io,json,os,resource,stat,sys,time
from research.reviews.H058_GPU0_full_training_successor_source_20261017_v8.common_v1 import *

def build_context_v1(record,spec):
    s,q,cfg,storage,protocol=validate_input_v1(record)
    require_work_window_v2(spec)
    require(s==spec,"Same actual admitted input")
    from arggnn import census_tsinfer_fit_v1 as fit
    reader=fit.ConsumerReaderV1();check_fit_active=root_fit_clock_view_v5(record,spec,q,cfg,fit)
    check_fit_active(spec)
    g=guard_v1()
    g.validate_storage_v9(spec,dict(output_cap_bytes=CAPS["reservation_bytes"]),storage,
        source_ref=ref(DRIVER),config_ref=spec["refs"]["config"],runtime_ref=spec["refs"]["runtime"])
    # Reuse the exact original strict writer body, changing only its local owned
    # serialization counter to enforce the inherited192MiB scientific cap.
    source=raw(GUARD).decode();tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="PilotOutputV9")
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="save")
    body=ast.get_source_segment(source,fn)
    before="self.used+planned<=self.storage['reserved_new_bytes']"
    require(body.count(before)==1,"Single original writer local-cap expression")
    body=body.replace(before,"self.used+planned<=min(self.storage['reserved_new_bytes'],192*2**20)")
    lazy="from arggnn.census_tsinfer_fit_v1 import _jsonable_v1"
    require(body.count(lazy)==1,"Exact lazy serializer source site")
    body=body.replace(lazy,"_jsonable_v1 = self._H058_jsonable")
    ns=dict(vars(g));exec(compile(body,"<original_selected_writer_with_H058_192MiB_local_cap>","exec"),ns)
    H058Output=type("H058OriginalOutput192MiBV1",(g.PilotOutputV9,),{"save":ns["save"]})
    output=H058Output(spec["output_directory"],storage,reader)
    counts=reader.decode_json(cfg["source_count_ref"])["parents"]
    original=reader.decode_json(protocol.H050_INPUT_REF)
    generation=reader.decode_json(original["refs"]["generation_config"])
    registration=reader.decode_json(original["refs"]["registration"])
    require(registration["config_ref"]==original["refs"]["generation_config"]
        and registration["parent_registry"]==generation["parent_registry"],"Exact original generation/registration")
    bindings=reader.decode_json(protocol.H050_TRAIN_BINDINGS_REF if spec["phase"]=="train" else protocol.H050_VAL_BINDINGS_REF)["parents"]
    started=time.monotonic()
    if spec["phase"]=="train":
        require(os.environ.get("CUDA_VISIBLE_DEVICES")==GPU0_UUID,"OnlyGPU0 actual visibility")
        from arggnn.native_argformer_exec import validate_lease,apply_and_validate_title
        lease=validate_lease();apply_and_validate_title();fit.require_logical_job_v1(lease,spec)
        require(lease["borrowed_physical_inode"] is True,"Truthful direct attachment semantics")
        base_fit=authenticated_base_fit_v2(reader)
        require_work_window_v2(spec)
        from arggnn.biological_candidate import configure_fit
        g.budget_check_v9(storage,boundary=True,required_write_bytes=CAPS["configure_fit_bytes"])
        device,runtime,qualified_lease=configure_fit("cuda",base_fit,output.path)
        import torch,torch_geometric
        require(torch.__version__=="2.6.0+cu124" and torch_geometric.__version__=="2.7.0","Exact original Torch/PyG")
        torch.cuda.set_per_process_memory_fraction(min(1.,cfg["max_cuda_reserved_bytes"]/
            torch.cuda.get_device_properties(device).total_memory),device)
        qualification_bytes=sum(max(p.stat().st_size,p.stat().st_blocks*512)+8192
            for p in output.path.rglob("*") if p.is_file())
        require(qualification_bytes<=CAPS["configure_fit_bytes"],"Existing configure_fit8MiB failstop, included in192MiB scientific output")
        output.used=qualification_bytes
        from arggnn import train_mutation_routed_whole_tree_gnn_v1 as h036
        checked_resources=root_resources_view_v5(h036,fit,check_fit_active)
        def resources():checked_resources(spec,qualified_lease)
    else:
        require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","Explicit noCUDA CPU evaluation")
        import torch,torch_geometric,setproctitle
        torch.set_num_threads(32);torch.set_num_interop_threads(1)
        require(torch.get_default_dtype()==torch.float32 and not torch.cuda.is_initialized()
            and torch.__version__=="2.6.0+cu124" and torch_geometric.__version__=="2.7.0","Same FP32 CPU runtime")
        setproctitle.setproctitle("AI Auto Research Pipeline Test")
        require(Path("/proc/self/cmdline").read_bytes().split(b"\0")[0].decode().strip()=="AI Auto Research Pipeline Test",
            "Exact actual CPU process title")
        device=torch.device("cpu")
        def resources():
            check_fit_active(spec)
            require(not torch.cuda.is_initialized(),"CPU must never initialize CUDA")
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=cfg["max_cpu_rss_bytes"],"Current128GiB RSS")
    def guard():
        active(spec);active(q);require_work_window_v2(spec);resources()
        require(time.monotonic()-started<CAPS["max_seconds"],"Original24h deadline")
        charged_census_v1(spec);g.budget_check_v9(storage,boundary=False)
    def math_guard():
        active(spec);active(q);require_work_window_v2(spec);resources()
        require(time.monotonic()-started<CAPS["max_seconds"],"Original24h deadline")
        charged_census_v1(spec)
    from arggnn import train_ordered_clade_episode_v1 as h45
    h45.science()
    def before_episode(pid,bank,declared):
        guard();actual=h45.episode_upper_bounds_v1(bank);row=counts[pid]
        require(declared==actual and actual["native_nodes"]==row["native_nodes"]
            and actual["native_holds"]==row["native_holds"] and actual["haplotypes"]==952,"Original complete-bank episode bounds")
        require(actual["compact_payload_upper_bound_bytes"]<=min(cfg["episode_payload_cap_bytes"],cfg["episode_cache_cap_bytes"]),
            "Whole episode inside both exact reviewed caps")
        estimate=actual["preparation_named_array_allowance_bytes"]+cfg["episode_python_scratch_allowance_bytes"]
        with open("/proc/self/statm") as f:rss=int(f.read().split()[1])*os.sysconf("SC_PAGE_SIZE")
        require(estimate<=cfg["episode_preparation_cap_bytes"] and rss+estimate<=cfg["max_cpu_rss_bytes"],
            "Prospective episode arrays plus Python scratch/RSS")
    ctx=dict(root_authorizer="root_agent",admitted=True,hypothesis_id="H058",phase=spec["phase"],
        device=device,physical_gpu=0 if spec["phase"]=="train" else None,cfg=cfg,
        config_ref=spec["refs"]["config"],reader=reader,output=output,guard=guard,
        before_episode_build=before_episode,registry=generation["parent_registry"],
        entry_refs=original["observation_entry_refs"],observation_source_refs=original["observation_cache_source_refs"],
        observation_cache_ref=protocol.OBSERVATION,bindings=bindings,
        worker_tmp_directory=str(local(spec["worker_auxiliary_scope"]+"/tmp")))
    if spec["phase"]=="evaluate":ctx["train_completion_ref"]=q["train_completion_ref"]
    # The original writer uses explicit local serializers, never uninitialized lazy globals.
    import numpy as np
    from research.reviews.H058_full_parent_CPU_execution_source_20261017_v1.writer_helpers_v1 import make_jsonable_v1
    output._H058_jsonable=make_jsonable_v1(torch,np)
    if spec["phase"]=="train":
        from .operational_v1 import make_math_gate_v1
        ctx["_H058_math_gate"]=make_math_gate_v1(guard,math_guard,torch,protocol.ARMS)
    guard();output.save("ROOT_started.json",dict(schema="H058_fixed_full_parent_ROOT_started_v1",
        input_ref=record,phase=spec["phase"],device=str(device),physical_gpu=ctx["physical_gpu"],
        scientific_output_cap_bytes=CAPS["scientific_output_bytes"],whole_scope_reservation_bytes=CAPS["reservation_bytes"],
        borrowed_physical_inode=spec["phase"]=="train",controller_inheritance_claimed=False))
    return ctx
