"""One exact legacy VAL001 CPU worker startup; inert for every other argv0."""
import sys
if sys.argv and sys.argv[0]=="/workspace-SR003.nfs2/ARG-GNN/research/reviews/C3_GNN_current980_executable_root_20261015_v1/worker_entry_v1.py":
    try:
        from pathlib import Path
        import hashlib,importlib.util,json,os
        root=Path("/workspace-SR003.nfs2/ARG-GNN")
        here="research/reviews/C3_GNN_VAL001_plan_metadata_20261016_v1"
        request_path=os.environ["ARGGNN_C3_VAL001_ROOT_REQUEST_PATH"]
        request_sha=os.environ["ARGGNN_C3_VAL001_ROOT_REQUEST_SHA256"]
        expected_request="research/reviews/C3_GNN_current980_executable_root_20261015_v1/request_plan_val_001.root_active_v1.json"
        if request_path!=expected_request:raise ValueError("Exact explicit VAL001 request path")
        p=root/request_path
        if p.resolve()!=p or p.is_symlink() or p.stat().st_size>65536:raise ValueError("Canonical bounded request")
        raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=request_sha:raise ValueError("Explicit request SHA")
        req=json.loads(raw);wrapper_ref=req["operational_wrapper_ref"]
        wrapper=root/(here+"/root_operational_bootstrap_v1.py")
        if wrapper_ref["path"]!=str(wrapper.relative_to(root)) or wrapper.resolve()!=wrapper or wrapper.is_symlink() or wrapper.stat().st_size>65536:raise ValueError("Exact bounded wrapper path")
        if hashlib.sha256(wrapper.read_bytes()).hexdigest()!=wrapper_ref["sha256"]:raise ValueError("Exact request-bound wrapper SHA")
        spec=importlib.util.spec_from_file_location("_C3_VAL001_startup_validation",wrapper)
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        request_ref=dict(path=request_path,sha256=request_sha)
        checked,runtime=helper.checked_request(request_ref)
        if len(sys.argv)!=6 or sys.argv[1:3]!=["--execute","--input"] or sys.argv[4]!="--expected-input-sha256":raise ValueError("Exact legacy worker argv")
        input_ref=dict(path=sys.argv[3],sha256=sys.argv[5])
        helper.checked_input(input_ref,request_ref)
        if os.environ.get("CUDA_VISIBLE_DEVICES")!="" or os.environ.get("PYTHONPATH")!=helper.CHILD_PATH or os.environ.get("PYTHONDONTWRITEBYTECODE")!="1" or os.environ.get("PYTHONNOUSERSITE")!="1":raise ValueError("Exact CPU child startup environment")
        if checked["runtime_bootstrap_ref"]!=helper._ref(here+"/startup/sitecustomize.py"):raise ValueError("Request binds this startup source")
        helper.install_renewal()
        print(json.dumps(dict(event="exact_VAL001_CPU_startup_renewal_installed",
            root_stage_request_ref=request_ref,input_ref=input_ref,runtime_bootstrap_ref=checked["runtime_bootstrap_ref"],
            operational_runtime_ref=checked["operational_runtime_ref"],renewal_certificate_ref=helper.CERT_REF,
            actual_child_PYTHONPATH=helper.CHILD_PATH)),flush=True)
    except BaseException as error:
        import os
        print("VAL001 startup refused: "+type(error).__name__+": "+str(error),flush=True)
        os._exit(78)
