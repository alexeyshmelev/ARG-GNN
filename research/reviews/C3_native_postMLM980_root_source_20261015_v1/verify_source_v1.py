"""Source/configuration checks only. Never execute bind/admit/launch/close paths."""
from pathlib import Path
import ast,importlib.util,json,os,subprocess,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/C3_native_postMLM980_root_source_20261015_v1"
D=ROOT/REL
def require(ok,message):
    if not ok:raise ValueError(message)
def load(name):
    spec=importlib.util.spec_from_file_location("_source_check_"+name,D/(name+".py"))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def main():
    require(Path.cwd()==ROOT and sys.dont_write_bytecode,"Repository source-only interpreter, no bytecode")
    sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts"),str(D),
        str(ROOT/"research/reviews/American_C3_native_MLM_full_fit_v8"),
        str(ROOT/"research/reviews/American_C3_native_MLM_full_source_v5"),
        str(ROOT/"research/reviews/American_C3_native_MLM_amendment_v12")]
    for path in D.glob("*.py"):compile(path.read_text(),str(path),"exec")
    c=load("root_native_contract_v1");native=c.native_contract()
    counts=c.counts_metadata_v1(c.COUNTS_REF,c.COUNTS_WAIT_REF,{})
    require(counts["contrastive_optimizer_updates"]==2105,"Actual count budget")
    for phase in c.PHASES:
        caps=json.loads((D/("caps_"+phase+".inactive.json")).read_text())
        cfg=c.make_config(dict(phase=phase),caps);native.validate_config(cfg,phase)
        require(cfg==dict(json.loads((D/("config_"+phase+".inactive.json")).read_text()),execution_enabled=True),"Exact inactive configuration")
        require(caps["phase_write_budget_bytes"]+caps["outer_auxiliary_cap_bytes"]+caps["root_packet_overhead_cap_bytes"]<=caps["reservation_cap_bytes"],"Whole reserved growth")
        require(sum(caps[k] for k in ("checkpoint_live_allowance_bytes","cache_allowance_bytes","temp_allowance_bytes","child_artifact_allowance_bytes"))<caps["phase_write_budget_bytes"],"Category plans fit the phase")
        require(len(str(ROOT/cfg["short_tmp_directory"]/("contrastive" if phase=="fit" else "RAM")/"pymp-12345678/listener-12345678").encode())<=107,"Legacy fallback socket pathname bound")
    for name in ("native_child_v1","launcher_postMLM980_v1","root_native_driver_v1","entry_postMLM_v1"):load(name)
    require("torch" not in sys.modules and "numpy" not in sys.modules,"No model/scientific runtime imports")
    driver=(D/"root_native_driver_v1.py").read_text();tree=ast.parse(driver)
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    for name in ("admit","close"):
        text=ast.get_source_segment(driver,functions[name])
        require("c.original_guard()" in text and "consumer_guard()" not in text,"Unwrapped mutation guards")
    require("c.consumer_guard()" in ast.get_source_segment(driver,functions["launch"]),"Read-only launch memo2")
    ctx=(D/"context_factory_v1.py").read_text();ct=ast.parse(ctx)
    fast=next(n for n in ct.body if isinstance(n,ast.FunctionDef) and n.name=="_fast_guard")
    fast_text=ast.get_source_segment(ctx,fast)
    require(not any(x in fast_text for x in ("consumer_guard","original_guard","budget_check","check_storage","configure_v2")),"No storage scan/journal guard in native fast loop")
    child=(D/"native_child_v1.py").read_text()
    require(child.count("module.train(config)")==1 and 'import_module("sequence_contrastive")' in child and 'import_module("main")' not in child,"Contrastive-only author training")
    require("bootstrap_native_execution_v1" in child and child.index('bootstrap=bootstrap_native_execution_v1')<child.index("    import torch"),"Abstract IPC before Torch")
    for name in ("launcher_postMLM980_v1.py","supervisor_v1.py","context_factory_v1.py","native_child_v1.py","root_native_driver_v1.py"):
        source=(D/name).read_text()
        require("import amendment_v12" not in source and "from amendment_v12" not in source,"No old execution authority imports")
    for name,args in (("root_native_driver_v1.py",[]),("entry_postMLM_v1.py",[]),("launcher_postMLM980_v1.py",["--help"]),("native_child_v1.py",["--help"])):
        run=subprocess.run([sys.executable,"-B",str(D/name),*args],cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1"),capture_output=True,text=True)
        require(run.returncode==0,name+": "+run.stderr[:2000])
    require(len(c.source_paths())==len(set(c.source_paths()))==37,"Complete unique transitive source path set")
    print(json.dumps(dict(status="pass_source_configuration_and_actual_counts_only",source_closure_paths=37,
        contrastive_optimizer_updates=2105,contrastive_dropped_train_remainder=2832,
        torch_imported=False,numpy_imported=False,guard_hardware_model_payload_admission_accounting_Git_executed=False,
        actual_MLM_fit_GPU_UUID_refs=None),sort_keys=True))
    return 0
if __name__=="__main__":raise SystemExit(main())
