"""Private GPU0 operational checks around unchanged H050/C3 scientific bodies."""
from pathlib import Path
import ast,hashlib,importlib.util,json,sys,types
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/GPU0_shared_execution_source_20261016_v1"
SOURCE=HERE+"/gpu0_runtime_adapter_v1.py"
CONTEXT=HERE+"/gpu0_h050_context_v1.py"
GUARD=HERE+"/gpu0_native_guard_v1.py"
H050_COMMON="research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1/root_bridge_v1/bridge_common_v1.py"
H050_CONTEXT="research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1/root_bridge_v1/context_factory_v1.py"
H050_PRODUCER="research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1/producer_v1.py"
H050_METADATA="research/reviews/H050_shared_admission_source_metadata_closure_fix_20261016_v1/context_metadata_fix_v1.py"
C3_CONTRACT="research/reviews/C3_GNN_prefetch_ramcache_isolated980_root_20261015_v3/root_contract_v1.py"
C3_ENTRY="research/reviews/C3_GNN_prefetch_ramcache_isolated980_root_20261015_v3/worker_entry_v1.py"
C3_FACTORY_REF=dict(path="research/reviews/C3_private_owned_RAM_cache_source_20261016_v1/tracking_factory_successor_v1.py",sha256="e9fd0155e42412b2f55af45d48e7f5c01bcfd0814406c94bbf69b59d60d6ff13")
FROZEN_PINS={'research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1/root_bridge_v1/bridge_common_v1.py': '5ec43d65bde777d18c071be2cdf4ee5c14ef2784bfe54ac27582f187ee93552b', 'research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1/root_bridge_v1/context_factory_v1.py': '58bf94db37f8fb63afeca64352fd362ddf580d7117ef794481dfe4395f087fa2', 'research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1/producer_v1.py': '1efd84ea51ed05b20e713f6a39ef33dfc08ce3921730d8ce8aa39ef184506130', 'research/reviews/H050_shared_admission_source_metadata_closure_fix_20261016_v1/context_metadata_fix_v1.py': '6c704a0b47d5b6f136479879ad9c11036f6d80760a93933ae3d0f222db3ce67d', 'research/reviews/C3_GNN_prefetch_ramcache_isolated980_root_20261015_v3/root_contract_v1.py': '8143dcd241c23f9bcf3379644c275f66febc8cf8dea01727e542d88651d734a0', 'research/reviews/C3_GNN_prefetch_ramcache_isolated980_root_20261015_v3/worker_entry_v1.py': '8901ac783449859db3f1a872932806a30768fe9c4e5e7f975e78bfd34e0f624d'}

def require(ok,message):
    if not ok:raise ValueError(message)

def ref(path):
    p=ROOT/path
    require(p.resolve()==p and p.is_relative_to(ROOT) and p.is_file() and not p.is_symlink(),"Canonical metadata/source path")
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def exact(path):
    r=ref(path);require(FROZEN_PINS.get(path)==r["sha256"],"Frozen operational/scientific source changed")
    return (ROOT/path).read_text()

def load_exact(path,label):
    exact(path)
    sp=importlib.util.spec_from_file_location(label,ROOT/path);m=importlib.util.module_from_spec(sp)
    sys.modules[label]=m;sp.loader.exec_module(m);return m

def function_source_v1(path,name,replacements):
    text=exact(path)
    n=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    value=ast.get_source_segment(text,n)
    for before,after in replacements:
        require(value.count(before)==1,"One declared GPU0 operational boundary:"+name)
        value=value.replace(before,after)
    return value

def private_h050_common_v1():
    adapter=load_exact(H050_METADATA,"_GPU0_original_H050_metadata_proxy")
    c=adapter.private_common_v1()
    c.CONTEXT=CONTEXT
    src=function_source_v1(H050_COMMON,"validate_spec_v1",[('spec.get("physical_gpu") in (4,5)','spec.get("physical_gpu")==0')])
    exec(compile(src,"<H050_original_validate_spec_GPU0_eligibility_only>","exec"),c.__dict__)
    return c

def build_h050_context_v1(record,spec):
    c=private_h050_common_v1()
    context=load_exact(H050_CONTEXT,"_GPU0_original_H050_context")
    source=function_source_v1(H050_CONTEXT,"build_context_v1",[('spec["physical_gpu"] in (4,5)','spec["physical_gpu"]==0')])
    namespace=dict(context.__dict__)
    for k in ("ROOT","HERE","CONTEXT","require","ref","cap_v1","worker","framework","load_guard_v1","POLICY_REF","LAUNCHER_REF","validate_spec_v1","count_plan_v1","validate_plan_caps_v1","authenticate_actual_cut_preparation_v1"):
        namespace[k]=getattr(c,k)
    exec(compile(source,"<H050_original_context_GPU0_eligibility_only>","exec"),namespace)
    return namespace["build_context_v1"](record,spec)

def run_h050_v1(j):
    # Relative producer imports retain their genuine package context.
    path=H050_PRODUCER;source=exact(path)
    before='in (4,5)'
    require(source.count(before)==2,"Exactly producer main/context GPU eligibility boundaries")
    source=source.replace(before,'in (0,)')
    name="research.reviews.H050_direct_readout_cut_memory_source_20261015_v1.producer_integration_v1._gpu0_private_producer"
    m=types.ModuleType(name);m.__file__=str(ROOT/path);m.__package__=name.rsplit(".",1)[0]
    sys.modules[name]=m;exec(compile(source,"<original_H050_producer_GPU0_eligibility_only>","exec"),m.__dict__)
    spec=json.loads((ROOT/j["input_ref"]["path"]).read_bytes())
    require(spec["root_context_module_ref"]==ref(CONTEXT),"Actual new GPU0 context declared")
    return m.main(j["args"])

def run_c3_v1(j):
    contract=load_exact(C3_CONTRACT,"_GPU0_private_original_C3_contract")
    source=function_source_v1(C3_CONTRACT,"validate_request",[('req["physical_gpu"] in (4,5)','req["physical_gpu"]==0')])
    exec(compile(source,"<original_C3_request_GPU0_eligibility_only>","exec"),contract.__dict__)
    make=contract.make_config
    def make_config(req,caps):
        value=make(req,caps);tmp=req["operational_short_tmp_directory"]
        require(isinstance(tmp,str) and tmp.startswith(".runtime/tmp/gpu0_shared/") and ".." not in Path(tmp).parts,"Fresh GPU0 private TMP namespace")
        value["short_tmp_directory"]=tmp
        return value
    contract.make_config=make_config
    require(ref(C3_FACTORY_REF["path"])==C3_FACTORY_REF,"Exact separately reviewed private-owned C3 cache repair")
    input0=json.loads((ROOT/j["input_ref"]["path"]).read_bytes())
    repair_pins={
        C3_FACTORY_REF["path"]:C3_FACTORY_REF["sha256"],
        "research/reviews/C3_private_owned_RAM_cache_source_20261016_v1/private_owned_cache_v1.py":"f2c1ba189436d5280b461d93822f4e75e54eb5205f1b95f490e4388d5b946dbf",
        "research/reviews/C3_private_owned_RAM_cache_source_20261016_v1/private_owned_tensors_v1.py":"3a647e5274c41b1859ddd66a50b43c7d7e596244f9f4c0ffccf06da9a8045e83"}
    require(all(input0["source_sha256"].get(k)==v and ref(k)["sha256"]==v for k,v in repair_pins.items()),"Actual private-owned repair closure in admitted C3 input")

    worker=load_exact(C3_ENTRY,"_GPU0_private_original_C3_worker")
    worker.c=contract
    source=function_source_v1(C3_ENTRY,"execute",[
        ('from research.reviews.C3_GNN_prefetch_ramcache_isolated980_root_20261015_v3.tracking_factory_v1 import TrackingObservationProviderFactoryV1',
         'from research.reviews.C3_private_owned_RAM_cache_source_20261016_v1.tracking_factory_successor_v1 import TrackingObservationProviderFactoryV1')])
    exec(compile(source,"<original_C3_worker_private_owned_cache_factory_only>","exec"),worker.__dict__)
    return worker.main(j["args"])

def invoke_job_v1(j):
    # Pool has authenticated all actual source/input/storage references before this call.
    require(ref(SOURCE)==j["runtime_entry_ref"],"Exact actual operational entry")
    require(ref(j["input_ref"]["path"])==j["input_ref"],"Actual original-format input bytes")
    spec=json.loads((ROOT/j["input_ref"]["path"]).read_bytes())
    require(spec["job_id"]==j["job_id"] and spec["refs"]["storage"]==j["storage_ref"]
        and spec.get("operational_GPU0_runtime_ref")==ref(SOURCE)
        and spec.get("operational_GPU0_guard_ref")==ref(GUARD),"Actual explicit GPU0 runtime/guard")
    from arggnn.native_argformer_exec import validate_lease,apply_and_validate_title
    require(validate_lease()["physical_gpu"]==0,"Inherited GPU0 guard installed by pool before scientific imports")
    apply_and_validate_title()
    if j["worker_kind"]=="H050":return run_h050_v1(j)
    require(j["worker_kind"]=="C3","Only H050/C3 algorithm wrappers here")
    return run_c3_v1(j)
