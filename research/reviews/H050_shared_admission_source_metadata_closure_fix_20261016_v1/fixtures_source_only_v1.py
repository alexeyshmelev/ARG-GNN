"""Pure classification/body/assembly fixtures. No guard or H050 import."""
from pathlib import Path
import ast,copy,hashlib,json
HERE=Path(__file__).resolve().parent
SOURCE=HERE/"context_metadata_fix_v1.py"
class MemoryPath:
    def __truediv__(self,name):return self
    def read_text(self):return json.dumps(proxy)
ns=dict(Path=Path,json=json,ROOT=MemoryPath())
tree=ast.parse(SOURCE.read_text())
nodes=[n for n in tree.body if isinstance(n,ast.Assign) and not any(isinstance(t,ast.Name) and t.id=="ROOT" for t in n.targets)
    or isinstance(n,ast.FunctionDef) and n.name in ("require","source_metadata_refs_v1")]
mod=ast.Module(body=nodes,type_ignores=[]);ast.fix_missing_locations(mod);exec(compile(mod,"<pure_H050_scientific_input_classification>","exec"),ns)
def fake_ref(name):return dict(path=name,sha256=hashlib.sha256(name.encode()).hexdigest())
ns["ref"]=fake_ref
proxy_ref=fake_ref("pure_fixture/scientific_input_proxy.json")
proxy=dict(schema="H050_scientific_input_transitive_closure_v1",hypothesis_id="H050",scientific_input_refs=[ns["WARM_REF"]],
    metadata_adapter_ref=fake_ref(ns["SOURCE"]),admission_successor_ref=fake_ref(ns["ADMISSION_SOURCE"]),
    original_common_ref=ns["COMMON_REF"],original_context_ref=ns["CONTEXT_REF"])
cfg=dict(warmstart_checkpoint_ref=ns["WARM_REF"],scientific_input_closure_ref=proxy_ref)
root=Path("/workspace-SR003.nfs2/ARG-GNN")
pending=root/"research/reviews/GPU45_C3cache_H050_migration_gate_successor_20261016_v2/admitted_H050_shared_train_v1/reserve_authority.json"
rows=json.loads(pending.read_text())["request"]["consumer_bindings"]
results=[]
def passed(name,ok):assert ok,name;results.append(dict(case=name,passed=True))
fixed=ns["source_metadata_refs_v1"](rows,cfg)
passed("exact_warm_pt_only_moves_to_JSON_proxy",ns["WARM_REF"] not in fixed and proxy_ref in fixed
    and all(r in fixed for r in rows if r!=ns["WARM_REF"]))
passed("all_runtime_sources_including_original_context_and_common_direct",
    all(r in fixed for r in [ns["COMMON_REF"],ns["CONTEXT_REF"],fake_ref(ns["SOURCE"]),fake_ref(ns["ADMISSION_SOURCE"])]))
passed("within_unchanged_64_source_metadata_limit",len(fixed)<=64 and all(Path(r["path"]).suffix in {".py",".sh",".json"} for r in fixed))
def rejects(name,call):
    try:call()
    except ValueError:passed(name,True)
    else:raise AssertionError(name)
rejects("unexpected_scientific_binding_not_silently_filtered",lambda:ns["source_metadata_refs_v1"](rows+[fake_ref("other.npz")],cfg))
bad=copy.deepcopy(cfg);bad["warmstart_checkpoint_ref"]=dict(ns["WARM_REF"],sha256="0"*64)
rejects("warm_checkpoint_pin_cannot_change",lambda:ns["source_metadata_refs_v1"](rows,bad))
before=proxy["scientific_input_refs"];proxy["scientific_input_refs"]=[]
rejects("proxy_must_retain_exact_original_scientific_ref",lambda:ns["source_metadata_refs_v1"](rows,cfg));proxy["scientific_input_refs"]=before
driver=HERE/"root_h050_shared_admission_fix_v1.py";source=driver.read_text();t=ast.parse(source)
admit=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=="admit")
changes=ast.literal_eval(next(n.value for n in admit.body if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=="changes" for v in n.targets)))
original=root/"research/reviews/H050_direct_readout_cut_memory_source_20261015_v1/producer_integration_v1/root_bridge_v1/root_driver_v1.py"
text=original.read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=="admit")
text=ast.get_source_segment(text,node)
for before,after in changes.items():assert text.count(before)==1;text=text.replace(before,after)
compile(text,"<H050_frozen_metadata_fix_admit_compile>","exec")
passed("frozen_original_admission_exact_single_adaptations_compile",True)
passed("fresh_H050_namespace_original_C3_sources_untouched",
    'H050_shared_direct_readout_train_seed101_v2' in text and "FIX_JOB" in text
    and 'root_driver_ref=ref(DRIVER)' in text and 'source_sha256=sources' in text)
assemble=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=="assemble")
changes=ast.literal_eval(next(n.value for n in assemble.body if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=="changes" for v in n.targets)))
base=root/"research/reviews/GPU45_C3cache_H050_migration_gate_successor_20261016_v2/root_shared_admission_v2.py"
original=base.read_text();node=next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name=="assemble")
text=ast.get_source_segment(original,node)
for before,after in changes.items():assert text.count(before)==1;text=text.replace(before,after)
compile(text,"<C3_unchanged_fresh_H050_assembly_compile>","exec")
passed("C3_worker_consumer_requirement_unchanged_in_assembly",
    'for x in operational_refs()+renewal_refs(r["marker_repair_renewal_ref"])+[ref(pin["path"])]' in text
    and text.count("EXTRA_METADATA_REFS")==2)
passed("actual_H050_job_used_fresh_controller_sources_bound",
    'jobs[MODULE]=h_spec["job_id"]' in source and 'controller_job="GPU4-C3cache-H050-shared-controller-v2"' in text)
context=root/ns["CONTEXT_REF"]["path"];original=context.read_text()
node=next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name=="build_context_v1")
compile(ast.Module(body=[node],type_ignores=[]),"<unchanged_H050_context_body_compile>","exec")
passed("original_context_body_compiles_unchanged_not_scientific_rewrite",True)
print(json.dumps(dict(schema="H050_shared_source_metadata_fix_pure_fixtures_v1",cases=results,cases_passed=len(results),
    pending_binding_count=len(rows),fixed_binding_count=len(fixed),
    actual_guard_calls_import_hooks_models_jobs_admissions_or_Git=0),indent=2,sort_keys=True))
