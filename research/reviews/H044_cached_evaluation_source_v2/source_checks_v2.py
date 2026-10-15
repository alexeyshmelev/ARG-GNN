"""Source-only static checks. No project modules, Torch, model/data or guards."""
from pathlib import Path
import ast,builtins,hashlib,json,datetime
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/H044_cached_evaluation_source_v2"

def body_text(source,name):
    node=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name==name)
    return "\n".join(source.splitlines()[node.lineno-1:node.end_lineno])+"\n"

def main():
    producer_path=HERE/"evaluate_cached_v2.py"
    cache_path=ROOT/"src/arggnn/marker_resolution_input_cache_v2.py"
    paths=[producer_path,cache_path]
    for path in paths:
        source=path.read_text();ast.parse(source);compile(source,str(path),"exec",dont_inherit=True)
        for node in ast.parse(source).body:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                names=[node.module] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names]
                assert all(n.split(".")[0] in {"pathlib","argparse","hashlib","importlib","io","json","dataclasses"} for n in names)
    source=producer_path.read_text();tree=ast.parse(source)
    assigned={node.targets[0].id:node.value for node in tree.body
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)}
    frozen_refs=ast.literal_eval(assigned["FROZEN_SOURCE_REFS"])
    cache_ref=ast.literal_eval(assigned["CACHE_SOURCE_REF"])
    for ref in [*frozen_refs,cache_ref]:
        assert hashlib.sha256((ROOT/ref["path"]).read_bytes()).hexdigest()==ref["sha256"]
    original=body_text((ROOT/"src/arggnn/train_marker_resolution_readout_v1.py").read_text(),"run_evaluate_v1")
    expected=original.replace("def run_evaluate_v1(ctx,input_ref):","def run_evaluate_cached_v2(ctx,input_ref):",1)
    aliases="""    from arggnn import train_marker_resolution_readout_v1 as frozen
    from arggnn.marker_resolution_input_cache_v2 import EvaluationMarkerBlockStoreV2,bind_cached_bundle_v2
    torch,core,h42,h038,fit=frozen.torch,frozen.core,frozen.h42,frozen.h038,frozen.fit
    ARMS,VAL,PASSES,SELECTION,CORE_REF=frozen.ARMS,frozen.VAL,frozen.PASSES,frozen.SELECTION,frozen.CORE_REF
    H044PhaseReaderV1=frozen.H044PhaseReaderV1
    DenseRAMMatrixV1=frozen.DenseRAMMatrixV1
    load_observations_v1=frozen.load_observations_v1
    device_parent_v1=frozen.device_parent_v1
    cache=EvaluationMarkerBlockStoreV2(**ctx["cache_policy"]["limits"])
"""
    expected=expected.replace("    from .mutation_routed_full_panel_primary_v1 import FullPanelPrimaryValidationV1\n",
        aliases+"    from arggnn.mutation_routed_full_panel_primary_v1 import FullPanelPrimaryValidationV1\n",1)
    line="        bundle,device_flags,_=device_parent_v1(ctx,cpu_bundle,flags_cpu)\n"
    expected=expected.replace(line,line+"        blocks=cache.get(pid,cpu_bundle)\n        bundle=bind_cached_bundle_v2(bundle,blocks)\n",1)
    expected=expected.replace("        del bundle,device_flags,cpu_bundle,flags_cpu\n",
        "        del bundle,device_flags,cpu_bundle,flags_cpu,blocks\n        cache.release(pid)\n",1)
    expected=expected.replace('    output.save("completed.json",result);return result\n',
        '    result["immutable_input_cache_v2"]=dict(execution_ref=ctx["cached_execution_ref"],\n'
        '        cache_policy_ref=ctx["cache_policy_ref"],cache=cache.receipt(),\n'
        '        same_original_probability_products=True,extra_fits=False)\n'
        '    output.save("completed.json",result);return result\n',1)
    actual=body_text(source,"run_evaluate_cached_v2")
    assert actual==expected,"Frozen evaluation flow changed outside declared cache insertions"
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="run_evaluate_cached_v2")
    loads={x.id for x in ast.walk(node) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Load)}
    stores={x.id for x in ast.walk(node) if isinstance(x,ast.Name) and isinstance(x.ctx,(ast.Store,ast.Del))}
    imports={a.asname or a.name.split(".")[0] for x in ast.walk(tree) if isinstance(x,(ast.Import,ast.ImportFrom)) for a in x.names}
    args={x.arg for x in ast.walk(node) if isinstance(x,ast.arg)}
    missing=loads-stores-imports-args-{"ROOT","require"}-set(dir(builtins))
    assert not missing,sorted(missing)
    assert actual.index("ram.verify();reader.truth_gate=True")>actual.index("for pid in VAL:")
    template=json.loads((HERE/"execution_input_inactive.json").read_text())
    assert template["status"]=="inactive" and template["root_authorizer"] is None
    assert template["root_context_module_ref"] is None and template["consumer_manifest_ref"] is None
    result=dict(schema="H044_cached_evaluation_source_checks_v2",created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_parse_compile=True,top_level_imports_stdlib_only=True,frozen_source_hashes=True,
        unchanged_evaluation_flow_except_declared_insertions=True,copied_evaluator_names_bound=True,
        original_all96_barrier_retained=True,execution_template_inactive=True,
        source_refs=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
        project_modules_imported=False,Torch_imported=False,model_or_data_tests=False,
        guard_or_process_state_access=False,admissions_jobs_or_Git=False)
    (HERE/"source_checks_receipt.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
    return 0

if __name__=="__main__":raise SystemExit(main())
