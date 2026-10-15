"""ROOT-only tiny synthetic CPU storage fixture; default never imports Torch."""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,os
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=Path(__file__).resolve().parent

def execute_fixture(output):
    # Only this fresh fixture process is CPU-only. No worker, model or CUDA API.
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    import torch
    from torch_geometric.data import Data
    sp=importlib.util.spec_from_file_location("_private_owned_storage_fixture_helper",HERE/"private_owned_tensors_v1.py")
    helper=importlib.util.module_from_spec(sp);sp.loader.exec_module(helper)
    model_src=(ROOT/"src/arggnn/mutation_routed_whole_tree_gnn_v1.py").read_text()
    klass=next(n for n in ast.parse(model_src).body if isinstance(n,ast.ClassDef) and n.name=="MutationRoutedWholeTreeInputsV1")
    bank_fields=tuple(n.target.id for n in klass.body if isinstance(n,ast.AnnAssign))
    block_tensors=("genomic_geometry","focal_rows","focal_hold","focal_sample_column")
    level_fields=("upward_levels","upward_children","upward_parent_rank","downward_levels")
    ns=dict(torch=torch,hashlib=hashlib,json=json,BANK_FIELDS=bank_fields,BLOCK_TENSORS=block_tensors)
    def require(ok,message):
        if not ok:raise ValueError(message)
    ns["require"]=require
    for source,name in (("src/arggnn/american_c3_gnn_stream_train_isolated_v2.py","_tensors"),
                        ("research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/prefetch_provider_isolated_v2.py","_buffer_feature_identity"),
                        ("src/arggnn/american_c3_gnn_stream_train_isolated_v2.py","_feature_identity")):
        text=(ROOT/source).read_text();fn=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),source,"exec"),ns)
    identity=ns["_buffer_feature_identity"];checks=[]
    def ck(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    def reject(name,fn):
        try:fn()
        except ValueError:checks.append(name);return
        raise AssertionError(name)
    block=Data()
    for name in bank_fields+block_tensors:
        if name in level_fields:block[name]=tuple(torch.arange(i+1,dtype=torch.long).share_memory_() for i in range(3))
        else:block[name]=torch.arange(6,dtype=torch.float32).share_memory_()
    base=torch.arange(12,dtype=torch.float32).share_memory_()
    parent=torch.arange(6,dtype=torch.long).share_memory_()
    block["edge_geometry"]=base.reshape(3,4).t()
    block["node_time"]=base[2:8]
    block["parent"]=parent
    block["upward_levels"]=(parent[:3],parent[3:],parent[:0])
    block["genotype"]=torch.tensor([[0,1],[1,0]],dtype=torch.int8).share_memory_()
    block["donor_call_mask"]=torch.tensor([True,False],dtype=torch.bool).share_memory_()
    block.parent_id="synthetic_C3_parent";block.block_index=0;block.start_tree=0;block.stop_tree=1
    block.sequence_length_bp=2000000.;block.complete_parent=True;block.plan_ref=dict(path="synthetic_plan.json",sha256="a"*64)
    block.observations_receipt=dict(nodes=6,trees=1,metadata=["synthetic",1]);block.worker_pid=123;block.num_nodes=6
    digest,size=identity(block);ck("original_and_buffer_feature_digest_equal",ns["_feature_identity"](block)==(digest,size))
    owned,record=helper.clone_block_private_v1(block,tensor_fields=bank_fields+block_tensors,expected_digest=digest,expected_size=size,identity=identity,torch_api=torch)
    ck("postclone_digest_and_logical_bytes_identical",identity(owned)==(digest,size))
    source_tensors=[];owned_tensors=[]
    for key in bank_fields+block_tensors:
        source_tensors.extend(ns["_tensors"](block[key]));owned_tensors.extend(ns["_tensors"](owned[key]))
    ck("same_tensor_count",len(source_tensors)==len(owned_tensors))
    for a,b in zip(source_tensors,owned_tensors):
        ck("value_dtype_shape_stride_offset_cpu_layout",torch.equal(a,b) and a.dtype==b.dtype and a.shape==b.shape and a.stride()==b.stride() and a.storage_offset()==b.storage_offset() and b.device.type=="cpu" and a.layout==b.layout)
        ck("storage_is_private_and_source_disjoint",not b.is_shared() and (not a.untyped_storage().nbytes() or a.untyped_storage()._cdata!=b.untyped_storage()._cdata))
    alias=lambda a,b:a.untyped_storage()._cdata==b.untyped_storage()._cdata
    ck("all_source_view_alias_relationships_preserved_privately",all(alias(a,b)==alias(c,d) for a,c in zip(source_tensors,owned_tensors) for b,d in zip(source_tensors,owned_tensors)))
    ck("PyG_container_and_metadata_independent",owned is not block and owned._store is not block._store and owned.plan_ref==block.plan_ref and owned.plan_ref is not block.plan_ref and owned.observations_receipt==block.observations_receipt)
    block.node_time[0]=999
    block.plan_ref["path"]="mutated_source_plan"
    ck("incoming_storage_and_metadata_mutation_cannot_change_cache",identity(owned)==(digest,size) and owned.plan_ref["path"]=="synthetic_plan.json")
    owned.node_time[0]=42
    ck("owned_internal_alias_still_operates",owned.edge_geometry[2,0].item()==42 and block.node_time[0].item()==999)
    ck("cached_mutation_is_still_caught_by_original_digest",identity(owned)[0]!=digest)
    ck("retained_private_physical_bytes_bounded_by_original_logical_bytes",record["private_storage_bytes"]<=size and record["incoming_shared_storage_count"]>0 and not record["incoming_storage_aliases_retained"])
    ck("ownership_report_has_no_tensor_or_storage_refs",all(type(v) in (int,bool) for v in record.values()))
    tiny=Data(parent=torch.arange(16,dtype=torch.float32).share_memory_()[::4])
    reject("oversized_backing_view_fails_without_silent_compaction",lambda:helper.clone_block_private_v1(tiny,tensor_fields=("parent",),expected_digest="x"*64,expected_size=16,identity=lambda b:("x"*64,16),torch_api=torch))
    receipt=dict(schema="C3_private_owned_CPU_cache_synthetic_storage_fixture_v1",status="PASS",checks=checks,
        check_count=len(checks),distinct_checks=sorted(set(checks)),torch_version=torch.__version__,synthetic_logical_bytes=size,
        private_storage_report=record,fixture_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_source_sha256=hashlib.sha256((HERE/"private_owned_tensors_v1.py").read_bytes()).hexdigest(),
        real_observation_tree_genotype_checkpoint_opened=False,worker_process_spawn=False,model_forward_or_update=False,
        GPU_API_call=False,guard_admission_resource_probe_or_training=False)
    with Path(output).open("x") as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write("\n")
    print(json.dumps(dict(status="PASS",check_count=len(checks),synthetic_logical_bytes=size)))
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--execute-fixture-root",action="store_true");p.add_argument("--output")
    a=p.parse_args()
    if not a.execute_fixture_root:
        print(json.dumps(dict(status="source_only_CPU_tensor_fixture_root_deferred",Torch_imported=False)));return 0
    if not a.output:raise ValueError("New explicit fixture receipt output required")
    execute_fixture(a.output);return 0

if __name__=="__main__":raise SystemExit(main())
