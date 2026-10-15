"""H049 observation-only compact cut-recipe preparation; inactive by default."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,time
from pathlib import Path
from research.reviews.H049_order2_cut_memory_root_source_20261015_v1 import producer_v1 as worker
from research.reviews.H049_order2_cut_memory_root_source_20261015_v1 import cut_recipe_codec_v1 as codec
ROOT=worker.ROOT
SOURCE="research/reviews/H049_order2_cut_memory_root_source_20261015_v1/prepare_v1.py"
PARENTS=worker.TRAIN+worker.VAL


def run_context_v1(ctx,input_ref):
    worker.science()
    torch=worker.torch
    from arggnn.prepare_marker_resolution_observations_v1 import ObservationReaderV1
    from arggnn.mutation_routed_full_panel_fast_reader_v1 import read_full_cache_index_v1,read_full_cache_parent_v1
    require=worker.require
    require(ctx.get("phase")=="prepare" and ctx.get("admitted") is True
        and ctx.get("root_authorizer")=="root_agent" and "bindings" not in ctx,
        "Admitted observation-only preparation")
    require(not torch.cuda.is_initialized(),"CPU preparation cannot initialize CUDA")
    reader=ObservationReaderV1(ctx["reader"]);out=ctx["output"];guard=ctx["guard"]
    index=read_full_cache_index_v1(ctx["observation_cache_ref"],reader,registry=ctx["registry"],
        entry_refs=ctx["entry_refs"],source_refs=ctx["observation_source_refs"])
    products={};started=time.monotonic()
    try:
        for pid in PARENTS:
            guard()
            item=read_full_cache_parent_v1(pid,index,reader)
            bank,parent=item["model_inputs"],item["scoring"]
            require(worker.h038.tensor_bytes_v1(bank)<=ctx["cfg"]["CPU_bank_cap_bytes"],
                "One complete original native CPU bank cap")
            ctx["before_cut_build"](pid,bank)
            plan,metadata=worker.core.prepare_cut_plan_v1(bank)
            arrays,receipt=codec.encode_v1(plan,metadata)
            limits=receipt["bounds"]
            require(limits["compressed_NPZ_upper_bytes"]<=ctx["cfg"]["cut_recipe_archive_cap_bytes"],
                "Compact recipe archive root cap")
            arrays_ref=out.save(pid+".cut_recipe.npz",arrays,"npz")
            require((ROOT/arrays_ref["path"]).stat().st_size<=limits["compressed_NPZ_upper_bytes"],
                "Actual compressed bytes satisfy compact count bound")
            reconstructed=codec.decode_v1(reader.read(arrays_ref),receipt,bank)
            require(all(torch.equal(a,b) for a,b in zip(plan.tensors(),reconstructed.tensors()))
                and plan.group_offsets==reconstructed.group_offsets
                and plan.episode_of_hold==reconstructed.episode_of_hold,
                "Production preparation losslessly reconstructs every logical cut tensor")
            del reconstructed
            receipt.update(lossless_roundtrip_verified=True,parent_id=pid,original_observed_entry_ref=ctx["entry_refs"][pid],
                original_cache_array_ref=index["parents"][pid]["array_ref"],
                arrays_ref=arrays_ref,codec_ref=worker.bound_source_ref_v1(codec.__file__),
                input_ref=input_ref)
            receipt_ref=out.save(pid+".cut_recipe.json",receipt)
            products[pid]=dict(arrays_ref=arrays_ref,receipt_ref=receipt_ref,
                compact_array_bytes=sum(a.nbytes for a in arrays.values()),
                compressed_archive_bytes=(ROOT/arrays_ref["path"]).stat().st_size,
                logical_cut_tensor_bytes=plan.tensor_bytes(),lossless_roundtrip_verified=True,
                counts=dict(native_nodes=plan.native_nodes,native_holds=plan.native_holds,
                    haplotypes=952,groups=plan.representatives.numel(),episodes=plan.episode_groups.shape[0]),
                source_fingerprints=dict(plan.source_fingerprints))
            print(json.dumps(dict(stage="H049_compact_cut_prepared",parent_id=pid,
                counts=products[pid]["counts"],compressed_bytes=products[pid]["compressed_archive_bytes"])),flush=True)
            del arrays,receipt,plan,metadata,bank,parent,item
            guard()
        result=dict(schema="H049_order2_observation_preparation_v1",
            hypothesis_id="H049",status="completed_all40_compact_cut_recipes",
            input_ref=input_ref,config_ref=ctx["config_ref"],parents=products,
            source_manifest_ref=ctx["cfg"]["source_manifest_ref"],
            registration_ref=ctx["cfg"]["registration_ref"],
            producer_ref=ctx["cfg"]["preparation_producer_ref"],
            marker_preparation_completion_ref=ctx["preparation_completion_ref"],
            marker_preparation_actual_wait_ref=ctx["preparation_actual_wait_ref"],
            observation_cache_ref=ctx["observation_cache_ref"],entry_refs=ctx["entry_refs"],
            observation_source_refs=ctx["observation_source_refs"],observations_only=True,
            targets_read=False,checkpoint_payloads_read=False,CUDA_initialized=False,
            format=codec.FORMAT,seconds=time.monotonic()-started)
        out.save("completed.json",result)
        return result
    except Exception as error:
        out.save("failed.json",dict(schema="H049_cut_preparation_failure_v1",hypothesis_id="H049",
            status="failed_partial_preserved",error_type=type(error).__name__,error=str(error),
            prepared_parent_ids=list(products),targets_read=False,automatic_replay=False))
        raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true");p.add_argument("--input")
    p.add_argument("--expected-input-sha256");args=p.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="source_only_not_executed",phase="prepare",
            parents=40,compact_format=codec.FORMAT,truth_access=False)));return 0
    worker.require(args.input and args.expected_input_sha256,"Pinned active H049 preparation capability")
    record=dict(path=args.input,sha256=args.expected_input_sha256)
    spec=json.loads(worker.bound(record))
    worker.require(spec.get("schema")=="H049_marker_grid_root_input_v1"
        and spec.get("hypothesis_id")=="H049" and spec.get("phase")=="prepare"
        and spec.get("status")=="active" and spec.get("producer_ref")==worker.bound_source_ref_v1(SOURCE),
        "Exact fresh active H049 preparation producer")
    cref=spec["root_context_module_ref"];worker.bound(cref)
    loader=importlib.util.spec_from_file_location("_H049_prepare_context",ROOT/cref["path"])
    module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    run_context_v1(module.build_context_v1(record,spec),record);return 0

if __name__=="__main__":raise SystemExit(main())
