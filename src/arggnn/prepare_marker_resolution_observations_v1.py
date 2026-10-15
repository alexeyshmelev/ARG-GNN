"""H044 standalone observation-only CPU sidecar preparation. Inert by default.

The root context owns admission, current storage guard and resource caps.
One complete original parent is decoded at a time. No truth is accepted.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,io,json,time
from pathlib import Path

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="src/arggnn/prepare_marker_resolution_observations_v1.py"
TRAIN=tuple("ap_expanded_train_%03d"%i for i in range(32))
VAL=tuple("ap_expanded_val_%03d"%i for i in range(8))
PARENTS=TRAIN+VAL
COORDINATE_KEYS=("positions_bp0","centers_bp","bin_edges_bp","intervals_bp","allele_cardinality")

def require(ok,message):
    if not ok:raise ValueError(message)

def bound(record):
    require(isinstance(record,dict) and set(record)=={"path","sha256"},"Exact source/capability reference")
    p=Path(record["path"])
    require(not p.is_absolute() and ".." not in p.parts and p.as_posix()==record["path"],"Canonical repository path")
    path=ROOT/p
    require(path.resolve()==path and path.is_file(),"Existing canonical file")
    data=path.read_bytes()
    require(hashlib.sha256(data).hexdigest()==record["sha256"],"Pinned source/capability bytes changed")
    return data

class ObservationReaderV1:
    """Deny all truth/oracle paths while retaining original source-pin reads."""
    def __init__(self,base):self.base,self.root=base,base.root
    def read(self,record,**kwargs):
        parts=tuple(x.lower() for x in Path(record["path"]).parts)
        require("truth" not in parts and not any("oracle" in x for x in parts),
            "H044 preparation is observation-only")
        return self.base.read(record,**kwargs)
    def decode_json(self,record):return json.loads(self.read(record))

def selected_rows_v1(registry):
    """Preserve original48 registry in cache checks, select only40 locally."""
    rows={r["parent_id"]:r for r in registry if r["split"]!="outer"}
    require(len(registry)==48 and len({r["parent_id"] for r in registry})==48
        and set(rows)==set(PARENTS),"Original48 registry with exact40 development parents")
    require(all(rows[p]["split"]==("train" if p in TRAIN else "val") for p in PARENTS),
        "Original development split labels")
    return rows

def coordinate_sidecar_v1(pid,loaded,bank,parent,index,ctx):
    """Exact original-loader/cache join; never infer physical positions from spans."""
    import numpy as np
    import torch
    from .marker_resolution_readout_v1 import MarkerCoordinatesV1,tensor_identity_v1,ORIGINAL_LOADER_REF
    checks={
        "genotype":np.array_equal(loaded.genotype,bank.genotype.numpy()),
        "reference_class":np.array_equal(loaded.reference_class,bank.reference_class.numpy()),
        "query_column":np.array_equal(loaded.query_ids,bank.query_column.numpy()),
        "sample_age":np.array_equal(np.asarray(loaded.sample_age,dtype=np.float32),bank.sample_age.numpy()),
        "centers":np.array_equal(loaded.readout_x_bp,parent["positions"]),
        "edges":np.array_equal(loaded.readout_bin_edges_bp,parent["edges"]),
        "query_slots":np.array_equal(loaded.scoring["query_slots"],parent["query_slots"]),
        "raw_query_nodes":np.array_equal(loaded.scoring["raw_query_nodes"],parent["raw_query_nodes"]),
        "cohorts":np.array_equal(loaded.scoring["cohorts"],parent["cohorts"]),
        "retained_indexes":np.array_equal(loaded.scoring["retained_indexes"],parent["retained_indexes"]),
        "site_hold":np.array_equal(np.searchsorted(loaded.intervals_bp[:,1],loaded.positions_bp0,side="right"),bank.site_hold.numpy()),
        "readout_hold":np.array_equal(np.searchsorted(loaded.intervals_bp[:,1],loaded.readout_x_bp,side="right"),bank.readout_hold.numpy()),
    }
    require(all(checks.values()),"Original observed-loader/cache ordered join differs: "+str(checks))
    arrays={
        "positions_bp0":torch.from_numpy(np.array(loaded.positions_bp0,dtype=np.float64,copy=True,order="C")),
        "centers_bp":torch.from_numpy(np.array(loaded.readout_x_bp,dtype=np.float64,copy=True,order="C")),
        "bin_edges_bp":torch.from_numpy(np.array(loaded.readout_bin_edges_bp,dtype=np.float64,copy=True,order="C")),
        "intervals_bp":torch.from_numpy(np.array(loaded.intervals_bp,dtype=np.float64,copy=True,order="C")),
        "allele_cardinality":torch.from_numpy(np.array(loaded.allele_count,dtype=np.int64,copy=True,order="C")),
    }
    identities={k:tensor_identity_v1(getattr(bank,k)) for k in
        ("genotype","reference_class","query_column","site_hold","readout_hold")}
    identities.update({k:tensor_identity_v1(v) for k,v in arrays.items()})
    binding=dict(schema="H044_original_observed_marker_coordinate_binding_v1",parent_id=pid,
        original_observed_entry_ref=ctx["entry_refs"][pid],original_loader_source_ref=ORIGINAL_LOADER_REF,
        original_cache_manifest_ref=ctx["observation_cache_ref"],
        original_cache_array_ref=index["parents"][pid]["array_ref"],
        support_rule="original_allele_cardinality",array_identities=identities,
        original_loader_receipt=loaded.receipt,original_loader_cache_checks=checks,
        original_loader_source_closure=ctx["observation_source_refs"],
        no_coordinate_reconstruction_from_FP32_spans=True,observations_only=True,targets_read=False)
    return MarkerCoordinatesV1(**arrays,binding=binding)

def run_context_v1(ctx,input_ref):
    import numpy as np
    import torch
    from . import train_anonymous_clade_flag_control_v1 as h42
    from .mutation_routed_full_panel_fast_reader_v1 import read_full_cache_index_v1,read_full_cache_parent_v1
    from .mutation_routed_full_panel_observations_v1 import load_full_parent_v1
    from .marker_resolution_readout_v1 import prepare_marker_evidence_v1,bind_marker_bank_v1,logical_bounds_v1
    require(ctx.get("root_authorizer")=="root_agent" and ctx.get("admitted") is True
        and ctx.get("phase")=="prepare" and callable(ctx.get("guard")),"Admitted CPU root preparation context")
    require(not torch.cuda.is_initialized(),"CPU preparation must not initialize CUDA")
    require(set(ctx["entry_refs"])==set(PARENTS) and "bindings" not in ctx,"Exact40 observed parents, no target capability")
    h42.science()
    reader=ObservationReaderV1(ctx["reader"]);output=ctx["output"];guard=ctx["guard"]
    index=read_full_cache_index_v1(ctx["observation_cache_ref"],reader,registry=ctx["registry"],
        entry_refs=ctx["entry_refs"],source_refs=ctx["observation_source_refs"])
    rows={r["parent_id"]:r for r in ctx["registry"]}
    require(set(rows)==set(PARENTS),"Original40-parent registry")
    products={};started=time.monotonic()
    try:
        for pid in PARENTS:
            guard()
            item=read_full_cache_parent_v1(pid,index,reader)
            bank,parent=item["model_inputs"],item["scoring"]
            require(h42.h038.tensor_bytes_v1(bank)<=ctx["cfg"]["CPU_bank_cap_bytes"],"One original bank exceeds CPU cap")
            entry=reader.decode_json(ctx["entry_refs"][pid])
            tree_path=ROOT/entry["refs"]["native_tree"]["path"]
            require(tree_path.stat().st_size<=ctx["cfg"]["CPU_scratch_cap_bytes"],"Original tree scratch exceeds admitted cap")
            ctx["scratch_guard"](ctx["cfg"]["CPU_scratch_cap_bytes"])
            loaded=load_full_parent_v1(rows[pid],ctx["entry_refs"][pid],reader,
                scratch_directory=ctx["scratch_directory"])
            sidecar=coordinate_sidecar_v1(pid,loaded,bank,parent,index,ctx)
            markers=prepare_marker_evidence_v1(bank,sidecar)
            bind_marker_bank_v1(bank,markers).validate()
            require(markers.tensor_bytes()<=ctx["cfg"]["marker_cache_cap_bytes"],"One prepared marker entry exceeds cap")
            s,h=bank.genotype.shape;t=len(bank.spans);p=len(parent["positions"])
            limits=logical_bounds_v1(s,t,p,genotype_bytes=bank.genotype.element_size())
            coordinates_ref=output.save(pid+".coordinates.npz",
                {k:getattr(sidecar,k).numpy() for k in COORDINATE_KEYS},"npz")
            binding_ref=output.save(pid+".binding.json",dict(sidecar.binding))
            products[pid]=dict(coordinates_ref=coordinates_ref,binding_ref=binding_ref,
                original_observed_entry_ref=ctx["entry_refs"][pid],
                original_cache_array_ref=index["parents"][pid]["array_ref"],
                counts=dict(sites=s,haplotypes=h,queries=686,native_holds=t,scoring_centers=p,
                    genotype_element_bytes=bank.genotype.element_size()),
                marker_tensor_bytes=markers.tensor_bytes(),logical_bounds=limits)
            output.save(pid+".prepared.json",products[pid])
            print(json.dumps(dict(stage="H044_original_observation_prepared",parent_id=pid,counts=products[pid]["counts"])),flush=True)
            del loaded,markers,sidecar,bank,parent,item
            guard()
        result=dict(schema="H044_marker_observation_preparation_v1",
            status="completed_all40_observation_only_sidecars",hypothesis_id="H044",
            input_ref=input_ref,config_ref=ctx["config_ref"],source_manifest_ref=ctx["source_manifest_ref"],
            observation_cache_ref=ctx["observation_cache_ref"],entry_refs=ctx["entry_refs"],
            observation_source_refs=ctx["observation_source_refs"],parents=products,parent_ids=list(PARENTS),
            coordinate_keys=list(COORDINATE_KEYS),observations_only=True,targets_read=False,
            checkpoint_payloads_read=False,CUDA_initialized=False,seconds=time.monotonic()-started,
            direct_wait_is_separate_root_evidence=True,non_mutating_tensor_alias_and_metadata_discipline_required=True)
        output.save("completed.json",result)
        return result
    except Exception as error:
        output.save("failed.json",dict(schema="H044_marker_preparation_failure_v1",hypothesis_id="H044",
            error_type=type(error).__name__,error=str(error),prepared_parent_ids=list(products),
            status="failed_partial_preserved",targets_read=False,automatic_replay=False))
        raise

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true");parser.add_argument("--input")
    parser.add_argument("--expected-input-sha256");args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="source_only_not_executed",source=SOURCE,
            phase="prepare",parents=40,truth_access=False,root_context_required=True)))
        return 0
    require(args.input and args.expected_input_sha256,"Pinned active CPU capability required")
    record=dict(path=args.input,sha256=args.expected_input_sha256);spec=json.loads(bound(record))
    require(spec.get("schema")=="H044_marker_root_execution_input_v1" and spec.get("hypothesis_id")=="H044"
        and spec.get("phase")=="prepare" and spec.get("root_authorizer")=="root_agent"
        and spec.get("status")=="active","Active root CPU preparation capability required")
    ref=spec["root_context_module_ref"];bound(ref)
    require(ref["path"].startswith("research/reviews/") and ref["path"].endswith(".py"),"Root context module")
    module_spec=importlib.util.spec_from_file_location("_H044_preparation_context",ROOT/ref["path"])
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    return 0 if run_context_v1(module.build_context_v1(record,spec),record) else 1

if __name__=="__main__":raise SystemExit(main())
