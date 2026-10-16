"""Root H053 TRAIN-only real-observation operator-support audit, no fit/GPU/labels."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,sys,time,traceback
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE="research/reviews/TGB_trade_H051_data_integration_root_20261016_v1"
def reference(p):
    p=Path(p);return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
    sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"scripts")]
    from research.reviews.TGB_trade_H051_data_integration_root_20261016_v1.observed_archive_v1 import download_observations_v1
    from research.reviews.TGB_trade_H051_training_source_20261016_v1.preparation_v1 import canonical_cut_v1
    from research.reviews.H051_conditional_innovation_design_20261016_v1.conditional_innovation_v1 import ConditionalInnovationTransportV1,make_plan_v1
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    start=time.monotonic()
    archive=download_observations_v1()
    assert not archive.train_label_member_opened
    dataset=archive.dataset_v1(tuple(range(1987,2010)))
    triple=ConditionalInnovationTransportV1(gate="fixed",alpha=1.,backend="pyg")
    marginal=ConditionalInnovationTransportV1(gate="fixed",alpha=0.,backend="pyg")
    basis=torch.eye(255,dtype=torch.float32)
    rows=[];a=b=None;previous_sha=None;cut_change_count=0;active_count=0
    for index,year in enumerate(dataset.forecast_years):
        graph=dataset[index];c=canonical_cut_v1(graph.cut_node_index)
        assert graph.released_year_mask[0,list(range(year-1986,31))].sum().item()==0
        row=dict(forecast_year=year,latest_observation_year=year-1,nodes=graph.num_nodes,
            edges=graph.edge_index.shape[1],cut_cells=int(torch.unique(c).numel()),
            missing_countries=int(graph.leaf_missing.sum()),stale_countries=int((graph.leaf_staleness_years>0).sum()),
            maximum_staleness=int(graph.leaf_staleness_years.max()),observation_sha256=graph.observation_sha256,
            cut_changed=False,conditional_increment_nonzero=False)
        if b is None:a=b=c
        elif not torch.equal(b,c):
            row["cut_changed"]=True;cut_change_count+=1
            digest=hashlib.sha256(json.dumps([previous_sha,graph.observation_sha256],separators=(",",":")).encode()).hexdigest()
            provenance=dict(schema="H051_observation_only_cut_source_v1",observed_trees_sha256=digest,
                builder_sha256=reference(ROOT/"research/reviews/TGB_trade_PyG_dataset_source_20261016_v1/trade_tree_core_v1.py")["sha256"],
                targets_or_outcomes_used=False,identity_namespace_sha256=None)
            plan=make_plan_v1(a,b,c,provenance=provenance)
            with torch.no_grad():
                delta=(triple(basis,plan,basis="leaf")-marginal(basis,plan,basis="leaf")).index_select(0,plan.leaf_bc)
            norm=float(torch.linalg.matrix_norm(delta))
            row.update(conditional_increment_frobenius=norm,conditional_increment_max_abs=float(delta.abs().max()),
                affected_destination_countries=int((delta.abs().amax(1)>1e-6).sum()),
                conditional_increment_nonzero=bool(delta.abs().max()>1e-6))
            active_count+=int(row["conditional_increment_nonzero"]);a,b=b,c
        previous_sha=graph.observation_sha256;rows.append(row)
    assert not archive.train_label_member_opened and not torch.cuda.is_initialized()
    report=dict(schema="H053_actual_TRAIN_observation_operator_support_v1",
        completed_utc=datetime.now(timezone.utc).isoformat(),source_ref=reference(Path(__file__).resolve()),
        parser_source_ref=reference(ROOT/HERE/"observed_archive_v1.py"),
        archive_sha256=archive.ARCHIVE_SHA if hasattr(archive,"ARCHIVE_SHA") else "697e9ca432c03aa6d4a998a75a1e49310bc78682dbe8cec2d8452b5a6604c9fe",
        archive_observation_rows=sum(archive.year_rows.values()),country_count=255,
        graph_forecast_years=list(dataset.forecast_years),cut_threshold=.25,cut_change_count=cut_change_count,
        nonzero_conditional_increment_years=active_count,by_year=rows,
        target_label_member_opened=False,VAL_TEST_graphs_constructed=False,
        trainable_model_constructed=False,optimizer_steps=0,ancestry_or_trade_accuracy_computed=False,
        GPU_context_created=False,torch_cuda_initialized=False,seconds=time.monotonic()-start,
        scientific_scope="Fixed support/geometry diagnostic on TRAIN observations only. Nonzero transport is not predictive utility or novelty.")
    path=ROOT/HERE/"actual_TRAIN_operator_support.root_v1.json"
    with path.open("x") as stream:json.dump(report,stream,indent=2);stream.write("\n")
    print(json.dumps({k:report[k] for k in ["cut_change_count","nonzero_conditional_increment_years","target_label_member_opened","GPU_context_created","seconds"]}),flush=True)
    return 0
if __name__=="__main__":
    try:raise SystemExit(main())
    except SystemExit:raise
    except BaseException:traceback.print_exc(file=sys.stdout);raise SystemExit(1)
