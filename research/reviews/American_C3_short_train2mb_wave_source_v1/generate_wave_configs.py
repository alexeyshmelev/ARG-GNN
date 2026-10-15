"""Generate inactive H041 20x2Mb TRAIN metadata only; never run science."""
from pathlib import Path
import hashlib,json
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
REL="research/reviews/American_C3_short_train2mb_wave_source_v1"
HERE=ROOT/REL
SOURCE=REL+"/generate_wave_configs.py"
ORIGINAL="research/reviews/ARGformer_American_C3_implementation_20261015_v1"
V5="research/reviews/American_C3_observation_root_execution_20261015_v5"
CONDITION="C3_independent20x2Mb_TRAIN_1600haps_inferred_unary_v1"


def ref(path):
    p=ROOT/path
    return dict(path=path,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def save(path,value):
    p=ROOT/path
    with p.open("x") as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write("\n")
    return ref(path)


def main():
    folder=HERE/"configs"
    if folder.exists():raise ValueError("Preserve existing generated config metadata")
    folder.mkdir()
    original=json.loads((ROOT/ORIGINAL/"parent_seed_registry_inactive.json").read_text())
    sim_base=json.loads((ROOT/ORIGINAL/"simulation_full30mb_inactive.json").read_text())
    obs_base=json.loads((ROOT/V5/"observation_full30mb_inactive.json").read_text())
    parents=[]
    for index in range(20):
        parent="american_c3_short_train_%03d"%index
        ancestry=22001+index;mutation=1022001+index;paths=92201+index
        sim=dict(sim_base,execution_enabled=False,parent_id=parent,split="train",
            sequence_length_bp=2000000,ancestry_seed=ancestry,mutation_seed=mutation,
            max_sites=100000,max_tables_bytes=250000000,max_truth_intervals=20000,
            output_cap_bytes=1000000000,
            interpretation="Prospective H041 independent2Mb TRAIN preparation,1600 haplotypes; not exact published Table1 reproduction; private query truth stays outside observations")
        obs=dict(obs_base,execution_enabled=False,observation_ref=None,max_sites=100000,
            max_tables_bytes=250000000,max_sample_store_bytes=750000000,max_node_map_bytes=250000000,
            output_cap_bytes=2000000000)
        sr=save(REL+"/configs/"+parent+".simulation.inactive.json",sim)
        ir=save(REL+"/configs/"+parent+".inference.inactive.json",obs)
        base="experiments/results/American_C3_short_train2mb_wave_v1/"+parent
        parents.append(dict(parent_id=parent,split="train",sequence_length_bp=2000000,
            ancestry_seed=ancestry,mutation_seed=mutation,path_sampling_seed=paths,
            sample_unit="haploid",sample_counts={"AFR":200,"EUR":200,"ASIA":200,"ADMIX":1000},
            simulation_config_ref=sr,inference_template_ref=ir,
            simulation_output_directory=base+"/simulation",
            inference_output_directory=base+"/inference"))
    ids={p["parent_id"] for p in parents}
    assert len(ids)==20 and not ids & {p["parent_id"] for p in original["parents"]}
    old_seeds={p[k] for p in original["parents"] for k in ("ancestry_seed","mutation_seed","path_sampling_seed")}
    new_seeds=[p[k] for p in parents for k in ("ancestry_seed","mutation_seed","path_sampling_seed")]
    assert len(set(new_seeds))==60 and not set(new_seeds)&old_seeds
    heldout=[p for p in original["parents"] if p["split"] in ("val","test")]
    assert len(heldout)==15
    save(REL+"/parent_registry.inactive.json",dict(
        schema="american_c3_independent_2Mb_TRAIN_wave_registry_v1",execution_enabled=False,
        hypothesis_id="H041",scientific_condition_id=CONDITION,parent_unit="independent simulation before window/path sampling",
        parents=parents,new_independent_TRAIN_count=20,
        class_order=["EUR","EAS","AFR"],sequence_length_bp=2000000,
        original_registry_ref=ref(ORIGINAL+"/parent_seed_registry_inactive.json"),
        deferred_heldout_30Mb_original_records=heldout,
        deferred_VAL_count=5,deferred_TEST_count=10,heldout_generation_authorized_by_this_wave=False,
        existing_30Mb_TRAIN=dict(parent_id="american_c3_train_000",ancestry_seed=21001,
            mutation_seed=1021001,split="train",included_in_new20=False,relabelled_as_TEST=False),
        maximum_concurrent_wave_parents=4,tsinfer_threads_per_parent=8,
        no_bulk_reservation=True,no_model_training_or_prediction_authority=True,
        meaningful_training_proposal_ref=ref("research/reviews/American_C3_GNN_adapter_source_v1/meaningful_1_2Mb_train_condition_inactive.json")))
    s,h,q,i=100000,1600,1000,20000
    sim_bound=2*(250000000+16777216)+s*h+8*s+128*s+i*(q+16)+16777216+8388608
    infer_bound=750000000+2*(250000000+16777216)+250000000+16777216
    overhead=134217728+16777216+1048576
    assert sim_bound+overhead<1000000000 and infer_bound+overhead<2000000000
    save(REL+"/resource_forecast.json",dict(schema="american_c3_2Mb_wave_source_count_bounds_v1",
        counts_are_rejection_bounds_not_measured_2Mb_results=True,
        simulation=dict(serialized_count_bound_bytes=sim_bound,total_plan_bytes=sim_bound+overhead,
            reservation_bytes=1000000000,max_sites=s,max_tables_bytes=250000000,max_truth_intervals=i,
            walltime_seconds=3600),
        inference=dict(serialized_count_bound_bytes=infer_bound,total_plan_bytes=infer_bound+overhead,
            reservation_bytes=2000000000,max_sites=s,max_tables_bytes=250000000,
            max_sample_store_bytes=750000000,max_node_map_bytes=250000000,
            walltime_seconds=10800,num_threads=8),
        combined_cap_per_parent=3000000000,max4_live_parent_cap_bytes=12000000000,
        hypothetical_all20_caps_bytes=60000000000,hypothetical_all20_is_not_admitted=True,
        metadata_overlay_charges_excluded_from_product_caps=True,
        per_event_overlay_metadata_bytes=524288,
        RAM_planning_bytes_per_active_parent=16000000000,RAM_measured_or_enforced=False,
        maximum_simultaneous_tsinfer_worker_threads=32,
        cap_failure="preserve full partial artifacts, stop stage, root versions/reviews bounds; no truncation or replacement parent"))
    save(REL+"/source_generation_receipt.json",dict(schema="american_c3_2Mb_wave_source_generation_v1",
        generator_ref=ref(SOURCE),source_config_generation_only=True,
        parent_ids=[p["parent_id"] for p in parents],ancestry_seeds=list(range(22001,22021)),
        mutation_seeds=list(range(1022001,1022021)),path_sampling_seeds=list(range(92201,92221)),
        original_VAL_TEST_records_exactly_copied=True,all60_new_seeds_unique_and_disjoint_from_original=True,
        scientific_data_opened=False,simulation_executed=False,inference_executed=False,
        admission_created=False,GPU_used=False))
    print(json.dumps(dict(status="inactive_source_metadata_generated",parents=20,sim_bound=sim_bound,infer_bound=infer_bound)))


if __name__=="__main__":main()
