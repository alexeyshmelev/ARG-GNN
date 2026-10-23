"""Authenticate and adapt the completed historical TRAIN0 without regeneration."""
from __future__ import annotations

import struct
import numpy as np
from . import paper_argmix_generator as g


def adopt_train0(config_path, output, *, guard):
    config, parent = g.parent_record(config_path, "published_recon_train_000")
    source = g.confined(config["existing_train0"]["path"])
    observed_receipt = source/"observed/observed_generation_receipt.json"
    g.need(g.sha(observed_receipt) == config["existing_train0"]["observed_receipt_sha256"], "TRAIN0 receipt drift")
    observed = g.metadata(observed_receipt)
    private_receipt = source/"private_receipts/generation_receipt.json"
    private = g.metadata(private_receipt)
    g.need(observed["complete"] and private["complete"] and
           private["registered_seeds"] == parent["new_seed_values"] and
           observed["parent_id"] == private["parent_id"] == parent["parent_id"], "Historical identity or seeds differ")
    for record in observed["files"].values():
        g.verify_ref(record)
        guard(0)
    for record in private["private_truth_files"].values():
        g.verify_ref(record)
        guard(0)
    g.need(g.metadata(source/"private_truth/demography.json") == config["demography"], "Historical demography differs")
    rows = g.roster(config)
    old_rows = g.metadata(source/"observed/roster.unbound.json")["rows"]
    g.need(len(old_rows) == len(rows) and all(
        old["diploid_id"] == new["sample_id"] and old["phase"] == new["haplotype"] and
        old["cohort"] == new["observed_cohort"] and old["age_generations"] == new["age_generations"]
        for old,new in zip(old_rows,rows)), "Historical semantic sample join differs")
    output = g._fresh_directory(output,guard)
    obs, truth = output/"observed", output/"truth"
    obs.mkdir(); truth.mkdir()
    budget = g.Writes(config["bounds"]["generation_bytes"],guard)
    clean_rows = [{k:v for k,v in row.items() if k != "sampling_population"} for row in rows]
    budget.json(obs/"sample_manifest.json",clean_rows)
    budget.text(obs/"sample_order.txt","\n".join(r["sample_id"] for r in rows[::2])+"\n")
    budget.text(obs/"sample_ages.generations","\n".join(str(r["age_generations"]) for r in rows)+"\n")
    budget.text(obs/"population_labels.txt","sample population group sex\n"+"".join(
        f'{r["sample_id"]} {r["observed_cohort"]} {r["observed_cohort"]} NA\n' for r in rows[::2]))
    posbuffer, genbuffer, count, previous = bytearray(), bytearray(), 0,-1
    with g.verify_ref(observed["files"]["observed.haps"]).open("rb") as stream, (obs/"positions.u4").open("xb") as positions, (obs/"genotypes.packbits").open("xb") as packed:
        for line in stream:
            fields=line.split(maxsplit=5)
            g.need(len(fields)==6 and fields[0]==b"22" and fields[3:5]==[b"A",b"C"], "Unexpected historical haps format")
            position=int(fields[2])-1
            genotype=np.fromstring(fields[5].decode("ascii"),dtype=np.uint8,sep=" ")
            g.need(len(genotype)==len(rows) and np.all(genotype <= 1) and
                   0 < int(genotype.sum()) < len(rows) and previous < position < config["map"]["length_bp"],
                   "Historical genotype shape or coordinate differs")
            previous=position
            posbuffer.extend(struct.pack("<I",position))
            genbuffer.extend(np.packbits(genotype,bitorder="little").tobytes())
            count+=1
            if len(posbuffer)+len(genbuffer)>=config["bounds"]["io_batch_bytes"]:
                budget.before(len(posbuffer)+len(genbuffer))
                positions.write(posbuffer); packed.write(genbuffer)
                posbuffer.clear(); genbuffer.clear(); budget.after()
        budget.before(len(posbuffer)+len(genbuffer))
        positions.write(posbuffer); packed.write(genbuffer); budget.after()
    g.need(count==observed["observed_sites"], "Historical site count differs")
    manifest=dict(schema="ARGMix_prospective1611_observations_v1",parent_id=parent["parent_id"],split="train",
                  config_ref=g.file_ref(config_path),source_ref=g.file_ref(__file__),counts=config["counts"],sites=count,
                  length_bp=config["map"]["length_bp"],observed_files={p.name:g.file_ref(p) for p in obs.iterdir()},
                  original_observation_receipt_ref=g.file_ref(observed_receipt),
                  external_observed_vcf_ref=observed["files"]["observed.vcf"],
                  error_applied_again=False,contains_genealogy_or_founder_truth=False,original_paper_reproduction=False)
    budget.json(obs/"observation_manifest.json",manifest)
    mapping=g.metadata(source/"private_truth/simulation_node_mapping.json")["rows"]
    g.need(len(mapping)==len(rows) and all(m["haplotype_id"]==old["haplotype_id"] for m,old in zip(mapping,old_rows)), "Raw truth identity join differs")
    truth_manifest=dict(schema="ARGMix_prospective1611_census_truth_v1",parent_id=parent["parent_id"],split="train",
                        config_ref=g.file_ref(config_path),census_time=config["demography"]["census_time"],
                        ancestry_file=private["private_truth_files"]["simulation.trees"],class_order=list(g.CLASSES),
                        population_id_to_name=g.metadata(source/"private_truth/population_mapping.json")["populations"],
                        sample_index_to_raw_node_id=[m["simulation_node_id"] for m in mapping],
                        original_private_receipt_ref=g.file_ref(private_receipt),truth_boundaries_forbidden_as_observed_model_inputs=True)
    budget.json(truth/"truth_manifest.json",truth_manifest)
    receipt=dict(schema="paper_argmix_adopted_train0_v1",parent_id=parent["parent_id"],split="train",
                 config_ref=g.file_ref(config_path),source_ref=g.file_ref(__file__),
                 observation_manifest_ref=g.file_ref(obs/"observation_manifest.json"),
                 truth_manifest_ref=g.file_ref(truth/"truth_manifest.json"),
                 regenerated=False,relate_executed=False,originals_modified=False)
    budget.json(output/"generation.json",receipt)
    return receipt
