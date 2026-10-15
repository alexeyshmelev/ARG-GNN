#!/usr/bin/env python3
"""Separate external-supervision constructor. Never imported by feature extraction.

CLI use is deferred to root; source tests call only injected tiny metadata trees.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from panel_adapter import (HERE,ROOT,Panel,LABELS,canonical_sha,sha256,load_json,
                           read_jsonl,write_json,validate_targets,validate_labels)

DEFINITION={
    "schema":"prospective1611_census_founder_definition_v1",
    "time_generations":350,
    "state":"population of the exact simulated ancestral node at census350",
    "founder_populations":["Ana","CHG","WHG","EHG"],
    "class_order":["Ana","CHG","WHG","EHG"],
    "coordinate":"exact shared inferred-tree interval midpoint; truth tree lookup is right-continuous",
    "sample_join":"explicit haplotype_id to simulation_node_id, separately from observed Relate IDs",
    "fallback":"none; missing census node, unknown founder or missing mapping is an error",
    "BAA_sample_count":0,
    "usage":"external training supervision, validation and scoring only; never observed features or donor eligibility"
}

def validate_cut_demography(doc):
    if doc.get("census_time")!=350 or doc.get("BAA_samples")!=0:
        raise ValueError("demography must declare census350 and zero BAA samples")
    if not set(LABELS)<=set(doc.get("populations_initial_size",{})):
        raise ValueError("four founder populations absent from demography")
    if any(not 0<float(e["time"])<350 for e in doc.get("admixtures",[])):
        raise ValueError("all retained admixtures must precede census350")
    if any(not float(e["time"])>350 for e in doc.get("splits",[])):
        raise ValueError("all retained founder divergences must be older than census350")
    for e in doc.get("admixtures",[]):
        if "BAA" in [e["derived"],*e["ancestral"]]:
            raise ValueError("this prospective condition removes the unused BAA event")
    if not doc.get("admixtures") or not doc.get("splits"):
        raise ValueError("explicit complete demography event lists required")
    return canonical_sha(doc)

def simulation_sample_map(document,panel):
    if document.get("schema")!="simulation_haplotype_node_mapping_v1":
        raise ValueError("wrong simulation mapping schema")
    rows=document.get("rows",[])
    if any(set(r)!={"haplotype_id","simulation_node_id"} for r in rows):
        raise ValueError("simulation mapping fields mismatch")
    if len(rows)!=len({r["haplotype_id"] for r in rows}):
        raise ValueError("duplicate simulation sample identity")
    mapping={r["haplotype_id"]:r["simulation_node_id"] for r in rows}
    if set(mapping)!=set(panel.by_hap):
        raise ValueError("simulation mapping must exactly cover observed haplotype identities")
    if any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in mapping.values()) or len(set(mapping.values()))!=len(mapping):
        raise ValueError("simulation node mapping must be one-to-one nonnegative integers")
    return mapping

def census_founder_label(tree,sample_node,population_names):
    """Require the real census node; never infer labels from cohort or index."""
    u=sample_node;visited=set()
    if not float(tree.time(u))<350:
        raise ValueError("sample must be younger than census350")
    while True:
        if u in visited:
            raise ValueError("cycle in simulation ancestry")
        visited.add(u)
        p=tree.parent(u)
        if p==-1 or float(tree.time(p))>350:
            break
        if float(tree.time(p))<float(tree.time(u)):
            raise ValueError("nonmonotone ancestral time")
        u=p
    if float(tree.time(u))!=350:
        raise ValueError("exact census350 ancestral node missing")
    pop=int(tree.population(u))
    name=population_names.get(pop)
    if name not in LABELS:
        raise ValueError("census node is not in an allowed founder population")
    return name

def construct_common_labels(truth_ts,simulation_mapping,population_mapping,targets,panel):
    validate_targets(targets,panel)
    mapping=simulation_sample_map(simulation_mapping,panel)
    if population_mapping.get("schema")!="simulation_population_mapping_v1":
        raise ValueError("wrong population mapping schema")
    names={int(k):v for k,v in population_mapping["populations"].items()}
    if len(names)!=len(population_mapping["populations"]):
        raise ValueError("duplicate integer population identity")
    if set(map(int,truth_ts.samples()))!=set(mapping.values()):
        raise ValueError("truth sequence samples differ from simulation mapping")
    for hid,nid in mapping.items():
        if float(truth_ts.node(nid).time)!=panel.by_hap[hid]["age_generations"]:
            raise ValueError("truth sample age differs from common observed roster")
        expected_population={"NF":"Neo","PostNeolithic":"present_bronze","Present":"present_bronze"}.get(panel.by_hap[hid]["cohort"],panel.by_hap[hid]["cohort"])
        if names.get(int(truth_ts.node(nid).population))!=expected_population:
            raise ValueError("truth sample population differs from common roster cohort")
    for pid,name in names.items():
        metadata=truth_ts.population(pid).metadata
        if not isinstance(metadata,dict) or metadata.get("name")!=name:
            raise ValueError("population-name mapping differs from simulation metadata")
    out=[]
    for row in targets:
        position=float(row["position"])
        if not 0<=position<truth_ts.sequence_length:
            raise ValueError("truth target outside simulation span")
        tree=truth_ts.at(position)
        if not float(tree.interval.left)<=position<float(tree.interval.right):
            raise ValueError("truth tree interval does not contain target coordinate")
        out.append({"row_id":row["row_id"],
                    "ancestry":census_founder_label(tree,mapping[row["haplotype_id"]],names)})
    validate_labels(out,targets)
    return out

def validate_supervision_contract(contract,*,parent_id,targets_path,roster_path,
                                  labels_path,observed_metadata):
    if contract.get("schema")!="prospective1611_external_supervision_contract_v1" or not contract.get("complete"):
        raise ValueError("missing/incomplete external founder supervision contract")
    if contract.get("parent_id")!=parent_id or contract.get("definition")!=DEFINITION:
        raise ValueError("wrong parent/ancestral-state definition")
    pairs=(("targets_sha256",targets_path),("roster_sha256",roster_path),
           ("labels_sha256",labels_path))
    for key,path in pairs:
        if contract.get(key)!=sha256(path):
            raise ValueError("supervision "+key+" mismatch")
    if contract.get("observed_trees_sha256")!=observed_metadata.get("observed_trees_sha256"):
        raise ValueError("supervision and observed tree bank identity differ")
    if contract.get("targets_sha256")!=observed_metadata.get("targets_sha256") or contract.get("roster_sha256")!=observed_metadata.get("roster_sha256"):
        raise ValueError("supervision/common observation metadata mismatch")
    if contract.get("target_rows")!=observed_metadata.get("target_rows"):
        raise ValueError("supervision target count differs")
    if contract.get("truth_definition_sha256")!=canonical_sha(DEFINITION):
        raise ValueError("truth definition identity differs")
    source=contract.get("constructor_source",{})
    if source.get("path")!=str((HERE/"truth_adapter.py").relative_to(ROOT)) or source.get("sha256")!=sha256(HERE/"truth_adapter.py"):
        raise ValueError("wrong census constructor source identity")
    provenance=contract.get("truth_input_sha256",{})
    if set(provenance)!={"simulation_trees","simulation_sample_mapping","population_mapping","demography"}:
        raise ValueError("external truth provenance incomplete")
    if any(not isinstance(v,str) or len(v)!=64 or any(c not in "0123456789abcdef" for c in v) for v in provenance.values()):
        raise ValueError("invalid truth input digest")
    validate_cut_demography(contract["demography"])
    if canonical_sha(contract["demography"])!=contract.get("demography_semantic_sha256"):
        raise ValueError("demography definition hash differs")
    return {"labels_sha256":contract["labels_sha256"],
            "supervision_contract_sha256":canonical_sha(contract),
            "truth_definition_sha256":canonical_sha(DEFINITION)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("roster","targets","simulation-trees","simulation-mapping",
                 "population-mapping","demography","observed-metadata","parent","output"):
        p.add_argument("--"+name,required=True)
    args=p.parse_args()
    panel=Panel(load_json(args.roster));targets=read_jsonl(args.targets)
    validate_targets(targets,panel,parent_id=args.parent)
    demography=load_json(args.demography);validate_cut_demography(demography)
    observed=load_json(args.observed_metadata)
    if observed.get("schema")!="prospective1611_native_features_v1" or not observed.get("complete") or observed.get("parent_id")!=args.parent or observed.get("targets_sha256")!=sha256(args.targets) or observed.get("roster_sha256")!=sha256(args.roster):
        raise ValueError("observed feature metadata does not bind common parent/roster/targets")
    # This is the only adapter entrypoint that opens simulation truth. It is a
    # separate externally scheduled supervision stage, never feature extraction.
    import tskit
    truth_ts=tskit.load(args.simulation_trees)
    if max(r["right"] for r in targets)>truth_ts.sequence_length:
        raise ValueError("inferred target interval exceeds truth simulation span")
    labels=construct_common_labels(truth_ts,load_json(args.simulation_mapping),
                                    load_json(args.population_mapping),targets,panel)
    with Path(args.output).open("x") as out:
        for row in labels:
            out.write(json.dumps(row,separators=(",",":"))+"\n")
    contract={"schema":"prospective1611_external_supervision_contract_v1","complete":True,
              "parent_id":args.parent,"target_rows":len(targets),"definition":DEFINITION,
              "truth_definition_sha256":canonical_sha(DEFINITION),
              "targets_sha256":sha256(args.targets),"roster_sha256":sha256(args.roster),
              "labels_sha256":sha256(args.output),
              "observed_trees_sha256":observed["observed_trees_sha256"],
              "truth_input_sha256":{"simulation_trees":sha256(args.simulation_trees),
                   "simulation_sample_mapping":sha256(args.simulation_mapping),
                   "population_mapping":sha256(args.population_mapping),
                   "demography":sha256(args.demography)},
              "demography":demography,"demography_semantic_sha256":canonical_sha(demography),
              "constructor_source":{"path":str((HERE/"truth_adapter.py").relative_to(ROOT)),
                                    "sha256":sha256(HERE/"truth_adapter.py")},
              "producer_stage":"separate_external_supervision_only",
              "AP_positive_six_head_target_indices":[0,1,2,3],
              "AP_no_positive_target_indices_native_BAA_paths":[4,5],
              "AP_method_label":"AncestralPaths-derived C4 adapter",
              "paper_reproduction":False}
    validate_supervision_contract(contract,parent_id=args.parent,targets_path=args.targets,
                                 roster_path=args.roster,labels_path=args.output,
                                 observed_metadata=observed)
    write_json(args.output+".contract.json",contract)

if __name__=="__main__":
    main()
