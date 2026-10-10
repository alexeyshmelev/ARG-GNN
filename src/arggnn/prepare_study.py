"""Versioned split-by-parent study preparation with common reference inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from .resources import ROOT,project_path,require_storage


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    import torch
    torch.set_num_threads(4)
    from .ancient_simulate import AncientSimulationConfig,simulate_ancient_replicate,save_ancient_replicate
    from .simulate import SimulationConfig,simulate_replicate,save_replicate
    from .reference_encoder import ReferenceFeatureAdapter
    source=project_path(args.config)
    target=project_path(args.output)
    if target.exists():
        raise RuntimeError("Use a fresh versioned dataset directory")
    settings=json.loads(source.read_text())
    if settings["benchmark"]=="ancient-europe-stdpopsim-0.3.0-oracle":
        config=AncientSimulationConfig(**settings["simulation"])
        simulate,save=simulate_ancient_replicate,save_ancient_replicate
    elif settings["benchmark"]=="three-source-neutral-pulse-oracle":
        config=SimulationConfig(**settings["simulation"])
        simulate,save=simulate_replicate,save_replicate
    else:
        raise ValueError("Unknown benchmark")
    if settings["splits"]["train"]<1 or settings["splits"]["val"]<1:
        raise ValueError("Training and validation require independent parents")
    target.mkdir(parents=True)
    manifest={"schema_version":2,"protocol":settings,"protocol_sha256":sha(source),
              "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              "git_dirty_at_generation":bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()),
              "generation_code_sha256":{str(Path(__file__).relative_to(ROOT)):sha(__file__)},
              "replicates":[]}
    for split in ("train","val","test"):
        for replicate in range(settings["splits"].get(split,0)):
            require_storage(headroom_bytes=500_000_000)
            case=simulate(config,master_seed=settings["master_seed"],split=split,replicate=replicate)
            directory=target/split/f"rep{replicate:03d}"
            record=save(case,directory)
            record["tensor_sha256"]=sha(record["tensor_file"])
            record["metadata_sha256"]=sha(record["metadata"])
            adapter=ReferenceFeatureAdapter(case.sequence.x.shape[1],**settings["reference_adapter"])
            prepared=adapter.precompute(case.sequence)
            cached=directory/"prepared.pt"
            provenance=adapter.manifest_provenance(case.metadata["feature_names"],record["tensor_sha256"])
            torch.save({"sequence":prepared.as_dict(),"adapter":provenance},cached)
            record.update(prepared_tensor_file=str(cached),prepared_tensor_sha256=sha(cached),
                          adapter_configuration_sha256=provenance["configuration_sha256"],
                          true_switches=case.metadata["total_true_switches"],
                          grid_switches=sum(case.metadata["grid_switch_counts_per_query"]))
            manifest["replicates"].append(record)
            (target/"manifest.partial.json").write_text(json.dumps(manifest,indent=2)+"\n")
            print(json.dumps({"event":"replicate_ready","split":split,"replicate":replicate,
                              "trees":prepared.num_graphs,"nodes":len(prepared.x),
                              "true_switches":record["true_switches"],"grid_switches":record["grid_switches"]}),flush=True)
    (target/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (target/"manifest.partial.json").unlink()
    print(json.dumps({"event":"dataset_complete","manifest":str(target/"manifest.json"),
                      "replicates":len(manifest["replicates"]),"storage_bytes":require_storage()}),flush=True)


if __name__=="__main__":
    main()
