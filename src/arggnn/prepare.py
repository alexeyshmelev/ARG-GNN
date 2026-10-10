"""Generate a versioned, split-by-parent simulation panel from configuration."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from .resources import ROOT, project_path, require_storage

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    torch.set_num_threads(4)
    from .simulate import SimulationConfig, generate_dataset
    source = project_path(args.config)
    target = project_path(args.output)
    if (target / "manifest.json").exists():
        raise RuntimeError("Dataset already exists. Use a new versioned output directory.")
    require_storage(headroom_bytes=5_000_000_000)
    settings = json.loads(source.read_text())
    config = SimulationConfig(**settings["simulation"])
    splits = settings["splits"]
    manifest = generate_dataset(target, config, master_seed=settings["master_seed"],
                                train_replicates=splits["train"],
                                val_replicates=splits["val"], test_replicates=splits["test"])
    manifest["protocol"] = settings
    manifest["protocol_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest["git_commit"] = subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    for record in manifest["replicates"]:
        record["tensor_sha256"] = hashlib.sha256(Path(record["tensor_file"]).read_bytes()).hexdigest()
    (target / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"event":"dataset_complete","manifest":str(target/"manifest.json"),
                      "replicates":len(manifest["replicates"]),
                      "storage_bytes":require_storage()}),flush=True)

if __name__ == "__main__":
    main()
