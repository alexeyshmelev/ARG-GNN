"""Refresh concise decisions and an exact, scalar-only result index.

Reads immediate experiment JSON receipts, never tensors, predictions, data,
or sealed confirmation directories. Existing source receipts stay untouched.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
METRICS = ("validation_nll", "mean_accuracy", "mae_cases", "rmse_cases",
           "parameters", "best_epoch", "training_seconds", "elapsed_seconds",
           "independent_replicates", "bootstrap_unit")
BINDINGS = ("stage", "interpretation", "input_kind", "evaluation",
            "evaluation_split", "training_seed", "protocol_sha256",
            "dataset_manifest_sha256", "dataset_directory", "git_commit",
            "physical_gpu", "environment", "test_evaluated", "final_test_evaluated")
EXCLUDE = {"protocol.json", "dataset_manifest.json", "progress.json", "results.json"}
DENIED = {"test", "tests", "heldout", "held-out", "confirmatory", "confirmation"}

def safe(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(ROOT)
    if path.is_symlink():
        raise ValueError("Symlink artifact forbidden")
    return resolved

def read_json(path: Path):
    raw = safe(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()

def receipt_ref(path: Path):
    safe(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        return {"path": str(path.relative_to(ROOT)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    # Immediate JSON receipts only, not recursive raw data or predictions.
    return {"path": str(path.relative_to(ROOT)), "kind": "receipt_directory",
            "json_receipts": [{"path": str(f.relative_to(ROOT)),
                              "sha256": hashlib.sha256(safe(f).read_bytes()).hexdigest()}
                             for f in sorted(path.glob("*.json"))]}

def scalar_metrics(obj):
    out = {key: obj[key] for key in METRICS
           if key in obj and (obj[key] is None or isinstance(obj[key], (str, int, float, bool)))}
    # The temporal runner stores equal-parent averages under mean_metrics.
    for container in ("mean_metrics", "metrics", "aggregate"):
        nested = obj.get(container)
        if isinstance(nested, dict):
            for key in ("accuracy", "nll", "brier", "macro_f1_all_classes_fixed_zero"):
                if key in nested and isinstance(nested[key], (int, float)):
                    out[container + "." + key] = nested[key]
    return out

def main():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("Run in the dedicated server repository")
    spec, spec_hash = read_json(ROOT / "research/experiment_decisions.json")
    ids = [entry["id"] for entry in spec["decisions"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate hypothesis ID")
    entries = []
    for entry in spec["decisions"]:
        if not all(entry.get(k) for k in
                   ("hypothesis", "status", "method", "conclusion", "repeat_policy", "evidence")):
            raise ValueError("Incomplete decision")
        item = dict(entry)
        item["evidence_receipts"] = [receipt_ref(ROOT / rel) for rel in entry["evidence"]]
        entries.append(item)

    inventory, rows = [], []
    result_root = ROOT / "experiments/results"
    for path in sorted(result_root.iterdir()):
        # Never inventory/read future sealed confirmation or test result banks.
        components = set(path.name.lower().replace("-", "_").split("_"))
        if components.intersection(DENIED):
            continue
        if path.is_file() and path.suffix == ".json":
            obj, digest = read_json(path)
            inventory.append({"path": str(path.relative_to(ROOT)), "sha256": digest,
                              "kind": "top_level_receipt",
                              "status": obj.get("status") if isinstance(obj, dict) else None})
            continue
        if not path.is_dir():
            continue
        context = {}
        context_refs = []
        for name in ("results.json", "progress.json", "protocol.json"):
            f = path / name
            if not f.exists():
                continue
            obj, digest = read_json(f)
            context_refs.append({"path": str(f.relative_to(ROOT)), "sha256": digest})
            if isinstance(obj, dict):
                for key in BINDINGS:
                    if key in obj and key not in context:
                        context[key] = obj[key]
        for f in sorted(path.glob("*.json")):
            obj, digest = read_json(f)
            if not isinstance(obj, dict):
                continue
            record = {"study": path.name, "artifact": str(f.relative_to(ROOT)),
                      "sha256": digest, "context_receipts": context_refs}
            if f.name in EXCLUDE:
                record["kind"] = "study_binding"
            else:
                metrics = scalar_metrics(obj)
                record.update(kind="reported_model_result" if metrics else "engineering_receipt",
                              metrics=metrics, bindings=context)
                for key in ("status", "checkpoint", "checkpoint_sha256",
                            "validation_predictions", "validation_predictions_sha256"):
                    if key in obj:
                        record[key] = obj[key]
                if metrics:
                    rows.append({"study": path.name, "method": f.stem,
                                 "artifact": record["artifact"], "sha256": digest,
                                 **{k: metrics.get(k) for k in METRICS},
                                 "evaluation_scope": context.get("evaluation_split",
                                                        context.get("evaluation", "see source receipt")),
                                 "training_seed": context.get("training_seed"),
                                 "protocol_sha256": context.get("protocol_sha256")})
            inventory.append(record)
        # Include named nonlearned comparators that only appear in the aggregate receipt.
        f = path / ("results.json" if (path / "results.json").exists() else "progress.json")
        if f.exists():
            obj, digest = read_json(f)
            for key in ("nearest_tmrca", "nearest_tmrca_validation", "classical_controls"):
                val = obj.get(key) if isinstance(obj, dict) else None
                if isinstance(val, dict):
                    inventory.append({"study": path.name, "artifact": str(f.relative_to(ROOT)),
                                      "sha256": digest, "kind": "aggregate_comparator",
                                      "method": key, "reported_values": val})
    record = {"schema_version": 1, "decision_spec_sha256": spec_hash,
              "scope": spec["scope"], "rules": spec["rules"],
              "metric_warning": "Reported source metrics retain their original scope. Different studies, "
                                "panels, inference pipelines or model information access are not "
                                "interchangeable. No automatic cross-study ranking.",
              "decisions": entries, "result_inventory": inventory}
    outputs = {
        ROOT / "research/experiment_ledger.json": json.dumps(record, indent=2, allow_nan=False) + "\n"}
    buf = io.StringIO()
    fields = ["study", "method", *METRICS, "evaluation_scope", "training_seed",
              "protocol_sha256", "artifact", "sha256"]
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    outputs[ROOT / "research/experiment_results.csv"] = buf.getvalue()
    lines = ["# Experiment and decision ledger", "",
             spec["scope"], "",
             "Before a substantive run, register its question and check this ledger. "
             "A repeat requires an explicit reason linked to the earlier ID. "
             "Complete result values and source hashes are in [experiment_results.csv]"
             "(experiment_results.csv) and [experiment_ledger.json](experiment_ledger.json). "
             "Method/configuration, dataset, source, seed and checkpoint bindings remain in the "
             "linked original receipts. Failures and partial runs are retained.", "",
             "Status is local to a question. A pilot observation is not a validated paper claim.", ""]
    for e in entries:
        links = ", ".join("[" + rel + "](../" + rel + ")" for rel in e["evidence"])
        lines.extend(["## " + e["id"] + ": " + e["hypothesis"], "",
                      "**" + e["status"] + ".** " + e["method"], "",
                      "**Conclusion:** " + e["conclusion"], "",
                      "**Next/repeat rule:** " + e["repeat_policy"], "",
                      "Evidence: " + links + ".", ""])
    outputs[ROOT / "research/experiment_ledger.md"] = "\n".join(lines)
    changed = []
    for path, contents in outputs.items():
        if path.exists() and path.read_text() == contents:
            continue
        path.write_text(contents)
        changed.append(str(path.relative_to(ROOT)))
    print(json.dumps({"decision_count": len(entries), "model_result_count": len(rows),
                      "indexed_receipt_count": len(inventory), "changed_files": changed}))

if __name__ == "__main__":
    main()
