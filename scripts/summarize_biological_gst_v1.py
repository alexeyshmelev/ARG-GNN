"""Summarize completed, hash-bound GST-style development metrics without arrays."""
from pathlib import Path
import datetime
import hashlib
import json
import statistics

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE = ROOT / "experiments/results/biological_gst_control_v1_seed101/receipt.json"
OUTPUT = ROOT / "experiments/results/biological_gst_control_v1_seed101_summary/summary.json"

def main():
    d = json.loads(SOURCE.read_text())
    if d["status"] != "completed_two_GST_conditions_one_seed":
        raise ValueError("Source fit incomplete")
    out = {
        "kind": "completed_biological_GST_style_development_summary_v1",
        "status": "two_reconstructed_controls_one_seed_completed",
        "hypothesis_id": "H009/H020",
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "native_author_reproduction": False,
        "method_scope": "Independently reconstructed GST-style architectural control with disclosed physical-span token weighting, gate initialization, full-tree F47 access and ARG per-query readout. Task adaptation, not published numerical reproduction.",
        "native_receipt": {"path": str(SOURCE.relative_to(ROOT)), "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest()},
        "training_seeds": [101],
        "independent_VAL_parents": 2,
        "TEST_parents": 0,
        "aggregation": "Equal-parent mean on exact retained-center clipped-bin physical query-bp support. Optimizer seeds do not add biological replication.",
        "conditions": {},
        "new_method_superiority_established": False,
    }
    candidate = json.loads((ROOT / "experiments/results/biological_candidate_pilot_v1_seed101_summary/summary.json").read_text())
    support = {r["parent_id"]: {g: v["denominator_query_bp"] for g,v in r["groups"].items()} for r in candidate["conditions"]["candidate_age_d2"]["rows"]}
    for name, m in d["models"].items():
        if m["status"] != "completed" or not m["optimizer_fit_completed"] or m["selected_maximum_parameter_delta_from_initial"] <= 0:
            raise ValueError("No completed changed model")
        rows = []
        for r in m["metrics"]:
            f = Path(r["path"])
            f.relative_to(ROOT)
            if f.is_symlink() or hashlib.sha256(f.read_bytes()).hexdigest() != r["sha256"]:
                raise ValueError("Metrics binding differs")
            x = json.loads(f.read_text())
            if x["parent_id"] != r["parent_id"] or x["split"] != "val" or x["selected_epoch"] != m["best_epoch"]:
                raise ValueError("Parent/split/selection differs")
            groups = x["C4_exact_support_metrics"]
            for g, denominator in support[r["parent_id"]].items():
                if groups[g]["denominator_query_bp"] != denominator:
                    raise ValueError("Common support differs")
            compact = {g: {key:v[key] for key in ("accuracy", "nll", "brier", "macro_f1", "infinite_nll_query_bp", "denominator_query_bp") if key in v} for g,v in groups.items()}
            rows.append({"parent_id":r["parent_id"], "metrics_source":str(f.relative_to(ROOT)), "metrics_source_sha256":r["sha256"], "groups":compact})
        if len(rows) != 2 or len({r["parent_id"] for r in rows}) != 2:
            raise ValueError("Two distinct development parents required")
        means = {group: {key: statistics.mean(r["groups"][group][key] for r in rows) for key in ("accuracy","nll","brier","macro_f1") if all(isinstance(r["groups"][group].get(key), (int,float)) for r in rows)} for group in rows[0]["groups"]}
        out["conditions"][name] = {"parameters":m["parameters"], "selected_epoch":m["best_epoch"], "epochs_completed":len(m["history"]), "optimizer_steps":m["optimizer_steps"], "selected_maximum_parameter_delta_from_initial":m["selected_maximum_parameter_delta_from_initial"], "rows":rows, "means":means}
    OUTPUT.parent.mkdir(exist_ok=True)
    if OUTPUT.exists():
        raise ValueError("Summary already exists, preserve it")
    OUTPUT.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"summary":str(OUTPUT.relative_to(ROOT)), "sha256":hashlib.sha256(OUTPUT.read_bytes()).hexdigest(), "means":{k:v["means"] for k,v in out["conditions"].items()}}, indent=2))

if __name__ == "__main__":
    main()
