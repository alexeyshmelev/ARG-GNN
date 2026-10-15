"""Read-only normalization of already-scored H040/H044 selected metric JSON.
No inference, models, truth arrays, admission or resource guards.
"""
from pathlib import Path
import hashlib,json,math,statistics
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCES={
 "H040":dict(path="experiments/results/H040_GPU45_frozen_evaluation_seed101_v3/completed.json",sha256="db98cff6c9412b4fddef850f0e38c6c09bec769f82811d2bae6836804466cf72"),
 "H044":dict(path="experiments/results/H044_marker_cached_evaluate_seed101_v2/completed.json",sha256="bc46c8edfaae2ef0894cd8ae057a69ce1d6844bda8fb9262b849709e4b4524a4")}
METRICS=("accuracy","balanced_accuracy","macro_f1","nll","brier","calibration_ece",
 "dosage_mse_per_diploid_bp_class","dosage_mae_per_diploid_bp_class")
def metric_stat(values,*,infinite_evidence=False):
    finite=[v for v in values if type(v) in (int,float) and math.isfinite(v)]
    n=len(finite);total=len(values)
    result=dict(mean=statistics.mean(finite) if n else None,
        sample_SD=statistics.stdev(finite) if n>1 else None,
        defined_parent_count=n,undefined_parent_count=total-n,n=total)
    if infinite_evidence or any(type(v) in (int,float) and v==math.inf for v in values):
        result.update(mean="∞",sample_SD=None,meaning="raw_NLL_positive_infinity;SD_undefined")
    return result
def normalize_scored_cells(cells):
    assert len(cells)==8 and {r["parent_id"] for r in cells}=={"ap_expanded_val_%03d"%i for i in range(8)}
    cohorts={}
    for cohort in cells[0]["groups"]:
        groups=[r["groups"][cohort] for r in cells]
        row={k:metric_stat([g[k] for g in groups],
            infinite_evidence=k=="nll" and any(g.get("infinite_nll_query_bp",0)>0 for g in groups)) for k in METRICS}
        for k in ("precision","recall","f1"):
            row["boundary_"+k]=metric_stat([g["boundaries"][k] for g in groups])
        cohorts[cohort]=row
    return cohorts
def load_normalized():
    output={}
    for tag,record in SOURCES.items():
        raw=(ROOT/record["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==record["sha256"]
        source=json.loads(raw)
        assert source["outer_truth_read"] is False and source["all96_RAM_seals_before_any_VAL_truth"] is True
        output[tag]=dict(source_ref=record,arms={arm:dict(cohorts=normalize_scored_cells(d["cells"]))
            for arm,d in source["selected_full_diagnostics"].items()})
    return output
if __name__=="__main__":print(json.dumps(load_normalized(),sort_keys=True,indent=2,allow_nan=False))
