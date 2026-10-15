"""Native scalar metric normalization, using already-recorded per-parent JSON only."""
from pathlib import Path
import hashlib,json,math,statistics
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE_REF=dict(path="research/reviews/native_common_cohort_diagnostics_20261015_v1/per_parent_metrics.json",
 sha256="f221906384c10e05803d06d0eaee8c2c40936bbb9c3063f410ad4d53501aebde")
def load_native_overrides():
    raw=(ROOT/SOURCE_REF["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==SOURCE_REF["sha256"]
    source=json.loads(raw);out={}
    for model,cells in source.items():
        assert len(cells)==8 and {r["parent_id"] for r in cells}=={"ap_expanded_val_%03d"%i for i in range(8)}
        cohorts={}
        for cohort in cells[0]["groups"]:
            groups=[r["groups"][cohort] for r in cells]
            infinity=[r for r in groups if r.get("infinite_nll_query_bp",0)>0]
            finite=[r["nll"] for r in groups if type(r["nll"]) in (int,float) and math.isfinite(r["nll"])]
            if infinity:
                assert len(finite)+len(infinity)==8
                nll=dict(mean="∞",sample_SD=None,n=8,defined_parent_count=8,undefined_parent_count=0,
                    finite_parent_count=len(finite),infinite_parent_count=len(infinity),
                    infinite_nll_query_bp=sum(r["infinite_nll_query_bp"] for r in infinity),
                    raw_infinity_not_missing=True)
            else:
                assert len(finite)==8
                nll=dict(mean=statistics.mean(finite),sample_SD=statistics.stdev(finite),
                    n=8,defined_parent_count=8,undefined_parent_count=0,infinite_parent_count=0)
            row=dict(nll=nll)
            for metric in ("precision","recall","f1"):
                values=[r["boundaries"][metric] for r in groups]
                finite=[v for v in values if type(v) in (int,float) and math.isfinite(v)]
                row["boundary_"+metric]=dict(mean=statistics.mean(finite) if finite else None,
                    sample_SD=statistics.stdev(finite) if len(finite)>1 else None,n=8,
                    defined_parent_count=len(finite),undefined_parent_count=8-len(finite),
                    conditional_on_defined_parents=(len(finite)<8))
            cohorts[cohort]=row
        out[model]=dict(source_ref=SOURCE_REF,cohorts=cohorts)
    return out
if __name__=="__main__":print(json.dumps(load_native_overrides(),indent=2,sort_keys=True,allow_nan=False))
