"""Inert metadata/schedule validator for a proposed causal TGB annual-trade diagnostic.

No dataset/network reader, model, dependency install, writer, guard or job entrypoint.
Caller-supplied records are metadata/synthetic fixture objects. Field checks do not
prove upstream feature provenance; future acquired bytes and builder need review.
"""
from pathlib import Path
import argparse,json,math,re

SCHEMA="TGB_trade_causal_partition_observation_v1"
COUNTRIES=255
THRESHOLD=.25

def require(ok,message):
    if not ok:raise ValueError(message)

def split_masks_v1(years):
    """Match source np.quantile default linear interpolation and tied-year masks."""
    require(isinstance(years,list) and years and all(type(y) is int for y in years),"Integer years")
    ordered=sorted(years)
    def quantile(p):
        at=(len(ordered)-1)*p;lo=int(math.floor(at));hi=int(math.ceil(at))
        return ordered[lo]+(ordered[hi]-ordered[lo])*(at-lo)
    a,b=quantile(.7),quantile(.85)
    return dict(q70=a,q85=b,train=[y<=a for y in years],
        val=[a<y<=b for y in years],test=[y>b for y in years])

def causal_year_groups_v1(years,forecast_year):
    """Return input row indexes grouped by complete past year; tied target rows excluded."""
    require(type(forecast_year) is int and isinstance(years,list)
        and all(type(y) is int for y in years),"Calendar years")
    grouped={}
    for i,year in enumerate(years):
        if year<forecast_year:grouped.setdefault(year,[]).append(i)
    return [(year,tuple(grouped[year])) for year in sorted(grouped)]

def validate_partition_observation_v1(record):
    fields={"schema","forecast_year","observation_year","country_catalogue_sha256","leaf_keys",
        "observed_edge_years","parent","merge_height","cut_membership","cut_threshold",
        "duration_years","missing_flags","features_observation_max_year","labels_used"}
    require(isinstance(record,dict) and set(record)==fields and record["schema"]==SCHEMA,
        "Exact observation-only interface; no target/label/ARG fields")
    year=record["forecast_year"];observed=record["observation_year"]
    require(type(year) is int and type(observed) is int and observed==year-1,
        "Forecast next calendar year")
    require(record["features_observation_max_year"]==observed and record["labels_used"] is False,
        "Past observations only; builder provenance still externally authenticated")
    years=record["observed_edge_years"]
    require(isinstance(years,list) and all(type(y) is int and y<=observed for y in years)
        and (not years or max(years)==observed),"No target-year/future edge")
    require(isinstance(record["country_catalogue_sha256"],str)
        and re.fullmatch("[0-9a-f]{64}",record["country_catalogue_sha256"]) is not None,
        "Fixed public country catalogue hash")
    require(record["leaf_keys"]==list(range(COUNTRIES)),"All255 public country schema IDs")
    missing=record["missing_flags"]
    require(isinstance(missing,list) and len(missing)==COUNTRIES
        and all(type(v) is bool for v in missing),"Explicit prefix-derived missing flags")
    require(type(record["cut_threshold"]) is float and record["cut_threshold"]==THRESHOLD
        and type(record["duration_years"]) is int and record["duration_years"]==1,
        "Proposed TV cut and calendar width; no ARG units")
    parent=record["parent"];height=record["merge_height"];group=record["cut_membership"]
    require(isinstance(parent,list) and COUNTRIES+1<=len(parent)<=2*COUNTRIES
        and isinstance(height,list) and len(height)==len(parent),
        "Complete bounded country dendrogram")
    n=len(parent);children=[[] for unused in range(n)];roots=[]
    for node,p in enumerate(parent):
        require(type(p) is int and -1<=p<n and p!=node,"Native parent")
        require(type(height[node]) in (int,float) and math.isfinite(height[node])
            and 0<=height[node]<=1,"Observed merge distance")
        if p<0:roots.append(node)
        else:
            require(height[p]>=height[node],"Nondecreasing merge distance")
            children[p].append(node)
    require(len(roots)==1 and height[roots[0]]==1,"One complete root at maximal distance")
    require({i for i,c in enumerate(children) if not c}==set(range(COUNTRIES))
        and all(height[i]==0 for i in range(COUNTRIES)),"All and only255 observed country leaves")
    representative=[-1]*n;seen=set();queue=[roots[0]]
    while queue:
        node=queue.pop();require(node not in seen,"Acyclic complete tree");seen.add(node)
        p=parent[node]
        if height[node]<=THRESHOLD:
            representative[node]=node if p<0 or height[p]>THRESHOLD else representative[p]
        queue.extend(children[node])
    require(len(seen)==n,"No hidden disconnected vertices")
    require(isinstance(group,list) and len(group)==COUNTRIES
        and all(type(g) is int and g>=0 for g in group),"Anonymous partition labels")
    forward={};backward={};counts={}
    for leaf,g in enumerate(group):
        rep=representative[leaf]
        require(rep>=0 and forward.get(rep,g)==g and backward.get(g,rep)==rep,
            "Partition is exactly the observation-derived threshold cut")
        forward[rep]=g;backward[g]=rep;counts[g]=counts.get(g,0)+1
    require(all(not missing[i] or counts[group[i]]==1 for i in range(COUNTRIES)),
        "Missing histories cannot create a spurious shared cohort")
    return dict(countries=COUNTRIES,retained_vertices=n,cut_cells=len(counts),
        forecast_year=year,maximum_input_year=observed,labels_or_ARG_semantics_used=False)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    print(json.dumps(dict(status="source_contract_only_payload_alignment_pending",
        dataset_download=False,models_or_jobs=False,record_schema=SCHEMA),sort_keys=True))
    return 0

if __name__=="__main__":raise SystemExit(main())
