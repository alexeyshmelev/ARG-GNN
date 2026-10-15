"""Observation-only country profiles and complete simultaneous single-linkage trees.

No I/O, label reader, model, Torch import, persistent internal IDs or ARG units.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
import numpy as np

OFFICIAL_COUNTRIES=255
OFFICIAL_MAPPING_SHA256="18dfd6a8a6a3a739a2f1653b1a5172e4dbf564c1073063608cc4d8373c972273"
CUT_THRESHOLD=.25
FORECAST_YEARS=tuple(range(1987,2017))
EDGE_YEARS=tuple(range(1986,2017))

def require(ok,message):
    if not ok:raise ValueError(message)

@dataclass(frozen=True)
class CountryCatalogueV1:
    names:tuple[str,...]
    sha256:str
    official:bool=False

    @classmethod
    def from_official_mapping(cls,mapping):
        # Same insertion-order JSON bytes as the root payload audit.
        require(isinstance(mapping,dict) and list(mapping.values())==list(range(OFFICIAL_COUNTRIES))
            and all(isinstance(k,str) and k for k in mapping),"Exact official first-appearance country order")
        raw=json.dumps(mapping,separators=(",",":"),ensure_ascii=False).encode()
        digest=hashlib.sha256(raw).hexdigest()
        require(digest==OFFICIAL_MAPPING_SHA256,"Audited 255-country catalogue hash")
        return cls(tuple(mapping),digest,True)

    @classmethod
    def synthetic_for_fixture(cls,names):
        names=tuple(names);require(len(names)>=2 and len(set(names))==len(names),"Synthetic unique names")
        digest=hashlib.sha256(json.dumps(dict(zip(names,range(len(names)))),separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
        return cls(names,digest,False)

@dataclass(frozen=True)
class ObservedYearV1:
    year:int
    profile:np.ndarray
    present:np.ndarray
    zero_mass:np.ndarray

def observed_year_v1(year,weights,source_present):
    """Normalize positive outgoing rows; absent rows and observed-zero differ."""
    require(type(year) is int,"Integer calendar year")
    w=np.asarray(weights,dtype=np.float64);p=np.asarray(source_present)
    require(w.ndim==2 and w.shape[0]==w.shape[1] and p.shape==(len(w),) and p.dtype==np.bool_,
        "Square country-by-destination weights and explicit boolean source presence")
    require(np.isfinite(w).all() and (w>=0).all() and not np.any(w[~p]),"Finite nonnegative observations; absent sources have no weights")
    mass=w.sum(axis=1,dtype=np.float64);require(np.isfinite(mass).all(),"Finite outgoing row masses");positive=mass>0
    profile=np.zeros(w.shape,dtype=np.float32)
    profile[positive]=(w[positive]/mass[positive,None]).astype(np.float32)
    return ObservedYearV1(year,profile,p.copy(),p & ~positive)

def observed_year_from_edges_v1(year,rows,catalogue):
    """In-memory complete-year (source ID,destination ID,weight) records only."""
    n=len(catalogue.names);w=np.zeros((n,n),dtype=np.float64);present=np.zeros(n,dtype=bool);seen=set()
    for source,destination,weight in rows:
        require(type(source) is int and type(destination) is int and 0<=source<n and 0<=destination<n,
            "Fixed catalogue row/column IDs")
        require((source,destination) not in seen,"No duplicate complete-year directed pair")
        seen.add((source,destination));w[source,destination]=weight;present[source]=True
    return observed_year_v1(year,w,present)

@dataclass(frozen=True)
class ProfileSnapshotV1:
    forecast_year:int
    observation_year:int
    profile:np.ndarray
    missing:np.ndarray
    zero_mass:np.ndarray
    last_observed_year:np.ndarray
    staleness_years:np.ndarray
    released_years:tuple[int,...]
    catalogue_sha256:str

def snapshot_before_v1(catalogue,forecast_year,available_years,year_loader):
    """The loader is called only for whole years strictly before the forecast."""
    require(type(forecast_year) is int and callable(year_loader),"Calendar forecast and caller-supplied observation loader")
    years=tuple(available_years)
    require(len(set(years))==len(years) and all(type(y) is int for y in years),"Unique available calendar years")
    n=len(catalogue.names);profile=np.zeros((n,n),dtype=np.float32)
    missing=np.ones(n,dtype=bool);zero=np.zeros(n,dtype=bool);last=np.full(n,-1,dtype=np.int64)
    released=tuple(sorted(y for y in years if y<forecast_year))
    for year in released:
        observation=year_loader(year)
        require(isinstance(observation,ObservedYearV1) and observation.year==year,"Loader returned the requested past year")
        # Revalidate the exact arrays without allowing absent-source hidden mass.
        require(observation.profile.shape==(n,n) and observation.present.shape==(n,)
            and observation.zero_mass.shape==(n,) and observation.present.dtype==np.bool_
            and observation.zero_mass.dtype==np.bool_,"Complete fixed-country observation dimensions")
        rows=observed_year_v1(year,observation.profile,observation.present)
        require(np.array_equal(rows.zero_mass,observation.zero_mass),"Observed-zero flag matches measured row mass")
        active=rows.present
        profile[active]=rows.profile[active];missing[active]=False
        zero[active]=rows.zero_mass[active];last[active]=year
    stale=np.zeros(n,dtype=np.int64);stale[~missing]=forecast_year-1-last[~missing]
    return ProfileSnapshotV1(forecast_year,forecast_year-1,profile,missing,zero,last,stale,released,catalogue.sha256)

@dataclass(frozen=True)
class CountryTreeV1:
    parent:np.ndarray
    merge_height:np.ndarray
    leaf_count:np.ndarray
    child_count:np.ndarray
    upward_level:np.ndarray
    downward_level:np.ndarray
    cut_node_index:np.ndarray
    root:int
    distance:np.ndarray

def total_variation_v1(profile,missing):
    p=np.asarray(profile,dtype=np.float64);missing=np.asarray(missing)
    require(p.ndim==2 and p.shape[0]==p.shape[1] and len(p)>=2
        and missing.dtype==np.bool_ and missing.shape==(len(p),),"Complete outgoing profiles and missing flags")
    require(np.isfinite(p).all() and (p>=0).all() and not np.any(p[missing]),"Finite profiles; missing histories are zero")
    sums=p.sum(1);require(np.all((sums==0)|np.isclose(sums,1,rtol=0,atol=2e-6)),"Outgoing profiles normalized or zero mass")
    n=len(p);distance=np.zeros((n,n),dtype=np.float64)
    # Avoid an [N,N,N] temporary; destination summation order is fixed catalogue order.
    for i in range(n-1):
        d=np.minimum(1.,.5*np.abs(p[i+1:]-p[i]).sum(1,dtype=np.float64))
        d[missing[i+1:]|missing[i]]=1.
        distance[i,i+1:]=d;distance[i+1:,i]=d
    return distance

def complete_single_linkage_v1(profile,missing,cut_threshold=CUT_THRESHOLD):
    """Kruskal component merges at one exact FP64 height are simultaneous.

    Each positive-size component merge creates one node with >=2 children.
    Equal-height internal chains never exist. Leaves retain their fixed catalogue
    indexes; newly appended internal indexes are anonymous within this tree.
    """
    require(float(cut_threshold)==CUT_THRESHOLD,"Fixed 0.25 total-variation cut")
    distance=total_variation_v1(profile,missing);n=len(distance)
    a,b=np.triu_indices(n,1);heights=distance[a,b];order=np.lexsort((b,a,heights))
    parent=[-1]*n;height=[0.]*n;counts=[1]*n;degrees=[0]*n;up=[0]*n
    uf=list(range(n));active=list(range(n));minimum=list(range(n));components=n
    def find(x):
        while uf[x]!=x:uf[x]=uf[uf[x]];x=uf[x]
        return x
    start=0
    while start<len(order) and components>1:
        end=start+1;h=float(heights[order[start]])
        while end<len(order) and heights[order[end]]==h:end+=1
        # Local connectivity among PRE-height components, before global union.
        local={}
        def local_find(x):
            local.setdefault(x,x)
            while local[x]!=x:local[x]=local[local[x]];x=local[x]
            return x
        for edge in order[start:end]:
            x,y=find(int(a[edge])),find(int(b[edge]))
            if x!=y:
                rx,ry=local_find(x),local_find(y)
                if rx!=ry:local[max(rx,ry)]=min(rx,ry)
        groups={}
        for x in local:groups.setdefault(local_find(x),[]).append(x)
        for members in sorted(groups.values(),key=lambda xs:min(minimum[x] for x in xs)):
            if len(members)<2:continue
            members.sort(key=lambda x:minimum[x]);children=[active[x] for x in members]
            node=len(parent);parent.append(-1);height.append(h)
            counts.append(sum(counts[x] for x in children));degrees.append(len(children))
            up.append(1+max(up[x] for x in children))
            for child in children:parent[child]=node
            representative=min(members)
            for x in members:uf[x]=representative
            active[representative]=node;minimum[representative]=min(minimum[x] for x in members)
            components-=len(members)-1
        start=end
    require(components==1,"One complete single-linkage tree")
    root=active[find(0)];down=np.zeros(len(parent),dtype=np.int64)
    # Every parent was appended after its children.
    for child in range(len(parent)-1,-1,-1):
        if parent[child]>=0:down[child]=down[parent[child]]+1
    cut=np.arange(n,dtype=np.int64)
    for leaf in range(n):
        node=leaf
        while parent[node]>=0 and height[parent[node]]<=cut_threshold:node=parent[node]
        cut[leaf]=node
    require(all(not missing[i] or np.count_nonzero(cut==cut[i])==1 for i in range(n)),
        "Missing histories are singleton below the fixed cut")
    return CountryTreeV1(np.asarray(parent,dtype=np.int64),np.asarray(height,dtype=np.float64),
        np.asarray(counts,dtype=np.int64),np.asarray(degrees,dtype=np.int64),
        np.asarray(up,dtype=np.int64),down,cut,root,distance)

def snapshot_digest_v1(snapshot,tree):
    """Target-free source witness; includes full profiles, flags and hierarchy."""
    h=hashlib.sha256()
    h.update(json.dumps(dict(forecast_year=snapshot.forecast_year,observation_year=snapshot.observation_year,
        released_years=snapshot.released_years,catalogue_sha256=snapshot.catalogue_sha256),
        sort_keys=True,separators=(",",":")).encode())
    for name,value in (("profile",snapshot.profile),("missing",snapshot.missing),("zero_mass",snapshot.zero_mass),
        ("last_observed_year",snapshot.last_observed_year),("staleness_years",snapshot.staleness_years),
        ("parent",tree.parent),("merge_height",tree.merge_height),("cut_node_index",tree.cut_node_index)):
        array=np.ascontiguousarray(value);h.update(name.encode());h.update(str(array.dtype).encode())
        h.update(json.dumps(list(array.shape)).encode());h.update(array.tobytes())
    return h.hexdigest()
