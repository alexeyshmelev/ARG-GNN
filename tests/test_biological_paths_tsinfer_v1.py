"""Handmade/source-only producer checks; never tsinfer/date or observed arrays."""
from __future__ import annotations
import builtins
import copy
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tskit

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
spec = importlib.util.spec_from_file_location("biological_tsinfer_owned", ROOT/"scripts/infer_biological_paths_tsinfer_v1.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def config():
    return json.loads((ROOT/"configs/biological_paths_tsinfer_v1.json").read_text())


def vcf(alleles=("A", "G"), gt=("0|1", "1|0"), pos=11, site_id=0):
    alt = ",".join(alleles[1:]) or "."
    return "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ta\tb\n" + f"22\t{pos}\t{site_id}\t{alleles[0]}\t{alt}\t.\tPASS\t.\tGT\t{gt[0]}\t{gt[1]}\n"


def parse(text, pos=11):
    return m.parse_vcf(io.StringIO(text), ["a", "b"],
                       [{"site_id":0, "tree_position_0based":pos-1, "VCF_position_1based":pos}], 100)


@pytest.mark.parametrize("alleles,gt,expected", [
    (("A","G"),("0|1","1|0"),[0,1,1,0]),
    (("T","C","A"),("2|1","0|2"),[2,1,0,2]),
    (("A",),("0|0","0|0"),[0,0,0,0]),
])
def test_all_original_site_kinds_retained(alleles,gt,expected):
    rows=parse(vcf(alleles,gt))
    assert len(rows)==1 and rows[0]["position"]==10
    assert rows[0]["genotypes"]==expected and rows[0]["alleles"]==list(alleles)
    assert rows[0]["ancestral_allele"]==0


@pytest.mark.parametrize("gt", [("0/1","0|0"),("0|.","0|0"),("2|0","0|0"),("0","0|0"),("-1|0","0|0")])
def test_bad_phase_or_missing_allele_aborts(gt):
    with pytest.raises(ValueError):
        parse(vcf(gt=gt))


@pytest.mark.parametrize("text", [
    vcf().replace("22\t11\t0","23\t11\t0"),
    vcf().replace("\ta\tb\n","\tb\ta\n"),
    vcf().replace("22\t11\t0","22\t12\t0"),
    vcf().replace("22\t11\t0","22\t11\t1"),
    vcf().replace("\tGT\t","\tDS\t"),
    vcf(("A","A")),
    vcf(("N","G")),
    vcf()+vcf().splitlines()[1]+"\n",
    vcf().splitlines()[0]+"\n",
])
def test_original_site_header_and_coordinate_contract(text):
    with pytest.raises(ValueError):
        parse(text)


def roster():
    rows=[]
    names=[f"d{i}" for i in range(140)]
    ages=[]
    for i in range(280):
        role="pure_source_reference" if i<96 else "mixed_reference_only" if i<160 else "query"
        age=0 if i<4 else 10+i//2
        ages.append(age)
        rows.append({"sample_index":i,"sample_id":names[i//2],"haplotype":i%2,
                     "age_generations":age,"observed_cohort":"handmade",
                     "role":role,"observed_reference_class":(i//2)%4 if role=="pure_source_reference" else None})
    return rows,names,ages


def test_full_handmade_panel_preserves_dates_and_roles():
    m.validate_roster(*roster())


@pytest.mark.parametrize("field,value,index", [
    ("sample_index",1,0),("sample_id","other",0),("haplotype",1,0),
    ("age_generations",0,10),("observed_reference_class",0,160),
    ("observed_reference_class",None,0),("observed_cohort","other",1),
    ("role","query",96)
])
def test_roster_drift_or_query_labels_rejected(field,value,index):
    rows,names,ages=roster()
    rows[index][field]=value
    with pytest.raises(ValueError):
        m.validate_roster(rows,names,ages)


def handmade_tree():
    t=tskit.TableCollection(100)
    t.nodes.metadata_schema=tskit.MetadataSchema.permissive_json()
    t.individuals.metadata_schema=tskit.MetadataSchema.permissive_json()
    t.populations.metadata_schema=tskit.MetadataSchema.permissive_json()
    pop=t.populations.add_row(metadata={"handmade":"strip"})
    ids=[t.individuals.add_row(metadata={"handmade":"strip"}) for _ in range(2)]
    for i,age in enumerate([0,0,10,10]):
        t.nodes.add_row(flags=tskit.NODE_IS_SAMPLE | 1024,time=age,individual=ids[i//2],population=pop,metadata={"handmade":"strip"})
    a=t.nodes.add_row(time=50,metadata={"handmade":"strip"})
    b=t.nodes.add_row(time=60,metadata={"handmade":"strip"})
    root=t.nodes.add_row(time=100,metadata={"handmade":"strip"})
    t.nodes.add_row(time=200,metadata={"handmade":"isolated"})
    for left,right,child_parent in [
        (0,50,[(0,a),(1,a),(a,root),(2,root),(3,root)]),
        (50,100,[(2,b),(3,b),(b,root),(0,root),(1,root)])
    ]:
        for child,parent in child_parent:
            t.edges.add_row(left,right,parent,child)
    for pos,node,time in [(10,0,1),(70,2,11)]:
        site=t.sites.add_row(pos,"A")
        t.mutations.add_row(site,node,"G",time=time)
    t.metadata_schema=tskit.MetadataSchema.permissive_json()
    t.metadata={"handmade":"strip"}
    t.provenances.add_row(record='{"handmade":true}')
    t.sort()
    return t.tree_sequence()


def sites():
    return [{"position":10,"alleles":["A","G"],"genotypes":[1,0,0,0]},
            {"position":70,"alleles":["A","G"],"genotypes":[0,0,1,0]}]


def test_exact_native_sample_genotype_and_strict_time_checks_handmade_only():
    ts=handmade_tree()
    assert m.check_samples(ts,[0,0,10,10])==[0,1,2,3]
    m.check_genotypes(ts,sites())
    m.strict_times(ts,[0,0,10,10])


def test_ancestral_or_phase_change_rejected_even_same_site_count():
    s=sites()
    s[0]["genotypes"]=[0,1,0,0]
    with pytest.raises(ValueError,match="phased"):
        m.check_genotypes(handmade_tree(),s)
    s=sites()
    s[0]["alleles"]=["T","G"]
    with pytest.raises(ValueError,match="ancestral"):
        m.check_genotypes(handmade_tree(),s)


def test_sanitation_clips_exact_support_and_keeps_global_ids_and_times():
    original=handmade_tree()
    clean=m.sanitize_clip(original,[0,0,10,10],10,90)
    assert clean.num_nodes==original.num_nodes
    assert clean.samples().tolist()==original.samples().tolist()
    assert np.array_equal(clean.tables.nodes.time,original.tables.nodes.time)
    assert clean.sequence_length==80
    assert clean.num_populations==clean.num_individuals==clean.num_provenances==0
    assert not clean.metadata
    assert np.all(clean.tables.nodes.population==-1) and np.all(clean.tables.nodes.individual==-1)
    for table in [clean.tables.nodes,clean.tables.edges,clean.tables.sites,clean.tables.mutations]:
        assert len(table.metadata)==0
    m.check_genotypes(clean,[{**s,"position":s["position"]-10} for s in sites()])
    assert m.require_identity(clean,1)["internal_reuse_occurrences"]==1


def test_connected_identity_ignores_isolated_nodes():
    result=m.connected_identity(handmade_tree())
    assert result["trees"]==2
    assert result["connected_internal_occurrences"]==4
    assert result["unique_connected_internal_ids"]==3
    assert result["internal_reuse_occurrences"]==1
    assert result["adjacent_shared_internal_min"]==1


def test_no_identity_repair_when_requirement_fails():
    with pytest.raises(ValueError,match="no ID repair"):
        m.require_identity(handmade_tree(),2)


def test_wrong_ancient_date_rejected():
    with pytest.raises(ValueError,match="age"):
        m.check_samples(handmade_tree(),[0,0,0,0])


@pytest.mark.parametrize("left,right",[(117,1998898),(19,1999991),(273,1999681),(116,1999724)])
def test_fixed100bp_joins_exact_half_open_support(left,right):
    got=m.grid_indexes(left,right)
    centers=np.arange(50,2000000,100)
    assert got==np.flatnonzero((centers>=left)&(centers<right)).tolist()


def test_bad_support_rejected():
    for left,right in [(0,0),(-1,10),(10,9),(0,float("inf"))]:
        with pytest.raises(ValueError):
            m.grid_indexes(left,right)


def test_source_only_never_imports_inference_or_opens_observed_arrays(monkeypatch):
    original=builtins.__import__
    def blocked(name,*args,**kwargs):
        if name in {"tsinfer","tsdate"}:
            raise AssertionError("inference import during source-only")
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",blocked)
    original_open=Path.open
    def guard(path,*args,**kwargs):
        if path.suffix in {".vcf",".npz",".trees"}:
            raise AssertionError("observed array opened during source-only")
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,"open",guard)
    result=m.source_only(config(),ROOT/"configs/biological_paths_tsinfer_v1.json")
    assert not result["runtime_imported"] and not result["observed_arrays_opened"]


@pytest.mark.parametrize("field,value",[("force_sample_times",False),("recombination_rate",1e-8),
                                      ("dating",{}),("preprocess",{}),
                                      ("polarization","REF_assumed_ancestral")])
def test_nondefault_scientific_drift_requires_new_condition(field,value):
    c=config()
    c[field]=value
    with pytest.raises(ValueError):
        m.validate_config(c)


def test_actual_mode_without_root_hashes_rejected_before_runtime(monkeypatch):
    monkeypatch.setattr(m,"runtime_inventory",lambda *x: (_ for _ in ()).throw(AssertionError("runtime visited")))
    args=SimpleNamespace(expected_config_sha256=None,expected_source_sha256=None,
                         admission=None,expected_admission_sha256=None)
    with pytest.raises(ValueError,match="root"):
        m.admission(config(),args,ROOT/"configs/biological_paths_tsinfer_v1.json")


def test_storage_guard_handmade_count(monkeypatch):
    c=config()
    monkeypatch.setattr(m,"project_bytes",lambda:450000000000)
    with pytest.raises(ValueError,match="storage"):
        m.storage_check(c)
    monkeypatch.setattr(m,"project_bytes",lambda:30)
    assert m.storage_check(c,10)==30


def test_polarization_requires_pinned_production_writer_header():
    coordinates=[{"site_id":0,"tree_position_0based":10,"VCF_position_1based":11}]
    got=m.parse_vcf(io.StringIO("##source=tskit 1.0.3\n"+vcf()),["a","b"],coordinates,100,expected_writer_source="tskit 1.0.3")
    assert got[0]["ancestral_allele"]==0
    for head in ["","##source=tskit 0.5.0\n","##source=ordinary_empirical_VCF\n"]:
        with pytest.raises(ValueError,match="writer"):
            m.parse_vcf(io.StringIO(head+vcf()),["a","b"],coordinates,100,expected_writer_source="tskit 1.0.3")

def test_equal_parent_child_times_rejected_without_adjustment():
    fake=SimpleNamespace(
        tables=SimpleNamespace(nodes=SimpleNamespace(time=np.array([0.,0.]))),
        samples=lambda:np.array([0],dtype=np.int32),
        edges=lambda:[SimpleNamespace(parent=1,child=0)],mutations=lambda:[])
    with pytest.raises(ValueError,match="strictly older"):
        m.strict_times(fake,[0],diploid=False)


def test_unknown_native_mutation_time_rejected():
    t=handmade_tree().dump_tables()
    t.mutations.time=np.array([tskit.UNKNOWN_TIME,11.])
    with pytest.raises(ValueError,match="mutation time"):
        m.strict_times(t.tree_sequence(),[0,0,10,10])


def native_flanked_handmade_tree(*, edge_span=(10, 61), internal_gap=None):
    """Manual native-style empty tails, preserving fixed qualification genotypes."""
    case=m.handmade_assets()
    t=tskit.TableCollection(2000000)
    individuals=[t.individuals.add_row() for _ in range(4)]
    for i,age in enumerate(case["ages"]):
        t.nodes.add_row(flags=tskit.NODE_IS_SAMPLE,time=age,individual=individuals[i//2])
    a=t.nodes.add_row(time=50)
    b=t.nodes.add_row(time=60)
    root=t.nodes.add_row(time=100)
    for original_left,original_right,pair,ancestor in [(10,35,[0,1],a),(35,61,[2,3],b)]:
        left=max(edge_span[0],original_left)
        right=min(edge_span[1],original_right)
        segments=[(left,right)]
        if internal_gap is not None:
            gl,gr=internal_gap
            segments=[(left,min(right,gl)),(max(left,gr),right)]
        for lo,hi in segments:
            if hi<=lo:
                continue
            for node in range(8):
                t.edges.add_row(lo,hi,ancestor if node in pair else root,node)
            t.edges.add_row(lo,hi,root,ancestor)
    for source in case["sites"]:
        site=t.sites.add_row(source["position"],"A")
        for node,g in enumerate(source["genotypes"]):
            if g:
                t.mutations.add_row(site,node,"G",time=case["ages"][node]+1)
    t.sort()
    return t.tree_sequence(),case


def test_identity_ignores_native_empty_tails_only_outside_exact_support():
    ts,case=native_flanked_handmade_tree()
    m.check_samples(ts,case["ages"])
    m.check_genotypes(ts,case["sites"])
    m.strict_times(ts,case["ages"])
    with pytest.raises(ValueError,match="inside identity support"):
        m.connected_identity(ts)
    before=ts.dump_tables()
    metrics=m.require_identity(ts,1,support=[10,61])
    assert metrics["trees"]==2 and metrics["native_trees_total"]==4
    assert metrics["evaluated_support_bp0"]==[10,61]
    assert metrics["internal_reuse_occurrences"]==1
    assert not metrics["uncovered_interval_inside_support_admitted"]
    assert ts.tables.equals(before)


def test_cpu_qualification_summary_accepts_native_style_empty_flanks():
    ts,case=native_flanked_handmade_tree()
    assert case["identity_support_original_bp0"]==[10,61]
    result=m.qualification_summary(config(),case,ts)
    assert result["status"]=="passed" and result["handmade_only"]
    assert result["identity"]["trees"]==2
    assert result["identity_support_original_bp0"]==[10,61]
    assert ts.sequence_length==2000000
    assert len(ts.tables.edges)==18


@pytest.mark.parametrize("span",[(11,61),(10,60)])
def test_native_coverage_must_contain_entire_qualification_support(span):
    ts,case=native_flanked_handmade_tree(edge_span=span)
    with pytest.raises(ValueError,match="inside identity support"):
        m.qualification_summary(config(),case,ts)


def test_internal_gap_is_rejected_even_between_scoring_centers():
    ts,case=native_flanked_handmade_tree(internal_gap=(35,35.25))
    # The only original 100 bp center in [10,61) is 50, outside this gap.
    with pytest.raises(ValueError,match="inside identity support"):
        m.require_identity(ts,1,support=[10,61])
    with pytest.raises(ValueError,match="inside identity support"):
        m.qualification_summary(config(),case,ts)


def test_observed_support_metrics_equal_final_clip_metrics_without_tail_extension():
    ts,case=native_flanked_handmade_tree()
    native=m.require_identity(ts,1,support=[19,55])
    clean=m.sanitize_clip(ts,case["ages"],19,55)
    local=m.require_identity(clean,1)
    assert native["trees"]==local["trees"]==2
    for key in ["connected_internal_occurrences","unique_connected_internal_ids",
                "internal_reuse_occurrences","adjacent_shared_internal_min","adjacent_shared_internal_max"]:
        assert native[key]==local[key]
    assert clean.num_nodes==ts.num_nodes
    assert np.array_equal(clean.tables.nodes.time,ts.tables.nodes.time)
    assert native["evaluated_support_bp0"]==[19,55]
    assert local["evaluated_support_bp0"]==[0,36]


def test_blocked_revision_admission_cannot_be_reused(monkeypatch):
    monkeypatch.setattr(m,"runtime_inventory",lambda *x: (_ for _ in ()).throw(AssertionError("runtime visited")))
    args=SimpleNamespace(expected_config_sha256=m.sha(ROOT/"configs/biological_paths_tsinfer_v1.json"),
                         expected_source_sha256="74f12e8576cb5a3d64fdca1f9363023ff67c91ccd984067d8a2327668e69a2ef",
                         admission="inactive",expected_admission_sha256="0"*64)
    with pytest.raises(ValueError,match="implementation/config bytes differ"):
        m.admission(config(),args,ROOT/"configs/biological_paths_tsinfer_v1.json")
