"""H038 full952 observed-only parent preparation. No work at import."""
from __future__ import annotations
from dataclasses import asdict
import inspect
import time
import io
import numpy as np
from . import mutation_routed_tree_observations_v1 as old
from . import mutation_routed_parent_loader_v1 as loader
from . import observed_variant_residual_v1 as variants
from .full_panel_native_baseline_contracts_v1 import validate_observed_roster
from .mutation_routed_whole_tree_gnn_v1 import MutationRoutedWholeTreeInputsV1, validate_inputs_v1

ROOT=loader.ROOT
QUERY_COUNTS={"present_day":182,"Bronze_Age":162,"BAA":14,"Neo":302,"Yam":26}
PROXY_COUNTS=(50,28,102,86)
ENTRY_FIELDS={"schema","parent_id","split","observations_only","truth_or_oracle_read",
              "refs","coordinate_contract","empirical_GRCh37_coordinate_claimed","samples",
              "query_haplotypes","pure_reference_haplotypes","original_native_inference_receipt"}
ENTRY_REFS={"source_roster","vcf","site_coordinates","scoring_coordinates","native_tree",
            "derived_roster","inference_receipt","observation_manifest"}


def validate_registry_v1(registry):
    if len(registry)!=48 or len({r["parent_id"] for r in registry})!=48:
        raise ValueError("Exactly48 unique independent parent records required")
    roles={s:[r["parent_id"] for r in registry if r["split"]==s] for s in ("train","val","outer")}
    if [len(roles[s]) for s in ("train","val","outer")]!=[32,8,8]:
        raise ValueError("Frozen32TRAIN/8VAL/8outer roles required")
    seeds=[r["seeds"][purpose] for r in registry for purpose in ("ancestry","mutations")]
    if any(type(s) is not int or not 0<s<2**32 for s in seeds) or len(set(seeds))!=96:
        raise ValueError("Unique explicit ancestry/mutation seed streams required")
    if any(not isinstance(r["parent_id"],str) or not r["parent_id"]
           or "/" in r["parent_id"] or ".." in r["parent_id"] for r in registry):
        raise ValueError("Safe registered parent IDs required")
    return roles


def _strict_source_clone(function,replacements,overrides=None):
    """Independent function namespace, with exact recognized production edits."""
    text=inspect.getsource(function)
    for before,after in replacements:
        if text.count(before)!=1:
            raise ValueError("Frozen numerical helper changed; count-contract edit must be reviewed")
        text=text.replace(before,after)
    namespace=dict(function.__globals__)
    namespace.update(overrides or {})
    exec(compile(text,function.__code__.co_filename+":full952_contract", "exec"),namespace)
    return namespace[function.__name__]


_full_join=_strict_source_clone(old.validate_observed_sample_join_v1,[
    ('count != 280 or role_counts != {"query": 120, "pure_source_reference": 96, "mixed_reference_only": 64}',
     'count != 952 or role_counts != {"query": 686, "pure_source_reference": 266, "mixed_reference_only": 0}'),
    ('[ref.count(c) for c in range(4)] != [24] * 4',
     '[ref.count(c) for c in range(4)] != [50,28,102,86]'),
    ('All280 observed haplotypes, pureR96, mixed64 and actualQ120 required',
     'Full952 observed haplotypes, pureR266 and actualQ686 required'),
    ('Observed pureR96 must have24 per class','Author pure-source counts must remain50/28/102/86')])
def _unused_sham_route_v1(parent,support,*args,**kwargs):
    return np.array(support,copy=True),dict(
        schema="H038_unused_branch_support_route_v1",placement_sham_computed=False,
        identity_placeholder=True,ablation="separately_zero_first3_true_fields_only")


_full_prepare=_strict_source_clone(old.prepare_mutation_routed_tree_observations_v1,[
    ('h != 280 or len(qi) != 120 or np.count_nonzero(ref >= 0) != 96',
     'h != 952 or len(qi) != 686 or np.count_nonzero(ref >= 0) != 266'),
    ('[np.count_nonzero(ref == c) for c in range(4)] != [24] * 4',
     '[np.count_nonzero(ref == c) for c in range(4)] != [50,28,102,86]'),
    ('Production requires280 observed haps, actualQ120 and only pureR96 labels',
     'Production requires952 observed haps, actualQ686 and only pureR266 labels'),
    ('Each native hold must retain all280 persistent observed sampled haps',
     'Each native hold must retain all952 persistent observed sampled haps')],
    {"shuffle_branch_support_v1":_unused_sham_route_v1})


def validate_full_roster_v1(rows):
    copied=[dict(r,inferred_tree_node_id=r["observed_node_id"]) for r in rows]
    validate_observed_roster(copied)
    return dict(samples=952,diploids=476,sample_names=[r["sample_id"] for r in rows[::2]],
        query_indexes=[r["sample_index"] for r in rows if r["role"]=="query"],
        reference_indexes=[r["sample_index"] for r in rows if r["role"]=="pure_source_reference"])


_parse_vcf=_strict_source_clone(variants.parse_observed_vcf_v1,[
    ('np.empty((len(rows), 280), dtype=np.int8)','np.empty((len(rows), 952), dtype=np.int8)'),
    ('len(fields) != 149','len(fields) != 485'),
    ('coordinates.get("empirical_GRCh37_coordinate_claimed")',
     'coordinates.get("empirical_GRCh37_coordinate_claimed",False)')],
    {"validate_observed_roster_v1":validate_full_roster_v1})


def join_full_rosters_v1(source,derived,mapping,nodes,times):
    validate_full_roster_v1(source)
    validate_observed_roster(derived)
    canonical_source,canonical_derived=[],[]
    for raw,row in zip(source,derived):
        a,b=dict(raw),dict(row)
        if a["observed_cohort"]=="Ana":
            a["observed_cohort"]=b["observed_cohort"]="ANA"
        canonical_source.append(a);canonical_derived.append(b)
    return _full_join(canonical_source,canonical_derived,mapping,nodes,times,fixture=False)


def validate_full_bank_v1(bank):
    validate_inputs_v1(bank,expected_haplotypes=952)
    if (len(bank.query_column)!=686
            or tuple(int((bank.reference_class==c).sum()) for c in range(4))!=PROXY_COUNTS
            or int((bank.reference_class<0).sum())!=686):
        raise ValueError("Exact full952 query/proxy count contract required")


def load_full_parent_v1(parent_row,entry_ref,reader,*,scratch_directory):
    """Read only observed assets and native inferred tree; no targets accepted."""
    boundary=loader._reader(reader)
    entry=boundary.decode_json(entry_ref)
    if (set(entry)!=ENTRY_FIELDS or entry["schema"]!="mutation_routed_full_panel_observed_entry_v1"
            or entry["parent_id"]!=parent_row["parent_id"] or entry["split"]!=parent_row["split"]
            or entry["observations_only"] is not True or entry["truth_or_oracle_read"] is not False
            or entry["empirical_GRCh37_coordinate_claimed"] is not False
            or entry["samples"]!=952 or entry["query_haplotypes"]!=686
            or entry["pure_reference_haplotypes"]!=266 or set(entry["refs"])!=ENTRY_REFS):
        raise ValueError("Strict full952 observed entry/registry binding required")
    refs={k:boundary.record(v) for k,v in entry["refs"].items()}
    source=boundary.decode_json(refs["source_roster"])
    derived=boundary.decode_json(refs["derived_roster"])
    inference=boundary.decode_json(refs["inference_receipt"])
    manifest=boundary.decode_json(refs["observation_manifest"])
    if (manifest["parent_id"]!=parent_row["parent_id"] or manifest["samples"]!=952
            or manifest["diploids"]!=476 or manifest["query_haplotypes"]!=686
            or manifest["pure_reference_haplotypes"]!=266
            or manifest["census_or_ancestry_annotations_present"] is not False
            or manifest["cleaned_oracle_exported"] is not False
            or inference["parent_id"]!=parent_row["parent_id"]
            or inference["truth_or_oracle_read"] is not False
            or {k:inference["original_observed_sample_manifest"][k] for k in ("path","sha256")}!=refs["source_roster"]
            or {k:inference["inferred_tree"][k] for k in ("path","sha256")}!=refs["native_tree"]):
        raise ValueError("Full observed inference/source roster lineage differs")
    if (inference.get("schema")!="mutation_routed_full_panel_inference_receipt_v1"
            or inference.get("native_sample_dates_unchanged") is not True
            or inference["coordinate_contract"]!=entry["coordinate_contract"]
            or {k:inference["observed_sample_manifest"][k] for k in ("path","sha256")}!=refs["derived_roster"]
            or inference["native_inference_receipt"]!=entry["original_native_inference_receipt"]):
        raise ValueError("Explicit native dates/coordinates/inference receipt join required")
    boundary.read(entry["original_native_inference_receipt"])
    contract=entry["coordinate_contract"]
    coverage=np.asarray(contract["coverage_original_bp0"],dtype=np.float64)
    if (contract["original_parent_interval_bp0"]!=[0,2000000]
            or contract["normalization_length_bp"]!=2000000
            or contract["physical_coordinate_transform"]!="physical_original = local_tree_position + coverage_left"
            or coverage.shape!=(2,) or not np.isfinite(coverage).all()
            or not 0<=coverage[0]<coverage[1]<=2000000
            or contract["local_to_original_offset_bp"]!=coverage[0]):
        raise ValueError("Original native inferred coverage/offset required")
    ts=loader._load_verified_ts_v1(boundary.read(refs["native_tree"]),boundary.root,
                                  scratch_directory=scratch_directory)
    nodes=np.asarray(ts.samples(),dtype=np.int64)
    join=join_full_rosters_v1(source,derived,inference["sample_index_to_inferred_tree_node_id"],
        nodes,np.asarray(ts.nodes_time,dtype=np.float64)[nodes])
    trees,intervals,topology=loader.extract_native_tree_holds_v1(ts,join,coverage,offset=float(coverage[0]))
    del ts
    alleles=_parse_vcf(boundary.read(refs["vcf"]).decode(),source,boundary.decode_json(refs["site_coordinates"]))
    if len(alleles.positions_bp0)!=manifest["sites"]:
        raise ValueError("Every original VCF site must survive")
    positions,edges,retained=loader._scoring_grid_v1(boundary.read(refs["scoring_coordinates"]),contract)
    query=join.query_id
    cohorts=np.asarray([source[int(i)]["observed_cohort"] for i in query])
    if {c:int((cohorts==c).sum()) for c in QUERY_COUNTS}!=QUERY_COUNTS:
        raise ValueError("Exact Present182/Ancient504 query strata required")
    raw=np.asarray([source[int(i)]["observed_node_id"] for i in query],dtype=np.int64)
    score=dict(parent_id=parent_row["parent_id"],split=parent_row["split"],samples=source,
        query_slots=query,raw_query_nodes=raw,cohorts=cohorts,positions=positions,edges=edges,
        retained_indexes=retained,coverage_original=coverage.tolist(),query_ids=query.copy(),
        positions_original=positions,bin_edges_original=edges,query_sample_index=query,
        original_query_nodes=raw,parent_manifest_ref=entry_ref,observed_parent_entry_ref=entry_ref)
    receipt=dict(schema="mutation_routed_full_panel_loaded_parent_v1",parent_id=parent_row["parent_id"],
        split=parent_row["split"],entry_ref=entry_ref,refs=refs,targets_read=False,
        observations_only=True,input_samples=952,labeled_pure_proxies=266,actual_query_readouts=686,
        topology=topology,alleles=alleles.metadata,original_holds_preserved=True)
    return loader.LoadedMutationRoutedParentV1(parent_row["parent_id"],parent_row["split"],
        trees,intervals,alleles.positions_bp0,alleles.genotypes,alleles.allele_counts,join.sample_id,
        join.reference_class,join.sample_age,query,positions,edges,score,receipt)


def prepare_full_collection_v1(registry,entry_refs,reader,*,scratch_directory,include_outer=False):
    """Prepare development only by default; freeze bins from all32TRAIN."""
    started=time.monotonic()
    roles=validate_registry_v1(registry)
    selected={r["parent_id"]:r for r in registry if include_outer or r["split"]!="outer"}
    if set(entry_refs)!=set(selected):
        raise ValueError("Exactly selected registered observation entry refs required")
    loaded={p:load_full_parent_v1(row,entry_refs[p],reader,scratch_directory=scratch_directory)
            for p,row in selected.items()}
    duration,count=[],[]
    for p in roles["train"]:
        d,c=old.observed_sham_geometry_v1(loaded[p].trees);duration.append(d);count.append(c)
    bins=old.fit_sham_bin_edges_v1(np.concatenate(duration),np.concatenate(count),split="TRAIN",bins=4)
    del duration,count
    items={}
    for p in selected:
        observed=loaded.pop(p)
        bank=_full_prepare(observed.trees,observed.intervals_bp,observed.positions_bp0,
            observed.genotype,observed.allele_count,observed.sample_ids,observed.reference_class,
            observed.sample_age,observed.query_ids,observed.readout_x_bp,sham_bin_edges=bins,fixture=False)
        inputs=MutationRoutedWholeTreeInputsV1.from_arrays(bank.model_arrays())
        validate_full_bank_v1(inputs)
        items[p]=dict(model_inputs=inputs,scoring=observed.scoring,
            observations_receipt=dict(observed.receipt,preprocessing=bank.metadata))
    return dict(parents=items,registry=registry,entry_refs=entry_refs,sham_bin_edges=asdict(bins),
        receipt=dict(schema="mutation_routed_full_panel_prepared_collection_v1",targets_read=False,
            observations_only=True,sham_bins_fit_from_parent_ids=roles["train"],
            original_native_holds_preserved=True,outer_included=include_outer,
            elapsed_observed_preparation_seconds=time.monotonic()-started,placement_sham_computed=False,
            sham_bins_legacy_metadata_only=True))
