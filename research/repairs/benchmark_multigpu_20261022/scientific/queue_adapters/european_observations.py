"""Complete European TreeLoom banks and dimension-bound tiled execution.

No launch, policy, lease, truth reader or checkpoint selection lives here.
All native trees/nodes/sites/samples enter the bank. Only sampled emissions are
selected; tree sweeps, stable-leaf history and genomic GRU context stay complete.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from arggnn.benchmarking import contracts as c
from arggnn.benchmarking import treeloom
from arggnn.benchmarking import american_preparation as generation
from arggnn.benchmarking import treeloom_prediction as american

PROTOCOLS = {
    "ancestralpaths_paper200mb_native6": "AP_paper_scale_200Mb_native6",
    "ancestralpaths_paper200mb_c4": "AP_paper_scale_200Mb_derived_C4_control",
    "european_c4_expanded": "benchmark_european_c4_expanded",
    "european_c4_prospective1611": "ARGMix_prospective1611",
}
from research.reviews.frozen_benchmark_training_ready_20261017_v1 import european_frozen_treeloom_v1 as implementation

NATIVE_SIX_CLASSES = ("Ana_nonBAA","CHG_nonBAA","WHG","EHG","Ana_viaBAA","CHG_viaBAA")
IDENTITY = american.IDENTITY
_STATE_KEYS = {"manifest", "ts", "rows", "nodes", "genotype", "positions", "allele_count",
               "labels", "ages", "query_columns", "coordinate_offset_bp", "source_refs"}


def _need(ok, message):
    if not ok:
        raise c.ContractError(message)


def requirements():
    manifest = treeloom.verify_implementation()
    return dict(source_refs=[dict(path=p,sha256=h) for p,h in manifest["source_sha256"].items()]
                + [treeloom.MANIFEST_REF, treeloom.EUROPEAN_MODEL_REF, c.file_ref(treeloom.__file__),
                   c.file_ref(__file__), c.file_ref(american.__file__),
                   c.file_ref(generation.__file__),
                   c.file_ref("src/arggnn/benchmarking/european_sampling.py"),
                   c.file_ref("src/arggnn/benchmarking/targets.py")],
                selected_hidden_width=32, two_shared_complete_tree_sweeps=True,
                native_six_class_aliases=dict(zip(
                    ("Ana","CHG","WHG","EHG","Ana_via_BAA","CHG_via_BAA"),NATIVE_SIX_CLASSES)),
                native_six_aliases_metadata_only_no_column_permutation=True,
                donor_classes=4, donor_age_cohorts=2, memory_rank=8, memory_parameters=513,
                full_length_bidirectional_GRU_per_haplotype_tile=True,
                no_detached_history_or_artificial_sequence_boundaries=True,
                analytical_parent_peak_RSS_proven=False, GPU_fit_qualified=False,
                scientific_payloads_opened=False, jobs_started=0)


def _implementation(protocol_id):
    _need(protocol_id in PROTOCOLS, "Explicit supported European endpoint required")
    if protocol_id == "european_c4_expanded":
        # Same metadata-only 2 Mb binding as the maintained public facade.
        condition = dict(haplotypes=952, queries=686, length_bp=2000000,
                         output_classes=("Ana", "CHG", "WHG", "EHG"), parameters=32325)
        existing = implementation.CONDITIONS.get(PROTOCOLS[protocol_id])
        _need(existing is None or existing == condition, "Conflicting expanded data binding")
        implementation.CONDITIONS[PROTOCOLS[protocol_id]] = condition
    _need(PROTOCOLS[protocol_id] in implementation.CONDITIONS,
          "An explicit existing frozen European condition is required")
    requirements()
    _need(c.file_ref(treeloom.EUROPEAN_MODEL_REF["path"]) == treeloom.EUROPEAN_MODEL_REF,
          "Exact unchanged ordinary European model source required")
    return implementation


def build_model(protocol_id):
    implementation = _implementation(protocol_id)
    import torch
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        model = implementation.FrozenEuropeanTreeLoomV1(condition=PROTOCOLS[protocol_id])
    if protocol_id == "ancestralpaths_paper200mb_native6":
        # Canonical native AP strings describe the same six unchanged class IDs.
        model.output_classes = NATIVE_SIX_CLASSES
    _need(all(p.dtype == torch.float32 and p.device.type == "cpu" for p in model.parameters()),
          "Original seed101 FP32 European model required")
    return model


def _bound(ref):
    american._public_ref(ref)
    path = generation._path(ref["path"])
    _need(path.is_file() and generation._reference(path) == ref, "Exact public observed file SHA required")
    return path


def _json_ref(ref, max_bytes):
    path = _bound(ref)
    _need(type(max_bytes) is int and max_bytes > 0 and path.stat().st_size <= max_bytes,
          "Bounded complete observed metadata required")
    with path.open("rb") as stream:
        raw = stream.read(max_bytes+1)
    _need(len(raw) <= max_bytes and hashlib.sha256(raw).hexdigest() == ref["sha256"],
          "Observed metadata changed during bounded snapshot read")
    return json.loads(raw)


def load_relate_observation(inference_ref, observed_vcf_ref, *, protocol_id,
                            max_sites, max_metadata_bytes, resource_check):
    """Join every retained native site to actual observed phased calls, never truth.

    AP's existing native receipt is concrete. Other protocols may supply this
    same observed receipt contract only after their genuine inference completes;
    this adapter does not invent ARGMix inputs, a map or an error mechanism.
    """
    implementation = _implementation(protocol_id)
    condition = implementation.CONDITIONS[PROTOCOLS[protocol_id]]
    _need(callable(resource_check) and type(max_sites) is int and max_sites > 0,
          "Explicit full-parent site cap and current worker callback required")
    resource_check("European_TreeLoom_before_observation_load")
    receipt = _json_ref(inference_ref, max_metadata_bytes)
    _need(receipt.get("status") == "post_five_iterations_inferred"
          and receipt.get("sequence_length_bp") == condition["length_bp"]
          and receipt.get("truth_or_oracle_read") is False,
          "Actual full-domain observed native inference receipt required")
    vcf_path = _bound(observed_vcf_ref)
    _need(receipt["input_observed_sha256"]["genotypes.vcf"] == observed_vcf_ref["sha256"],
          "Original observed VCF must match actual native inference input")
    rows = _json_ref(receipt["observed_sample_manifest"], max_metadata_bytes)
    import numpy as np
    import tskit
    ts = tskit.load(str(_bound(receipt["inferred_tree"])))
    tables = ts.tables
    samples = np.asarray(ts.samples(), dtype=np.int64)
    _need(ts.num_samples == condition["haplotypes"] and 0 < ts.num_sites <= max_sites
          and len(rows) == len(samples)
          and [r["sample_index"] for r in rows] == list(range(len(rows)))
          and samples.tolist() == [r["observed_node_id"] for r in rows]
          and ts.num_migrations == ts.num_populations == ts.num_individuals == ts.num_provenances == 0
          and not tables.metadata and (tables.nodes.population == -1).all()
          and (tables.nodes.individual == -1).all()
          and all(len(table.metadata) == 0 for table in
                  (tables.nodes, tables.edges, tables.sites, tables.mutations)),
          "Complete sanitized native sample/node/site identities required")
    post = receipt["post_five_iterations"]
    left, right = post["coverage_original_bp0"]
    _need(0 <= left < right <= condition["length_bp"]
          and right-left == ts.sequence_length
          and post["local_to_original_offset_bp"] == left,
          "Genuine native coverage and exact physical offset required")
    join = post["site_join"]
    _need(len(join) == ts.num_sites
          and [r["native_site_id"] for r in join] == list(range(ts.num_sites))
          and [r["local_position_0based"] for r in join] == tables.sites.position.tolist()
          and all(r["original_position_0based"] == r["VCF_position_1based"]-1
                  and r["local_position_0based"] == r["original_position_0based"]-left for r in join),
          "Every native SNP must retain its original VCF/physical coordinate")
    wanted = {r["VCF_position_1based"]: r["native_site_id"] for r in join}
    _need(len(wanted) == len(join), "Unique native retained observed SNPs required")
    h = len(rows)
    genotype, allele_count = np.empty((ts.num_sites,h),dtype=np.int8), np.empty(ts.num_sites,dtype=np.int8)
    seen, header, previous = set(), None, 0
    with vcf_path.open() as stream:
        for line in stream:
            if line.startswith("##"):
                continue
            fields = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                header = fields[9:]
                _need(h % 2 == 0 and header == [r["sample_id"] for r in rows[::2]]
                      and all(rows[i]["sample_id"] == rows[i+1]["sample_id"]
                              and [rows[i]["haplotype"],rows[i+1]["haplotype"]] == [0,1]
                              for i in range(0,h,2)), "Original phased diploid sample columns required")
                continue
            _need(header is not None and len(fields) == len(header)+9, "Complete original VCF rows required")
            position = int(fields[1])
            _need(position > previous, "Original observed VCF positions must remain ordered")
            previous = position
            index = wanted.get(position)
            if index is None:
                continue
            if len(seen) % 4096 == 0:
                resource_check("European_TreeLoom_complete_VCF_join")
            alleles = [fields[3]] + ([] if fields[4] == "." else fields[4].split(","))
            _need(0 < len(alleles) <= 4 and all(a in {"A","C","G","T"} for a in alleles)
                  and len(set(alleles)) == len(alleles), "Original nucleotide allele order required")
            format_names = fields[8].split(":")
            _need("GT" in format_names, "Actual observed GT calls required")
            gt_index = format_names.index("GT")
            calls = []
            for sample in fields[9:]:
                gt = sample.split(":")[gt_index]
                _need("|" in gt, "Observed haplotypes must remain phased")
                pair = gt.split("|")
                _need(len(pair) == 2, "Two actual haplotypes per VCF sample required")
                for call in pair:
                    value = -1 if call == "." else int(call)
                    _need(-1 <= value < len(alleles), "Observed nucleotide genotype index required")
                    calls.append(value)
            genotype[index] = calls
            allele_count[index] = len(alleles)
            seen.add(index)
    _need(header is not None and len(seen) == ts.num_sites, "Every native SNP requires original observed calls")
    labels, queries, ages = [], [], []
    for row in rows:
        label = row["observed_reference_class"]
        _need(row["role"] in ("query","pure_source_reference")
              and (label is None if row["role"] == "query" else type(label) is int and 0 <= label < 4),
              "Only original pure-reference labels; all queries unknown")
        labels.append(-1 if label is None else label)
        if row["role"] == "query":
            queries.append(row["sample_index"])
        ages.append(row["age_generations"])
    _need(len(queries) == condition["queries"] and np.array_equal(ts.nodes_time[samples], ages),
          "All original query identities and sample dates required")
    source_refs = dict(inference_ref=inference_ref, observed_vcf_ref=observed_vcf_ref,
                       sample_manifest_ref=receipt["observed_sample_manifest"])
    resource_check("European_TreeLoom_complete_observation_loaded")
    return dict(manifest=receipt,ts=ts,rows=rows,nodes=samples,genotype=genotype,
                positions=np.asarray(tables.sites.position,dtype=np.float64),allele_count=allele_count,
                labels=np.asarray(labels,dtype=np.int64),ages=np.asarray(ages,dtype=np.float64),
                query_columns=np.asarray(queries,dtype=np.int64),coordinate_offset_bp=left,
                source_refs=source_refs)


def _sampled_records(ref, *, observation_ref=None, parent_id=None, split=None,
                     max_metadata_bytes=128*2**20):
    import numpy as np
    import zipfile
    path = _bound(ref)
    if path.suffix == ".json":
        from arggnn.benchmarking import european_sampling
        registry = _json_ref(ref,max_metadata_bytes)
        _need(registry.get("observation_ref") == observation_ref
              and registry.get("parent_id") == parent_id and registry.get("split") == split
              and registry.get("labels_present") is False and registry.get("query_truth_read") is False,
              "Sampled registry must bind the exact actual native inference observation")
        return european_sampling.registry_arrays(registry)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        allowed = {key + ".npy" for key in (*IDENTITY, "first_occurrence_intervals_bp")}
        required = {key + ".npy" for key in IDENTITY}
        _need(required <= set(names) <= allowed and len(names) == len(set(names))
              and sum(x.file_size for x in archive.infolist()) <= 32*2**20,
              "Bounded immutable sampled registry required")
    with np.load(path,allow_pickle=False) as archive:
        return {key:archive[key].copy() for key in archive.files}


@dataclass(frozen=True)
class EuropeanWholeParent:
    original: object
    focal_hold: object
    focal_sample_column: object
    sampled_registry_ref: dict
    focal_identity_sha256: str
    focal_stamps: tuple

    @property
    def bank(self): return self.original.bank
    @property
    def episodes(self): return self.original.episodes
    @property
    def cut(self): return self.original.cut
    @property
    def parent_id(self): return self.original.parent_id
    @property
    def split(self): return self.original.split
    @property
    def native_holds(self): return self.bank.sample_index.shape[0]

    def validate(self):
        import torch
        self.original.validate()
        memory = implementation.memory_core
        _need(self.focal_stamps == memory._stamps((self.focal_hold,self.focal_sample_column))
              and self.focal_hold.dtype == self.focal_sample_column.dtype == torch.long
              and self.focal_hold.device.type == self.focal_sample_column.device.type == "cpu"
              and self.focal_hold.ndim == 1 and self.focal_hold.numel() > 0
              and self.focal_hold.shape == self.focal_sample_column.shape
              and bool(((self.focal_hold >= 0) & (self.focal_hold < self.native_holds)).all())
              and bool(((self.focal_sample_column >= 0)
                        & (self.focal_sample_column < self.bank.sample_index.shape[1])).all())
              and bool(torch.isin(self.focal_sample_column,self.bank.query_column).all()),
              "Immutable complete original sampled query/hold join required")
        return self.bank


def prepare_observation(state, *, protocol_id, parent_id, split, source_refs,
                        sampled_registry_ref, max_nodes, max_holds,
                        max_episode_payload_bytes, resource_check,
                        max_sampled_metadata_bytes=128*2**20):
    implementation = _implementation(protocol_id)
    condition = implementation.CONDITIONS[PROTOCOLS[protocol_id]]
    _need(callable(resource_check) and type(state) is dict and set(state) == _STATE_KEYS
          and all(type(v) is int and v > 0 for v in (max_nodes,max_holds,max_episode_payload_bytes))
          and source_refs == state["source_refs"]
          and state["manifest"]["parent_id"] == parent_id
          and state["manifest"]["split"] == split,
          "Actual same-parent observed state, split, refs and complete bounds required")
    import numpy as np
    import torch
    from research.repairs.benchmark_recovery_20261020.runtime import observations
    from research.repairs.benchmark_recovery_20261020.runtime import episodes as episode_preparation
    from research.repairs.benchmark_recovery_20261020.runtime import cuts as cut_preparation
    from arggnn.ordered_clade_episode_gnn_v1 import OrderedCladeEpisodesV1
    from arggnn.american_c3_gnn_targets_v1 import identity_sha
    ts, h = state["ts"], condition["haplotypes"]
    _need(ts.num_trees <= max_holds and ts.num_samples == h
          and state["genotype"].shape == (ts.num_sites,h)
          and np.array_equal(state["positions"],ts.tables.sites.position)
          and np.array_equal(state["nodes"],ts.samples())
          and len(state["query_columns"]) == condition["queries"]
          and state["manifest"]["sequence_length_bp"] == condition["length_bp"]
          and state["manifest"]["truth_or_oracle_read"] is False,
          "Every original observed native tree/site/sample must remain present")
    trees, intervals, nodes = [], [], 0
    workspace = observations.native_workspace_v1(ts)
    node_time = np.asarray(ts.nodes_time,dtype=np.float64)
    for tree in ts.trees():
        if tree.index % 256 == 0:
            resource_check("European_TreeLoom_complete_native_tree_list")
        native = observations.native_tree(tree,state["nodes"],node_time=node_time,workspace=workspace)
        nodes += len(native.parent)
        _need(nodes <= max_nodes, "Complete native parent exceeds node cap; never truncate")
        trees.append(native)
        intervals.append(tuple(tree.interval))
    bank, observation_metadata = observations.encode_block(
        trees,intervals,state["genotype"],state["allele_count"],state["positions"],
        state["labels"],state["ages"],num_classes=4,age_split=350.,
        query_columns=state["query_columns"],resource_check=resource_check)
    del trees
    _need(bank.parent.numel() == nodes and len(bank.spans) == ts.num_trees
          and len(bank.genotype) == ts.num_sites, "Full original observed bank required")
    isolated = tuple(observation_metadata.get("isolated_zero_site_holds",()))
    resource_check("European_TreeLoom_global_episodes")
    episodes, episode_metadata = episode_preparation.build_episodes(
        bank,max_payload_bytes=max_episode_payload_bytes,isolated_hold_indices=isolated)
    resource_check("European_TreeLoom_global_cut_plan")
    cut, cut_metadata = cut_preparation.prepare_cut_plan_v1(
        bank,expected_haplotypes=h,max_nodes=max_nodes,max_holds=max_holds,isolated_hold_indices=isolated)
    refs = dict(source_refs,parent_id=parent_id,query_truth_in_inputs=False,
                all_observed_trees_nodes_sites_samples_retained=True,
                supported_length_bp=float(bank.spans.double().sum()),
                requested_length_bp=condition["length_bp"])
    native_split = {"validation":"val_prediction","test":"evaluation_prediction",
                    "evaluation":"evaluation_prediction"}.get(split,split)
    original = implementation.FrozenEuropeanBundleV1(
        bank,episodes,cut,PROTOCOLS[protocol_id],parent_id,native_split,refs,
        implementation.memory_core._stamps(implementation._bank_tensors(bank)),
        implementation.memory_core._stamps(implementation._episode_tensors(episodes)))
    original.validate()
    records = sampled_records(sampled_registry_ref,protocol_id=protocol_id,
        observation_ref=source_refs["inference_ref"],parent_id=parent_id,split=split,
        max_metadata_bytes=max_sampled_metadata_bytes)
    output_classes = (NATIVE_SIX_CLASSES if protocol_id == "ancestralpaths_paper200mb_native6"
                      else condition["output_classes"])
    n = len(records.get("record_ids",[]))
    _need(set(IDENTITY) <= set(records) and n > 0
          and all(records[k].dtype == np.int64 and records[k].shape == (n,)
                  for k in ("record_ids","sample_nodes","sample_indices"))
          and len(set(records["record_ids"].tolist())) == n
          and records["positions_bp0"].dtype == np.float64 and records["positions_bp0"].shape == (n,)
          and records["parent_id"].shape == () and records["parent_id"].item() == parent_id
          and records["class_order"].tolist() == list(output_classes)
          and np.isin(records["sample_indices"],state["query_columns"]).all()
          and np.array_equal(records["sample_nodes"],state["nodes"][records["sample_indices"]]),
          "Exact original sampled row/sample/class identity required; repeated draws preserved")
    local = records["positions_bp0"] - state["coordinate_offset_bp"]
    _need(np.isfinite(local).all() and ((local >= 0) & (local < ts.sequence_length)).all(),
          "No invented native tail or coordinate extrapolation")
    holds = np.searchsorted(np.asarray(list(ts.breakpoints())),local,side="right")-1
    focal_hold = torch.as_tensor(holds.copy(),dtype=torch.long)
    focal_sample = torch.as_tensor(records["sample_indices"].copy(),dtype=torch.long)
    bundle = EuropeanWholeParent(original,focal_hold,focal_sample,dict(sampled_registry_ref),identity_sha(records),
        implementation.memory_core._stamps((focal_hold,focal_sample)))
    bundle.validate()
    return bundle, dict(parent_id=parent_id,protocol_id=protocol_id,
        sampled_registry_ref=sampled_registry_ref,focal_identity_sha256=bundle.focal_identity_sha256,
        focal_records=n,episode_metadata=episode_metadata,cut_metadata=cut_metadata,
        observation_metadata=observation_metadata,private_truth_opened=False,
        all_native_trees_nodes_sites_samples_retained=True,
        structural_recipes_global_before_partition=True,artificial_block_boundaries=False)


def _general_history(memory, forcing, theta, spans, plan, *, device,
                     holds_per_window=128, resource_check, profile=None):
    """Pinned stable-leaf algorithm with only the actual panel-width assertion generalized."""
    import torch
    from research.reviews.TreeLoom_GPU_history_source_20261017_v1 import stable_leaf_history_v1 as history
    offsets = tuple(plan.group_offsets)
    _need(memory.arm == "stable_leaf_memory"
          and forcing.device.type == theta.device.type == spans.device.type == "cpu"
          and forcing.dtype == theta.dtype == spans.dtype == torch.float32
          and type(holds_per_window) is int and 1 <= holds_per_window <= 128
          and len(offsets) == plan.native_holds+1 and offsets[0] == 0
          and offsets[-1] == len(forcing)
          and plan.leaf_groups.shape == (plan.native_holds,plan.haplotypes)
          and plan.haplotypes in (952,3222) and forcing.shape[1] == 8
          and all(0 < b-a <= plan.haplotypes for a,b in zip(offsets,offsets[1:])),
          "Exact complete European FP32 disjoint cut-group partition required")
    profile = profile or history.NullProfileV1()
    def direction(reverse):
        ell = 100.+(100000.-100.)*theta.sigmoid()
        rho = torch.exp(-spans/ell)
        source = -torch.expm1(-spans/ell)
        return history._Direction.apply(forcing,rho,source,plan.leaf_groups,offsets,reverse,
            holds_per_window,torch.device(device),resource_check,profile)
    left, right = direction(False), direction(True)
    with profile.stage("history_original_CPU_pool"):
        leaf = .5*(left+right)
        read = memory._mean(leaf.flatten(0,1),plan.leaf_groups.flatten(),
                            plan.group_counts,forcing.shape[0])
        return read-forcing



def sampled_records(ref, *, protocol_id, **kwargs):
    records = _sampled_records(ref, **kwargs)
    if protocol_id == "ancestralpaths_paper200mb_c4":
        import numpy as np
        _need(records["class_order"].tolist() == list(NATIVE_SIX_CLASSES),
              "Original AP coordinate registry identity remains native six")
        records = dict(records,class_order=np.asarray(("Ana","CHG","WHG","EHG")))
    return records


def predict_logits(model,bundle,metadata,*,resource_check):
    from .european_execution import predict_logits as run
    return run(model,bundle,metadata,resource_check=resource_check)


def train_sampled_step(model,bundle,targets,optimizer,metadata,*,resource_check):
    from .european_execution import train_sampled_step as run
    return run(model,bundle,targets,optimizer,metadata,resource_check=resource_check)
