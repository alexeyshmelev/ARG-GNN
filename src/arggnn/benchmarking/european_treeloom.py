"""Complete European TreeLoom banks and dimension-bound tiled execution.

No launch, policy, lease, truth reader or checkpoint selection lives here.
All native trees/nodes/sites/samples enter the bank. Only sampled emissions are
selected; tree sweeps, stable-leaf history and genomic GRU context stay complete.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from . import contracts as c
from . import treeloom
from . import american_preparation as generation
from . import treeloom_prediction as american

PROTOCOLS = {
    "ancestralpaths_paper200mb_native6": "AP_paper_scale_200Mb_native6",
    "ancestralpaths_paper200mb_c4": "AP_paper_scale_200Mb_derived_C4_control",
    "european_c4_expanded": "benchmark_european_c4_expanded",
    "european_c4_prospective1611": "ARGMix_prospective1611",
}
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
    requirements()
    return treeloom._european_module()


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
        from . import european_sampling
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
        memory = treeloom._european_module().memory_core
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
    observations = importlib.import_module(
        "research.reviews.TreeLoom_preparation_reuse_source_20261017_v1.optimized_observations_v1")
    episode_preparation = importlib.import_module(
        "research.reviews.H060_isolated_hold_compatibility_source_20261017_v1.episode_preparation_v1")
    cut_preparation = importlib.import_module(
        "research.reviews.H060_isolated_hold_compatibility_source_20261017_v1.cut_preparation_v1")
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
    arrays, episode_metadata = episode_preparation.build_clade_episode_arrays_v1(
        bank.parent.numpy(),bank.sample_index.numpy(),bank.spans.numpy(),
        max_payload_bytes=max_episode_payload_bytes,isolated_hold_indices=isolated)
    episodes = OrderedCladeEpisodesV1.from_verified_arrays(arrays,episode_metadata)
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
    records = _sampled_records(sampled_registry_ref,
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
    history = importlib.import_module(
        "research.reviews.TreeLoom_GPU_history_source_20261017_v1.stable_leaf_history_v1")
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


_EXECUTION_CLASS = None


def _executor_class():
    """Create a private class from authenticated executor AST, changing dimensions only.

    Original C3 modules remain untouched. Reuse exact sweep/layout/flag/own-state
    bodies. The four explicit changes are actual H, observed4 versus C3 donor
    one-hot, dynamic sampled/output count, and the frozen European bidirectional
    GRU call. CPU construction exists for tiny numerical-parity tests; public
    training/prediction APIs below require the admitted CUDA0 model.
    """
    global _EXECUTION_CLASS
    if _EXECUTION_CLASS is not None:
        return _EXECUTION_CLASS
    requirements()
    import ast
    import copy
    original = importlib.import_module(treeloom.EXECUTION_MODULE)
    implementation = treeloom._european_module()
    source = Path(original.__file__).read_text()
    module = ast.parse(source)
    cls = copy.deepcopy(next(node for node in module.body
                             if isinstance(node,ast.ClassDef) and node.name == "BoundedExecutionV1"))
    changes = dict(panel=0,minimum=0,onehot=0,observed_width=0,output=0,test_device=0)
    class Dimensions(ast.NodeTransformer):
        method = None
        def visit_FunctionDef(self,node):
            previous = self.method
            self.method = node.name
            if node.name == "_decode":
                new = ast.parse(
                    "def _decode(self,context,geometry,focal_hold,focal_column):\n"
                    " self.tick('bounded_full_length_European_GRU')\n"
                    " context=context.to(self.device);geometry=geometry.to(self.device)\n"
                    " t,h=context.shape[:2]\n"
                    " sequence=torch.cat((context,geometry[:,None].expand(t,h,3)),-1)\n"
                    " history,_=self.model.genomic_gru(sequence)\n"
                    " if len(focal_hold):\n"
                    "  hold=focal_hold.to(self.device);column=focal_column.to(self.device)\n"
                    "  logits=self.model.classifier(torch.cat((context[hold,column],history[hold,column]),-1))\n"
                    " else:logits=context.new_empty((0,len(self.model.output_classes)))\n"
                    " return logits,history.sum()*0.\n").body[0]
                new.decorator_list = node.decorator_list
                self.method = previous
                return ast.copy_location(new,node)
            result = self.generic_visit(node)
            self.method = previous
            return result
        def visit_Constant(self,node):
            if self.method == "_sites" and node.value == 1600:
                changes["panel"] += 1
                return ast.copy_location(ast.parse("self.bank.sample_index.shape[1]",mode="eval").body,node)
            if self.method == "__init__" and node.value == 1600:
                changes["minimum"] += 1
                return ast.copy_location(ast.parse("bundle.bank.sample_index.shape[1]",mode="eval").body,node)
            return node
        def visit_Compare(self,node):
            if self.method == "__init__" and ast.unparse(node) == "self.device.type == 'cuda'":
                changes["test_device"] += 1
                return ast.copy_location(ast.parse("self.device.type in ('cpu','cuda')",mode="eval").body,node)
            return self.generic_visit(node)
        def visit_Call(self,node):
            node = self.generic_visit(node)
            if self.method == "_base" and isinstance(node.func,ast.Attribute):
                if node.func.attr == "one_hot" and len(node.args) == 2 and isinstance(node.args[1],ast.Constant) and node.args[1].value == 3:
                    node.args[1] = ast.Constant(4);changes["onehot"] += 1
                if node.func.attr == "expand" and len(node.args) == 3 and isinstance(node.args[2],ast.Constant) and node.args[2].value == 5:
                    node.args[2] = ast.Constant(6);changes["observed_width"] += 1
            if self.method == "forward" and isinstance(node.func,ast.Attribute) and node.func.attr == "empty":
                if node.args and isinstance(node.args[0],ast.Tuple) and ast.unparse(node.args[0]) == "(10000, 3)":
                    node.args[0] = ast.parse(
                        "(self.bundle.focal_hold.numel(),len(self.model.output_classes))",mode="eval").body
                    changes["output"] += 1
            return node
    cls = Dimensions().visit(cls)
    cls.name = "EuropeanBoundedExecutionV1"
    _need(changes == dict(panel=1,minimum=1,onehot=1,observed_width=1,output=1,test_device=1),
          "Pinned executor dimension adaptations must remain exact and reviewable")
    namespace = dict(original.__dict__)
    namespace.update(frozen=SimpleNamespace(
        AmericanTreeLoomC3FullV1=implementation.FrozenEuropeanTreeLoomV1,
        AmericanTreeLoomWholeParentV1=EuropeanWholeParent,
        _bank_tensors=implementation._bank_tensors),
        stable_leaf_history_v1=_general_history)
    tree = ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[]))
    exec(compile(tree,__file__+":dimensions_only","exec"),namespace)
    _EXECUTION_CLASS = namespace[cls.name]
    return _EXECUTION_CLASS


def _model_binding(model, bundle, metadata, resource_check):
    import torch
    implementation = treeloom._european_module()
    _need(type(model) is implementation.FrozenEuropeanTreeLoomV1
          and type(bundle) is EuropeanWholeParent
          and model.condition == bundle.original.condition
          and callable(resource_check) and os.environ.get("ARGGNN_PHYSICAL_GPU") == "0"
          and next(model.parameters()).device.type == "cuda"
          and next(model.parameters()).device.index in (None,0)
          and all(p.dtype == torch.float32 for p in model.parameters())
          and metadata.get("parent_id") == bundle.parent_id
          and metadata.get("sampled_registry_ref") == bundle.sampled_registry_ref
          and metadata.get("focal_identity_sha256") == bundle.focal_identity_sha256
          and metadata.get("private_truth_opened") is False,
          "Exact supervised GPU0 European model/observed sampled bundle required")
    bundle.validate()
    resource_check("European_TreeLoom_complete_parent")


def predict_logits(model, bundle, metadata, *, resource_check):
    _model_binding(model,bundle,metadata,resource_check)
    _need(bundle.split in ("val_prediction","evaluation_prediction","iid_prediction"),
          "Held-out observations only for European prediction")
    import torch
    model.eval()
    with torch.no_grad():
        return _executor_class()(model,bundle,metadata,resource_check=resource_check,
                                 history_holds_per_window=128).forward()


def train_sampled_step(model, bundle, targets, optimizer, metadata, *, resource_check):
    """One exact mean-CE update on every immutable sampled row, including repeats."""
    _model_binding(model,bundle,metadata,resource_check)
    import torch
    from torch.nn import functional as F
    from .targets import SampledFocalTargets
    _need(bundle.split == "train" and isinstance(targets,SampledFocalTargets)
          and targets.class_order == tuple(model.output_classes)
          and type(optimizer) is torch.optim.AdamW and len(optimizer.param_groups) == 1,
          "Same-parent sampled TRAIN targets and original AdamW required")
    targets.validate_binding(parent_id=bundle.parent_id,split="train",
                             sampled_registry_ref=bundle.sampled_registry_ref)
    rows = torch.tensor(targets.readout_rows,dtype=torch.long)
    columns = torch.tensor(targets.query_columns,dtype=torch.long)
    _need(rows.shape == bundle.focal_hold.shape and torch.equal(rows,bundle.focal_hold)
          and bool((columns < bundle.bank.query_column.numel()).all())
          and torch.equal(bundle.bank.query_column[columns],bundle.focal_sample_column),
          "Sampled labels must preserve every exact registered native hold/query draw")
    group, parameters = optimizer.param_groups[0], list(model.parameters())
    _need(group["lr"] == .001 and group["weight_decay"] == .0001
          and group["betas"] == (.9,.999) and group["eps"] == 1e-8
          and group.get("maximize",False) is False and group.get("amsgrad",False) is False
          and len(group["params"]) == len(parameters)
          and {id(p) for p in group["params"]} == {id(p) for p in parameters},
          "Exact original FP32 AdamW settings and all European parameters required")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    executor = _executor_class()(model,bundle,metadata,resource_check=resource_check,
                                 history_holds_per_window=128)
    with torch.autocast(device_type="cuda",enabled=False):
        logits = executor.forward()
        labels = torch.tensor(targets.labels,dtype=torch.long,device=logits.device)
        loss = F.cross_entropy(logits,labels,reduction="mean")
    _need(bool(torch.isfinite(loss)), "Finite complete sampled-row mean CE required")
    resource_check("European_TreeLoom_complete_parent_backward")
    loss.backward()
    _need(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters),
          "Every original European parameter requires connected finite gradients")
    norm = torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
    resource_check("European_TreeLoom_complete_parent_optimizer")
    optimizer.step()
    _need(all(bool(torch.isfinite(p).all()) for p in parameters), "Finite updated European parameters required")
    return dict(parent_id=bundle.parent_id,one_complete_parent_optimizer_update=True,
                sampled_records=len(targets.labels),loss=float(loss.detach()),gradient_norm=float(norm),
                private_truth_opened_by_observation_or_executor=False,ancestry_accuracy=None,
                all_native_trees_nodes_sites_samples_retained=True,
                detached_history=False,artificial_block_boundaries=False)
