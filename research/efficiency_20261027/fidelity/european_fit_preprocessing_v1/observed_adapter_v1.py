"""Exact vectorized observed-call reader for our input adapter.

No simulation, native competitor, tree inference, or model operation changes.
Noncanonical integer spellings and unsupported fast layouts use the original
per-sample interpretation. The full public-input validation remains unchanged.
"""
from research.repairs.benchmark_multigpu_20261022.scientific.queue_adapters import european_observations as original
from research.preprocessing_optimization_20261026.observed.parser import read_observed_calls

_implementation = original._implementation
PROTOCOLS = original.PROTOCOLS
_need = original._need
_json_ref = original._json_ref
_bound = original._bound

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
    genotype, allele_count = read_observed_calls(
        vcf_path, rows, wanted, ts.num_sites, resource_check=resource_check)
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

