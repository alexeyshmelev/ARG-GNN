"""Pinned official FLARE conventions. No native acquisition or execution on import."""
from pathlib import Path
import gzip
import hashlib
import json
from .american_c3_common_v1 import CLASSES, require, bound, read, ref

PIN_REPOSITORY = "https://github.com/browning-lab/flare"
PIN_TAG = "v0.6.0"
PIN_COMMIT = "ec0ad295aeebfacf10d8cbfb6486ab78981de1b1"
EXPECTED_VERSION = "flare version 0.6.0 [04ad73fd0 31-Oct-2025]"
PROJECTION = "nearest_native_marker_midpoints_ties_right_edges_to_domain_v1"
NATIVE_OPTIONS = {
    "probs": True, "em": True, "update-p": False, "array": False,
    "min-maf": .005, "min-mac": 50, "gen": 10, "panel-probs": False,
    "nthreads": 4, "seed": -99999,
}
MODEL_INPUT_ROLES = {"pure_source_reference", "query"}


def observation_ref(record):
    require(isinstance(record, dict), "Explicit observed metadata binding required")
    require(not any(p in {"truth", "oracle"} for p in Path(record["path"]).parts),
            "No private truth/oracle file accepted by FLARE observation stages")
    return record


def native_mac_threshold(reference_haplotypes):
    import numpy as np
    require(reference_haplotypes % 2 == 0 and reference_haplotypes // 2 > 50,
            "Native min-mac50 must be below reference diploid count; tiny canary is incompatible")
    # Exact float32 multiply followed by ceil, as pinned AdmixReader.minMac.
    threshold = int(__import__("math").ceil(float(
        np.float32(np.float32(.005) * np.float32(reference_haplotypes)))))
    return max(50, threshold)


def pair_roster(rows):
    require(len(rows) % 2 == 0 and [r["sample_index"] for r in rows] == list(range(len(rows))),
            "Complete original genotype-column roster required")
    pairs = []
    for i in range(0, len(rows), 2):
        a, b = rows[i:i+2]
        require(a["sample_id"] == b["sample_id"] and (a["haplotype"], b["haplotype"]) == (0, 1)
                and a["role"] == b["role"] and a["role"] in MODEL_INPUT_ROLES,
                "Exact phased diploid pairing/role required")
        cls = a["observed_reference_class"]
        require(cls == b["observed_reference_class"]
                and (cls in range(3) if a["role"] == "pure_source_reference" else cls is None),
                "Known reference class only; no query ancestry class")
        pairs.append(dict(
            vcf_sample_id="c3i_%06d" % (i//2), original_sample_id=a["sample_id"],
            sample_indices=[i, i+1], observed_node_ids=[a["observed_node_id"], b["observed_node_id"]],
            role=a["role"], observed_reference_class=cls,
        ))
    refs = sorted((p for p in pairs if p["role"] == "pure_source_reference"),
                  key=lambda p: (p["observed_reference_class"], p["sample_indices"][0]))
    queries = [p for p in pairs if p["role"] == "query"]
    require(refs and queries and {p["observed_reference_class"] for p in refs} == set(range(3)),
            "All three reference ancestries and admixed queries required")
    require(len({p["original_sample_id"] for p in pairs}) == len(pairs),
            "Diploid sample identity must be unique")
    return refs, queries


def validate_focals(registry, manifest, roster):
    import numpy as np
    require(registry["parent_id"].item() == manifest["parent_id"]
            and registry["class_order"].tolist() == list(CLASSES),
            "Common observation-only focal parent/class identity required")
    for i, node in zip(registry["sample_indices"], registry["sample_nodes"]):
        require(0 <= i < len(roster) and roster[int(i)]["role"] == "query"
                and roster[int(i)]["observed_node_id"] == int(node),
                "Focals must use exact original query-column/inferred-node bridge")
    require(((registry["positions_bp0"] >= 0)
             & (registry["positions_bp0"] < manifest["sequence_length_bp"])).all(),
            "All original focals must remain on the complete declared domain")


def raw_bound(markers, samples):
    # Genotypes are single-digit allele codes0..3. Header/ID allowance is explicit.
    return 4194304 + markers * (256 + 4 * samples)


def gzip_bound(raw_bytes):
    # Allow expansion, never assume a compression saving.
    return int(raw_bytes + (raw_bytes + 99)//100 + 65536)


class GzipVCF:
    def __init__(self, writer, name, raw_cap):
        self.writer, self.raw_cap, self.raw_bytes = writer, int(raw_cap), 0
        self.cap = gzip_bound(raw_cap)
        self.path = writer.prepare(name, self.cap)
        self.raw = self.path.open("xb")
        self.stream = gzip.GzipFile(filename="", mode="wb", fileobj=self.raw, mtime=0)

    def write(self, text):
        data = text.encode("ascii")
        self.raw_bytes += len(data)
        require(self.raw_bytes <= self.raw_cap, "VCF uncompressed count bound exceeded")
        self.stream.write(data)

    def close(self):
        self.stream.close()
        self.raw.close()
        return self.writer.finish(self.path, self.cap)


def stage_completion(record, status):
    value = read(observation_ref(record))
    require(value.get("status") == status and value.get("author_Table1_exact_reproduction") is False,
            "Exact declared native reconstruction stage required")
    return value


def native_file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()
