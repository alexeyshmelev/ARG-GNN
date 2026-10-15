"""Source-only H018 baseline bridge core. No file/network/native/model I/O.

Root admits and supplies exact pinned bytes and the qualified40-row metadata
roster. This module never executes on real observations during source review.
"""
import csv
import gzip
import hashlib
import io
import json
import math

BEGIN, END = 20_000_000, 30_000_000
CANDIDATES, RETAINED, SUBJECTS = 94892, 4148, 20
MAX_INPUT_COMPRESSED = 80_000_000
MAX_QC_BYTES = 80_000_000
MAX_EXPANDED_VCF = 1_000_000_000
MAX_LINE = 1_000_000
BASES = frozenset("ACGT")


def require(ok, text):
    if not ok:
        raise ValueError(text)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def frozen_identity(row):
    """QC-table identity, never selected from current-query allele frequency."""
    p = int(row["POS_bp1"])
    require(row["chrom"] == "22" and BEGIN <= p-1 < END, "Frozen chr22 region")
    require(int(row["absolute_bp0"]) == p-1 and int(row["region_bp0"]) == p-1-BEGIN,
            "Exact absolute/regional coordinate join")
    ref, alt = row["REF"], row["ALT"]
    require(ref in BASES and alt in BASES and len(ref) == len(alt) == 1 and ref != alt,
            "Original biallelic nucleotide identity")
    return "22", p, ref, alt


def frozen_mask(qc_raw):
    require(isinstance(qc_raw, bytes) and len(qc_raw) <= MAX_QC_BYTES, "Bounded QC bytes")
    rows = csv.DictReader(io.StringIO(qc_raw.decode("utf-8")), delimiter="\t")
    mask, all_ids, previous = {}, [], 0
    for row in rows:
        identity = frozen_identity(row)
        require(identity[1] > previous, "Every candidate QC row uniquely ordered")
        previous = identity[1]; all_ids.append(identity)
        flag = row["H018_fixed_QC_pass"]
        require(flag in ("True", "False"), "Exact original serialized QC Boolean")
        if flag == "True":
            require(row["pan_troglodytes"] in identity[2:], "Frozen polarization evidence")
            mask[identity] = row["pan_troglodytes"]
    require(len(all_ids) == CANDIDATES and len(mask) == RETAINED, "Exact94892/4148 frozen mask")
    return all_ids, mask


def decode_fixed_HS_pair(hs, draw):
    """Future sensitivity only: same draw per subject across every ligated site."""
    require(type(hs) is int and 0 <= hs < 2**30 and type(draw) is int and 0 <= draw < 8,
            "Actual NMAIN15/30bit and fixeddraw0..7")
    return (hs >> (2*draw)) & 1, (hs >> (2*draw+1)) & 1


def polarize_GT_record(fields, expected_ancestral):
    """Biological nucleotide/phase equivalence before and after AA/DA recoding."""
    require(len(fields) == 9+SUBJECTS and fields[0] == "22", "Exact20 diploid columns")
    p, ref, alt = int(fields[1]), fields[3], fields[4]
    require(ref in BASES and alt in BASES and len(ref) == len(alt) == 1 and ref != alt,
            "Single-base original alleles")
    tags = fields[8].split(":")
    require(len(tags) == len(set(tags)) and "GT" in tags and "AP" not in tags,
            "Unique GT-bearing AP-free FORMAT")
    info = {}
    for item in fields[7].split(";"):
        if item in ("", "."):
            continue
        k, sep, value = item.partition("=")
        require(k not in info and k != "AP", "Unique AP-free INFO")
        info[k] = value if sep else True
    ancestor = info.get("pan_troglodytes")
    require(ancestor in BASES and ancestor in (ref, alt) and ancestor == expected_ancestral,
            "Exact pinned single-base source ancestry; missing/third/ambiguous fails")
    source = []
    gt_at = tags.index("GT")
    for sample in fields[9:]:
        values = sample.split(":")
        require(gt_at < len(values), "Callable original GT field")
        gt = values[gt_at]
        require(gt in ("0|0", "0|1", "1|0", "1|1"), "Exact phased callable source GT")
        source.extend((int(gt[0]), int(gt[2])))
    flip = int(ancestor == alt)
    derived = ref if flip else alt
    recoded = [x ^ flip for x in source]
    original_nt = "".join((ref, alt)[x] for x in source)
    inference_nt = "".join((ancestor, derived)[x] for x in recoded)
    require(original_nt == inference_nt and len(source) == 2*SUBJECTS,
            "Every actual haplotype nucleotide/order must survive polarization")
    name = f"H018_22_{p}_{ref}_{alt}"
    haps = " ".join(["22", name, str(p-BEGIN), ancestor, derived] + list(map(str, recoded)))+"\n"
    audit = dict(chrom="22", source_POS_bp1=p, regional_bp0=p-1-BEGIN,
        haps_POS_bp1=p-BEGIN, source_REF=ref, source_ALT=alt, source_ancestral=ancestor,
        allele_orientation_flipped=bool(flip), source_GT_haplotype_codes=source,
        native_haps_codes=recoded, nucleotide_equivalence_sha256=sha(original_nt.encode()),
        exact_nucleotide_and_phase_order_equivalence=True,
        monomorphic_among_actual40=sum(recoded) in (0, len(recoded)))
    return haps.encode(), audit


def assemble_baseline(vcf_gz_raw, qc_raw, roster, *, expected_vcf_sha256, expected_qc_sha256):
    """Pure in-memory preparation; native admissibility remains an explicit gate."""
    require(isinstance(vcf_gz_raw, bytes) and len(vcf_gz_raw) <= MAX_INPUT_COMPRESSED,
            "Bounded original regional BGZF bytes")
    require(sha(vcf_gz_raw) == expected_vcf_sha256 and sha(qc_raw) == expected_qc_sha256,
            "Root's pinned completed acquisition bytes")
    require(len(roster) == 2*SUBJECTS and [x["sample_index"] for x in roster] == list(range(2*SUBJECTS)),
            "Qualified40 haplotype metadata order")
    names = []
    for i in range(SUBJECTS):
        a, b = roster[2*i:2*i+2]
        require(a["subject_id"] == b["subject_id"] and (a["haplotype"], b["haplotype"]) == (0, 1)
                and a["age_generations"] == b["age_generations"]
                and math.isfinite(a["age_generations"]) and a["age_generations"] > 0
                and a["query_ancestry_truth"] is None and b["query_ancestry_truth"] is None,
                "True original diploid identity/date order; no query truth")
        names.append(a["subject_id"])
    require(len(set(names)) == SUBJECTS
            and sum(x["role"] == "reference" for x in roster) == 16
            and sum(x["role"] == "query" for x in roster) == 24,
            "Unique R16/Q24, no duplicated/padded references")
    all_ids, mask = frozen_mask(qc_raw)
    observed_ids, haps, audits, header_names, expanded = [], [], [], None, 0
    nmain15 = False
    with gzip.GzipFile(fileobj=io.BytesIO(vcf_gz_raw)) as source:
        while True:
            raw = source.readline(MAX_LINE+1)
            if not raw:
                break
            expanded += len(raw)
            require(len(raw) <= MAX_LINE and expanded <= MAX_EXPANDED_VCF, "VCF line/expansion cap")
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("##"):
                if line.startswith("##NMAIN="):
                    require(line == "##NMAIN=15" and not nmain15, "Exact unique NMAIN15")
                    nmain15 = True
                require(not line.startswith(("##FORMAT=<ID=AP,", "##INFO=<ID=AP,")),
                        "AP definitions forbidden before bridge")
                continue
            if line.startswith("#CHROM\t"):
                require(header_names is None, "Unique VCF header")
                header_names = line.split("\t")[9:]
                require(header_names == names and nmain15, "Exact source/qualified diploid header and NMAIN15 join")
                continue
            require(header_names is not None and not line.startswith("#"), "Original VCF header required")
            f = line.split("\t")
            require(len(f) == 9+SUBJECTS, "Exact source diploid columns")
            identity = (f[0], int(f[1]), f[3], f[4])
            observed_ids.append(identity)
            if identity in mask:
                native, audit = polarize_GT_record(f, mask[identity]); haps.append(native); audits.append(audit)
    require(observed_ids == all_ids and len(audits) == RETAINED, "Exact all-candidate and4148-site identity/order join")
    identities = [list(x) for x in mask]
    ages = "".join(format(x["age_generations"], ".17g")+"\n" for x in roster).encode()
    sample = ("ID_1 ID_2 missing\n0 0 0\n"+"".join(f"{n} {n} 0\n" for n in names)).encode()
    mono = [x["source_POS_bp1"] for x in audits if x["monomorphic_among_actual40"]]
    report = dict(schema="H018_baseline_observed_bridge_prepared_v1",
        status="prepared_GT_inputs_not_native_admitted",candidate_sites=CANDIDATES,retained_sites=RETAINED,
        haplotypes=2*SUBJECTS,reference_haplotypes=16,query_haplotypes=24,
        source_GT_biological_nucleotides_and_phase_unchanged=True,AP_used=False,
        source_VCF_sha256=sha(vcf_gz_raw),source_QC_sha256=sha(qc_raw),
        retained_identity_sha256=sha(json.dumps(identities,separators=(",",":")).encode()),
        polarization_flipped_sites=sum(x["allele_orientation_flipped"] for x in audits),
        monomorphic_sites=mono,native_baseline_admissible_if_no_other_blocker=not mono,
        monomorphic_policy="Stop native admission if present; no silent drop or rescue mask",
        original_QC_failure_replaced=False,HS_draws_materialized=False,
        map_qualification_required=True,native_inference_executed=False,ancestry_truth_available=False)
    payloads = {"baseline.haps": b"".join(haps), "baseline.sample": sample,
        "sample_ages.generations": ages,
        "frozen_site_identities.json": (json.dumps(identities,indent=2)+"\n").encode(),
        "polarization_equivalence.json": (json.dumps(audits,indent=2)+"\n").encode(),
        "bridge_preparation.json": (json.dumps(report,indent=2)+"\n").encode()}
    require(sum(map(len, payloads.values())) <= 25_000_000, "Baseline preparation output cap")
    return payloads, report
