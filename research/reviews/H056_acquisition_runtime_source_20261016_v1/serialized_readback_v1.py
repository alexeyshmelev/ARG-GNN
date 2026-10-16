"""Pure independent H056 serialized readback and root postwait seal.
No bridge/extractor/model imports, IO, acquisition, or execution on import.
"""
import csv, gzip, hashlib, io, json, math, re
BEGIN, END, H = 20000000, 30000000, 36
SUBJECTS = ("Bar8","Bon002","KK1","SATP","Loschbour","Bichon","Sidelkino","NEO88",
            "NEO866","NEO945","NEO28","NEO942","NEO790","NEO739","Gjerrild5","NEO735","NEO92","NEO870")
AGES = (8071,10078,9720,13255,8050,13665,11258.5,7858,5457,5444,5461,5495,5668,3971,4110,3995,4218,4255)
CORE_NAMES = {"baseline.haps","baseline.sample","sample_ages.generations",
              "frozen_site_identities.json","polarization_equivalence.json","bridge_preparation.json"}

def require(ok, message):
    if not ok:
        raise ValueError(message)
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
def compact(value):
    return json.dumps(value, separators=(",",":"), allow_nan=False).encode()
def decode(raw):
    def pairs(items):
        out = {}
        for k,v in items:
            require(k not in out, "Unique readback JSON keys")
            out[k] = v
        return out
    def invalid(v):
        raise ValueError("No nonfinite JSON: " + v)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
def map_v1(raw, qualification, chromosome):
    require(isinstance(raw, bytes) and len(raw) <= 2**20, "Bounded qualified map bytes")
    require(qualification["root_authorizer"] == "root_agent"
            and qualification["chromosome"] == chromosome
            and qualification["assembly"] in ("GRCh37", "GRCh37_b37_hs37d5")
            and qualification["physical_unit"] == "bp"
            and qualification["genetic_unit"] == "cM"
            and qualification["map_sha256"] == sha(raw)
            and qualification["byte_identity_verified"] is True
            and qualification["source_semantics_qualified"] is True
            and qualification["source_map_brackets_full_requested_interval"] is True
            and qualification["source_boundary_bp1"] == [BEGIN+1, END+1]
            and qualification["local_boundary_bp1"] == [1, END-BEGIN+1]
            and qualification["uniform_map_substitution"] is False
            and qualification["source_positions_or_distances_rescaled"] is False,
            "Root-qualified exact chromosome/build/unit/bracketing map")
    lines = raw.decode("ascii").splitlines()
    require(lines and lines[0] == "Position(bp) Rate(cM/Mb) Map(cM)", "Original map header")
    rows = []
    for line in lines[1:]:
        values = line.split()
        require(len(values) == 3 and re.fullmatch(r"[0-9]+", values[0]), "Map fields")
        p, rate, cm = int(values[0]), float(values[1]), float(values[2])
        require(math.isfinite(rate) and math.isfinite(cm) and rate >= 0 and cm >= 0
                and (not rows or p > rows[-1][0] and cm >= rows[-1][2]), "Strict map positions/nondecreasing finite cM")
        rows.append((p,rate,cm))
    require(len(rows) == qualification["output_rows"] and len(rows) >= 2
            and rows[0][0] == 1 and rows[-1][0] == END-BEGIN+1
            and rows[0][2] == 0 and rows[-1][2] > 0, "Exact full regional map endpoints and origin")
    return rows

def verify_v1(vcf_gz, qc_raw, roster, payloads, map_raw, qualification, *,
              chromosome, counts, extraction, protocol_ref):
    require(chromosome in ("20","21","22") and set(counts) == {"candidate_sites","sites","constants"}
            and all(type(v) is int for v in counts.values())
            and counts["candidate_sites"] >= counts["sites"] >= 1000
            and 0 <= counts["constants"] <= counts["sites"], "Actual independent base count seal")
    require(isinstance(vcf_gz,bytes) and len(vcf_gz) <= 24*2**20
            and isinstance(qc_raw,bytes) and len(qc_raw) <= 80000000, "Bounded completed source bytes")
    require(set(payloads) == CORE_NAMES and sum(map(len,payloads.values())) <= 16*2**20, "Exact bounded bridge assets")
    require(isinstance(roster,list) and len(roster) == H, "Exact36 source roster")
    for i,row in enumerate(roster):
        j, label = i//2, i//4 if i < 16 else None
        require(row["sample_index"] == i and type(row["sample_index"]) is int
                and row["subject_id"] == SUBJECTS[j] and row["haplotype"] == i%2
                and row["role"] == ("reference" if i < 16 else "query")
                and row["observed_reference_class"] == label
                and row.get("query_ancestry_truth") is None and row.get("native_tree_node_id") is None
                and row["age_mean_cal_BP"] == AGES[j]
                and row["age_generations"] == AGES[j]/28.
                and row["date_zero"] == "calBP1950" and row["years_per_generation"] == 28,
                "Independent frozen subject/pair/reference/query/date join")
    sample = ("ID_1 ID_2 missing\n0 0 0\n" + "".join(f"{n} {n} 0\n" for n in SUBJECTS)).encode()
    ages = "".join(format(AGES[i//2]/28., ".17g")+"\n" for i in range(H)).encode()
    require(payloads["baseline.sample"] == sample and payloads["sample_ages.generations"] == ages,
            "Actual serialized exact sample/haplotype/date text")
    qc = csv.DictReader(io.StringIO(qc_raw.decode("utf-8")), delimiter="\t")
    require(qc.fieldnames and len(qc.fieldnames) == len(set(qc.fieldnames)), "Unique QC header")
    haps = iter(payloads["baseline.haps"].decode("ascii").splitlines())
    identities, audits = decode(payloads["frozen_site_identities.json"]), decode(payloads["polarization_equivalence.json"])
    require(isinstance(identities,list) and isinstance(audits,list)
            and len(identities) == len(audits) == counts["sites"], "Serialized retained identity/audit cardinality")
    vectors, mono, retained, all_candidates = hashlib.sha256(), [], [], 0
    expanded, previous, header, nmain = 0, 0, False, False
    with gzip.GzipFile(fileobj=io.BytesIO(vcf_gz)) as src:
        while True:
            raw = src.readline(1000001)
            if not raw:
                break
            expanded += len(raw)
            require(len(raw) <= 1000000 and expanded <= 200000000, "Independent VCF logical/line bound")
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("##"):
                require(not line.startswith(("##FORMAT=<ID=AP,", "##INFO=<ID=AP,")), "No AP source declaration")
                if line.startswith("##NMAIN="):
                    require(not nmain and line == "##NMAIN=15", "Unique exact NMAIN15")
                    nmain = True
                continue
            if line.startswith("#CHROM\t"):
                require(not header and nmain and line.split("\t")[9:] == list(SUBJECTS), "Independent18-subject VCF header")
                header = True
                continue
            require(header and not line.startswith("#"), "Unique VCF header before records")
            f = line.split("\t")
            require(len(f) == 27 and f[0] == chromosome, "Exact candidate source columns/chromosome")
            p, ref, alt = int(f[1]), f[3], f[4]
            require(BEGIN <= p-1 < END and p > previous and ref and alt, "All candidate coordinates/order/allele text")
            previous = p
            row = next(qc, None)
            require(isinstance(row,dict) and None not in row and row["chrom"] == chromosome
                    and int(row["POS_bp1"]) == p and int(row["absolute_bp0"]) == p-1
                    and int(row["region_bp0"]) == p-1-BEGIN and row["REF"] == ref and row["ALT"] == alt
                    and row["H018_fixed_QC_pass"] in ("0","1"), "Every candidate exact QC0/1 serialized identity/order join")
            all_candidates += 1
            tags = f[8].split(":")
            require(tags == ["GT","GP","HS","DS"], "Unchanged AP-free original GT/GP/HS/DS FORMAT")
            info = {}
            for item in f[7].split(";"):
                if item in ("","."):
                    continue
                key, sep, value = item.partition("=")
                require(key not in info and key in ("INFO","RAF","MASK_1000G","pan_troglodytes"), "AP-free unique original INFO")
                info[key] = value if sep else True
            if row["H018_fixed_QC_pass"] == "0":
                continue
            require(ref in "ACGT" and alt in "ACGT" and len(ref) == len(alt) == 1 and ref != alt,
                    "QC1 original biallelic ACGT")
            ancestor = row["pan_troglodytes"]
            require(ancestor in (ref,alt) and info.get("pan_troglodytes") == ancestor, "Original ancestry identity")
            codes = []
            for text in f[9:]:
                values = text.split(":")
                require(len(values) == 4 and values[0] in ("0|0","0|1","1|0","1|1"), "Original phased callable GT36")
                codes.extend((int(values[0][0]), int(values[0][2])))
            flip = int(ancestor == alt)
            derived = ref if flip else alt
            native = [v ^ flip for v in codes]
            nt = "".join((ref,alt)[v] for v in codes)
            require(nt == "".join((ancestor,derived)[v] for v in native), "Biological nucleotide/phase invariance")
            identity = [chromosome,p,ref,alt]
            at = len(retained)
            require(identities[at] == identity, "Exact serialized frozen retained identity")
            expected_haps = " ".join([chromosome,"H056_"+chromosome+"_"+str(p)+"_"+ref+"_"+alt,
                                      str(p-BEGIN),ancestor,derived]+list(map(str,native)))
            require(next(haps,None) == expected_haps, "Actual serialized AA/DA36 haplotype site/phase order")
            audit = dict(chrom=chromosome,source_POS_bp1=p,regional_bp0=p-1-BEGIN,haps_POS_bp1=p-BEGIN,
                source_REF=ref,source_ALT=alt,source_ancestral=ancestor,allele_orientation_flipped=bool(flip),
                source_GT_haplotype_codes=codes,native_haps_codes=native,nucleotide_equivalence_sha256=sha(nt.encode()),
                exact_nucleotide_and_phase_order_equivalence=True,monomorphic_among_panel=len(set(native))==1)
            require(audits[at] == audit, "Every actual serialized polarization audit field")
            if audit["monomorphic_among_panel"]:
                mono.append(p)
            vectors.update((str(p)+":"+nt+"\n").encode())
            retained.append(identity)
    require(header and next(qc,None) is None and next(haps,None) is None
            and all_candidates == counts["candidate_sites"] and len(retained) == counts["sites"]
            and len(mono) == counts["constants"], "Actual independently observed C/S/K; no dropped candidates/constants")
    vector_sha, identity_sha, mono_sha = vectors.hexdigest(), sha(compact(retained)), sha(compact(mono))
    require(extraction["counts"] == counts and extraction["qualified_identity_sha256"] == identity_sha
            and extraction["original_nucleotide_vectors_sha256"] == vector_sha, "Independent extraction identity/nucleotide digest equality")
    report = decode(payloads["bridge_preparation.json"])
    require(report["schema"] == "H056_H36_observed_bridge_prepared_v1"
            and report["protocol_ref"] == protocol_ref and report["chromosome"] == chromosome
            and report["counts"] == counts and report["retained_identity_sha256"] == identity_sha
            and report["original_nucleotide_vectors_sha256"] == vector_sha
            and report["monomorphic_source_POS_bp1_sha256"] == mono_sha and report["monomorphic_sites"] == mono
            and report["source_VCF_sha256"] == sha(vcf_gz) and report["source_QC_sha256"] == sha(qc_raw)
            and report["native_inference_executed"] is False and report["AP_used"] is False,
            "Exact independent bridge/source/count/digest join")
    maps = map_v1(map_raw,qualification,chromosome)
    require(all(maps[0][0] <= row[1]-BEGIN <= maps[-1][0] for row in retained), "Every retained local site joins exact map bracket")
    return dict(schema="H056_acquisition_serialized_readback_worker_v1",
        status="actual_serialized_readback_passed_root_wait_seal_required",protocol_ref=protocol_ref,
        chromosome=chromosome,counts=dict(counts),candidate_sites=all_candidates,retained_sites=len(retained),
        original_nucleotide_vectors_sha256=vector_sha,retained_identity_sha256=identity_sha,
        monomorphic_source_POS_bp1_sha256=mono_sha,
        actual_output_sha256={**{name:sha(raw) for name,raw in payloads.items()},"qualified.regional.map":sha(map_raw)},
        actual_serialized_site_nucleotide_phase_order_verified=True,
        actual_serialized_sample_haplotype_date_join_verified=True,
        actual_qualified_map_bytes_and_local_site_join_verified=True,
        map_rows=len(maps),source_QC_encoding="BGZF; bridge and readback consume decompressed logical bytes",
        root_preparation_wait_verified=False,root_owned_group_absent=False,
        native_inference_executed=False,model_inference_executed=False)

def seal_root_postwait_v1(worker_readback, worker_readback_ref, worker_completion_ref,
                         actual_root_wait, actual_root_wait_ref):
    """Root calls only after its genuine external terminal wait0 and owned PID/group absence.
    The function validates the supplied evidence shape; it cannot observe its parent's wait.
    Root must bind actual_root_wait_ref to its persisted genuine observation under the existing guard.
    """
    require(worker_readback["schema"] == "H056_acquisition_serialized_readback_worker_v1"
            and worker_readback["root_preparation_wait_verified"] is False
            and worker_readback["root_owned_group_absent"] is False, "Unsealed actual worker readback")
    for key in ("actual_serialized_site_nucleotide_phase_order_verified",
                "actual_serialized_sample_haplotype_date_join_verified",
                "actual_qualified_map_bytes_and_local_site_join_verified"):
        require(worker_readback[key] is True, "Actual worker serialization proof")
    require(actual_root_wait["schema"] == "H056_root_actual_preparation_wait_v1"
            and actual_root_wait["root_authorizer"] == "root_agent"
            and actual_root_wait["actual_tool_call_id"]
            and actual_root_wait["actual_tool_terminal_consumed"] is True
            and actual_root_wait["actual_tool_exit_code"] == 0
            and type(actual_root_wait["worker_pid"]) is int and actual_root_wait["worker_pid"] > 1
            and actual_root_wait["process_group"] == actual_root_wait["worker_pid"]
            and actual_root_wait["actual_worker_PID_absent"] is True
            and actual_root_wait["actual_owned_group_absent"] is True
            and actual_root_wait["worker_readback_ref"] == worker_readback_ref
            and actual_root_wait["worker_completion_ref"] == worker_completion_ref,
            "Root genuine completed tool wait0 and exact owned PID/group absence evidence")
    for ref in (worker_readback_ref,worker_completion_ref,actual_root_wait_ref):
        require(set(ref) == {"path","sha256"} and isinstance(ref["path"],str)
                and re.fullmatch("[0-9a-f]{64}",ref["sha256"]), "Exact persisted root/worker refs")
    return dict(worker_readback, schema="H056_acquisition_serialized_readback_root_sealed_v1",
        status="root_actual_wait0_drain_and_serialized_readback_sealed",
        worker_readback_ref=worker_readback_ref,worker_completion_ref=worker_completion_ref,
        preparation_actual_wait_ref=actual_root_wait_ref,
        root_preparation_wait_verified=True,root_owned_group_absent=True)


def seal_container_root_postwait_v1(worker, worker_ref, completion_ref, actual_wait, actual_wait_ref):
    """Pure root-only seal after actual external successful tool wait and group absence."""
    require(worker["schema"] == "H056_container_worker_qualification_v1" and worker["passed"] is True
            and worker["root_preparation_wait_verified"] is False, "Actual unsealed container qualification")
    require(actual_wait["schema"] == "H056_root_actual_container_wait_v1"
            and actual_wait["root_authorizer"] == "root_agent"
            and actual_wait["actual_tool_call_id"] and actual_wait["actual_tool_terminal_consumed"] is True
            and actual_wait["actual_tool_exit_code"] == 0
            and type(actual_wait["worker_pid"]) is int and actual_wait["worker_pid"] > 1
            and actual_wait["process_group"] == actual_wait["worker_pid"]
            and actual_wait["actual_worker_PID_absent"] is True
            and actual_wait["actual_owned_group_absent"] is True
            and actual_wait["worker_qualification_ref"] == worker_ref
            and actual_wait["worker_completion_ref"] == completion_ref, "Genuine root container wait0/drain")
    for ref in (worker_ref, completion_ref, actual_wait_ref):
        require(set(ref) == {"path","sha256"} and isinstance(ref["path"],str)
                and re.fullmatch("[0-9a-f]{64}",ref["sha256"]), "Exact persisted container evidence refs")
    return dict(schema="H056_container_root_actual_wait_seal_v1",root_authorizer="root_agent",passed=True,
        worker_qualification_ref=worker_ref,worker_completion_ref=completion_ref,root_actual_wait_ref=actual_wait_ref,
        root_preparation_wait_verified=True,root_owned_group_absent=True,
        scientific_QC_or_model_qualification=False)
