"""Pure H056 map qualifier. H018 copy extends only chromosome/asset identities.
No network, model, genotype or storage accounting side effects on import.
"""
import bisect,gzip,hashlib,io,json,math
from pathlib import Path
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
BEGIN,END=20000000,30000000
MAP_EXPANDED_CAP=8000000

def require(ok,message):
    if not ok:raise ValueError(message)

def sha(raw):return hashlib.sha256(raw).hexdigest()

def read_ref(ref,cap=4000000):
    require(isinstance(ref,dict) and set(ref)=={"path","sha256"},"Exact semantic evidence reference")
    rel=Path(ref["path"])
    require(not rel.is_absolute() and ".." not in rel.parts,"Contained evidence path")
    p=ROOT/rel
    require(p.resolve()==p and p.is_file() and not p.is_symlink() and p.stat().st_size<=cap,"Bounded canonical evidence")
    raw=p.read_bytes()
    require(sha(raw)==ref["sha256"],"Exact semantic evidence bytes")
    return raw

def inspect_candidate_map_bytes_v1(raw, asset):
    """Exact candidate identity and bounded header inspection, no semantic claim."""
    require(isinstance(raw, bytes) and len(raw) == asset["bytes"], "Exact candidate map byte count")
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    require(blob == asset["git_blob_sha1"], "Exact Git source blob identity required")
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        expanded = stream.read(MAP_EXPANDED_CAP + 1)
    require(len(expanded) <= MAP_EXPANDED_CAP, "Expanded map cap")
    text = expanded.decode("ascii")
    lines = [line for line in text.splitlines() if line.strip()]
    require(len(lines) >= 3, "Map header and >=2 rows required")
    return text, dict(source_compressed_bytes=len(raw), source_sha256=sha(raw),
        git_blob_sha1=blob, expanded_bytes=len(expanded), expanded_sha256=sha(expanded),
        nonempty_lines=len(lines), first_three_nonempty_lines=lines[:3],
        byte_identity_verified=True, source_semantics_qualified=False)

def parse_map_text_v1(text, contract):
    """Fail closed until root has reviewed the actual source units/coordinates."""
    require(contract.get("status") == "root_reviewed_source_semantics"
            and contract.get("assembly") == "GRCh37"
            and contract.get("chromosome") in ("20", "21", "22")
            and contract.get("physical_units") == "bp"
            and contract.get("cumulative_distance_units") == "cM"
            and contract.get("position_basis") in ("absolute_bp1", "absolute_bp0")
            and contract.get("source_format") in ("pos_chr_cM", "pos_rate_cM")
            and contract.get("semantic_evidence_refs"), "Inactive or unreviewed source map semantics")
    for ref in contract["semantic_evidence_refs"]:
        read_ref(ref)
    lines = [line.split() for line in text.splitlines() if line.strip()]
    require(lines[0] == contract.get("exact_header_tokens"), "Exact root-reviewed source map header")
    positions, distance, reported = [], [], []
    for row in lines[1:]:
        require(len(row) == 3, "Exactly three source map columns")
        require(row[0].lstrip("+").isdigit(), "Integer source physical position")
        pos = int(row[0]) + (1 if contract["position_basis"] == "absolute_bp0" else 0)
        require(pos >= 1 and (not positions or pos > positions[-1]), "Strictly increasing bp1 map positions")
        if contract["source_format"] == "pos_chr_cM":
            require(row[1] in (contract["chromosome"], "chr" + contract["chromosome"]), "Exact chromosome source map rows")
        else:
            rate = float(row[1])
            require(math.isfinite(rate) and rate >= 0, "Finite nonnegative cM/Mb map rates")
            reported.append(rate)
        cm = float(row[2])
        require(math.isfinite(cm) and cm >= 0 and (not distance or cm >= distance[-1]),
                "Finite nondecreasing source genetic positions in cM")
        positions.append(pos); distance.append(cm)
        require(len(positions) <= 200000, "Source map row cap")
    require(len(positions) >= 2 and positions[0] <= BEGIN+1 and positions[-1] >= END+1,
            "Public map must bracket both regional boundaries; no extrapolation")
    if reported:
        tolerance = contract.get("rate_rounding_tolerance_cM_per_Mb")
        require(type(tolerance) in (int, float) and math.isfinite(tolerance) and 0 <= tolerance <= 0.02,
                "Explicit reviewed source rate-rounding tolerance required")
        for i in range(len(positions)-1):
            slope = (distance[i+1]-distance[i])*1e6/(positions[i+1]-positions[i])
            require(abs(reported[i]-slope) <= tolerance, "Rate and cumulative map units disagree")
    return positions, distance

def crop_piecewise_map_v1(positions, distance):
    """Preserve the public piecewise-linear cM curve; shift origin only."""
    left, right = BEGIN+1, END+1
    def at(pos):
        i = bisect.bisect_right(positions, pos) - 1
        if i == len(positions)-1:
            require(pos == positions[-1], "No map extrapolation")
            return distance[-1]
        require(i >= 0, "No map extrapolation")
        return distance[i] + (distance[i+1]-distance[i]) * ((pos-positions[i])/(positions[i+1]-positions[i]))
    knots = [left] + [p for p in positions if left < p < right] + [right]
    zero = at(left)
    cumulative = [at(p)-zero for p in knots]
    require(cumulative[-1] > 0, "Positive genetic distance over the admitted region")
    rates = [(cumulative[i+1]-cumulative[i])*1e6/(knots[i+1]-knots[i])
             for i in range(len(knots)-1)]
    rates.append(rates[-1])  # Terminal sentinel rate is unused past final boundary.
    out = "Position(bp) Rate(cM/Mb) Map(cM)\n"
    out += "".join(f"{p-BEGIN} {r:.17g} {cm:.17g}\n"
                   for p, r, cm in zip(knots, rates, cumulative))
    return out.encode(), dict(source_boundary_bp1=[left, right], local_boundary_bp1=[1, END-BEGIN+1],
        source_cM_at_left=zero, regional_distance_cM=cumulative[-1],
        regional_distance_Morgans=cumulative[-1]/100.0,
        native_source_knots_inside_region=len(knots)-2, output_rows=len(knots),
        operation="piecewise-linear interpolation only at boundaries; preserve all interior source knots; subtract constant genetic origin",
        uniform_map_substitution=False, source_positions_or_distances_rescaled=False,
        source_map_brackets_full_requested_interval=True,
        source_build_and_units_depend_on_root_semantic_receipt=True)

def qualify_candidate_map_bytes_v1(raw, semantics, asset):
    text, identity = inspect_candidate_map_bytes_v1(raw, asset)
    positions, distance = parse_map_text_v1(text, semantics)
    output, crop = crop_piecewise_map_v1(positions, distance)
    return output, dict(identity, source_semantics_qualified=True, **crop,
        map_sha256=sha(output), sites_and_haps_map_join_qualified=False,
        inference_or_model_executed=False)
