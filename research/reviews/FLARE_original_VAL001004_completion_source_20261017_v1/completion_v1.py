"""Complete four missing original FLARE VAL cells. No native runner or admission here.

Reuse original converter/run_v2/seal_native source unchanged. This module supplies
inactive config templates, authenticates the four-cell pretruth barrier and adds
raw rounded ANP NLL plus five-parent descriptive statistics to original hard-call metrics.
VAL000 truth was already exposed and remains explicitly exploratory.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import copy,gzip,hashlib,json,math,re,statistics,zipfile
HERE=Path(__file__).resolve().parent
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
VAL=tuple("american_c3_val_%03d"%i for i in range(5))
MISSING=VAL[1:]
CLASSES=("EUR","EAS","AFR")
IDENTITY=("record_ids","sample_nodes","sample_indices","positions_bp0","parent_id","class_order")
RAW_FIELDS=("native_ANP_probabilities_C3_order","native_ANP_decimal_strings","native_ancestry_order",
 "native_ancestry_indices","native_hard_calls_C3","native_marker_positions_bp0")
METHOD="FLARE_native_v0.6.0_tag_CURRENT_RECONSTRUCTION"
PROJECTION="nearest_native_marker_midpoints_ties_right_edges_to_domain_v1"
COMMIT="ec0ad295aeebfacf10d8cbfb6486ab78981de1b1"
OPTIONS={"probs":True,"em":True,"update-p":False,"array":False,"min-maf":.005,
 "min-mac":50,"gen":10,"panel-probs":False,"nthreads":4,"seed":-99999}

def require(ok,message):
    if not ok:raise ValueError(message)

def _tools():
    from arggnn.american_c3_common_v1 import read,bound,ref
    return read,bound,ref

def _local(name):
    return json.loads((HERE/name).read_text())

def _local_ref(name):
    return _tools()[2](str((HERE/name).relative_to(ROOT)))

def _sources():
    _,_,ref=_tools()
    for row in _local("source_pins.v1.json")["refs"]:
        require(ref(row["path"])==row,"Exact unchanged FLARE/completion source dependency")

def _context(record,parent_id=None):
    read,_,_=_tools();ctx=read(record);known=_local("original_remaining_inputs.v1.json")["known_inputs"]
    require(ctx.get("schema")=="FLARE_original_VAL_completion_ROOT_context_v1"
        and ctx.get("root_authorizer")=="root_agent" and ctx.get("fixed_VAL000_basis_ref")==_local_ref("VAL000_basis.v1.json")
        and set(ctx["targets"])==set(MISSING)
        and set(ctx["remaining_seal_execution_evidence_refs"])==set(MISSING)
        and ctx.get("remaining_VAL_truth_not_previously_opened") is True
        and ctx.get("algorithm_options_unchanged_after_VAL000") is True,
        "Allfour genuine original targets required;VAL004unbound and earliertruth exposure cannot be concealed")
    parents=MISSING if parent_id is None else (parent_id,)
    require(all(parent in MISSING and isinstance(ctx["targets"][parent],dict)for parent in parents),
        "Selected original targets must genuinely complete;VAL004unbound cannot be substituted")
    for parent in parents:
        row=ctx["targets"][parent]
        if known[parent] is not None:
            require(all(row.get(k)==v for k,v in known[parent].items()),"Exact original C3 observation/focals/counts")
        require(row.get("parent_id")==parent and row.get("haplotypes")==1600
            and type(row.get("input_sites")) is int and row["input_sites"]>0,"Complete original1600haplotype SNP count")
        known_sim=_local("original_remaining_inputs.v1.json")["known_simulation_completion_refs"][parent]
        require(row.get("simulation_completion_ref")==known_sim,"Exact already completed originalVALsimulation metadata;no rerun")
        simulation=read(known_sim)
        m=read(row["observation_ref"]);prep=read(row["native_preparation_ref"]);simcfg=read(row["simulation_config_ref"])
        require(simulation.get("status")=="completed_C3_simulation_reconstruction"
            and simulation.get("parent_id")==parent and m.get("source_observation_ref")==simulation.get("observation_ref"),
            "Original inferred observation descends from exact originalVALsimulation")
        index=int(parent[-3:])
        require(m.get("schema")=="american_c3_observed_v1" and m.get("parent_id")==parent and m.get("split")=="val"
            and m.get("sequence_length_bp")==30000000 and m.get("class_order")==list(CLASSES)
            and m.get("counts_haploid")=={"AFR":200,"EUR":200,"ASIA":200,"ADMIX":1000}
            and m.get("query_ancestry_truth_in_inputs") is False
            and m.get("genealogy_kind")=="inferred_tsinfer_tsdate_CURRENT_RECONSTRUCTION_not_simulated_true_ARG"
            and prep.get("status")=="completed_native_C3_path_preparation" and prep.get("parent_id")==parent
            and prep.get("observation_ref")==row["observation_ref"] and prep.get("focals_ref")==row["focal_registry_ref"]
            and prep.get("query_focal_count")==10000 and prep.get("query_ancestry_truth_opened") is False,
            "Exact original inferred SNPs and10000native query focals;no alternate tree/focal namespace")
        require(simcfg.get("schema")=="american_c3_simulation_reconstruction_v1" and simcfg.get("parent_id")==parent
            and simcfg.get("split")=="val" and simcfg.get("sequence_length_bp")==30000000
            and simcfg.get("ancestry_seed")==31001+index and simcfg.get("mutation_seed")==1031001+index
            and simcfg.get("recombination_rate")==1e-8 and simcfg.get("mutation_rate")==1.25e-8
            and simcfg.get("sample_unit")=="haploid"
            and simcfg.get("sample_counts")=={"AFR":200,"EUR":200,"ASIA":200,"ADMIX":1000},
            "Same original simulation seeds/map/sample units")
        roster=read(m["roster_ref"])["samples"]
        require(len(roster)==1600 and [r["sample_index"] for r in roster]==list(range(1600))
            and [r["observed_reference_class"] for r in roster]==[2]*200+[0]*200+[1]*200+[None]*1000
            and [r["role"] for r in roster]==["pure_source_reference"]*600+["query"]*1000,
            "Unchanged EUR/EAS/AFR labels;600reference and1000query original columns")
    return ctx

def inactive_stage_configs_v1(context_ref,parent_id,*,conversion_ref=None,native_run_ref=None):
    """Return only inactive copies of exact VAL000 source configs; never run native code."""
    require(parent_id in MISSING,"Only missing originalVAL001..004;VAL000native rerun forbidden")
    _sources();ctx=_context(context_ref,parent_id=parent_id);row=ctx["targets"][parent_id]
    basis=_local("VAL000_basis.v1.json")["configurations"]
    convert=copy.deepcopy(basis["conversion_config"]);native=copy.deepcopy(basis["native_config"]);seal=copy.deepcopy(basis["seal_config"])
    for cfg in (convert,native,seal):
        cfg["execution_enabled"]=False;cfg["registration_ref"]=None;cfg["count_manifest_ref"]=row["count_manifest_ref"]
        cfg["focal_registry_ref"]=row["focal_registry_ref"];cfg["max_sites_bound"]=row["input_sites"]
        cfg["external_truth_ref"]=None
    convert.update(observation_ref=row["observation_ref"],simulation_config_ref=row["simulation_config_ref"],max_sites=row["input_sites"])
    native.update(conversion_ref=conversion_ref,max_sites_bound=row["input_sites"])
    # Keep the frozen native options/JAR/Java/heap. Native count-derived bounds are recalculated by author wrapper after conversion.
    markers=row["input_sites"]
    if conversion_ref is not None:
        converted=_tools()[0](conversion_ref)
        require(converted.get("status")=="completed_C3_FLARE_observation_conversion"
            and converted.get("parent_id")==parent_id and converted.get("observation_ref")==row["observation_ref"]
            and converted.get("focal_registry_ref")==row["focal_registry_ref"] and converted.get("native_options")==OPTIONS
            and converted.get("native_source_commit")==COMMIT and converted.get("input_sites")==row["input_sites"]
            and converted.get("query_ancestry_truth_opened") is False
            and type(converted.get("native_expected_markers"))is int
            and 0<converted["native_expected_markers"]<=row["input_sites"],
            "Actual completed unmodified conversion metadata for native marker/output count")
        markers=converted["native_expected_markers"]
    require(native_run_ref is None or conversion_ref is not None,"Native seal binds its actual completed conversion")
    native["max_native_vcf_bytes"]=_native_output_bound(markers)["vcf_bytes"]
    native["output_cap_bytes"]=_native_output_bound(markers)["total_bytes"]
    convert["output_cap_bytes"]=_conversion_bound(row["input_sites"])
    seal.update(native_run_ref=native_run_ref,max_markers=markers,prediction_ref=None,
        write_derived_marker_tracts=False,max_tract_records=0)
    return dict(conversion=convert,native=native,seal=seal,ROOT_actual_capacity_and_approval_still_required=True)

def _gzip_bound(raw):return raw+(raw+99)//100+65536

def _conversion_bound(sites):
    raw=lambda samples:4194304+sites*(256+4*samples)
    return _gzip_bound(raw(300))+_gzip_bound(raw(500))+16*sites+16777216

def _native_output_bound(markers):
    vcf=_gzip_bound(4194304+markers*(256+64*500))
    return dict(vcf_bytes=vcf,total_bytes=vcf+2*1048576+3*16777216+1073741824)

def _load_npz(record,cap):
    import numpy as np
    _,bound,_=_tools();path=bound(record)
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        require(len(names)==len(set(names)) and sum(i.file_size for i in z.infolist())<=cap
            and all(n.endswith(".npy") and "/"not in n for n in names),"Bounded unique flat NPZ")
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def _missing_keys(rows):
    require(isinstance(rows,dict) and set(rows)==set(MISSING)
        and all(isinstance(r,dict) for r in rows.values()),"Allfour original remaining seals required before their truth")
    return rows

def _header_ancestry(native,ancestry_order):
    """Authenticate author header labels/indexes; never infer labels from ANP magnitude."""
    _,bound,_=_tools();mapping=None;seen=False;total=0
    with gzip.open(bound(native["native_output_refs"][".anc.vcf.gz"]),"rt",encoding="ascii")as stream:
        for _ in range(1024):
            line=stream.readline(65537);total+=len(line)
            require(line and len(line)<=65536 and total<=1048576,"Bounded complete native VCFheader")
            if line.startswith("##ANCESTRY="):
                match=re.fullmatch(r"##ANCESTRY=<([^>]+)>\n?",line)
                require(match is not None and mapping is None,"Exactly one author ancestry header")
                parts=[part.split("=")for part in match.group(1).split(",")]
                require(len(parts)==3 and all(len(part)==2 for part in parts),"Exact3native ancestry header entries")
                mapping={name:int(index)for name,index in parts}
                require(set(mapping)==set(CLASSES) and set(mapping.values())==set(range(3)),
                    "Native EUR/EAS/AFR labels indexed0,1,2 without population relabelling")
            elif line.startswith("#CHROM"):
                seen=True;break
            else:require(line.startswith("##"),"Header must precede every native data record")
    require(seen and mapping is not None and [next(name for name,index in mapping.items()if index==i)for i in range(3)]==ancestry_order,
        "Raw posterior ancestryorder exactly matches authenticated native header")
    return mapping

def _cell(parent,seal_ref,ctx,cap):
    import numpy as np
    from arggnn.american_c3_evaluate_v1 import registry,verify_prediction
    read,_,_=_tools();basis=_local("VAL000_basis.v1.json");seal=read(seal_ref)
    require(seal.get("status")=="sealed_C3_native_FLARE_calls_and_raw_posteriors"
        and seal.get("parent_id")==parent and seal.get("method")==METHOD and seal.get("projection")==PROJECTION
        and seal.get("class_order")==list(CLASSES) and seal.get("records")==10000
        and seal.get("native_AN1_AN2_calls_used_unchanged") is True
        and seal.get("one_hot_is_native_call_transport") is True and seal.get("common_focal_registry_unchanged") is True
        and seal.get("rounded_ANP_renormalized") is False and seal.get("raw_rounded_ANP_modified") is False
        and seal.get("native_VCF_modified") is False and seal.get("query_ancestry_truth_opened") is False,
        "Original unmodified author hard calls and separately preserved raw rounded ANP")
    native=read(seal["native_run_ref"]);converted=read(native["conversion_ref"]);pin=basis["configurations"]["native_config"]
    require(native.get("status")=="completed_C3_native_FLARE" and native.get("parent_id")==parent
        and native.get("source_commit")==COMMIT and native.get("native_options")==OPTIONS
        and native.get("native_version")==basis["native_identity"]["reported_version"]
        and native.get("class_order")==list(CLASSES) and native.get("focal_registry_ref")==seal["focal_registry_ref"]
        and seal.get("conversion_ref")==native["conversion_ref"]
        and seal.get("raw_native_output_refs")==native["native_output_refs"]
        and converted.get("status")=="completed_C3_FLARE_observation_conversion" and converted.get("parent_id")==parent
        and converted.get("common_focal_registry_unchanged")is True
        and converted.get("native_MAC_threshold")==50
        and seal.get("native_markers")==converted.get("native_expected_markers")
        and native.get("jar_ref")==pin["jar_ref"] and native.get("binary_provenance_ref")==pin["binary_provenance_ref"]
        and native.get("native_outputs_modified") is False and native.get("query_ancestry_truth_opened") is False
        and native.get("projection")==PROJECTION and converted.get("native_options")==OPTIONS
        and converted.get("native_source_commit")==COMMIT and converted.get("query_ancestry_truth_opened") is False
        and converted.get("reference_haplotypes")==600 and converted.get("query_haplotypes")==1000
        and converted.get("common_focal_records")==10000 and converted.get("unsupported_sites_dropped")==0,
        "Same authorcommit/JAR/native options and complete600reference/1000query conversion")
    wait=read(native["native_actual_wait_ref"]);argv=read(native["native_argv_ref"])
    require(wait.get("actual_direct_native_wait") is True and wait.get("returncode")==0
        and wait.get("owned_process_group_ESRCH") is True and wait.get("operational_failure") is None
        and argv.get("source_commit")==COMMIT and argv.get("native_options")==OPTIONS
        and "-Xmx16g"in argv["argv"] and argv["argv"][0]==pin["java_executable"]
        and native["java_version_ref"]["sha256"]==pin["java_version_output_sha256"],
        "Genuine native0/drain and unchanged Java16GiB identity/options")
    if parent in MISSING:
        row=ctx["targets"][parent]
        require(converted["observation_ref"]==row["observation_ref"]
            and seal["focal_registry_ref"]==converted["focal_registry_ref"]==row["focal_registry_ref"]
            and converted["input_sites"]==row["input_sites"],"Exact original full observed input support")
        evidence=read(ctx["remaining_seal_execution_evidence_refs"][parent])
        require(evidence.get("schema")=="FLARE_remaining_cell_ROOT_terminal_evidence_v1"
            and evidence.get("root_authorizer")=="root_agent" and evidence.get("parent_id")==parent
            and evidence.get("seal_completion_ref")==seal_ref and evidence.get("native_run_ref")==seal["native_run_ref"]
            and evidence.get("native_wait_ref")==native["native_actual_wait_ref"]
            and evidence.get("genuine_seal_direct_wait0") is True and evidence.get("all_owned_drained") is True
            and evidence.get("actual_outer_terminal_consumed0") is True
            and all(isinstance(evidence.get(k),dict) for k in ("seal_actual_wait_ref","seal_outer_terminal_ref")),
            "Future ROOT-certified genuine remaining-cell seal wait0/drain/outer0, no fabricated completion")
        # Underlying immutable OS/outer receipts must confirm the ROOT certification.
        seal_wait=read(evidence["seal_actual_wait_ref"]);terminal=read(evidence["seal_outer_terminal_ref"])
        completion_ref=evidence["seal_completion_envelope_ref"]
        if completion_ref!=seal_ref:require(read(completion_ref).get("result_ref")==seal_ref,"Actual outer completion binds exact original seal result")
        require(seal_wait.get("returncode")==0 and seal_wait.get("actual_direct_worker_wait") is True
            and seal_wait.get("all_owned_processes_and_groups_drained") is True and seal_wait.get("operational_failure") is None
            and seal_wait.get("completed_ref")==completion_ref
            and terminal.get("actual_terminal_consumed") is True
            and (terminal.get("actual_exit_code")==0 or terminal.get("actual_tool_exit_code")==0)
            and terminal.get("completed_ref")==completion_ref
            and (terminal.get("direct_wait_ref")==evidence["seal_actual_wait_ref"]
                 or terminal.get("actual_direct_wait_ref")==evidence["seal_actual_wait_ref"]),
            "Actual successful seal wait/drain/consumed outer0, never an unknown/failed completion")
    else:require(seal_ref==basis["refs"]["seal_completion"],"Reuse exact existingVAL000seal, no rerun")
    records=registry(seal["focal_registry_ref"]);prediction=_load_npz(seal["prediction_ref"],cap);calls=verify_prediction(prediction,records)
    raw=_load_npz(seal["raw_focal_posterior_ref"],cap)
    require(set(raw)==set(IDENTITY)|set(RAW_FIELDS)
        and all(np.array_equal(raw[k],records[k])for k in IDENTITY)
        and len(calls)==10000 and records["parent_id"].item()==parent
        and np.isin(records["sample_indices"],np.arange(600,1600)).all()
        and ((records["positions_bp0"]>=0)&(records["positions_bp0"]<30000000)).all()
        and np.array_equal(prediction["probabilities"],np.eye(3,dtype=np.float32)[calls]),
        "Identical original10000query identities and unchanged onehot hardcall transport")
    probabilities=raw["native_ANP_probabilities_C3_order"];hard=raw["native_hard_calls_C3"];codes=raw["native_ancestry_indices"]
    ancestry=raw["native_ancestry_order"].tolist();strings=raw["native_ANP_decimal_strings"]
    require(probabilities.dtype==np.float32 and probabilities.shape==(10000,3) and np.isfinite(probabilities).all()
        and ((probabilities>=0)&(probabilities<=1)).all() and (np.abs(probabilities.sum(-1)-1)<=.0151).all()
        and len(ancestry)==3 and set(ancestry)==set(CLASSES)
        and hard.dtype==codes.dtype==np.uint8 and hard.shape==codes.shape==(10000,)
        and (codes<3).all() and np.array_equal(hard,calls)
        and strings.dtype==np.dtype("U32") and strings.shape==(10000,)
        and raw["native_marker_positions_bp0"].dtype==np.int64
        and raw["native_marker_positions_bp0"].shape==(10000,)
        and ((raw["native_marker_positions_bp0"]>=0)&(raw["native_marker_positions_bp0"]<30000000)).all(),
        "Native hard calls stay separate from raw rounded ANP;no normalization/argmax replacement")
    header_mapping=_header_ancestry(native,ancestry)
    inverse=np.asarray([CLASSES.index(name)for name in ancestry],dtype=np.uint8)
    class_order=[ancestry.index(name)for name in CLASSES]
    require(np.array_equal(inverse[codes],hard),"Exact native ancestrylabel-to-C3 mapping")
    for i,text in enumerate(strings):
        parsed=np.asarray([float(x)for x in str(text).split(",")],dtype=np.float32)
        require(parsed.shape==(3,) and np.array_equal(parsed[class_order],probabilities[i]),"Raw decimal ANP preserved exactly in C3order")
    return dict(seal_ref=seal_ref,seal=seal,converted=converted,records=records,calls=calls,raw=probabilities,
        native_header_ancestry_mapping=header_mapping,
        coordinate_join="original_bp0_to_VCF_POS_plus1;native_POS_minus1_exactgrid;focals_bp0;nearest_midpoints_ties_right_domain_edges")

def freeze_remaining_predictions_v1(context_ref,remaining_seal_refs,*,max_npz_uncompressed_bytes,writer,runtime_guard):
    """No truth resolver. Allfour new cells plus originalVAL000payloads authenticate."""
    _missing_keys(remaining_seal_refs);require(type(max_npz_uncompressed_bytes)is int and max_npz_uncompressed_bytes>0
        and callable(runtime_guard) and callable(getattr(writer,"json",None)),"Explicit NPZbound and ROOTwriter/resource callbacks")
    _sources();ctx=_context(context_ref);basis=_local("VAL000_basis.v1.json");cells=[]
    for parent in VAL:
        runtime_guard("FLARE_original_remaining_pretruth_payload_check")
        seal_ref=basis["refs"]["seal_completion"]if parent==VAL[0]else remaining_seal_refs[parent]
        cell=_cell(parent,seal_ref,ctx,max_npz_uncompressed_bytes)
        cells.append(dict(parent_id=parent,seal_completion_ref=seal_ref,
            prediction_ref=cell["seal"]["prediction_ref"],raw_focal_posterior_ref=cell["seal"]["raw_focal_posterior_ref"],
            focal_registry_ref=cell["seal"]["focal_registry_ref"],
            native_header_ancestry_mapping=cell["native_header_ancestry_mapping"],coordinate_join=cell["coordinate_join"]))
    return writer.json("FLARE_original_remaining4_pretruth_barrier.json",dict(
        status="sealed_FLARE_original_remaining4_before_their_VAL_truth_v1",context_ref=context_ref,
        source_manifest_ref=_local_ref("source_manifest.v1.json"),fixed_VAL000_basis_ref=_local_ref("VAL000_basis.v1.json"),
        cells=cells,max_npz_uncompressed_bytes=max_npz_uncompressed_bytes,
        remaining_original_parents=list(MISSING),all4_remaining_sealed_before_their_truth=True,
        VAL000_truth_previously_exposed=True,VAL000_exploratory_retained=True,VAL000_native_rerun=False,TEST_deferred=True))

def _truth(record,records,cap):
    import numpy as np
    value=_load_npz(record,cap)
    require(set(value)=={"intervals_bp","query_sample_indices","source_classes_0based","parent_id","class_order"}
        and np.array_equal(value["parent_id"],records["parent_id"])
        and np.array_equal(value["class_order"],records["class_order"]),"Exact external originalC3truth parent/classes")
    intervals,samples,labels=value["intervals_bp"],value["query_sample_indices"],value["source_classes_0based"]
    require(intervals.dtype==np.float64 and intervals.ndim==2 and intervals.shape[1:]==(2,) and len(intervals)>0
        and np.isfinite(intervals).all() and intervals[0,0]==0 and intervals[-1,1]==30000000
        and (intervals[:,1]>intervals[:,0]).all() and np.array_equal(intervals[:-1,1],intervals[1:,0])
        and samples.dtype==np.int64 and np.array_equal(samples,np.arange(600,1600,dtype=np.int64))
        and labels.dtype==np.uint8 and labels.shape==(len(intervals),1000) and (labels<3).all(),"Fullhalfopen30Mbtruth/all1000queries")
    return labels[np.searchsorted(intervals[:,1],records["positions_bp0"],side="right"),records["sample_indices"]-600].astype(np.int64)

def _summary(values):
    require(len(values)==5 and all(math.isfinite(x)for x in values),"Allfiveparent values required;no partial pooled result")
    return dict(per_parent=dict(zip(VAL,values)),equal_parent_mean=statistics.mean(values),
        sample_SD=statistics.stdev(values),parents=5,SD_ddof=1)

def score_five_original_VAL_v1(barrier_ref,*,authority_ref,max_npz_uncompressed_bytes,max_truth_uncompressed_bytes,
        writer,resolve_val_truth,runtime_guard):
    """Future separately ROOT-authorized score only, no native rerun and noTESTcapability."""
    read,_,_=_tools();barrier=read(barrier_ref);authority=read(authority_ref)
    require(all(type(x)is int and x>0 for x in (max_npz_uncompressed_bytes,max_truth_uncompressed_bytes))
        and callable(resolve_val_truth) and callable(runtime_guard) and callable(getattr(writer,"json",None)),
        "Separate bounded VAL-only writer/resource capabilities")
    now=datetime.now(timezone.utc);created=datetime.fromisoformat(authority["created_at_utc"]);expires=datetime.fromisoformat(authority["expires_at_utc"])
    require(created.tzinfo and expires.tzinfo and created<=now<expires
        and authority.get("schema")=="FLARE_original_remaining_VAL_ROOT_score_authority_v1"
        and authority.get("root_authorizer")=="root_agent" and authority.get("active")is True
        and authority.get("barrier_ref")==barrier_ref and authority.get("source_manifest_ref")==_local_ref("source_manifest.v1.json")
        and authority.get("context_ref")==barrier["context_ref"] and authority.get("all4remaining_seals_before_truth")is True
        and authority.get("max_npz_uncompressed_bytes")==max_npz_uncompressed_bytes
        and authority.get("max_truth_uncompressed_bytes")==max_truth_uncompressed_bytes
        and barrier.get("max_npz_uncompressed_bytes")==max_npz_uncompressed_bytes
        and authority.get("VAL001004_truth_authorized")is True and authority.get("VAL000_supplemental_raw_NLL_authorized")is True
        and authority.get("native_rerun_authorized")is False and authority.get("TEST_authorized")is False,
        "Separate current ROOTauthority;retain earlierVAL000exposure;no unexposed5parent claim")
    require(barrier.get("status")=="sealed_FLARE_original_remaining4_before_their_VAL_truth_v1"
        and barrier.get("source_manifest_ref")==_local_ref("source_manifest.v1.json")
        and barrier.get("fixed_VAL000_basis_ref")==_local_ref("VAL000_basis.v1.json")
        and barrier.get("all4_remaining_sealed_before_their_truth")is True
        and barrier.get("VAL000_truth_previously_exposed")is True and barrier.get("VAL000_exploratory_retained")is True
        and barrier.get("VAL000_native_rerun")is False and len(barrier["cells"])==5,"Exact descriptive fiveparent completion barrier")
    by_parent={r["parent_id"]:r for r in barrier["cells"]}
    require(set(by_parent)==set(VAL),"All5unique originalparents")
    _sources();ctx=_context(barrier["context_ref"]);basis=_local("VAL000_basis.v1.json");cells={}
    # Authenticate ALLremaining predictions/rawANP before the first missing-parent truth resolver.
    for parent in VAL:
        runtime_guard("FLARE_all5_original_payloads_before_any_new_truth")
        cell=_cell(parent,by_parent[parent]["seal_completion_ref"],ctx,max_npz_uncompressed_bytes)
        require(all(by_parent[parent][k]==cell["seal"][k]for k in ("prediction_ref","raw_focal_posterior_ref","focal_registry_ref"))
            and by_parent[parent]["native_header_ancestry_mapping"]==cell["native_header_ancestry_mapping"]
            and by_parent[parent]["coordinate_join"]==cell["coordinate_join"],"Unchanged exact pretruthcells and header/coordinatejoin")
        cells[parent]=cell
    import numpy as np
    from arggnn.american_c3_evaluate_v1 import metrics
    results={};score_refs={}
    for parent in VAL:
        runtime_guard("FLARE_external_original_native_focal_score_after_remaining4barrier")
        cell=cells[parent]
        sim_ref=(_local("VAL000_upstream.v1.json")["simulation_completion_ref"]
            if parent==VAL[0]else ctx["targets"][parent]["simulation_completion_ref"])
        simulation=read(sim_ref);observed=read(cell["converted"]["observation_ref"])
        require(simulation.get("status")=="completed_C3_simulation_reconstruction"
            and simulation.get("parent_id")==parent and observed.get("source_observation_ref")==simulation.get("observation_ref"),
            "Truth derives from exact original inferred-observation ancestor")
        truth_ref=resolve_val_truth(parent)
        require(truth_ref==simulation["private_truth_ref"],"Exact original sameparent VALtruth handle")
        labels=_truth(truth_ref,cell["records"],max_truth_uncompressed_bytes)
        confusion=np.zeros((3,3),dtype=np.int64);np.add.at(confusion,(labels,cell["calls"]),1)
        result=metrics(confusion)
        if parent==VAL[0]:require(result==basis["VAL000_prior_score"]["result"],"OriginalVAL000hardcall metrics unchanged")
        raw_true=cell["raw"][np.arange(10000),labels].astype(np.float64);zero=int(np.count_nonzero(raw_true==0))
        result["raw_rounded_ANP_NLL"]=dict(mean_nats=None if zero else float(-np.log(raw_true).mean()),
            positive_infinity=bool(zero),zero_true_class_probabilities=zero,
            rounded_native_ANP=True,renormalized=False,probability_floor=False,
            one_hot_transport_used_for_NLL=False,hard_call_argmax_replacement=False)
        results[parent]=result
        score_refs[parent]=writer.json("FLARE_"+parent+"_original_common_metrics_and_raw_NLL.json",dict(
            status="completed_FLARE_original_VAL_common_hardcalls_and_raw_NLL_v1",parent_id=parent,method=METHOD,
            barrier_ref=barrier_ref,seal_completion_ref=cell["seal_ref"],prediction_ref=cell["seal"]["prediction_ref"],
            raw_focal_posterior_ref=cell["seal"]["raw_focal_posterior_ref"],focal_registry_ref=cell["seal"]["focal_registry_ref"],
            external_truth_ref=truth_ref,records=10000,class_order=list(CLASSES),projection=PROJECTION,result=result,
            native_header_ancestry_mapping=cell["native_header_ancestry_mapping"],coordinate_join=cell["coordinate_join"],
            VAL000_previously_exposed=parent==VAL[0],native_rerun=False,TEST_deferred=True))
    require(set(results)==set(VAL),"No fiveparent aggregate from partialcells")
    rates={}
    for key in ("macro_F1","accuracy"):
        rates[key]=_summary([100*results[p][key]for p in VAL])
    for name in CLASSES:
        for key in ("precision","recall","F1"):
            rates[name+"_"+key]=_summary([100*results[p]["per_class"][name][key]for p in VAL])
    nll=[results[p]["raw_rounded_ANP_NLL"]for p in VAL]
    nll_summary=(dict(equal_parent_mean_nats=None,positive_infinity=True,sample_SD_nats=None,
        SD_status="undefined_with_nonfinite_parent_NLL")if any(r["positive_infinity"]for r in nll)
        else dict(equal_parent_mean_nats=statistics.mean(r["mean_nats"]for r in nll),positive_infinity=False,
            sample_SD_nats=statistics.stdev(r["mean_nats"]for r in nll)))
    return writer.json("FLARE_original5VAL_descriptive_aggregate.json",dict(
        status="completed_FLARE_original5VAL_descriptive_completion_v1",method=METHOD,barrier_ref=barrier_ref,
        source_manifest_ref=_local_ref("source_manifest.v1.json"),score_authority_ref=authority_ref,
        per_parent=results,score_refs=score_refs,rates_percent=rates,raw_rounded_ANP_NLL=nll_summary,
        equal_parent_not_pooled=True,SD_ddof=1,uncertainty_unit="5simulationparents;not_trainingseeduncertainty",
        completed_missing_original_cells=list(MISSING),native_options=OPTIONS,source_commit=COMMIT,
        VAL000_truth_previously_exposed=True,VAL000_exploratory_retained=True,VAL000_native_rerun=False,
        all4remaining_predictions_before_their_truth=True,all5_unexposed_claimed=False,
        validation_development_only=True,independent_TEST_confirmation=False,TEST_deferred=True,
        author_Table1_exact_reproduction=False))
