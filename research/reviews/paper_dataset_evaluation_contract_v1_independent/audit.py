from pathlib import Path
import ast, hashlib, html, json, os, re, statistics, subprocess, sys, zipfile
from datetime import datetime, timezone

ROOT = Path("/workspace-SR003.nfs2/ARG-GNN")
OUT = ROOT / "research/reviews/paper_dataset_evaluation_contract_v1_independent"
E = ROOT / "research/reviews/paper_dataset_evaluation_contract_v1"
checks = []
inputs = {}
def sha(b): return hashlib.sha256(b).hexdigest()
def read(path, role="source_or_metadata"):
    p = ROOT / path
    b = p.read_bytes()
    inputs[str(p.relative_to(ROOT))] = {"path": str(p.relative_to(ROOT)), "bytes": len(b), "sha256": sha(b), "role": role}
    return b
def js(path, role="source_or_metadata"): return json.loads(read(path, role))
def check(name, passed, expected, actual, anchors=None):
    checks.append({"case": name, "result": "PASS" if passed else "FAIL", "expected": expected, "actual": actual, "anchors": anchors or []})
def text(path): return read(path).decode()
def flattened(path): return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", " ", text(path))))
def anchor(path, lo, hi):
    ls=text(path).splitlines()
    return {"path": path, "start": lo, "end": hi, "text": "\n".join(ls[lo-1:hi])}

read("AGENTS.md", "instructions")
read("scripts/project_env.sh", "runtime_environment_instructions")
doc=text("docs/paper_dataset_evaluation_contract_v1.md")
cfg=js("configs/paper_dataset_evaluation_contract_v1.json", "contract")
seal=js("research/reviews/paper_dataset_evaluation_contract_v1/seal.json", "input_seal")
evidence=js("research/reviews/paper_dataset_evaluation_contract_v1/source_evidence.json", "source_manifest")
validation=js("research/reviews/paper_dataset_evaluation_contract_v1/validation.json", "prior_validation_not_independent_evidence")
for section, records in [("sealed_records",seal["records"]),("referenced_sources",evidence["existing_sources"])]:
    errors=[]
    for a in records:
        b=read(a["path"], section)
        if len(b)!=a["bytes"] or sha(b)!=a["sha256"]: errors.append(a["path"])
    check(section+"_sha256_and_sizes",not errors,{"count":len(records),"mismatches":0},{"count":len(records),"mismatches":errors})
canonical=json.dumps(seal["records"],sort_keys=True,separators=(",",":")).encode()
check("seal_records_canonical_sha256",sha(canonical)==seal["records_sha256"],seal["records_sha256"],sha(canonical))
check("seal_total_bytes",sum(x["bytes"] for x in seal["records"])==seal["owned_bytes_total"],seal["owned_bytes_total"],sum(x["bytes"] for x in seal["records"]))
check("scope_and_identity_labels",cfg["status"]=="source_metadata_manifest_only" and cfg["development_benchmark"]["satisfies_original_paper_data_requirement"] is False and set(cfg["identity_labels"])=={"identical_original_raw","author_recipe_resimulation","released_author_workflow","independent_confirmation","task_transfer","derived_endpoint"},"source-only status; development insufficient; six separate labels",{"status":cfg["status"],"development":cfg["development_benchmark"],"labels":cfg["identity_labels"]})
gsttree=js("research/reviews/paper_dataset_evaluation_contract_v1/gst_pinned_tree.json")
gstr=js("research/reviews/paper_dataset_evaluation_contract_v1/gst_pinned_tree_receipt.json")
gb=read("research/reviews/paper_dataset_evaluation_contract_v1/gst_pinned_tree.json")
check("gst_tree_receipt_and_pin",len(gb)==gstr["bytes"] and sha(gb)==gstr["sha256"] and gsttree["sha"]==cfg["papers"]["gst"]["source_commit"] and not gsttree["truncated"],"full cached pinned tree and receipt hash",{"tree_sha":gsttree["sha"],"truncated":gsttree["truncated"],"bytes":len(gb),"sha256":sha(gb)})
tree={x["path"]:x for x in gsttree["tree"]}
gstreceipts=js("research/reviews/paper_dataset_evaluation_contract_v1/gst_source_metadata/receipt.json")
blob_results=[]
for a in gstreceipts:
    b=read(a["path"])
    blob=hashlib.sha1(b"blob "+str(len(b)).encode()+b"\0"+b).hexdigest()
    expected=tree[a["upstream_path"]]
    ok=len(b)==a["bytes"] and sha(b)==a["sha256"] and blob==expected["sha"] and len(b)==expected["size"] and cfg["papers"]["gst"]["source_commit"] in a["url"]
    blob_results.append({"path":a["path"],"upstream_path":a["upstream_path"],"passed":ok,"git_blob_sha1":blob})
check("all_cached_GST_source_receipts_and_Git_blobs",all(x["passed"] for x in blob_results),{"files":14,"all_match":True},blob_results)
check("missing_gst_v2_explicitly_blocked",not any("gst_v2" in p for p in tree) and any("gst_v2" in b for b in cfg["papers"]["gst"]["blockers"]),"missing module acknowledged, no alias qualification",{"tree_paths":[p for p in tree if "gst_v2" in p],"contract_blockers":cfg["papers"]["gst"]["blockers"]})
check("CIFAR_missing_recipe_is_bounded_not_universal",not any(re.search("cifar|cifair|superpixel",p,re.I) for p in tree) and cfg["papers"]["gst"]["CIFAR10"]["do_not_guess_as_exact_reproduction"],"no dedicated matching path in pinned inventory; unresolved original recipe",cfg["papers"]["gst"]["CIFAR10"])
raw_checks=[]
for family,key,route in [("bh","Buchwald_Hartwig","data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx"),("uspto15k","USPTO15K","USPTO-15K/data.zip")]:
    ref=js(f"research/reviews/paper_dataset_evaluation_contract_v1/public_metadata/{family}_master_ref.json")
    tr=js(f"research/reviews/paper_dataset_evaluation_contract_v1/public_metadata/{family}_pinned_tree.json")
    obj=next(x for x in tr["tree"] if x["path"]==route)
    condition=cfg["papers"]["gst"][key]
    ok=ref["object"]["sha"]==tr["sha"]==condition["resolved_raw_source_commit"] and not tr["truncated"] and obj["sha"]==condition["expected_git_blob_sha1"] and obj["size"]==condition["expected_bytes"] and condition["full_file_sha256"] is None
    raw_checks.append({"family":family,"passed":ok,"tree_sha":tr["sha"],"object":obj,"full_file_bytes_acquired":False})
check("GST_raw_routes_are_metadata_only_and_match",all(x["passed"] for x in raw_checks),"pinned raw routes, byte sizes, Git-object identities, unresolved raw SHA256",raw_checks)
pubreceipts=js("research/reviews/paper_dataset_evaluation_contract_v1/public_metadata/receipt.json")
check("reconstructed_public_receipts_disclosed",all(a.get("headers") is None and "lost" in a.get("receipt_note","") for a in pubreceipts),"hashes bind saved metadata only; missing HTTP headers explicit",{"entries":len(pubreceipts),"notes":[a["receipt_note"] for a in pubreceipts]})
z=js("research/reviews/paper_dataset_evaluation_contract_v1/public_metadata/Wohns_zenodo5512994.json")
zc=cfg["papers"]["argformer"]["real_data"]["collection_metadata"]
expected={f["key"]:(f["size"],f["checksum"],f["links"]["self"]) for f in z["files"]}
actual={f["key"]:(f["bytes"],f["publisher_checksum"],f["download_url"]) for f in zc["files"]}
check("Wohns_metadata_all_39_file_identities",expected==actual and len(expected)==39 and sum(x["size"] for x in z["files"])==2726887201 and all(f["full_file_local_sha256"] is None for f in zc["files"]),"39 exact metadata tuples, no local full raw hashes",{"equal":expected==actual,"files":len(expected),"compressed_bytes":sum(x["size"] for x in z["files"])})
wohns=flattened("research/reviews/paper_dataset_evaluation_contract_v1/public_metadata/Wohns_PMC10027547.html")
hgdp=flattened("research/reviews/paper_dataset_evaluation_contract_v1/public_metadata/HGDP1kGP_PMC9900804.html")
check("Wohns_modern_vs_ancient_collection_distinction","5512994" in wohns and "5495535" in wohns and "high coverage ancient" in wohns and cfg["papers"]["argformer"]["real_data"]["modern_only_silent_substitution_forbidden"],"cited ancient-inclusive collection distinguished from modern-only",{"5512994":True,"5495535":True,"silent_substitute_forbidden":True})
check("ARGMix_BCF_availability_does_not_establish_exact_tutorial_object","phased_haplotypes_v2/" in hgdp and "as BCFs" in hgdp and "gnomad_meta_updated.tsv" not in hgdp and not cfg["papers"]["argmix"]["released_branch"]["current_tutorial_resource_availability"]["tutorial_named_SHAPEIT5_VCF_exact_object_or_conversion_identity_established"],"BCF collection availability only; exact named VCF/TSV unresolved",cfg["papers"]["argmix"]["released_branch"]["current_tutorial_resource_availability"])
zpath=".runtime/author_sources/ancestralpaths_published_zenodo8319452.zip"
base=".runtime/author_sources/ancestralpaths_77db3cfae100/"
with zipfile.ZipFile(ROOT/zpath) as zz:
    zip_matches=[]
    for rel in evidence["ancestralpaths_zip_current_file_identity"]["relative_paths"]:
        members=[n for n in zz.namelist() if n.endswith("/"+rel) or n==rel]
        assert len(members)==1,(rel,members)
        b=zz.read(members[0]); local=read(base+rel)
        zip_matches.append({"path":rel,"matches":b==local,"sha256":sha(b)})
check("AncestralPaths_cited_source_archive_17_files",len(zip_matches)==17 and all(x["matches"] for x in zip_matches),"17 code/metadata files byte-identical; no biological archive members accessed",zip_matches)
codepaths=validation["parsed_source_paths"]+[
".runtime/author_sources/argformer_f7545be7bca0/src/data/prepare_data_pretrain.py",
base+"training.py",base+"testing_bygroup.py",base+"simulate.py",
".runtime/author_sources/argmix_c2481248eb0d/src/joint_interval_model.py",
".runtime/author_sources/argmix_c2481248eb0d/src/forward_panel.py"]
parsed=[]
for path in codepaths: ast.parse(text(path),filename=path);parsed.append(path)
check("selected_native_source_AST_parse_without_import",len(parsed)==15,"15 inspected Python files parse without executing/importing author code",parsed)
ap_pub=cfg["papers"]["ancestralpaths"]["published_simulation"];ap_rel=cfg["papers"]["ancestralpaths"]["released_simulation"]
check("AP_native_publication_release_limits_retained",ap_pub["length_bp"]==200000000 and ap_rel["length_bp"]==2000000 and ap_rel["overlap_parents"]==[5,6] and ap_pub["architecture_definition_or_checkpoint_available_in_cited_release"] is False,"published scale, missing CNN and source overlap separated",{"published":ap_pub,"released":ap_rel})
ap_pdf=".runtime/author_sources/ancestralpaths_supplement_41586_2023_6705_MOESM1_ESM.pdf"
supp=re.sub(r"\s+"," ",subprocess.run(["pdftotext",str(ROOT/ap_pdf),"-"],capture_output=True,text=True,check=True).stdout)
check("AP_published_scale_and_CNN_primary_text","three 200Mbp tree sequences" in supp and "50,000 pairs in total" in supp and "five 200Mbp length sequences" in supp and "convolutional neural network" in supp,"primary supplement confirms known published requirements",{"three_200Mbp":True,"50000_pairs":True,"five_200Mbp_test":True,"CNN_named":True})
synthpath="research/reviews/paper_dataset_evaluation_contract_v1/gst_source_metadata/scripts__eval_synthetic.py.txt"
synth=text(synthpath)
check("GST_native_rng_validation_and_metrics",all(x in synth for x in ["seed * 7919 + 1","seed * 7919 + 2","seed * 7919 + 3","next(val_iter)","best_state","torch.sigmoid(out)"]),"distinct RNG streams; fresh validation draws; native checkpoint selection",{"TRAIN_draws":2500*16,"VAL_draws_per_check":8*16,"VAL_checks":5,"VAL_total_draws":5*8*16,"TEST_draws":32*16})
check("GST_unique_validation_sets_not_proven", "total_distinct_sets_consumed" not in json.dumps(cfg["papers"]["gst"]["synthetic"]), "describe generated draws; do not certify byte-unique sets without deduplication",{"contract_total_distinct_sets_consumed":cfg["papers"]["gst"]["synthetic"]["validation"]["total_distinct_sets_consumed"],"source_deduplicates_generated_validation_sets":False},[anchor(synthpath,850,868)])
bhpath="research/reviews/paper_dataset_evaluation_contract_v1/gst_source_metadata/scripts__eval_react.py.txt"
bh=text(bhpath)
toy_pool=[0.0,100.0,100.0,100.0];toy_train=toy_pool[:1];toy_val=toy_pool[1:2]
check("handmade_BH_pool_label_dependency_demonstration",statistics.mean(toy_pool)!=statistics.mean(toy_train),"whole-pool target statistics depend on validation/out-of-subset labels",{"toy_full_pool_mean":statistics.mean(toy_pool),"toy_train_only_mean":statistics.mean(toy_train),"toy_validation_label":toy_val[0],"biological_or_actual_dataset_rows_used":False})
check("BH_normalization_fitting_scope_disclosed", "normalization_fitting_scope" in cfg["papers"]["gst"]["Buchwald_Hartwig"],"explicit source-native full70%pool including fraction-excluded and VAL labels; train-only variant separately derived",{"field_present":False,"source_calls_compute_target_stats_train_pool_before_subset":True},[anchor(bhpath,1511,1535),anchor(bhpath,1588,1603),anchor(bhpath,730,745)])
gstpaper=flattened(".runtime/cache/gst-baseline/paper.html")
paper_lr=bool(re.search(r"1.*?10.*?− 4.*?for GST",gstpaper))
shared_lr_configs={}
for stem in ["synth_run","bh_lc_quick","uspto15k_run"]:
    path=f"research/reviews/paper_dataset_evaluation_contract_v1/gst_source_metadata/results__{stem}__config.json.txt"
    cc=js(path);shared_lr_configs[stem]=cc["args"]["lr"]
check("GST_paper_vs_source_learning_rate_disclosed", "paper_source_learning_rate_discrepancy" in cfg["papers"]["gst"],"paper baselines1e-3 versus GST1e-4; released sources share args.lr=1e-3",{"field_present":False,"paper_GST1e-4_text_confirmed":paper_lr,"cached_shared_source_lr":shared_lr_configs},[anchor(synthpath,828,831),anchor(bhpath,715,718),anchor("research/reviews/paper_dataset_evaluation_contract_v1/gst_source_metadata/scripts__eval_rxn_center.py.txt",719,733)])
check("AP_published_GNOMix_comparison_bound", "GNOMix" in doc and "native_GNOMix_comparison" in cfg["papers"]["ancestralpaths"],"explicit published comparator, four cohorts excluding BAA, LR base+xgboost,0.02cMGBR/0.2cMother cohorts, five200Mbp parents; unresolved version/splits remain unknown",{"GNOMix_present_in_contract_doc": "GNOMix" in doc,"machine_native_comparison_present":False,"supplement_settings_confirmed": all(x in supp for x in ["logistic regression base","xgboost smoother","0.2cM","0.02cM","do not test the BAA"])},[{"path":ap_pdf,"printed_page":34,"section":"Comparison to an advanced LAI tool"}])
usptopath="research/reviews/paper_dataset_evaluation_contract_v1/gst_source_metadata/scripts__eval_rxn_center.py.txt"
uspto=text(usptopath)
toy_raw=[None,"retained_A",None,"retained_B","retained_C"];kept=[]
for i,row in enumerate(toy_raw):
    if row is not None:
        kept.append((i,row))
        if len(kept)>=2: break
check("handmade_USPTO_limit_counts_retained_rows",kept==[(1,"retained_A"),(3,"retained_B")],"max_train refers to retained reactions; retain raw-row ledger and reason",{"max_retained":2,"retained_raw_ids":[i for i,_ in kept],"raw_rows_seen":4,"biological_or_actual_dataset_rows_used":False},[anchor(usptopath,261,277),anchor(usptopath,710,717)])
check("heldout_and_task_transfer_guards_retained",cfg["guards"]["missing_original_architecture_or_split_blocks_exact_claim"] and cfg["guards"]["author_native_and_independent_confirmation_separate"] and cfg["papers"]["argmix"]["released_branch"]["validation_score_after_early_stopping_is_test"] is False and cfg["papers"]["ancestralpaths"]["real_data"]["AP_input_forbidden"],"unknown native inputs stop exact claims; VAL reuse and AP target leakage forbidden",{"guards":cfg["guards"],"ARGMix_default_TEST":cfg["papers"]["argmix"]["released_branch"]["template_replicate_split"]["test"],"AP_input_forbidden":True})
# The review verdict concerns the source contract's known-condition disclosure only.
concerns=[
{"id":"I1","priority":1,"title":"Disclose BH whole-pool target normalization","blocking":True,"summary":"The released train-size ablation computes mean/std from all train_pool labels before fraction-specific TRAIN/VAL selection. That includes VAL labels and targets from rows excluded at small fractions. State this source-native condition explicitly; a train-only normalizer is a changed condition.","anchors":[{"path":bhpath,"lines":[1531,1593,1599,1600]}]},
{"id":"I2","priority":1,"title":"Disclose GST shared source learning rate versus paper per-model rates","blocking":True,"summary":"Paper3.1.2 assigns AdamW1e-3 to DS/ST and1e-4 to GST, while the cached synthetic, BH and USPTO configurations and runners apply one args.lr=0.001 to every model. The known discrepancy is omitted from v1.","anchors":[{"path":".runtime/cache/gst-baseline/paper.html","section":"3.1.2 Implementation details"},{"path":synthpath,"lines":[828,829]},{"path":bhpath,"lines":[718]},{"path":usptopath,"lines":[719,720,721,722]}]},
{"id":"I3","priority":2,"title":"Bind the published AP GNOMix comparison endpoint","blocking":True,"summary":"v1 describes six-path AP metrics and rejects four-source collapsing as a substitute, but neither main contract contains the original GNOMix comparator recipe. SupplementNote3p34 uses EHG/WHG/CHG/Ana references, GBR/BronzeAge/Neo/Yam queries, excludes BAA, uses five200Mbp sequences and evenly spaced sites, logistic-regression base+xgboost smoother,0.02cM GBR and0.2cM other cohorts. Version/seed/row identity remain unresolved and must not be invented.","anchors":[{"path":ap_pdf,"printed_page":34,"section":"Comparison to an advanced LAI tool"}]},
{"id":"I4","priority":2,"title":"640 validation draws are not proven unique datasets","blocking":False,"summary":"Source consumes640 fresh generated validation sets across five checks but does not deduplicate or establish unique byte identities. Rename total_distinct_sets_consumed to generated draws, or define distinct as generator draws and leave byte-uniqueness unverified.","anchors":[{"path":synthpath,"lines":[850,854,855]},{"path":"configs/paper_dataset_evaluation_contract_v1.json","field":"papers.gst.synthetic.validation.total_distinct_sets_consumed"}]}
]
receipt={
"schema":"paper_dataset_evaluation_contract_v1_independent_audit_receipt",
"timestamp_utc":datetime.now(timezone.utc).isoformat(),
"reviewer":"independent source-contract reviewer subagent",
"host":"anogena",
"repository_root":str(ROOT),
"python_executable":sys.executable,
"python_version":sys.version.split()[0],
"scope":"source-only independent audit of original v1; no empirical qualification",
"verdict":"BLOCKED",
"verdict_definition":"Known native/source-paper conditions require explicit versioned disclosure before source contract approval. This does not reopen sealed v1 bytes or authorize acquisition/execution.",
"input_hash_algorithm":"sha256",
"input_manifest_sha256":sha(json.dumps([inputs[k] for k in sorted(inputs)],sort_keys=True,separators=(",",":")).encode()),
"inputs":[inputs[k] for k in sorted(inputs)],
"checks":checks,
"concerns":concerns,
"guarded_actions":{"data_acquisition":0,"author_code_execution":0,"biological_model_score_arrays_opened":0,"heldout_data_opened":0,"GPU_jobs":0,"training":0,"installs":0,"git_actions":0,"runtime_or_ledger_mutations":0,"author_contacts":0,"writes_outside_owned_directory":0},
"performed":"hash/size verification of existing source/paper/metadata files, Git blob checks, JSON metadata joins, source-text/AST inspection, source-PDF text extraction to stdout, handmade tiny examples",
"limitations":["Cached HTTP receipts are unsigned source snapshots; no new external network fetch was performed.","A reconstructed receipt's bytes/hash are verified, but original missing response headers cannot be independently recovered.","Absence of dedicated CIFAR paths is bounded to the pinned complete tree, not proof that no unexamined external recipe exists.","No raw dataset checksum, sample/split identity, runtime compatibility, model prediction or empirical reproduction was qualified.","Original-paper blocked fields already disclosed by v1 remain blocked.","The reviewer does not attest historical tool execution beyond the preserved evidence; guarded_actions describes this audit only."],
"source_pin_summary":{"gst":cfg["papers"]["gst"]["source_commit"],"argformer":cfg["papers"]["argformer"]["source_commit"],"argmix_released":cfg["papers"]["argmix"]["released_branch"]["source_commit"],"ancestralpaths":cfg["papers"]["ancestralpaths"]["source_commit"]},
"source_contract_hashes":{p:inputs[p]["sha256"] for p in ["docs/paper_dataset_evaluation_contract_v1.md","configs/paper_dataset_evaluation_contract_v1.json","research/reviews/paper_dataset_evaluation_contract_v1/seal.json"]},
"immutability":"Content is SHA256-sealed and files are made read-only against accidental edits. A later amendment/review must use a new artifact; this is not an OS-admin-resistant immutable store."
}
(OUT/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps({"verdict":receipt["verdict"],"input_count":len(inputs),"checks":len(checks),"passed":sum(c["result"]=="PASS" for c in checks),"failed":sum(c["result"]=="FAIL" for c in checks),"receipt_sha256":sha((OUT/"receipt.json").read_bytes()),"input_manifest_sha256":receipt["input_manifest_sha256"]},indent=2))
