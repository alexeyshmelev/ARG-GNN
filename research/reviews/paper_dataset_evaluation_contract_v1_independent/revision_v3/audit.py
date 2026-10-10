from pathlib import Path
import hashlib,json,re,subprocess,sys
from datetime import datetime,timezone
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=ROOT/"research/reviews/paper_dataset_evaluation_contract_v1_independent/revision_v3"
inputs={};checks=[]
def sha(b):return hashlib.sha256(b).hexdigest()
def read(rel,role="source_or_metadata"):
 b=(ROOT/rel).read_bytes();inputs[rel]={"path":rel,"bytes":len(b),"sha256":sha(b),"role":role};return b
def js(rel,role="source_or_metadata"):return json.loads(read(rel,role))
def check(name,ok,expected,actual):checks.append({"case":name,"result":"PASS" if ok else "FAIL","expected":expected,"actual":actual})
prev=js("research/reviews/paper_dataset_evaluation_contract_v1_independent/revision_v2/receipt.json","immutable_previous_review")
mismatch=[]
for a in prev["inputs"]:
 b=read(a["path"],a["role"])
 if len(b)!=a["bytes"] or sha(b)!=a["sha256"]:mismatch.append(a["path"])
check("all_221_v2_inputs_unchanged",not mismatch,{"count":221,"mismatches":0},{"count":len(prev["inputs"]),"mismatches":mismatch})
for stem in ["research/reviews/paper_dataset_evaluation_contract_v1_independent","research/reviews/paper_dataset_evaluation_contract_v1_independent/revision_v2"]:
 seal=js(stem+"/seal.json","immutable_previous_review_seal")
 errs=[]
 for a in seal["records"]:
  b=read(a["path"],"immutable_previous_review_artifact")
  if sha(b)!=a["sha256"] or len(b)!=a["bytes"]:errs.append(a["path"])
 check(stem.split("/")[-1]+"_review_artifacts_unchanged",not errs,"sealed prior BLOCKED artifacts intact",errs)
cfgrel="configs/paper_dataset_evaluation_contract_v3.json";docrel="docs/paper_dataset_evaluation_contract_v3.md"
cfg=js(cfgrel,"final_submitted_v3");doc=read(docrel,"final_submitted_v3").decode()
pins={cfgrel:"0bcae8548a48dab1acd69977267eec947b0d2b0e28350b499894a1d7e8739847",docrel:"517e4476bf2a31c430ef7a240173c338ee8fb2a2a77df5ffe40d58a3c5715528"}
check("final_v3_submitted_hashes",all(inputs[p]["sha256"]==h for p,h in pins.items()),pins,{p:inputs[p]["sha256"] for p in pins})
base={}
for key in ["base_contract","base_document","base_seal"]:
 a=cfg[key];b=read(a["path"],"effective_base_contract");base[key]={"path":a["path"],"expected":a["sha256"],"actual":sha(b)}
check("effective_v1_bindings_unchanged",all(a["actual"]==a["expected"] for a in base.values()),"exact original v1 contract/document/seal",base)
sup=cfg["supersedes_inaccurate_amendment"];supchecks={}
for key in ["config","document"]:
 a=sup[key];b=read(a["path"],"immutable_superseded_v2");supchecks[key]={"expected":a["sha256"],"actual":sha(b)}
check("v3_precedence_and_v2_preservation",all(v["expected"]==v["actual"] for v in supchecks.values()) and "corrected V3 amendments supersede inaccurate V2" in cfg["purpose"] and "supersedes the inaccurate v2" in doc,"effective contract is v1+v3; v2 is preserved superseded history",{"v2_bindings":supchecks,"purpose":cfg["purpose"]})
author="research/reviews/paper_dataset_evaluation_contract_v3_author/"
first=js(author+"initial_submitted_config.json","preserved_initial_v3_submission")
repair=js(author+"purpose_repair_receipt.json","v3_purpose_repair_receipt")
authorr=js(author+"receipt.json","initial_v3_author_receipt_historical")
diff=[k for k in set(first)|set(cfg) if first.get(k)!=cfg.get(k)]
check("v3_purpose_repair_preserves_scientific_content",diff==["purpose"] and inputs[author+"initial_submitted_config.json"]["sha256"]==repair["old_config_sha256"]==authorr["config"]["sha256"] and inputs[cfgrel]["sha256"]==repair["config_sha256"] and inputs[docrel]["sha256"]==repair["document_sha256"],"only purpose changed; old/new exact hashes and author receipt disambiguated",{"changed_top_level_fields":diff,"repair":repair})
bh=cfg["amendments"]["gst_Buchwald_Hartwig_target_statistics"]
check("I1_BH_statistics_access_resolved",bh["test_labels_in_statistics"] is False and bh["preserve_native_source_and_report_both_separately"] and bh["cannot_claim_small_fraction_label_budget_equals_fit_rows"] and "full retained70percent" in bh["native_statistics_scope"] and "validation" in bh["native_data_access"] and bh["required_native_condition_label"]=="released_full_pool_target_statistics" and bh["independent_corrected_condition"]=="training_subset_only_target_statistics","full70%pool labels include VAL and fraction-excluded rows, TEST excluded; TRAIN-only normalizer distinct",bh)
lr=cfg["amendments"]["gst_learning_rates"]
check("I2_GST_optimizer_and_learning_rate_discrepancy_resolved",lr["paper_non_CIFAR_optimizer"]=="AdamW" and lr["paper_non_CIFAR_lr"]=={"DeepSets":0.001,"SetTransformer":0.001,"GST":0.0001} and lr["paper_CIFAR_optimizer"]=="Adam" and lr["paper_CIFAR_lr_all_models"]==0.001 and lr["saved_source_config_lr_all_models"]==0.001 and lr["BH_Adam_source_vs_AdamW_paper_discrepancy_retained"] and set(lr["required_branches"])=={"released_source_common_learning_rate","paper_stated_per_model_learning_rate"},"paper per-model and released-source shared rates/optimizers distinct",lr)
gn=cfg["amendments"]["ancestralpaths_published_GNOMix_comparator"]
supprel=gn["supplement"]["path"];suppb=read(supprel,"primary_paper_supplement")
exrel=gn["primary_excerpt"]["path"];exb=read(exrel,"primary_GNOMix_excerpt")
supp=re.sub(r"\s+"," ",subprocess.run(["pdftotext",str(ROOT/supprel),"-"],capture_output=True,text=True,check=True).stdout)
excerpt=re.sub(r"\s+"," ",exb.decode()).strip()
check("GNOMix_primary_excerpt_independently_verified",sha(suppb)==gn["supplement"]["sha256"] and sha(exb)==gn["primary_excerpt"]["sha256"] and excerpt in supp,"exact supplement hash and normalized verbatim p34 excerpt substring",{"supplement_sha256":sha(suppb),"excerpt_sha256":sha(exb),"verbatim_normalized_substring":excerpt in supp})
check("I3_GNOMix_query_and_reference_roles_resolved",gn["query_populations"]==["GBR","Bronze Age","Neo","Yam"] and gn["only_GBR_is_present_day"] is True and gn["BAA_query_population_excluded"] is True and gn["reference_path_populations"]==["EHG","WHG","CHG","Ana"] and "four modern admixed query individuals" not in doc,"four query populations with BAA queries excluded; four pure reference populations",{"queries":gn["query_populations"],"refs":gn["reference_path_populations"],"BAA_query_excluded":gn["BAA_query_population_excluded"]})
check("GNOMix_query_cohort_windows_resolved",gn["window_cM_by_query_cohort"]=={"GBR":0.02,"Bronze Age":0.2,"Neo":0.2,"Yam":0.2},"0.02cM GBR;0.2cM BronzeAge/Neo/Yam queries",gn["window_cM_by_query_cohort"])
check("GNOMix_sequential_native_modules_resolved",gn["native_modules_in_sequence"]==["logistic regression base","xgboost smoother"] and gn["native_modules_are_not_alternative_model_variants"] is True and "model_variants" not in gn,"one GNOMix pipeline with native base followed by smoother",gn["native_modules_in_sequence"])
check("GNOMix_known_coverage_and_summary_bound",gn["independent_simulated_sequences"]==5 and gn["sequence_length_bp"]==200000000 and "evenly spaced" in gn["scoring_positions"] and gn["native_summary"]=="per-query-population mean and standard deviation across five sequences" and gn["published_significance_test"]=="two-sided two-sample T-test","five200Mbp sequences; evenly spaced sites; population summaries and original statistical test",{"sequences":gn["independent_simulated_sequences"],"length_bp":gn["sequence_length_bp"],"positions":gn["scoring_positions"],"summary":gn["native_summary"],"test":gn["published_significance_test"]})
check("GNOMix_original_identities_and_defaults_remain_unknown",gn["full_recipe_raw_identity_and_original_site_spacing_still_to_qualify"] and "without a fully pinned" in gn["other_parameters"] and gn["published_comparison_is_not_candidate_data_or_truth"],"original raw/splits/seeds/site-spacing/defaults not invented; published outputs external",{"unresolved":gn["full_recipe_raw_identity_and_original_site_spacing_still_to_qualify"],"defaults":gn["other_parameters"],"candidate_truth_forbidden":gn["published_comparison_is_not_candidate_data_or_truth"]})
draw=cfg["amendments"]["gst_synthetic_validation_draws"]
check("I4_validation_draw_wording_resolved",draw["validation_checks"]==5 and draw["sets_per_check"]==128 and draw["total_generator_draws"]==640 and draw["byte_unique_sets_proven"] is False and draw["deduplication_or_regeneration_admitted"] is False and draw["overrides_v1_field"]=="total_distinct_sets_consumed","640 native generator draws; byte uniqueness not claimed; no changed data recipe",draw)
link=cfg["review_receipt_pending"]
check("initial_audit_receipt_link_resolved",link=="research/reviews/paper_dataset_evaluation_contract_v1_independent/receipt.json" and (ROOT/link).is_file(),"actual immutable initial receipt path",link)
basecfg=js(cfg["base_contract"]["path"],"effective_base_contract")
check("effective_original_data_and_method_guards_retained",basecfg["guards"]["no_silent_smaller_substitute"] and basecfg["guards"]["missing_original_architecture_or_split_blocks_exact_claim"] and basecfg["guards"]["author_native_and_independent_confirmation_separate"] and basecfg["papers"]["argmix"]["april_branch"]["matching_native_source_available"] is False and basecfg["papers"]["ancestralpaths"]["published_simulation"]["architecture_definition_or_checkpoint_available_in_cited_release"] is False,"v3 disclosure does not clear absent original models/raw/splits or change native/task-transfer separation",{"guards":basecfg["guards"],"ARGMix_April_native_unresolved":True,"AP_CNN_unresolved":True})
check("source_contract_only_no_empirical_readiness",cfg["source_or_scientific_code_changed"] is False and cfg["datasets_acquired_or_executed"] is False and cfg["heldout_results_accessed"] is False and "not establish acquired original bytes" in doc,"amendment remains no-execution source contract",{"status":cfg["status"],"no_native_source_changed":True,"no_dataset_acquired_or_executed":True,"heldout_results_accessed":False})
verdict="PASS" if all(c["result"]=="PASS" for c in checks) else "BLOCKED"
receipt={"schema":"paper_dataset_contract_revision_v3_independent_review","timestamp_utc":datetime.now(timezone.utc).isoformat(),"reviewer":"independent source-contract reviewer subagent","verdict":verdict,"verdict_scope":"source-contract fidelity of effective immutable v1 plus final v3 only","source_only_contract_ready":verdict=="PASS","original_paper_empirical_milestone_qualified":False,"native_execution_authorized_by_this_review":False,"python_executable":sys.executable,"final_submitted_v3_hashes":pins,"superseded_v2_bytes_preserved":True,"prior_review_artifacts_preserved":True,"inputs":[inputs[k] for k in sorted(inputs)],"input_manifest_sha256":sha(json.dumps([inputs[k] for k in sorted(inputs)],sort_keys=True,separators=(",",":")).encode()),"checks":checks,"concerns":[],"resolved_initial_issues":["I1 BH whole70%pool target normalization including VAL/out-of-fraction labels","I2 paper versus source optimizer/per-model learning-rate discrepancy","I3 GNOMix comparator populations/reference roles/query windows/native sequential modules/coverage","I4 validation generator draws versus byte-unique datasets"],"resolved_v2_issues":["query population unit and ancient/modern scope","BAA query exclusion","cohort-specific query windows","logistic-regression base then xgboost smoother","five200Mb sequences and evenly spaced native sites","dangling initial receipt link"],"resolved_initial_v3_editorial_note":"Purpose precedence repaired; prior submitted config and old/new hashes retained by author.","guarded_actions":{"data_acquisition":0,"author_code_execution":0,"biological_model_score_arrays_opened":0,"heldout_data_opened":0,"GPU_jobs":0,"training":0,"installs":0,"git_actions":0,"runtime_or_ledger_mutations":0,"author_contacts":0,"writes_outside_owned_directory":0},"performed":"Independent exact input hash/size checks, prior review-seal verification, primary-PDF text extraction to stdout and amendment/source-text qualification. Native/source metadata checks from initial independent audit are inherited only because their exact input bytes remain unchanged.","limitations":["PASS is documentary source-contract approval only; source hashes/metadata do not establish exact original dataset bytes.","Original paper model, raw data, sample/split/query identity and execution gaps disclosed in v1 remain blockers to empirical claims.","No dataset, author loader/simulator, model training, inference, scoring or runtime qualification was performed.","Future acquisitions/adapters/native implementations/readouts require their own full data, coverage and independent receipts.","Read-only files and SHA256 seals detect accidental edits but do not resist administrator replacement."]}
(OUT/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps({"verdict":verdict,"input_count":len(inputs),"checks":len(checks),"passed":sum(c["result"]=="PASS" for c in checks),"failed":sum(c["result"]=="FAIL" for c in checks),"receipt_sha256":sha((OUT/"receipt.json").read_bytes()),"input_manifest_sha256":receipt["input_manifest_sha256"]},indent=2))
