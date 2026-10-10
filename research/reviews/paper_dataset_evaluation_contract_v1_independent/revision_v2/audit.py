from pathlib import Path
import hashlib,json,re,subprocess,sys
from datetime import datetime,timezone
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OUT=ROOT/"research/reviews/paper_dataset_evaluation_contract_v1_independent/revision_v2"
inputs={};checks=[]
def sha(b):return hashlib.sha256(b).hexdigest()
def read(rel,role="source_or_metadata"):
 b=(ROOT/rel).read_bytes();inputs[rel]={"path":rel,"bytes":len(b),"sha256":sha(b),"role":role};return b
def js(rel,role="source_or_metadata"):return json.loads(read(rel,role))
def check(name,ok,expected,actual):checks.append({"case":name,"result":"PASS" if ok else "FAIL","expected":expected,"actual":actual})
initial=js("research/reviews/paper_dataset_evaluation_contract_v1_independent/receipt.json","immutable_initial_audit")
read("research/reviews/paper_dataset_evaluation_contract_v1_independent/seal.json","immutable_initial_audit_seal")
mismatches=[]
for a in initial["inputs"]:
 b=read(a["path"],a["role"])
 if len(b)!=a["bytes"] or sha(b)!=a["sha256"]:mismatches.append(a["path"])
check("all_initial_217_inputs_unchanged",not mismatches,{"count":217,"mismatches":0},{"count":len(initial["inputs"]),"mismatches":mismatches})
docrel="docs/paper_dataset_evaluation_contract_v2.md";cfgrel="configs/paper_dataset_evaluation_contract_v2.json"
doc=read(docrel,"submitted_amendment").decode();cfg=js(cfgrel,"submitted_amendment")
pins={docrel:"707794ed5842b6676601dc90f3e13dc94e5f4f931c2de15474e6c6dbc790afcb",cfgrel:"80eefce5625517f0359ae28c3f8d14007253b45d27743e9ff09905801a9e5f97"}
check("submitted_v2_exact_hashes",all(inputs[p]["sha256"]==h for p,h in pins.items()),pins,{p:inputs[p]["sha256"] for p in pins})
base={}
for key in ["base_contract","base_document","base_seal"]:
 a=cfg[key];b=read(a["path"],"effective_base_contract");base[key]={"path":a["path"],"expected":a["sha256"],"actual":sha(b)}
check("effective_base_v1_hash_bindings",all(a["actual"]==a["expected"] for a in base.values()),"v1 contract/document/seal exact unchanged SHA256",base)
bh=cfg["amendments"]["gst_Buchwald_Hartwig_target_statistics"]
check("BH_whole_pool_normalization_disclosure_resolves_I1",bh["test_labels_in_statistics"] is False and bh["preserve_native_source_and_report_both_separately"] and bh["cannot_claim_small_fraction_label_budget_equals_fit_rows"] and "full retained70percent" in bh["native_statistics_scope"] and "validation" in bh["native_data_access"],"full70%pool labels include VAL/fraction-excluded rows; TEST excluded; correction separate",bh)
lr=cfg["amendments"]["gst_learning_rates"]
check("GST_learning_rate_disclosure_resolves_I2",lr["paper_non_CIFAR_optimizer"]=="AdamW" and lr["paper_non_CIFAR_lr"]=={"DeepSets":0.001,"SetTransformer":0.001,"GST":0.0001} and lr["paper_CIFAR_optimizer"]=="Adam" and lr["paper_CIFAR_lr_all_models"]==0.001 and lr["saved_source_config_lr_all_models"]==0.001 and lr["BH_Adam_source_vs_AdamW_paper_discrepancy_retained"],"paper per-model rates and shared released-source rates separate",lr)
supprel=".runtime/author_sources/ancestralpaths_supplement_41586_2023_6705_MOESM1_ESM.pdf"
supp=re.sub(r"\s+"," ",subprocess.run(["pdftotext",str(ROOT/supprel),"-"],capture_output=True,text=True,check=True).stdout)
needle="Comparison to an advanced LAI tool GNOMix"
start=supp.index(needle)
extract=supp[start:supp.index("Table 15.",start)]
gn=cfg["amendments"]["ancestralpaths_published_GNOMix_comparator"]
check("GNOMix_query_population_unit_and_age_scope","four modern admixed query individuals" not in gn["query_scope"] and "four modern admixed query individuals" not in doc and all(pop in gn["query_scope"] for pop in ["GBR","Bronze","Neo","Yam"]),"four query populations GBR/BronzeAge/Neo/Yam; only GBR present-day; original individual identities unknown",{"submitted_query_scope":gn["query_scope"],"primary_text":"only test the four admixed populations as query sequences ... GBR, Bronze Age, Neo, and Yam"})
check("GNOMix_BAA_exclusion_is_query_scope","BAA_query_population_excluded" in gn and gn["BAA_query_population_excluded"] is True,"BAA query population excluded; references explicitly EHG/WHG/CHG/Ana",{"submitted_BAA_reference_class_excluded":gn.get("BAA_reference_class_excluded"),"query_exclusion_field_present":"BAA_query_population_excluded" in gn,"primary_text":"do not test the BAA"})
check("GNOMix_windows_apply_to_query_cohorts","other_compared_sources" not in gn["window_cM"] and all(k in gn["window_cM"] for k in ["GBR","Bronze_Age","Neo","Yam"]),"0.02cM GBR;0.2cM BronzeAge/Neo/Yamnaya query cohorts",{"submitted_windows":gn["window_cM"],"primary_text":"window size of 0.2cM for Bronze Age, Neo and Yamnaya samples; ... GBR ... window size to 0.02cM"})
check("GNOMix_modules_are_sequential_base_and_smoother","model_variants" not in gn and gn.get("base_module")=="logistic_regression" and gn.get("smoother_module")=="xgboost","logistic-regression base AND xgboost smoother, not alternative variants",{"submitted_model_variants":gn.get("model_variants"),"primary_text":"default logistic regression base and xgboost smoother modules"})
check("GNOMix_known_coverage_binding_explicit","test_sequences" in gn and gn.get("test_sequences")==5 and gn.get("sequence_length_bp")==200000000 and gn.get("scoring_site_selection")=="evenly_spaced_sites","bind five200Mbp sequences and evenly spaced sites to published comparator while identities remain unknown",{"submitted_fields":sorted(gn),"primary_text":"five 200Mbp length sequences ... evenly spaced sites across the five sequences"})
link=cfg["review_receipt_pending"]
check("v2_initial_audit_receipt_link_exists",(ROOT/link).is_file(),"bind existing immutable initial receipt.json",{"submitted":link,"exists":(ROOT/link).is_file(),"actual_initial":"research/reviews/paper_dataset_evaluation_contract_v1_independent/receipt.json"})
check("v2_scope_keeps_empirical_limits",cfg["source_or_scientific_code_changed"] is False and cfg["datasets_acquired_or_executed"] is False and cfg["heldout_results_accessed"] is False and "not establish acquired original bytes" in doc,"source-only amendment does not qualify empirical milestone",{"status":cfg["status"],"source_changed":False,"datasets_acquired_or_executed":False,"heldout_results_accessed":False})
concerns=[
{"id":"V2-I1","priority":1,"blocking":True,"title":"Correct GNOMix populations, modules and query-window scope","summary":"The main amendment and JSON falsely name four modern admixed query individuals. Primary supplement p34 describes four admixed populations GBR, Bronze Age, Neo, Yam, with BAA queries excluded. Logistic regression is the base and xgboost the smoother of one pipeline. Windows apply to query cohorts, not reference sources. Bind the five200Mbp test sequences and evenly spaced scoring sites; preserve unknown raw/split/version identities.","source":{"path":supprel,"sha256":inputs[supprel]["sha256"],"printed_page":34,"section":"Comparison to an advanced LAI tool"},"fields":["query_scope","BAA_reference_class_excluded","window_cM.other_compared_sources","model_variants"]},
{"id":"V2-I2","priority":2,"blocking":False,"title":"Repair dangling initial-audit receipt link","summary":"review_receipt_pending points to review_receipt.json, which does not exist. The actual immutable initial artifact is receipt.json with its separate seal.json.","field":"review_receipt_pending"},
{"id":"V2-I3","priority":2,"blocking":False,"title":"Original synthetic draw-uniqueness wording remains unresolved","summary":"Effective v1+v2 still says total_distinct_sets_consumed=640. Native source certifies generated draws, not byte-unique datasets; clarify in the next amendment.","source_issue":"I4 from initial review"}
]
receipt={"schema":"paper_dataset_contract_revision_v2_independent_review","timestamp_utc":datetime.now(timezone.utc).isoformat(),"reviewer":"independent source-contract reviewer subagent","verdict":"BLOCKED","scope":"effective immutable v1 plus submitted v2, source-contract-only","python_executable":sys.executable,"submitted_v2_hashes":pins,"inputs":[inputs[k] for k in sorted(inputs)],"input_manifest_sha256":sha(json.dumps([inputs[k] for k in sorted(inputs)],sort_keys=True,separators=(",",":")).encode()),"checks":checks,"resolved_initial_issues":["I1 BH full-pool normalization","I2 GST shared source versus paper per-model learning rates"],"concerns":concerns,"primary_GNOMix_source_excerpt":extract,"guarded_actions":{"data_acquisition":0,"author_code_execution":0,"biological_model_score_arrays_opened":0,"heldout_data_opened":0,"GPU_jobs":0,"training":0,"installs":0,"git_actions":0,"runtime_or_ledger_mutations":0,"author_contacts":0,"writes_outside_owned_directory":0},"limitations":["No original data/checkpoint/split qualification or empirical reproduction performed.","v1 and v2 submitted bytes and initial review are preserved; a subsequent amendment needs a distinct review receipt.","File read-only protection and SHA256 seals detect accidental changes, not administrator-resistant immutability."],"effective_original_milestone":"still unqualified regardless of source-contract amendment verdict"}
(OUT/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps({"verdict":receipt["verdict"],"inputs":len(inputs),"checks":len(checks),"passed":sum(c["result"]=="PASS" for c in checks),"failed":sum(c["result"]=="FAIL" for c in checks),"receipt_sha256":sha((OUT/"receipt.json").read_bytes())},indent=2))
