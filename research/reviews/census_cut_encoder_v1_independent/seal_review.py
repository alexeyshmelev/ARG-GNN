from pathlib import Path
import csv,datetime,hashlib,json
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');OUT=ROOT/'research/reviews/census_cut_encoder_v1_independent'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
concerns=[
 {'id':'C01','priority':'P1','scope':'native integration admission','finding':'Forward trusts a prequalified donor bank and does not bind it to all current tree/reference geometry.','evidence':'encoder lines332-345; prequalification_boundary_exposure: stale donor max error0.19736, logits difference0.0004473; changed persistent R dates accepted','action':'Certify one cached preparation artifact binding native tree topology/times, graph order, actual Q/R identities/roles, donor map and feature/scaler inputs before fitting.','status':'blocked external join qualification; disclosed prepared-input boundary, no narrow source failure'},
 {'id':'C02','priority':'P1','scope':'native integration admission','finding':'Shape/finite/encoded-age F47 checks cannot verify genotype/label/scaler semantics or reference purity.','evidence':'cut lines217-234; encoder lines127-145; synthetic arbitrary feature marker accepted and influences logits; no label argument in forward','action':'Provide audited observed-only F47 feature roster and genotype/scaler/role joins. Exclude sampling_population, hidden census labels and ancestry truth from learned inputs. Freeze scaler fitting split.','status':'blocked external feature provenance qualification'},
 {'id':'C03','priority':'P2','scope':'native coordinate/event admission','finding':'Separately rounded arithmetic can split a physically intended anchor/boundary/readout collision under exact half-open semantics.','evidence':'preserved attempt01 translation by2.13 yields10 to11 evaluations and target difference0.20475; attempt02 supplied-coordinate dense parity1.78e-15','action':'Canonicalize shared original-bp/2Mb event coordinates and qualify actual collision joins. Do not silently add epsilon or claim exact translation invariance for separately rounded inputs.','status':'qualified exact supplied-coordinate implementation; canonical native join remains blocked'},
 {'id':'C04','priority':'P2','scope':'native unary/F47 admission','finding':'All nonroot nonsample unary rows are suppressed, including their stored F47. Endpoint-duration joining alone does not preserve arbitrary unary feature semantics.','evidence':'cut lines235-254; independent serial unary rows with distinct synthetic features disappear while retained outputs remain exactly equal','action':'Document why native unary rows are anonymous/disposable and qualify retained F47 semantics or use a separately reviewed representation when unary observations carry meaningful information.','status':'qualified declared anonymous-unary contract; native feature semantics unqualified'},
 {'id':'C05','priority':'P1','scope':'scientific and efficiency claims','finding':'Handmade numerical qualification establishes neither biological benefit nor fitting efficiency. Whole-tree batching still has level depth/full-state copy traffic.','evidence':'no project arrays, model fit or profiling; standard diffusion/OU/backbone; docs biological_kernel_design_v2.md admission gates','action':'Before new outcomes, freeze native observed support distributions, matched eta0/permuted/raw-diagonal/local fits, horizon/tile/grid sensitivity,10x matched20-epoch speed gate and independent parent-level NLL/calibration/accuracy confirmation.','status':'unvalidated scientific claim; no fitting admission'},
]
with (OUT/'concerns.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(concerns[0]));w.writeheader();w.writerows(concerns)
bindings=json.loads((OUT/'binding_checks.json').read_text());before=json.loads((OUT/'source_hashes_before.json').read_text())
after={p:sha(ROOT/p) for p in before};assert after==before
(OUT/'source_hashes_final.json').write_text(json.dumps(after,indent=2)+'\n')
frozen={}
for namespace in ['census_cut_exchange_v1_author','census_tree_encoder_v1_author']:
 r=json.loads((ROOT/'research/reviews'/namespace/'receipt.json').read_text())
 for p,row in r['frozen_source_integrity'].items():
  h=sha(ROOT/p);assert h==row['expected_sha256'],p
  frozen[p]={'expected_sha256':row['expected_sha256'],'actual_sha256':h,'matches':True}
metrics=json.loads((OUT/'independent_metrics_attempt_02.json').read_text());assert metrics['failed_checks']==0 and metrics['passed_checks']==15
permutation=json.loads((OUT/'permutation_metrics.json').read_text());assert permutation['passed']
pytest=json.loads((OUT/'author_pytest_attempt_01.json').read_text());assert pytest['exit_code']==0 and '66 passed' in (OUT/'author_pytest_attempt_01.txt').read_text()
expected_pair={p:after[p] for p in ['src/arggnn/census_cut_exchange_v1.py','src/arggnn/census_tree_encoder_v1.py']}
artifact_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in ['receipt.json','review_binding.json']}
receipt={
 'schema':'h021_census_cut_encoder_v1_independent_receipt',
 'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'project_root':str(ROOT),'hypothesis':'H021','condition':'census_cut_exchange_proto_v1_source_only',
 'status':'qualified_source_and_handmade_CPU_for_exact_sealed_pair',
 'native_integration_and_fitting_admission':'blocked_pending_observed_F47_scaler_role_donor_tree_identity_and_canonical_coordinate_joins',
 'scientific_claim_status':'unvalidated_no_benefit_speed_or_generic_kernel_novelty_established',
 'qualification_contract':'Exact sealed source, CPU handmade fixtures, qualified prepared inputs, complete sample-ancestor union, sampled leaves only, anonymous nonroot unary suppression, all Q age<census350, older R as proxies, exact supplied half-open coordinates. No author source corrections required within this contract.',
 'sealed_pair_sha256':expected_pair,
 'author_binding_checks':{k:{'all_match':v['all_match'],'binding_sha256':v['binding_sha256'],'entries_checked':len(v['artifacts'])} for k,v in bindings.items()},
 'frozen_source_integrity':frozen,
 'before_after_artifact_integrity':{'all_match':True,'entries_checked':len(before),'before':'source_hashes_before.json','after':'source_hashes_final.json'},
 'results':{'author_pytest_cases':66,'author_pytest_exit_code':pytest['exit_code'],'independent_final_check_groups':15,'independent_final_failures':0,'full_native_sample_graph_permutation_passed':True,'metrics':'independent_metrics_attempt_02.json','permutation_metrics':'permutation_metrics.json'},
 'preserved_failures':[
  {'kind':'auditor SSH source-write quoting failure','evidence':'launch_diagnostic_01.txt; *_attempt_00_shell_corrupted.py','resolution':'Corrected raw SSH stdin writing only inside audit ownership.'},
  {'kind':'auditor invalid arithmetic-translation raw-shape invariance assertion','evidence':'independent_checks_attempt_01.py/.txt; independent_metrics_attempt_01.json','resolution':'Retained14 passes/1 failure. Exact transformed-coordinate dense parity independently checked in attempt02, physical scaling independently tested without collisions. Target difference0.20475 retained as canonical-coordinate join concern. No tolerance or sealed source change.'}],
 'concerns':concerns,
 'inherited_math_and_backbone':'Partition diffusion, OU/covariance, finite-anchor kernels, TMRCA pooling and whole-tree GNN are inherited. Specific observed census horizon/support-loss biological constraint is unvalidated with unresolved priority. No independent comprehensive prior-art search in this audit.',
 'actions':{'SSH_alias':'anogena','read_AGENTS_first':True,'writes_only_this_review_directory':True,'handmade_CPU_forwards_backward':True,'actual_project_or_holdout_arrays':False,'actual_prepared_bank_arrays':False,'stochastic_genealogy_simulation':False,'native_inference':False,'fitting_or_optimizer':False,'CUDA_or_GPU_work':False,'performance_profiling':False,'packages_installed':False,'ledger_or_decision_edits':False,'git_used':False,'author_or_frozen_sources_edited':False,'Mac_project_file_writes':False},
 'artifact_sha256':artifact_sha256,
}
path=OUT/'receipt.json';assert not path.exists();path.write_text(json.dumps(receipt,indent=2)+'\n')
binding={'schema':'h021_census_cut_encoder_v1_independent_binding','receipt_sha256':sha(path),'sealed_pair_sha256':expected_pair,'artifact_sha256':artifact_sha256}
(OUT/'review_binding.json').write_text(json.dumps(binding,indent=2)+'\n')
print(json.dumps({'status':receipt['status'],'native_integration_and_fitting_admission':receipt['native_integration_and_fitting_admission'],'receipt_sha256':sha(path),'binding_sha256':sha(OUT/'review_binding.json'),'author_cases':66,'independent_checks':15,'permutation_passed':True,'frozen_sources_checked':len(frozen),'before_after_entries':len(before)},indent=2))
