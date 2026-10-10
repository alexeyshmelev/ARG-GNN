"""Seal the final audit only after final replay and tests have completed."""
from pathlib import Path
import csv,hashlib,json,re
from datetime import datetime,timezone
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN');OUT=ROOT/'research/reviews/integrity_fix_reaudit_v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
replay=json.loads((OUT/'replay.json').read_text())
assert replay['all_checks_passed'] and len(replay['checks'])==50
post=json.loads((OUT/'coherent_settings_drift_post_repair.json').read_text())
assert post['accepted'] is False and post['rejection']['message']=='Family settings differ from captured original config bytes'
prep=json.loads((OUT/'active_preparation_freeze_check.json').read_text())
assert prep['all_declared_files_match']
files=['src/arggnn/train_canonical_gru.py','src/arggnn/development_training_bindings.py','src/arggnn/train_anchor_development.py',
       'src/arggnn/train_anchor_temporal_development.py','tests/test_canonical_gru.py','tests/test_tree_canonicalization.py',
       'tests/test_development_training_bindings.py','tests/test_train_anchor_development.py','tests/test_train_anchor_temporal_development.py',
       'tests/test_train_anchor_temporal.py','tests/test_anchor_temporal_baselines.py','scripts/bind_development_training.py',
       'configs/canonical_inferred_cohort64_pilot_v1.json','docs/development_integrity_fix_ad03_ad04_v1.json',
       'docs/development_training_bindings_v1.md','AGENTS.md','scripts/project_env.sh']
hashes={name:sha(ROOT/name) for name in files}
for name,value in replay['loaded_source_sha256'].items():assert hashes['src/arggnn/'+name]==value
assert hashes['configs/canonical_inferred_cohort64_pilot_v1.json']==post['original_config_sha256']
(OUT/'audit_scope_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
suites=[]
for name,passed,deselected in [('pytest_development.log',104,0),('pytest_temporal_order.log',4,0),('pytest_canonical.log',77,1)]:
 content=(OUT/name).read_text();match=re.search(r'(\d+) passed(?:, (\d+) deselected)? in ([0-9.]+)s',content)
 assert match and int(match[1])==passed and int(match[2] or 0)==deselected
 suites.append({'log':name,'passed':passed,'deselected':deselected,'seconds':float(match[3])})
rows=[
 {'id':'AD-01','status':'resolved','priority':'P1','scope':'Canonical held-out admission','finding':'Config/manifest/source held-out paths reject before any control bytes, tensor loading or device query.','evidence':'replay.json canonical_heldout_* and canonical_startup_*; pytest_canonical.log'},
 {'id':'AD-02','status':'resolved after additional repair','priority':'P2','scope':'Canonical synthesis','finding':'Original common-hash/wrong-spec reproduction rejects. Fresh coherent settings+records drift was accepted, then repaired by mandatory original config_bytes SHA and parsed settings equality.','evidence':'coherent_settings_drift_pre_repair.json; coherent_settings_drift_post_repair.json; replay.json canonical_wrong_common_*'},
 {'id':'AD-03','status':'resolved','priority':'P2','scope':'Development source integrity','finding':'Actual freeze and seed registry, all declared source/raw/cache artifacts, epoch stat identities and ending full hashes are consumed.','evidence':'replay.json real_seed_registry_catalog_admission and source_mutation_*; source guard call sites in review.md'},
 {'id':'AD-04','status':'resolved','priority':'P2','scope':'Candidate config integration','finding':'Candidate consumes completion and imported/current dependency locks before full loader/device admission; overrides must match completion.','evidence':'replay.json candidate_real_admission_with_stubbed_full_loaders and candidate_startup_*; pytest_development.log'},
 {'id':'ORDER-05','status':'resolved during reaudit','priority':'P2','scope':'Temporal and canonical device ordering','finding':'Both initially queried the device before mandatory admission. Owners relocated admission; final CPU/CUDA-request spy regressions reject bad inputs without any device query.','evidence':'pytest_temporal_order.log; replay.json gru_startup_* and canonical_startup_*'},
 {'id':'LIMIT-06','status':'scope limitation','priority':'none','scope':'Audit reach','finding':'Positive full boundary/age loaders in rich synthetic admission were stubbed. No certificate of the active fresh cohort completion, real-data performance, GPU execution, or final-confirmatory readiness.','evidence':'replay.py FakeDataset and loader stubs; test and no-access scope in review.md'}]
with (OUT/'concerns.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=['id','status','priority','scope','finding','evidence']);writer.writeheader();writer.writerows(rows)
report=f'''# Independent AD01–AD04 integrity fix reaudit v1

The four original integrity findings are resolved in the final source snapshot. This reaudit found two additional gaps, device queries before mandatory source admission and synthesis accepting coherently mutated in-memory settings under the original config digest. The implementation owners repaired both and independent regression replay now rejects the same cases. No unresolved actionable defect remains within this narrow audited scope.

All project writes stayed on anogena inside `/workspace-SR003.nfs2/ARG-GNN`. The auditor read AGENTS.md, sourced scripts/project_env.sh, used `.runtime/env/bin/python`, and kept CUDA hidden. No GPU job, package installation, Git operation, simulation, native inference, real held-out read or implementation edit was performed by the auditor. Original review material in canonical_development_audit_v1 and development_integration_audit_v1 was read as evidence. Review output and temporary synthetic fixtures are separate from active preparation and pilots. Exact final hashes are in audit_scope_hashes.json.

## Independent evidence

The final test record contains **185 passed tests and one intentionally deselected test**: 104 development tests in {suites[0]['seconds']:.2f}s, four temporal admission-order tests in {suites[1]['seconds']:.2f}s, and 77 canonical/canonicalization tests in {suites[2]['seconds']:.2f}s. The deselected canonical smoke test loads the actual pinned TRAIN input, which this reaudit deliberately avoided. The development suite includes tiny synthetic CPU fitting and serialization, with production cohort admission mocked where the fixture has fewer parents. These are correctness checks rather than ancestry results. The author's earlier 104 relevant and 90 canonical test counts were background reports, not additional independent counts here.

The independent replay.py completes **50 checks** on final source bytes. Its richer catalog uses the actual installed seed-helper outputs for all 80 TRAIN/VAL parents, without mocking expected_parent_registry. It binds **1,053 artifact paths** across four linked stage manifests, actual synthetic preparation freeze/config/source bytes, oracle/inferred metadata/raw/prepared bytes, original oracle archives, native inferred/execution/preparation/input files, supervision, bank and age caches. These are fake byte payloads, not simulations or tensors from the cohort. Positive full boundary/age loader contracts were explicitly stubbed to 64/16 cases and 640 cache records. Real catalog admission, path guards, byte hashes, signatures and implementation locks remained unmocked.

Twenty-three representative artifact mutations each fail the stat-only epoch check. After deliberately refreshing the mutated artifact's cached stat signature, the ending full-hash check still rejects its wrong bytes. The stat-only check is separately exercised with the hash function replaced by a failing spy, proving no source hashing occurs in that check. The original inferred-manifest mutation now rejects. Failed/missing completion, dependency drift and wrong frozen stage directory reject before loader and device spies in both development entry points.

## Finding dispositions

AD-01: Canonical pilot_path rejects confirmatory, confirmation, final_test, final-test and the exact test path component before reading either supplied config or manifest. Source artifact paths receive the same guard. Five namespace probes observe zero Path.read_bytes calls. Both CPU and CUDA-request entry-point probes reject missing or held-out data before CUDA spies. No hypothetical held-out path was created or opened.

AD-02: Synthesis now requires an explicit expected original config SHA, captured config_bytes hashing to that SHA, and parsed settings equality before any model/count/record work. Every required configuration/seed must occur exactly once. Row specs, parameter count, dataset manifest, complete implementation locks and raw/prepared/metadata source locks must match the byte-bound frozen settings. Original wrong-common-hash and wrong-record-spec cases reject.

A fresh probe first retained the original real config bytes/hash, changed only the caller's settings width32 to width16 and made all24 record specs/counts coherent with those changed settings. Before the additional repair, the helper accepted and selected width16 under the h32 model ID. The original config file was untouched. That accepted evidence is preserved as coherent_settings_drift_pre_repair.json. With mandatory original config_bytes, the same probe now rejects with `Family settings differ from captured original config bytes`, recorded in coherent_settings_drift_post_repair.json. The archived pre-repair replay files preserve their historical source snapshot instead of presenting them as final-byte verification.

AD-03: capture_completion_sources binds the actual freeze bytes and declared config/source hashes, the exact parent registry, frozen stage paths, linked manifests, per-parent file paths/roles/seeds and all declared input/cache hashes. SourceSnapshot checks stat device/inode/size/mtime/ctime identities during epochs and hashes every bound artifact at full seals, with stat checks before and after hashing. check_live also enforces captured config and imported/current implementation locks. verify_completion performs actual catalog admission before the full boundary/age loaders and rechecks full source bytes afterward.

The production call sites were inspected: temporal admission precedes require_device at lines80/82, epoch full_sources=False is at line134, per-model and final full_sources=True are at lines168/186. Candidate admission precedes require_device at lines212–216, epoch stat checks are at line292, and model/final full checks are at lines363/377. Both include ending_input_artifact_sha256 in the accepted results. These line references correspond to audit_scope_hashes.json. The direct source-lock replay proves the checker semantics, while existing tiny fit tests and code inspection verify runner use. No real fresh-cohort run was admitted or certified here.

AD-04: checked_development_protocol requires completion/freeze/dependency fields, and candidate main invokes the same admit_run used by temporal execution. Config bytes are captured and immutable. Supplied data/age overrides must match the completion-bound paths. Missing/incomplete receipt, dependency mismatch and wrong stage catalog reject before loaders and device queries. The candidate no longer relies only on freshly recorded unbound current code hashes.

ORDER-05: Initial inspection found temporal require_device before admit_run and canonical CUDA checks before capture_run_snapshot. That mattered because require_device can query torch.cuda.device_count. The implementation owners relocated those checks. Final no-CUDA monkeypatch regressions independently prove the corrected order, including both CPU/CUDA requests for canonical and temporal. No actual CUDA query was used as a probe.

## Scope and remaining limits

active_preparation_freeze_check.json independently rehashes the 19 config/source files declared by the active preparation freeze. Every file still matches the freeze SHA `{prep['freeze_sha256']}`. This verifies that the audited fixes did not change its declared files. It does not certify that the active CPU preparation has completed or that its current real artifacts pass admission.

The catalog positive tests stub deserialization/full-loader contracts, and the saved fake cache bytes are deliberately not usable training data. The audit establishes admission and mutation behavior, with the tested production call sites, rather than end-to-end fitting on the fresh cohort. No acceptance, novelty, superiority, predictive utility, GPU memory/runtime or final-confirmatory readiness follows. The prior information-reachability limitations remain relevant and were not retested here. Complete family search and final sealed evaluation remain separate work.

concerns.csv contains machine-readable dispositions. receipt.json binds the final source snapshot, verified test counts and all review artifacts.
'''
(OUT/'review.md').write_text(report)
artifacts={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in {'receipt.json','seal_report.log'}}
receipt={'kind':'independent_integrity_fix_reaudit_v1','created_utc':datetime.now(timezone.utc).isoformat(),
         'source_sha256':hashes,'test_suites':suites,'independent_passed_tests':185,'deselected_real_train_smoke_tests':1,
         'synthetic_replay_checks':50,'synthetic_replay_passed':True,'bound_synthetic_artifacts':1053,
         'actual_seed_registry_parents':80,'all_active_preparation_declared_files_match':True,
         'original_findings_disposition':{r['id']:r['status'] for r in rows},'unresolved_actionable_findings':[],
         'no_real_fresh_completion_admitted':True,'no_gpu_jobs':True,'no_installs':True,'no_git_operations':True,
         'no_simulation_or_native_inference':True,'no_real_heldout_access':True,'auditor_implementation_edits':False,
         'scope_limitations':[rows[-1]['finding']],'artifact_sha256':artifacts}
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'receipt':str(OUT/'receipt.json'),'passed_tests':185,'replay_checks':50,'unresolved_actionable_findings':0}))
