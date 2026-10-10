"""Seal an honest audit only after a completed passing joint run and source recheck."""
from pathlib import Path
import csv
from datetime import datetime,timezone
import hashlib
import json
import re
import xml.etree.ElementTree as ET

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OUT=ROOT/'research/reviews/confirmatory_workflow_audit_v1'
sha=lambda body:hashlib.sha256(body).hexdigest()
manifest=json.loads((OUT/'final_source_manifest.json').read_text())
for record in manifest['files']:
    assert sha((ROOT/record['path']).read_bytes())==record['sha256'],record['path']
xml=ET.parse(OUT/'final_joint_junit.xml').getroot()
cases=xml.findall('.//testcase')
failures=xml.findall('.//failure');errors=xml.findall('.//error');skipped=xml.findall('.//skipped')
assert cases and len(failures)==2 and not errors and not skipped
failed_cases=[case for case in cases if case.find('failure') is not None]
assert all('confirmatory_workflow_audit_v1' in case.attrib['classname'] for case in failed_cases)
assert {case.attrib['name'] for case in failed_cases}=={'test_analysis_exact_fixed_tile_edges_should_reject','test_secondary_shortfall_retains_adjusted_results_but_blocks_primary_confirmation'}
log=(OUT/'final_joint.log').read_text()
match=re.search(r'(\d+) failed, (\d+) passed in ([0-9.]+)s',log)
assert match and int(match[1])+int(match[2])==len(cases)
rerun_xml=ET.parse(OUT/'independent_final_junit.xml').getroot()
rerun_cases=rerun_xml.findall('.//testcase')
assert rerun_cases and not rerun_xml.findall('.//failure') and not rerun_xml.findall('.//error') and not rerun_xml.findall('.//skipped')
rerun_match=re.search(r'(\d+) passed in ([0-9.]+)s',(OUT/'independent_final.log').read_text())
assert rerun_match and int(rerun_match[1])==len(rerun_cases)
assert (OUT/'independent_final.exit_status').read_text().strip()=='0'
assert {(c.attrib['classname'],c.attrib['name']) for c in rerun_cases}=={(c.attrib['classname'],c.attrib['name']) for c in cases if 'confirmatory_workflow_audit_v1' in c.attrib['classname']}
counts={}
for case in cases:
    key=case.attrib['classname']
    counts[key]=counts.get(key,0)+1
summary=json.loads((OUT/'integrated_summary.json').read_text())
assert summary['genuine_scientific_evidence'] is False
assert summary['positive_cuda_receipts_explicitly_mocked'] is True
assert summary['actual_cpu_terminal_receipts_rejected'] is True
assert summary['cells']==24 and summary['real_cpu_forward_grid_calls']==48
initial_counts=json.loads((OUT/'test_counts_intermediate.json').read_text())
concerns=[
    {'id':'CW01','severity':'major','type':'actual_defect','status':'repaired_and_retested','concern':'CUDA UUID literal spelling rejects matching raw/prefixed device identities','source':'initial predictor initialize_execution; frozen bf390043…','evidence':'root actual formatting observation; independent canonical-format CPU injection plus final UUID module cases','resolution':'Strict canonical UUID equality with optional GPU- prefix; malformed and mismatched identities still reject'},
    {'id':'CW02','severity':'major','type':'actual_defect','status':'repaired_and_retested','concern':'Lock/predictor pending path replacement publishes unrelated bytes and removes replacement','source':'initial lock/predictor publishers','evidence':'publication_initial.log and lock_publication_initial.log expected-rejection failures; final replacement probes pass','resolution':'Owned inode guards, exclusive links and owned-file cleanup'},
    {'id':'CW03','severity':'major','type':'actual_defect','status':'repaired_and_retested','concern':'Same-inode staged byte drift publishes bytes differing from reported SHA','source':'all four initial publishers','evidence':'publication_recheck.log three failures; lock_bytes_initial.log one failure; final all staged-byte probes pass','resolution':'Staged byte/size/stat/hash verification plus ownership and published-artifact checks where implemented'},
    {'id':'CW04','severity':'major','type':'actual_defect','status':'repaired_and_retested','concern':'Success NPZ join omits operational terminal receipt and admits CPU-validation-only output','source':'initial shared roster and analysis admission','evidence':'24 actual synthetic CPU cell outputs structurally eligible initially; final actual CPU terminals reject before targets; explicit mocked CUDA consumer path passes','resolution':'Mandatory terminal references, shared strict runtime/cell/source checker and custody roster sealer'},
    {'id':'CW05','severity':'moderate','type':'actual_defect','status':'repaired_and_retested','concern':'Analysis accepts arbitrary internal tile edges instead of fixed 125000 bp multiples','source':'initial analysis identity admission','evidence':'probes_initial.log fixed-edge expected-rejection failure; final independently mutated edge rejection passes','resolution':'Exact fixed tile-edge comparison in shared schema/analysis'},
    {'id':'CW06','severity':'minor','type':'compatibility_check','status':'verified','concern':'Predictor source-pin map varies by import context','source':'predictor dynamic imported-source inventory','evidence':'prediction_pin_context_difference.json: standalone 48 versus custody 49, shared hashes identical; final mock-positive receipts use fresh standalone inventory','resolution':'Checker verifies mandatory frozen dependencies and every reported source while allowing extra valid context pins'},
    {'id':'CW07','severity':'minor','type':'policy_ambiguity','status':'unresolved_policy_clarification','concern':'Secondary adjusted favorable flags can remain visible during a primary admission/precision shortfall','source':'v1 secondary multiplicity and primary decision rules','evidence':'probes_initial.log failed audit expectation; primary robust confirmation remains false','resolution':'Not classified as a code defect: protocol clearly gates primary precision, but does not unambiguously extend it to all secondary flags'},
    {'id':'CW08','severity':'major','type':'declared_scope_limit','status':'not_scientifically_qualified','concern':'No real DEV selection, VAL temperature fitting, published ID lock or genuine cohort/runtime qualification in this audit','source':'explicit mocked positive lock/cohort/temperature/CUDA fixture boundaries','evidence':'receipt.json scope and integrated_summary.json mock flags; real validator rejects handmade lock','resolution':'Complete and review actual development, calibration, runtime and lock evidence separately'},
    {'id':'CW09','severity':'major','type':'declared_scope_limit','status':'not_scientifically_qualified','concern':'Current bounded ten-family ID workflow excludes genuine ARGformer and AncestralPaths main competitors','source':'root user clarification and pinned official AncestralPaths README','evidence':'ancestralpaths_README.md at official commit 77db3cfae100641c35218b87e1451e14f3074666','resolution':'Separate author-source/data-protocol assessment and reviewed evaluation-scope amendment; no author execution by reviewer'},
    {'id':'CW10','severity':'minor','type':'declared_scope_limit','status':'explicitly_preserved','concern':'Generation, OS isolation, global prior-history completeness, one scientific join and remaining robustness/OOD/query/sensitivity/illustrations are outside this bounded audit','source':'workflow docs and full_protocol_readiness=false','evidence':'frozen docs/schema and final report flags','resolution':'Operational responsibilities and future workflows remain explicit; not implementation defects'},
    {'id':'CW11','severity':'minor','type':'audit_fixture_error','status':'corrected_with_original_preserved','concern':'Initial independent anchor assertion expected288 instead of384 preparations','source':'test_workflow_probes_initial.py integrated test','evidence':'probes_initial.log first failure; deterministic smoother also uses fixed 32 anchors','resolution':'Corrected 24 cells × 16 tile preparations; final real CPU integrated test passes'},
]
with (OUT/'concerns.csv').open('w',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(concerns[0]));writer.writeheader();writer.writerows(concerns)
core={r['path']:r['sha256'] for r in manifest['files'] if r['path'] in ('src/arggnn/confirmatory_lock.py','src/arggnn/confirmatory_prediction_runner.py','src/arggnn/confirmatory_analysis_runner.py','src/arggnn/confirmatory_custodian.py')}
total=len(cases);elapsed=float(match[3]);independent=len(rerun_cases);rerun_elapsed=float(rerun_match[2]);production=total-independent
text=f'''# Independent confirmatory workflow audit

## Summary and verdict

The repaired bounded ID implementation passed **{production} production cases** in the joint CPU run. That run had 301 passes and 2 audit-harness failures in {elapsed:.2f} seconds. After correcting only the audit harness, **{independent} independent probes passed in {rerun_elapsed:.2f} seconds**. This verifies {total} distinct cases across those two runs, not a single all-passing joint run. The five concrete implementation findings identified below were repaired and rechecked. No unresolved implementation failure remains within these probes. This is evidence about implementation guards and synthetic interoperability, not a genuine confirmatory experiment or scientific-readiness verdict.

The final frozen sources are recorded in `final_source_manifest.json` and preserved in `final_source_snapshot/`. Core SHA256s are:

| Source | SHA256 |
|---|---|
| Lock | {core['src/arggnn/confirmatory_lock.py']} |
| Predictor | {core['src/arggnn/confirmatory_prediction_runner.py']} |
| Analysis | {core['src/arggnn/confirmatory_analysis_runner.py']} |
| Custodian | {core['src/arggnn/confirmatory_custodian.py']} |

## Major concerns and resolutions

1. **CUDA UUID spelling.** The original predictor compared Torch's raw UUID literally with a `GPU-`-prefixed lease UUID, rejecting the same device. Root observed that formatting difference. Canonical-format injection and malformed/mismatched identity regressions now pass without this reviewer initializing CUDA.
2. **Publication ownership.** Direct probes replaced the predictor and lock binder pending path after staging. Both published unrelated replacement bytes, returned the intended-content digest and removed the replacement. Their expected-rejection tests failed in `publication_initial.log` and `lock_publication_initial.log`. The repaired inode/cleanup guards pass.
3. **Publication byte drift.** Separate probes changed staged bytes while preserving the inode. All four publishers accepted content inconsistent with their returned or enclosing digests. `publication_recheck.log` records three failures and `lock_bytes_initial.log` the direct binder failure. Final staged-byte probes pass after byte/size/stat/hash and ownership checks.
4. **Runtime evidence at joining.** Initially a successful NPZ carried no bound terminal receipt, so real CPU-validation-only outputs were structurally eligible for the joined analysis. The repaired schema requires success/failed terminal refs, and the shared checker binds method, seed, parent, checkpoint, protocol, grids, observed identity, ledger, source bytes and strict operational runtime metadata. The custodian sealer consumes immutable terminal records. Actual CPU success receipts now reject before target reads in both analysis and sealing.
5. **Fixed tile edges.** Initial analysis accepted any seven increasing internal edges, including [1,2,3,4,5,6,7] bp, although custody and prediction require 125000-bp multiples. The independently mutated-edge rejection now passes.

These were implementation defects, not observed errors in an actual final cohort. Original and intermediate source snapshots and all failing logs remain preserved. `concerns.csv` distinguishes repaired defects, audit errors, policy questions and scope limits.

## Validation evidence and limits

The independent integrated fixture uses two handmade observed star-tree parents and exhausts 192 synthetic admission attempts. It performs real electrical/canonical QA, real model construction and checkpoint loading for candidate, deterministic reference smoother, gain-matched diagonal and static-local controls across seeds 101/202/303, and **24 actual CPU prediction cells**. It observes **48 separate primary 128/dense 4096 forwards**, **384 fixed 32 anchor preparations**, zero prediction target reads and four analysis target reads. The initial complete 24-cell unresolved roster blocks joining before any array join.

Every actual CPU terminal receipt is retained and rejected for operational joining. The positive consumer-interoperability path uses those real CPU logits but **explicitly mocked CUDA execution metadata**, then passes custody sealing and joined diagnostics. It never constitutes a genuine GPU terminal, cohort, lock, temperature fit or scientific result. Positive lock/cohort/development/calibration admission is explicitly mocked. Existing binder fixtures also mock producer/factory/secondary admission where disclosed. A separate negative path uses the real published-lock validator and rejects the handmade lock.

The final positive mock receipts use a fresh standalone prediction-CLI source inventory. Separate metadata-only processes showed 48 CLI pins versus 49 pins after custody import, with custody as the only extra module and identical shared hashes. The checker accepts valid extra import-context pins while verifying the mandatory frozen dependency set and every reported source.

The combined run exercises complete method×seed×parent rosters, first 160 or all 192 admission, checkpoint/source identity, terminal failure preservation, unresolved-cell rejection, independent whole-parent averaging, uncapped float64 raw NLL, frozen VAL temperatures, 15-bin calibration, dosage pairs, census agreement, dense support/censoring, fixed Holm family size, exclusive publication and primary positive-claim completeness/precision guards. Raw and VAL-temperature diagnostics remain distinct. Dense metrics remain descriptive and approximate midpoint quadrature, not extra independent parents.

The initial five-module suite passed 204 tests in 160.41 seconds. The first independent 9-test run had 5 passes and 4 failures: two concrete defects, one audit anchor-count error and one unsupported policy expectation. The smoother also prepares anchors, so the correct count is 384 rather than 288. The secondary shortfall probe expected all adjusted favorable flags to be suppressed, but v1 clearly gates primary precision without unambiguously imposing that rule on every secondary flag. The implementation's primary robust confirmation remains false. Both initial audit corrections and the original failures are preserved. The final joint run additionally retained two harness errors: string quotes lost in an earlier mutation edit and the old unsupported secondary assertion not replaced. All 288 production cases and 13 of 15 independent cases passed in that run. After the owned harness correction, all 15 independent probes passed; unchanged production sources were rechecked rather than repeating their passed tests. Repeated focused runs overlap those cases and must not be summed as independent checks.

## Scientific and operational scope

There is no real DEV selection, VAL calibration, published ID lock, cohort or runtime qualification in this audit. The workflow imports already prepared artifacts, and native inference return codes remain upstream recorded claims. OS isolation, reviewed historical registry completeness and one scientific joined execution remain operational responsibilities. Robustness, demographic OOD, query-count, finite-source/tile sensitivity and illustrations remain explicitly pending with `full_protocol_readiness=false`.

The user clarified the main scientific competitors as actual ARGMix, genuine ARGformer and AncestralPaths. The current ten-family bounded ID workflow excludes genuine ARGformer and AncestralPaths, so adequacy as the final main-competitor study is unestablished. The pinned official AncestralPaths README describes a Relate-based neural path-ancestry classifier and painting pipeline. Its documented painting output stores assigned path/chosen softmax and interval coordinates. Reviewed label mapping, observed/sample-role adaptation and full raw-logit export are needed for this four-class endpoint. README inspection alone establishes no licence, dependency/runtime compatibility or exact reproduction. No author code was installed or executed.

## Method and artifact record

All project work stayed on anogena in `/workspace-SR003.nfs2/ARG-GNN`, with writes owned by this review directory. No production edits, Git operations, installation, GPU launch or actual cohort/data/test-target reads were performed. Probe processes used the existing `.runtime/env/bin/python`, sourced `scripts/project_env.sh`, empty CUDA visibility and disabled bytecode writes. The review follows the Summary / Major concerns / Minor concerns / Verdict structure requested by AGENTS.md, adapted from the local paper-review skill to a GPT implementation audit. There is no manuscript/literature verdict or Claude review-launch claim.

Exact per-module counts, source hashes, overlapping historical runs and artifact digests are in `receipt.json`. The definitive evidence is `final_joint.log`, `final_joint_junit.xml`, `integrated_summary.json`, source snapshots and the preserved failed probes. No result in this bundle establishes superiority, novelty, publication readiness or scientific confirmation.
'''
(OUT/'review.md').write_text(text)
artifacts={}
for name in ['review.md','concerns.csv','final_joint.log','final_joint_junit.xml','final_source_manifest.json','integrated_summary.json','independent_final.log','independent_final_junit.xml','independent_final.exit_status','first_final_joint.log','first_final_joint_junit.xml','first_final_source_manifest.json','test_workflow_probes_before_harness_correction.py',
'existing_tests_initial.log','probes_initial.log','publication_initial.log','lock_publication_initial.log','publication_recheck.log',
'lock_bytes_initial.log','lock_predictor_publication_final.log','custodian_directory_initial.log','test_workflow_probes.py',
'prediction_pin_context_difference.json','ancestralpaths_README.md']:
    body=(OUT/name).read_bytes();artifacts[name]={'sha256':sha(body),'bytes':len(body)}
receipt={'schema_version':1,'kind':'independent_bounded_confirmatory_workflow_audit_receipt',
    'created_utc':datetime.now(timezone.utc).isoformat(),'status':'passed_synthetic_cpu_audit_with_declared_scientific_limits',
    'project_root':str(ROOT),'owned_directory':str(OUT),'final_source_hashes':core,'all_manifest_hashes_rechecked_after_tests':True,
    'tests':{'joint':{'passed':301,'failed':2,'errors':0,'skipped':0,'elapsed_seconds':elapsed,'per_module':counts,'production_cases_passed':production,'independent_cases_passed':independent-2,'failure_classification':'two_owned_audit_harness_errors_only','process_exit_code':1},
             'independent_final':{'passed':independent,'failed':0,'errors':0,'skipped':0,'elapsed_seconds':rerun_elapsed,'process_exit_code':0},
             'unique_cases_verified_after_harness_corrections':total,
             'historical_overlapping_runs':initial_counts,'lock_predictor_publication_final':{'passed':4,'failed':0,'elapsed_seconds':19.92},
             'custodian_directory_initial':{'passed':1,'failed':0,'deselected':14,'elapsed_seconds':4.89}},
    'integrated_probe':summary,'production_sources_edited':False,'git_operations':False,'installation':False,'gpu_launch':False,
    'actual_data_cohort_or_test_targets_opened':False,'positive_lock_admission_explicitly_mocked':True,
    'positive_cuda_execution_evidence_explicitly_mocked':True,'genuine_dev_selection_calibration_id_lock_or_runtime_qualification':False,
    'scientific_readiness_established':False,'genuine_argformer_ancestralpaths_main_competitor_adequacy_established':False,
    'full_protocol_readiness':False,'review_structure_adapted_from_local_paper_review_skill':True,
    'ancestralpaths_readme':{'url':'https://raw.githubusercontent.com/AliPearson/AncestralPaths/77db3cfae100641c35218b87e1451e14f3074666/README.md',
        'commit':'77db3cfae100641c35218b87e1451e14f3074666','author_code_installed_or_executed':False},
    'artifacts':artifacts,'declared_limit':'Synthetic source/consumer guard evidence only, not an actual ID workflow or scientific result.'}
(OUT/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
print(json.dumps({'production_cases_passed':production,'independent_probes_passed':independent,'unique_cases_verified':total,'joint_elapsed_seconds':elapsed,'independent_elapsed_seconds':rerun_elapsed,'receipt_sha256':sha((OUT/'receipt.json').read_bytes())}))
