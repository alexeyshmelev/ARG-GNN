# Independent confirmatory workflow audit

## Summary and verdict

The repaired bounded ID implementation passed **288 production cases** in the joint CPU run. That run had 301 passes and 2 audit-harness failures in 366.54 seconds. After correcting only the audit harness, **15 independent probes passed in 90.13 seconds**. This verifies 303 distinct cases across those two runs, not a single all-passing joint run. The five concrete implementation findings identified below were repaired and rechecked. No unresolved implementation failure remains within these probes. This is evidence about implementation guards and synthetic interoperability, not a genuine confirmatory experiment or scientific-readiness verdict.

The final frozen sources are recorded in `final_source_manifest.json` and preserved in `final_source_snapshot/`. Core SHA256s are:

| Source | SHA256 |
|---|---|
| Lock | da843329cab737b6c6bf5fffab318efe58db988439964246997688477681d977 |
| Predictor | a63bd3dc24954dc9f35e8b30401d07f82ec437b64dbec32ce9449232e6f2dfce |
| Analysis | dfde4cf9616c0e5deea3340c38efac369191a83182411df5f68a5c5047e5dd11 |
| Custodian | efddcdaac64608ea1bd1b57886940d4bcb3a9fddbaa41fddbdec8b4ae331b7b3 |

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
