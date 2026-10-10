# Independent confirmatory workflow audit — working record

## Summary and scope

This is a bounded source and synthetic CPU implementation audit of the ID custody, prediction and joined-analysis workflow. It is not a peer review of a manuscript or evidence that a confirmatory experiment occurred. All project reads and writes were on anogena under `/workspace-SR003.nfs2/ARG-GNN`. The reviewer owns only this audit directory. No production code, Git, environment installation or GPU launch was performed, and no actual cohort/data/test targets were opened.

The review uses the Summary / Major concerns / Minor concerns / Verdict structure requested by AGENTS.md, adapted from the local paper-review skill to a GPT implementation reviewer. No Claude review launcher, paper extraction or literature review applies to this source-level task. The reviewer did not seek a prescribed verdict.

The primary initial sources were frozen lock d7d1f61985a4b7b3675d537a3a46a2cb5b63273409a1c5acb74375be30283b18, predictor bf3900433f690cffc94cf4bda6aa49dc064acb7bf201ce2b3ee66fe5603a5188, analysis dbba5641e94e785442344d46f0391bd98791e8c7af7ed318b2ac93018f3ca1b0 and custodian fbe252ec1a50855a7392160f05eff5a628952b7583c55d51f99f03ed6074a07c. Original sources, intermediate repairs, failed probes and fixture errors are preserved. Final sources and joint counts are pending the shared freeze.

## Major concerns discovered

1. **CUDA UUID formatting:** the original predictor compares Torch's raw UUID with a GPU-prefixed lease UUID literally. The main audit observed exactly those two spellings for the same physical device. This blocks legitimate CUDA execution before predictions. Root repaired the shared comparator. No GPU was initialized by this reviewer.
2. **Temporary path ownership:** predictor and lock emission could link a replaced temporary path, return the SHA of the intended content and delete unrelated replacement bytes. Direct expected-rejection probes failed for both, while analysis's inode guard rejected the same race. Root's initial inode repair passed all three replacement probes.
3. **Staged byte drift:** modifying a staged file without replacing its inode is still accepted by all four publishers. Predictor, analysis and custodian direct probes failed, and a direct binder probe also failed. The published bytes then differ from the digest in the returned artifact or enclosing receipt. This remains separate from inode ownership. Repairs and final rechecks are pending.
4. **Execution provenance at join:** successful raw prediction NPZs do not bind a terminal runtime receipt. Real synthetic CPU-only predictions are structurally eligible for the current join, even though their terminal receipts explicitly say CPU validation only. This disconnect must be closed before qualification of a frozen CUDA workflow. Root assigned a coordinated schema/checker/sealer repair.

## Minor concern and policy clarification

The analysis validates any seven sorted internal tile edges, while the predictor and custodian require the fixed 125000-bp multiples. An isolated analysis fixture with [1,2,3,4,5,6,7] bp is admitted. Those wrong edges would alter descriptive edge/interior metrics. A direct rejection probe failed. This is a consumer validation gap, not evidence that a real custodian release used wrong edges.

A proposed probe expected secondary adjusted favorable flags to be suppressed for a three-parent shortfall. The implementation retains those flags, while primary robust confirmation is false. The v1 protocol clearly gates primary precision but does not unambiguously extend that rule to every secondary adjusted flag. The failed expectation is preserved and classified as policy ambiguity, not a demonstrated code defect.

## Strengths and evidence limits

The existing five-module CPU suite passed 204 tests in 160.41 seconds before joint repair. Source inspection shows complete method × seed × admitted-parent roster checks, contiguous first160-or-all192 admission, independent parent averaging, raw uncapped float64 NLL, fixed Holm family size, external census agreement, mandatory frozen VAL temperatures, failure preservation and a primary robust-positive guard requiring completeness and precision. Unresolved cells reject joining before target reads. Every successful prediction is preflighted before the first target opens.

The independent integrated fixture executed actual observed star-tree electrical/canonical QA, 24 real CPU frozen-model cells for two handmade parents, 48 separate 128/4096 grid forwards and 384 fixed32 anchor preparations, without target reads during prediction. Its first run stopped at an erroneous audit assertion expecting 288 anchor preparations: the deterministic smoother also uses fixed anchors. That is a reviewer fixture error and is not attributed to production. Joined execution will be rerun after the terminal-receipt schema freezes.

All positive fixtures explicitly mock genuine completed-cohort, development selection, temperature-fitting and published-lock admission. Some existing binder tests additionally mock producer/factory/secondary admission. Handmade checkpoint files and real CPU forwards establish implementation behavior only. There is no real DEV selection/calibration/ID lock/runtime qualification here. The actual lock validator separately rejects the handmade lock before imports or targets.

The workflow deliberately imports already prepared artifacts rather than generating them. Native inference return codes remain upstream recorded claims. OS permission isolation, historical registry completeness and one scientific joined execution remain operational/review responsibilities. Robustness, OOD, query-count, finite-source/tile sensitivity and illustrations are explicitly pending, with full_protocol_readiness=false. These declared limits are not defects.

## Verdict

Pending final coordinated source freeze and joint synthetic CPU rechecks. The original implementation is not ready for workflow qualification because of the concrete publication and execution-provenance defects. No scientific outcome, superiority, novelty or genuine confirmatory workflow claim follows from these tests.
