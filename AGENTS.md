# ARG-GNN research instructions

## Scope and isolation
All project work is performed on anogena in /workspace-SR003.nfs2/ARG-GNN.
All server writes must remain inside this repository, including environments, package caches, temporary files, data, checkpoints, logs and manuscript outputs.
Do not write project files on the controlling Mac.
Reading installed tools, the provided SSH identity, and public source material is allowed.
Do not edit external user files, shared environments or global Git configuration.
Use the existing conda executable at /home/jovyan/miniconda3/bin/conda with scripts/project_env.sh.
Do not write credentials or key passphrases into project files, Git, command logs or research artifacts.

## Resources
Total repository storage, including ignored files and the environment, must remain below 1 TB, as authorized by the user.
Use a 950 GB warning threshold and stop bulk writes before reaching 1 TB.
At most two distinct physical GPUs may be used by this project at once.
The user explicitly allows multiple model worker processes to share those devices.
Use an audited shared-lease pool with memory reservations, separate logical job records and prescribed GPU process titles for concurrent workers.
Preserve existing pinned model/runtime guards while introducing scheduling changes.
New resource admissions use research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py with research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json and the bound root marker-repair authority. This prospective guard was activated by root from actual completed-credit-v3 prefix110, projection949921787041bytes. Original v3 source/policy/journal/charges and live consumers remain unchanged. All future old-v3 reserve/close mutations are cooperatively revoked. New accounting warns at950GB and rejects at980GB, retaining more than20GB below the human1TB limit. Old parent prefixes must remain frozen. Existing admitted old-guard writers may finish within their original caps. All future accounting admissions through capacity-v2 are revoked. Existing live GPU 4/5 workers retain their original guards and full scope charges. Completed-scope credits require exact original completion/direct-wait evidence and targeted root process/revocation certificates, without deletion or historical journal changes. The successor preserves every historical guard and journal, rejects the malformed historical reservation and the retired original failed H039 scope, and retains all failed artifacts. All previous-guard future launches are revoked. Fresh reservations must bind the actual project_budget_v9_consumer.py marker, repair helper and repair authority, capacity-v1 and capacity-v2 sources, and actual runtime/producer/launcher sources. Root owns admissions and execution.
For the single never-spawned allocation H041/American-C3-native-MLM-full30mb-qualify_mlm-v2, root may activate one execution-only v6 amendment with separately authenticated actual sources and an explicit root directive. All original v4/v5 launch, resume and write capabilities for this key are revoked. The original allocation, 15032385536-byte charge, scopes, science, guards and absolute expiry remain unchanged. No new reservation or manufactured completion/credit is allowed. The original outer OS wait remains unknown. Frozen old entrypoints do not mechanically consult the amendment, so this exception uses explicit cooperative root revocation and a fresh complete old-writer absence certificate. All new execution boundaries must validate the active single-use v6 attempt and inherited lifetime lock. Root owns activation and execution.
The v6 execution-only attempt for that same key failed before any leased/native child because an independent read-only exclusive flock probe raises EBADF on NFS. Its consumed token, attempt and genuine outer wait1 remain immutable. Root may authorize one separately authenticated v7 operational successor with a writable exclusion probe, requiring the actual NFS probe receipt, exact preserved v6 failure lineage and fresh absence of both old roots. All v4/v5/v6 launch, resume and write capabilities for this key are revoked. V7 does not reset any old token, original storage charge, science, scope or expiry. It needs a new explicit root directive, authority and single-use inherited lifetime lock. No automatic replay or failed-attempt credit is allowed.
Only physical GPUs 4 and 5 are eligible under the latest user instruction on 2026-10-15. The user authorized sharing with other projects on 2026-10-12. Verify available memory and preserve all external processes. The current project scheduler uses physical GPUs 4 and 5, with at most two project GPUs at once. Previous assignments to GPUs 2 and 3 are revoked.
GPU 0, GPU 1, GPU 2, GPU 3 and GPU 6 are excluded.
GPU jobs must set the process title AI Auto Research Pipeline Test before creating CUDA contexts.
Use the project GPU launcher, record physical GPU assignment and verify the process name in NVTOP.
Do not terminate or change other users' processes.

## Scientific integrity
All new experiments with our models use PyTorch and PyTorch Geometric. Competitor experiments use the authors' repository code, with necessary adaptations disclosed.
Separate hypotheses, implementations, pilot observations and validated claims.
Never fabricate results, references or reviewer verdicts.
Split simulation replicates and subjects before windowing or sampling.
Keep ground-truth annotations out of model input features.
Keep tuning, validation and test data separate. Freeze evaluation protocols before full experiments.
Record dataset provenance, licences, source commits, environments, seeds and configuration.
Before each substantive experiment, consult research/experiment_ledger.md and register its hypothesis or an explicit repeat reason in research/experiment_decisions.json.
Link repeats to prior hypothesis IDs and identify the new condition, implementation correction, unfinished cell or independent confirmation. Do not rerun an unchanged answered pilot.
Record concise outcomes, including negative and partial findings, then refresh scripts/update_research_ledger.py and commit/push compact receipts with meaningful progress.
Do not claim an exact baseline reproduction when code or parameters differ from the paper.
The manuscript focuses on supported contributions and informative results. Preserve complete experiment records separately.

## Collaboration and publishing
The primary agent owns Git commits, pushes, integration and GPU scheduling.
Other agents must not commit, push, install packages, or launch GPU jobs without explicit assignment.
Use dedicated file ownership when agents implement in parallel.
Make regular meaningful commits and push progress to origin.
Paper reviewers start with fresh context and receive the paper plus auditable evidence, without desired verdict instructions.
Use the paper-review skill's review structure and concern synthesis, adapting its Claude-specific execution to available agents when necessary and disclosing that adaptation.

## Manuscript
Use the official AAAI LaTeX template and provide an Overleaf-ready source bundle.
Aim for about 20 pages total or less and respect tighter limits of the selected venue.
Avoid semicolons in manuscript prose.
All tables and figures must trace to actual recorded experiment outputs.

## Required progress reporting and parallel work

The user requires current scores and experiment status both in chat and the main README, without needing repeated reminders. After each meaningful completed evaluation, training milestone, failure or repaired execution, run scripts/update_research_ledger.py and scripts/update_current_scoreboard.py, inspect the report, commit and push. Report the measured result and practical implication in chat. Include every current setup, its dataset/cohort/support, accuracy and other available metrics, sample SD with its unit, optimizer/training budget, and explicit pending/running/failed states. Do not report training loss or engineering checks as ancestry accuracy. Keep current-score details and the complete historical metric receipt inventory linked prominently in README.

Use all available independent workers in parallel when useful. Root orchestrates actual resource admissions/execution, integration, Git and reporting. Give workers disjoint source/repair tasks and keep other work moving while one repair is under review. Do not serialize independent work merely because one benchmark remains unfinished. Retain the two-physical-GPU limit and existing execution guards.

For shared-data comparisons, derive each native method's required representation from the same frozen observed data and reference/query split, then score the same held-out queries against the same truth/support. Preserve author algorithms/settings and disclose adaptations. Internal tensor/path/VCF formats can differ. Method-specific representations do not justify postponing native comparisons. Distinguish pipeline performance from architecture-only attribution and exact original-paper reproduction.

## Narrow native ARGformer operational successor

Root authorizes a source-only v10 operational successor to inactive v9 for the original same-charge full30Mb native MLM allocation. Preserve all frozen v8/v9 files, original allocation/expiry, native science and guards. The host denies reading some unrelated same-UID process metadata, so v9's complete-metadata absence claim is unachievable and must not be asserted. V10 may instead use exact known-old-root/group ESRCH, complete stable current same-UID PID/start enumeration, no exact old job/source/scope tie in every readable metadata field, exact own kernel-listener absence, and a root-approved allowlist of specific opaque PID/start identities and denied fields. No blanket unreadable-process skip is authorized. Unexpected opaque identities, any observable match, identity churn, live known roots/groups or kernel listeners must block. Root accepts the explicitly recorded residual uncertainty for those exact unrelated opaque fields plus cooperative future v8/v9 capability revocation. Historical descendants remain incomplete and prior native exits/optimizer updates remain UNKNOWN. Only root may approve the actual opaque identities, archive, activate and launch. No new storage charge, scientific changes, privilege escalation or external modifications are authorized by this exception.

## Native ARGformer v11 isolated namespace execution

Root authorizes the reviewed v11 operational namespace successor for original key H041/American-C3-native-MLM-full30mb-mlm-fit-v8. All prior launch/resume/write attempts are cooperatively revoked. Preserve every old regular/scientific file and the two partial-archive metadata files. Root may unlink only the eight exact previously recorded original TMP socket nodes after preserving metadata, fresh known-root/group ESRCH, exact lstat/canonical checks and exact kernel endpoint absence before each unlink. Do not claim historical drain or complete old-writer absence. No global unrelated-process FD stability prerequisite applies. Use fresh child output/TMP/AUX namespaces inside the original charged parents. Every original parent remains fully measured by unchanged guards. Original14GiB native and5MiB admin charges and expiries remain unchanged. Native model/objective/corpus/configuration/seed/batch/precision/5epoch/224775update science stays exact. Current-attempt wait/drain/counters remain mandatory, and parent completion may be written only after genuine successful child completion and launcher wait0. Root owns actual cleanup, activation, launch and close. Read-only admin boundary checks may use the pinned consumer memo2, with no accounting mutation or guard-source changes.
