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
New resource admissions use research/reviews/H039_missing_marker_repair_v1/marker_repair_v1.py configured with the root repair_authority.json. Original guards cannot replay the one preserved malformed reservation. All old-guard future launches are revoked. Fresh reservations must include the actual project_budget_v9_consumer.py marker, repair helper and repair authority. Root owns this transition.
Only physical GPUs 2, 3, 4 and 5 are eligible. The user authorized sharing with other projects on 2026-10-12. Verify available memory and preserve all external processes. The current project scheduler uses physical GPUs 2 and 3, with at most two project GPUs at once.
GPU 0, GPU 1 and GPU 6 are excluded.
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
