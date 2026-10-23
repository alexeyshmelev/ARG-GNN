# GST file and paper agreement audit

Reviewed 24 October 2026. Paper: arXiv:2606.05116v1. Current author source: `6469d56aa131463325058b897b1e435c6d2edcd2`. A fresh check of the arXiv version history still identifies v1.

## Scope and review depth

The current public tree contains 166 files. Every payload has a coverage row and a verified hash matching the public Git tree. All 154 text/empty files were scanned, including 20 Python scientific routes, 52 JSON records, 20 CSV records and two TOML files. Twelve PNG files were inspected by hash and image header, not visually. This is static source and artifact review. It does not certify a successful runtime or replicate published scores.

The prior audit retains all 18 reachable public commits and the deleted GSTv2/CIFAR material. This pass reuses those acquisitions and records exact source references in `agreement_matrix.json`.

## Details now resolved

- For each model, task cell and seed, the saved synthetic schedule accounts for 40,000 training sets, 640 cumulative validation sets and 512 test sets. Validation visits use fresh batches. The 640 total is not a fixed validation set reused at every visit.
- The reaction-yield learning-curve branch, selected with `--train-size-ablation`, reserves 30 percent for TEST. Its inner validation fraction is 10 percent, capped at 50 records. Calculation from the saved 3,955 reactions exactly reproduces all seven printed training counts: 90, 179, 357, 741, 1136, 1928 and 2718. The default 70/10/20 branch is a different supported condition.
- Graph convolutions, graph normalization, graph-token attention, set masks, target functions and reaction atom labels have source definitions. These must be preserved for a paired comparison.
- Deleted CIFAR scripts and GSTv2 are recoverable from commit `d8fd56a797998527b32e85b6b7469bf307161d60`. They are not missing public code, but they are not automatically interchangeable with the current model.

## Conflicts and conditions that remain material

The paper's equation uses two gates. Current `gst.py` uses one set gate and an ungated home token. Historical GSTv2 contains two gates but also adds local feedforward and LayerScale operations. An undefined intermediate in the paper leaves the exact published architecture unresolved. A module alias cannot establish equivalence.

The paper gives GST a learning rate of 0.0001 and its pipeline baselines 0.001. Current runners expose one all-model rate. The BH runner uses Adam where the paper prose says AdamW. Native summaries report Student-t confidence half-widths, while paper tables describe standard deviations. These quantities must keep distinct labels.

The committed USPTO run uses only the first 1,000 retained training reactions. It is an author pilot, not evidence for the full Table 3 condition. Current BH and synthetic-v2 scripts import a deleted module, requiring an explicitly pinned historical composition before execution.

BH normalization accesses labels from the entire 70 percent training pool before drawing each low-data subset. It uses no TEST labels, but its label access exceeds the subset size. The source-native condition must disclose this. A subset-only normalization experiment would be a separate sensitivity analysis.

Model construction occurs before per-training seeding. CIFAR sets can reuse graph members and include repeated sampled members. These rules affect reproducibility and dependence, and must be fixed identically for both paired methods.

## Prepared next steps

`remediation.json` specifies inactive, separately named native synthetic, full-data USPTO, paper-rate, BH historical-model and CIFAR historical-source conditions. It lists the missing interface and qualification work. No source pins, running jobs, scientific settings or model weights were changed by this audit.

Our candidate still needs valid adapters and full paired evaluation on GST's tasks. In unordered molecular sets, persistent ancestry identities or an artificial genomic order cannot be invented. GST is absent from the active LAI queue. Neither code availability nor this audit closes that benchmark requirement.

## Evidence

- `file_coverage.json` and `.csv`: every current file, hash and review mode.
- `agreement_matrix.json`: 21 paper/source items with exact references and resolution status.
- `count_derivations.json`: all synthetic totals and seven BH count calculations.
- `paper_version_check.json`: fresh arXiv retrieval receipt.
- `remediation.json`: proposed conditions, all inactive.
- Prior source/history inventory: `research/reviews/author_repository_deep_audit_20261023_v1/gst/` from the project root.
