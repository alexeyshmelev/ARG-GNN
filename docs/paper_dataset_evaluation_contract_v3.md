# Paper-dataset evaluation contract v3

This corrected amendment supersedes the inaccurate v2 amendment and supplements the immutable [v1 contract](paper_dataset_evaluation_contract_v1.md). Its machine-readable counterpart is [v3](../configs/paper_dataset_evaluation_contract_v3.json). It changes no native source, acquires no dataset and executes no experiment. Independent source review is pending. V1's original data/model/split limitations remain unresolved.

Final comparisons require original-paper data and tasks, genuine native implementations and explicit records of every change. Original raw identity, author-recipe resimulation, released workflow, independent confirmation and task transfer remain distinct. Small development results do not fulfill this milestone.

## GST target normalization and label access

The released Buchwald–Hartwig runner computes target statistics from the full 70% training pool before drawing a fraction-specific subset and its validation split. Smaller training fractions therefore use target labels outside their optimizer training rows, including later validation rows. The held-out 30% TEST labels are excluded from these statistics.

A native reproduction retains this behavior and reports its actual label access as released_full_pool_target_statistics. A separately reported correction fits statistics using the final training subset only. That correction changes the scientific training protocol. It must not silently replace the native condition, and the native condition cannot claim a label budget equal to its optimizer rows alone.

Source anchors: eval_react.py target-statistics call at line 1531, fraction/subset/validation construction at lines 1593–1600. Immutable source evidence is preserved under the v1 evidence directory.

## GST training settings

Paper Section 3.1.2 states AdamW with learning rate 0.001 for DeepSets and SetTransformer and 0.0001 for GST outside CIFAR. CIFAR uses Adam and 0.001 for all models. The three cached released task configurations use a single 0.001 argument, applied to every model in the synthetic, reaction-yield and reaction-centre runners. The reaction-yield source also uses Adam, while the paper states AdamW.

Keep released-source common-learning-rate and paper-stated per-model-learning-rate conditions separate. Binding an experiment to one condition does not establish the other. Changing learning rates or optimizers is a scientific training change.

## AncestralPaths published comparator

Supplementary Note 3, printed page 34, compares GNOMix on four admixed query populations: GBR, Bronze Age, Neo and Yam. Only GBR is present-day. BAA is excluded as a query population. References represent EHG, WHG, CHG and Ana path populations. The window is 0.02 cM for GBR and 0.2 cM for Bronze Age, Neo and Yam. Default logistic-regression base and xgboost smoother are sequential modules. They are not alternative model variants.

The paper simulates five 200 Mb sequences and scores evenly spaced sites. It reports per-population means and standard deviations across those sequences and two-sided two-sample T-tests. Exact original seeds, raw inputs, sample identities, site spacing, pinned defaults and complete executable settings remain to be qualified. Published GNOMix/AP outputs never supply ground truth or candidate features. The primary text and its supplement checksum are bound in the v3 machine-readable amendment.

## GST synthetic validation draws

Five checks draw 128 graph sets each from a rolling validation generator: 640 generator draws. Their exact bytes have not been shown to be all distinct. The v1 total_distinct_sets_consumed field is superseded by this draw count. No deduplication or regeneration is authorized by this wording correction.

## Review and execution status

The initial independent audit found omissions in v1 and factual GNOMix wording errors in v2. Both BLOCKED source-contract reviews and any later v3 review are preserved separately. Native source and the historical v1/v2 bytes remain unchanged. A favorable source-contract review would establish documented protocol fidelity only. It would not establish acquired original bytes, a runnable native method, matched scoring or empirical performance.
