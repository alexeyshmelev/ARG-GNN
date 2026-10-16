# Minimal inactive v2 train-step boundary repair

This successor changes only the three train-step assertions requested by independent findings TL-C3-01 and TL-C3-02. The AdamW group must have maximize=False and amsgrad=False. Every model parameter must currently require gradients. Gradient checks now inspect the complete parameter list without filtering frozen parameters.

The exact two text edits are in source_delta.v2.patch. Static reversal restores v1 byte-for-byte, and every module AST node outside train_complete_parent_step_v1 is identical. Existing parameter identity/count checks, finite/nonzero gradient checks, encoder, SNP features, topology recipes,350cut semantics, shared sweeps, stable-leaf memory, genomic geometry, model, mean CE and optimizer update math remain unchanged. Checkpoint family and scientific condition are unchanged.

The actual new group predicate was extracted and evaluated using49opaque Python parameter tokens. The declared minimizing/default AdamW variant passes. maximize=True, amsgrad=True and frozen parameters fail before computation. The gradient comprehension has no requires_grad exclusion. No Torch, PyG, NumPy, model, data, checkpoint or truth package/payload was imported or opened.

Sealed v1 and all its sources/artifacts remain intact. Scientific closure v2 replaces only the own-source pin; the48external scientific source pins are unchanged. This packet grants no execution or admission.

Next action: independent delta review, then ROOT's separately authorized practical CPU FP32 implementation check. No jobs, guards, GPU or Git actions occurred.
