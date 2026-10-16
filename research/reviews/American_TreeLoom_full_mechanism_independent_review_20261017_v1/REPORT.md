# Independent American full TreeLoom source review

SOURCE_CHANGES_REQUIRED for model SHA256 `3d5a7b29b2b6eb9cd62696d051324a4b5219ccbc8374492d952558da273b2f01`. Two narrow training-interface defects need correction before this callable can support its declared training contract.

- **P2, lines422–431:** `AdamW(maximize=True)` passes the optimizer checks and ascends focal CE. Require `group["maximize"] is False`. For the claimed original variant, also require `group["amsgrad"] is False`.
- **P2, lines426–445:** frozen memory/structural weights remain in the optimizer and pass parameter identity/count checks. The gradient check then filters them out. Require current `all(p.requires_grad for p in model.parameters())` and inspect gradients across the complete parameter list.

The source encoder splice matches final European H050 after harmless AST normalization. It retains two shared complete-tree cavity sweeps, both rank8 local residuals, exact stable-haplotype memory and full-parent bidirectional GRU equations. The C3 observation adaptation preserves the original600-reference/1000-query mask, age350 rules, alpha0.5, physical memory scale and half-open focal join. No direct feature/label leakage was found inside the trusted external observation/target authentication boundary. The simpler saved600 AmericanC3WholeTreeGNNV1 remains a distinct result.

Independent checks authenticated all9 source-manifest entries and all49 import-closure entries, compared the final encoder splice and genomic geometry ASTs, derived32132 parameters/49 tensors algebraically, checked handmade physical-decay identities, and evaluated the actual optimizer predicate against each counterexample using opaque tokens. No scientific module or numerical runtime was imported.

Candidate manifest SHA256 `1c7dfc4975c771fba9d53e40e309c8d19b319e1d69e56a03e46c24fa7ceaf26f`; handoff SHA256 `4a378ab3dd21a1ce22fe6762f7e6a9e8ee9d5c58607ee611970a4a854e400ecb`. Detailed pins and evidence are in `review.v1.json` and `static_checks.v1.json`.

Limits: source only. No data, truth, checkpoints, model construction, forward/backward, CUDA, jobs, guards or Git. Resource fit, real gradients, staged training/initializer provenance and a new30-cell pretruth seal remain unqualified external integration. The proposed focal-CE2400-update transfer is disclosed as different from Europe's span-CE3840-update lineage. This review grants no execution or scientific result.
