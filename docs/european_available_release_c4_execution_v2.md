# European available-release C4 comparison

This is a new executable common-data comparison. It is separate from the native six-path AncestralPaths endpoint and is not a reproduction of the April ARGMix paper tables.

- Data: the same three independent 200 Mb TRAIN parents and five TEST parents as the native AP benchmark, with all 952 haplotypes, 266 pure source references and 686 queries. BAA remains a query cohort.
- Validation: one additional independent 200 Mb parent, `argmix_release_ap200mb_validation_000`, with ancestry, mutation and Relate seeds frozen in recipe v2. Only ARGMix uses it for its native checkpoint selection.
- Endpoint: the four original source ancestries at observed SNPs. C4 TreeLoom is genuinely trained with a new four-output head for 30 passes and 90 updates. No native-six predicted head or checkpoint is collapsed.
- Support: before private labels or predictions, each C4 parent has fixed cohort quotas sampled from retained SNPs in the interior marginal trees. This preserves the available author's default first/last-tree exclusion. The registry records eligible and excluded SNP/tree/bp counts. The primary AP native-six registry is unchanged.
- Truth: canonical sampled C4 labels are joined directly to the original source-class truth and checked against the six-path ontology. Native ARGMix supervision contains every retained observed SNP, including the original resolver's first-SNP semantics and no purity filtering.
- ARGMix: unchanged available-release model, chunk length 1024, stride 512, all native chunks, batch 16, seed 50, original reference augmentation, native loss, AdamW and minimum-validation-loss selection with patience two and at most 20 epochs.
- Archive IO: original Rust ZIP_STORED NPY members are memory mapped in place. A logical concatenation retains parent order and every native chunk. Only indexed rows are copied and cast. The author's TensorDataset, DataLoader, shuffle and complete trainer main bytecode remain intact. No merged or extracted duplicate tensor corpus is written.
- Scoring: both endpoint scorers require all four method/endpoint products on all five TEST parents before any TEST truth. Equality of sampled query/site support is enforced within each endpoint. Metrics include per-cohort and per-class accuracy, precision, recall, F1, raw NLL, Brier and confusion matrices, with equal-parent means and sample SD.

## Integration

`european_argmix_workflow.build_tasks` returns existing AP tasks plus the new C4 stages, replacing only the AP score descriptor with an all-endpoint barrier wrapper. Its 44 new stages bring an isolated AP+C4 DAG to 88 tasks. ROOT supplies runtime qualification receipts, finite stage limits and the current executor callbacks.

Qualified native interpreter: `.runtime/official_argmix_abi210_v1/cu129/env/bin/python`.

Focused CPU fixtures verify direct same-SNP source truth, TEST denial before private metadata, half-open interval boundaries, no prediction denominator edits, exact observed edge eligibility and quotas, actual frozen-DAG factory signatures, and parity against the author's dense archive loader including global shuffled batches and the final partial batch.

No scientific run or completed comparison is claimed by these integration tests. The full native archive size remains data dependent. The adapter fails at explicit resource caps and never reduces chunks, training budgets or common support to fit.

## Remaining published-paper gaps

The April ARGMix paper's exact 1611-person age/ID roster, chr22 map identity/build/domain, genotype error operation, demographic perturbation distributions and historical training/source archive remain unresolved. The available release on the declared AP panel is labelled accordingly.

The GPU adapter applies the admitted allocator limit before native model allocation. Stock CLI inference retains the original child script argv and source files; a sitecustomize.py written only inside the new stage sets the required process title and admitted GPU0 UUID/allocator limit before the GPU child source runs. The explicit CPU TorchScript export does not import Torch or initialize CUDA through this hook. Startup failures exit before the native script, the inference receipt records the exact generated hook reference and SHA256, and the parent PYTHONPATH is restored when the CLI returns or raises.

The CLI command boundary uses a private globals binding of the unchanged native cmd_infer code object. Only its _run_local callback is replaced with the existing ROOT bounded runner, preserving each original argv and working directory while inheriting the owned process group. Every child requires a genuine successful direct wait; the outer supervisor retains final group-drain ownership. The original module binding is restored on return or failure, and each child wait is included in the inference receipt.
