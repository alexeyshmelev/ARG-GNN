# Inactive source correction review

These diffs were authored and checked as text only. They have not been applied to any author or execution source tree. Both require the exact source SHA256 in `patches_manifest.json`, which also records the hypothetical target SHA256 and an independent in-memory application check.

`attention_key_j.inactive.patch` changes the relative-key term from `k_i dot P_key[psi(i,j)]` to the paper equation's `k_j dot P_key[psi(i,j)]`. The transposed gather restores the original asymmetric distance after indexing key j. It does not replace psi(i,j) with psi(j,i). A term constant across an attention row would cancel in softmax, but this release term generally varies across j through distance buckets. No model accuracy consequence has been measured.

`fast_linear_clamp.inactive.patch` clips the wide absolute time difference before conversion to u16. At a cap of 1,000, the existing linear option returns zero for a 65,536-generation difference and one for 65,537. The proposed expression returns 1,000 for both. Missing-LCA behavior is retained. The default log-scale available-release condition is not changed by this proposal.

These are deterministic, narrow source proposals with scientific consequences requiring separate qualification. The key-index change defines an equation-aligned algorithm variant. The clamp change repairs the raw linear option. Neither proves original paper reproduction, restores missing weights or justifies a performance claim. The unchanged available-release baseline must remain available and accurately named.

No imports of author code, compilation, author tests, numerical model tests, simulation, training, inference, GPU allocation or process changes occurred. The only executed checker was this visible standard-library text transformer. Root's standard-library arithmetic examples are independent logical illustrations, not author runtime validation. Independently, Root ran `git apply --check` for each exact diff and recorded exit 0 in `../static_counterexamples.json` under `inactive_patch_applicability`; these applicability checks applied neither patch and are separate from this worker's in-memory text checker.
