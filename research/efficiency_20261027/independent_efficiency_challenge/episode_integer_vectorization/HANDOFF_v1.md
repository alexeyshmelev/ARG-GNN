# Isolated exact episode integer writes v1

Status: 13 deterministic CPU tests pass. No full-parent run, deployment, active-source edit, GPU work, or throughput claim.

The candidate preserves the original proper-clade traversal, insertion order, episode numbering, arrival/departure flags, public input guards, span reductions, hashes, and metadata. Only the membership loop's integer buffer writes are batched within each native hold. The Python dictionary loop remains. All floating-point operations retain the original function and order.

`integer_writes_v1.py` assigns IDs in original dictionary order, stores node/hold/episode slices, and increments int32 sizes with `np.add.at`, retaining repeated-index behavior. Per-hold temporary arrays/lists are O(H); there are no new full-parent arrays. If conservative member or episode bounds could be reached, the exact original loop preserves validation order and partial mutations. A conservative near-int32-overflow test also returns to that loop to preserve scalar warning behavior. Valid whole-builder inputs have episode lengths at most T < 2**31 and cannot overflow.

`episodes_v1.py` is an isolated builder. It neither replaces live module attributes nor installs hooks. The AST test proves its whole array-builder body matches the original after substituting only the membership block. Another AST test proves the fallback loop is the original source. Structural kernels and FP32 span denominator function identities are checked.

The 13 CPU tests cover tiny complete binary and unary forests, persistent/disappearing/reappearing clades, isolated native roots and empty memberships, all arrays byte-for-byte and complete metadata equality, input rejection messages, buffer-bound rejection order and partial mutations, episode zero, repeated episode IDs, exact int32 boundaries, and warnings. Tests use no scientific inputs. CUDA visibility is cleared before imports. Test body time was 0.016 seconds, excluding imports; this is correctness evidence only.

Next qualification: under ROOT admission, compare the original and candidate complete episode outputs on the same fixed full native parent, preserving all holds, sampled columns, original vertices and spans. Require all array hashes and full metadata to match, with explicitly recorded source pins and memory/timing receipts. Then measure complete learning-cycle wall time and attributable GPU utilization under the intended producer configuration. Synthetic correctness and the existing GIL-contention hypothesis do not establish a throughput benefit.

Invoke tests remotely from the project root with `source scripts/project_env.sh`, `PYTHONPATH=src:.`, `CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`, and `$ARGGNN_PYTHON -B research/efficiency_20261027/independent_efficiency_challenge/episode_integer_vectorization/test_equivalence_v1.py`.
