# Categorical execution of the current TreeLoom SNP encoder

**Prototype passed the authorized bounded CPU comparison. No live integration was performed.** This is an exact execution factorization of the existing model, with ordinary FP32 rounding tolerance; it is not a new scientific architecture, an approximation or a change to observations.

## Change

Current C3 _site_chunk constructs the same 15-dimensional MLP input separately for each site/haplotype row. Those features depend on the site, observed allele and called status: six donor log frequencies, six donor-availability flags, panel log probability, site entropy and a called flag. Haplotype identity is absent from the MLP input.

categorical_site.py constructs those features once per site/allele category, evaluates the unchanged current site_mlp, and gathers the encoded values into the original site/haplotype order. It then uses the original index_add sums/counts and scatter_reduce amax, including the original reduction order and max-tie multiplicity. There is no runtime global hash and no learned-value cache across calls or parameter updates. The allele axis remains general; the measured data happened to use three allele slots.

Missing calls gather safely but remain excluded from every sum, maximum and count. Unused category slots receive finite neutral inputs: panel probability is replaced by one before log, then the unused feature row is zeroed. This prevents log(0) padding from contaminating parameter gradients.

The factoring contract is the current row-independent Linear/GELU/Linear/Tanh MLP. Dropout or batch-dependent layers would require reevaluation.

## Bounded evidence

Inputs were the unchanged observed first 128 American TRAIN000 trees / 681 SNPs / 1,600 haplotypes packet and the authenticated update94 checkpoint. One real 256-site chunk received forward and parameter-gradient comparisons. One small 8-site / 32-haplotype case covered an all-missing hold, mixed calls, unavailable donor masks, repeated identical rows giving max ties, and an unused fourth allele with zero panel frequency. One forward per method covered the complete 681-site slice with the original global256-site chunk partition, cross-chunk hold pooling and trailing169-site chunk.

Everything used FP32 on four CPU threads. No CUDA initialization, optimizer, truth access, source patch, new training run or simulation change occurred.

| Comparison | Result |
|---|---:|
| Real chunk MLP rows | 409,600 → 768 |
| Real chunk sums / maxima / counts | Exact outputs |
| Real chunk maximum parameter-gradient absolute error | 2.35e-07 |
| Mixed-missing forward outputs | Exact |
| Mixed-missing maximum parameter-gradient absolute error | 3.35e-08 |
| Complete 681-site projected output | Exact |
| Required tolerance | atol=rtol=1e-4 |

The gradient probe used a weighted mean of per-hold mean and maximum channels. Gather backward accumulates all original haplotype uses into each category before MLP backward. Small FP32 gradient differences are expected from changing accumulation and GEMM shapes; bitwise gradient equality is not claimed.

## CPU timing

First-call figures are the first invocation of each method in one shared process. Warm forward medians use three interleaved measurements on the same real chunk; forward/backward has one measured probe per method.

| Scope | Current | Categorical | Ratio |
|---|---:|---:|---:|
| Chunk first forward call | 222.9 ms | 111.6 ms | 2.00× |
| Chunk warm forward median | 189.3 ms | 107.0 ms | 1.77× |
| Chunk forward + parameter backward | 525.6 ms | 265.9 ms | 1.98× |
| 681-site projected slice forward | 578.7 ms | 234.3 ms | 2.47× |

Peak process RSS was 1.40 GB, including the existing Python/Torch runtime and both models. These are encoder CPU measurements, not GPU or whole-training speedups. The large MLP-row reduction does not remove the original gathered tensor or pooling work.

## Integration remains open

The diagnostic pooling adapter preserves source chunk boundaries and reductions but does not add the existing training checkpoint wrapper; production memory and whole-update performance remain unmeasured. A separately reviewed successor should retain that wrapper and all full-parent structural/history behavior. The packet's terminal partial global256-site chunk omits neighboring holds, as documented in the packet receipt; no full-parent floating-kernel parity claim follows.

Authoritative hashes, parameters, per-tensor gradient errors, timings, scope and limits are in measurement.json and artifact_manifest.json. The independent source review is owned separately by the model agent. No further tests or benchmarks were run after this passing comparison.
