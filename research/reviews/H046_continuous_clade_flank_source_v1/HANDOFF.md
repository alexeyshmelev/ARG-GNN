# H046 continuous clade flanks: source handoff

Source and synthetic CPU qualification only. Root owns admission, device choice, execution and actual resource qualification. H045 live sources and outputs are unchanged.

The model is src/arggnn/continuous_clade_flank_gnn_v1.py; the paired worker is src/arggnn/train_continuous_clade_flank_v1.py. The inactive integration packet is in research/reviews/H046_GPU 45_flank_root_source_v1/.

The complete trainable H042 pass30 body retains both actual PyG sweeps, SNP encoding and the genomic BiGRU. Auxiliary flanks read the first-sweep proper-clade states. Exact descendant sets form maximal consecutive episodes; absence splits an episode, roots and singletons are excluded, deepest unary representatives are used, and all original vertices remain. No H045 implementation import is used.

Two work-efficient recursive affine scans compute exclusive left/right exponential integrals over original source-native supports. The entire focal original native hold is excluded. Distance from the focal hold edge to the grid position adds attenuation. Integrals are normalized by tau, never by foreign mass. Tau is constrained to100–100000bp, initialized5000bp, and learned in TRAIN. The own-state control substitutes focal projected state times the same foreign mass and retains the same computed pools and parameterization.

The exact refinement scope holds the learned field, original focal support, episode membership and physical positions fixed. It is not whole-body or arbitrary native-hold subdivision invariance.

Local evidence uses original integer SNP positions, categorical Jeffreys reference frequencies and original allele cardinality, with both 1000bp sides clipped to the focal original hold. A site is usable only when the query and all four reference classes are callable. FP32 prefixes restart every 1024 SNPs. The 27-feature head is averaged under flank swap; its output projection starts at zero, and a gate forces zero correction if either side lacks usable SNPs. The correction has magnitude at most the gate per logit. Added parameters 377; total 31165.

The worker consumes the immutable H044 all40 batched observation-only coordinate/allele products, prepares independent RAM-only episode arrays, and seals all phase observation recipes before phase truth. Both arms start from identical H042 weights with separate reset AdamW states. Each of 30 passes uses the registered 32-parent order, for 960 actual updates per arm. Targets stream once per original-grid parent and are shared between arms. Full[P,686,4] logits and their autograd graph are materialized. First-parent GPU qualification occurs inside the continuous schedule; there are no extra pilot updates.

Evaluation seals all 96 original dense grid probability products before any VAL truth. Selected models retain the original full diagnostics and fixed Present182 callability strata min(nL,nR):0,1–2,3–5,>=6 usable SNPs. Strata report original span support, accuracy, NLL, missing support and across-parent variability; they never revise selection. The root clarification defines the practical criterion as equal-parent mean accuracy gain>=.001 and equal-parent mean boundary-F1 loss<=.01 across 8 VAL parents. It is not a worst-parent bound or significance claim.

APIs: build_clade_flank_arrays_v1; ContinuousCladeFlanksV1.from_arrays/.to/.tensor_bytes/.device_bytes; prepare_continuous_clade_bank_v1; H046BoundCladeBankV1.validate/.to; segmented_flank_integrals_v1; ContinuousCladeFlankGNNV1; caller-payload-only load_h042_flank_warmstart_v1; make_matched_flank_family_v1; logical_flank_bounds_v1. Worker entry is run_context_v1(ctx,input_ref), requiring before_flank_build(pid,bank,declared) and original H044 preparation completion/direct-wait refs.

The model focused CPU check passed 34 checks. The worker focused CPU check passed 28 checks, including actual synthetic updates, barrier/privacy failures and actual observation-prefix selected diagnostics across query chunks. Receipts and check sources are in this directory. No actual data, pretrained checkpoint, GT/VAL content, GPU, admission or guard action was used for these checks.

Resource formulas are named-tensor bounds, not an allocator/autograd or GPU-fit proof:
- CPU compact payload:6N+14M+4E+4T+40P bytes; upper bound24N+4T+40P, since M,E<=N.
- GPU compact payload excludes CPU windows:6N+14M+4E+4T+16P.
- Projected member table16M; two 5-channel endpoint tables40M; conservative scan allowance1024M.
- CPU prefix20*ceil(S/1024)*1025*B; construction allowance128*(S+1024)*B.
- Retained CPU local checkpoint features40*P*686, up to 548800000B per parent.
- One full logits or target tensor16*P*686, up to 219520000B.
- All96 dense probability products at P=20000:21073920000B.

The inactive packet proposes 128 GiB CPU-bank and 32 GiB prediction-RAM caps, bounded independent marker/flank LRUs, 512000000000B process RSS and 50000000000B CUDA reserved limits. Its phase output cap is100000000B. The output planner uses supplied original count metadata and H044 preparation counts; no corpus counts were executed here. Root must review actual counts, freeze caps/status, select physical GPU 4 or 5 and retain the current completed-v3 guard/memo/original-v4/launcher pins. Actual first-parent GPU and whole-body autograd peak remain unqualified.
