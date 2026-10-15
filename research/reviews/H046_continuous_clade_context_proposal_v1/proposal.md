**H046 proposal: continuous clade-flank context with SNP-supported boundary correction**

Source/design only. No implementation, experiment, admission or accuracy claim is authorized by this note. H045 remains unchanged. The candidate scientific question is whether distance-weighted context from topology-equivalent clades helps resolve SNP-supported ancestry changes inside a native tree hold. The proposed distinction is a support-separated conditional readout, not a new averaging or bilinear kernel.

Use the complete learned H042 pass-30 body, including its flag adapter, two full PyG tree sweeps, SNP encoder and genomic BiGRU. Keep every original native tree and vertex. The auxiliary channel reads first-sweep proper-clade states, before genomic recurrence, and changes the original 100-bp-grid readout. It does not inject another bilinear adapter between tree sweeps. Both arms receive identical local SNP features, readout capacity, training exposure and original-grid target mass.

**Physical operator and exclusion contract.** A source-native hold is a fixed observed support J_i=[l_i,r_i). Its identity is part of the original observation contract, not an ancestral-node identity. Keep that support distinct from any smaller numerical integration cells. For each maximal consecutive exact proper-clade episode e, define h_e(y)=h_i on its original holds and f_e(y)=A h_e(y), A:32→4, bias-free. Absent clades break episodes. Root/singleton clades are excluded and unary representatives follow H042. No internal identity is inferred.

For x in J_i inside episode [a_e,b_e), use

    v_L(x) = (1/tau) integral_[a_e,l_i) exp(-(x-y)/tau) f_e(y) dy
    v_R(x) = (1/tau) integral_[r_i,b_e) exp(-(y-x)/tau) f_e(y) dy.

Compute c_L,c_R by replacing f with 1. These dimensionless integrals retain physical support/reliability mass. They are deliberately not divided by c: normalization would cancel the distance-to-focal-hold attenuation on each side. No division by an empty or underflowing pool is needed.

The cross channel excludes *all of the original J_i*. If computing full forward/backward integrals first, subtract the exact focal pieces tau*(1-exp(-(x-l_i)/tau))*f_i and tau*(1-exp(-(r_i-x)/tau))*f_i before dividing by tau. Prefer equivalent endpoint-prefix/suffix formulas to avoid subtracting nearly equal FP32 values.

Use one globally shared, differentiable scale per arm, tau=100+(100000-100)*sigmoid(theta) bp, initialized to 5000 bp. These are concrete registration defaults, not tuned findings. Learn theta on TRAIN only and freeze it for evaluation.

**Exact proof scope.** For a constant field f on a cell of width w, the forward unscaled integral update is F_out=exp(-w/tau)*F_in+tau*(1-exp(-w/tau))*f. Composing widths w1,w2 gives the width-(w1+w2) update. Therefore splitting/merging numerical cells leaves the integral, fixed-support subtraction, fixed-position readout, and their parameter derivatives unchanged in exact arithmetic, provided the field, original J_i, episode membership and learned parameters are unchanged. The statement holds at every TRAIN parameter value under that contract. FP32 accumulation order can change rounding.

This is **integration-mesh invariance**, not whole-model invariance to redefining source-native holds. If refinement also replaces J_i by its smaller child, self-removal changes the estimand and invariance fails. The H045 counterexample still applies. Recomputing hold-pooled SNP/span features or inserting extra BiGRU steps changes h_e(y), so the current backbone has no end-to-end refinement theorem. Keeping the original body execution and semantic supports fixed is essential, not a technical footnote.

A stronger arbitrary-partition contract would use a fixed-bp exclusion [x-delta,x+delta], independent of holds, and a genuinely partition-independent input field. That can retain same-native-tree material outside the exclusion. It must not be called strict cross-native-only context. Do not silently substitute that variant into H046.

**Second research aspect: evidence-supported boundary correction.** Foreign context alone cannot locate an unknown ancestry switch inside an otherwise fixed focal hold. Different within-hold switches can have the same flank context and the same pooled local features. Preserve the actual SNP positions.

At each scored query/center, form four-class reference-only log-compatibility means ell_L,ell_R from the called SNPs in J_i intersected with [x-1000,x) and [x,x+1000) bp. Estimate allele frequencies from observed proxy references only, with a fixed Jeffreys pseudocount of 0.5 per allele. Use sites with a called query and at least one callable reference in each class; missing sites contribute no evidence. Center each four-vector across classes. No query ancestry label or target mass enters these features. Reuse existing H044 observation/allele-cardinality primitives; this is not a new genotype representation.

Let d=ell_R-ell_L. Set g=0 if either side has no usable SNP; otherwise g=||d||²/(||d||²+1). This deterministic gate has no learned weights. It suppresses the added correction when local class-change evidence is absent. It is not an uncertainty calibration or a guarantee of biological correctness: reference misspecification, LD and finite-SNP noise can make d misleading.

For a query, select its deepest proper-clade representative whose episode includes at least two source-native holds. Carry this representative in one root-to-leaf traversal; do not remove nodes. With no eligible representative, the correction is zero. Use the following 27-dimensional feature vector:

    [base_logits(4), mean(ell_L,ell_R)(4), d(4),
     v_L(4), v_R(4), abs(v_R-v_L)(4), c_L, c_R, local_callability(1)].

Local callability is log1p(min(n_L,n_R))/log(100001). Let z average tanh(H*s) and tanh(H*swap(s)), H:27→8. The swap exchanges L/R context and masses and negates d. Add g*tanh(O*z) to the base logits, O:8→4 initialized exactly zero. This gives an exact bound |added logit|<=g and zero added correction when g=0, for every parameter value. The readout is invariant to swapping the supplied flank views. The inherited BiGRU does not thereby acquire a chromosome-reversal theorem.

This tests a bounded, local-evidence-dependent correction using physically meaningful flank contrast. A benefit must be attributed to this particular conditional model; the gate, MLP and boundary-localized SNP evidence are established ideas. No general robustness or recovery guarantee is claimed.

**Matched experiment.** Primary arms are foreign_flank and own_state. Both compute the same foreign integrals. The own arm uses v_L=c_L*A*h_i and v_R=c_R*A*h_i at the same eligible nodes, with the same missing-side handling, masses and local evidence. Retain a zero-gradient foreign link if useful for matching computation. Independent theta values can learn geometry in both arms through c and the mass-scaled features.

Added bias-free weights are A:128, H:216 and O:32, plus theta:1, totaling 377; complete models have 31,165 parameters. Share H/O between the swapped views. Identical new initial weights, exact O-zero, the same H042 state, independent reset AdamW, seed101/FP32 and full-body training are required. Use the existing H044 streamed original-grid exact-span NLL and target/privacy barriers; native-hold mass is no longer sufficient for varying grid outputs. No auxiliary boundary target or extra optimization step is proposed.

Freeze the original 32 TRAIN/eight VAL parents, 30 passes/960 updates, six checkpoints per arm, all96 prediction seals before VAL truth, existing selection/cohorts/scoring support, and no outer access. Retain the practical .001 mean Present182 accuracy gain and <=.01 boundary-F1 loss rule as development criteria, not significance. Report boundary/short-tract effects and local callability strata. A future distance-matched non-clade routing arm would be needed to establish clade-route specificity; the primary pair only tests foreign-state use.

**Work and prior-art limits.** Project states to four channels before integrating. Two segmented affine scans over M clade occurrences cost O(M*4) arithmetic, with O(M*4+E) working tables; affine-map composition permits a work-efficient parallel scan. Do not use a positive-exponent global prefix formula that can overflow, or a dense tree-pair/episode-length matrix. Existing whole-tree work is unchanged. Local SNP prefix statistics cost O(S*Q*4), and grid heads cost O(P*Q) with fixed-width factors; compute in bounded query/center blocks. Exact-clade preparation also has bitset width/hash costs, not a proved node-only bit complexity. GPU scan/autograd efficiency and actual peak memory remain unqualified; no 100-MB or GPU admission is certified here. Only compact receipts/checkpoints should be durable.

Continuous state-space recurrence/convolution is established, including S4 §2.1/§2.3/§2.4. S4 §2.3 uses bilinear discretization, not the exact scalar hold integration derived here. That elementary scalar ODE semigroup is not proposed as a new principle. GMN §3.2 already exchanges cross-graph states before local computation. GST §3.1 already pools/exchanges/broadcasts graph context. Whitehouse2024 Fig.1 already combines tree GCNs and a tree-sequence GRU; ARGformer §2.1 already exploits shared genealogical structure, and ARGMix Methods4.1/4.3 already targets tree-based ancestry. H046's narrower candidate distinction is source-canonical clade alignment plus support-excluded physical flank context used for SNP-supported, bounded within-hold corrections. Its priority and utility remain unestablished. Register that scientific condition, not a claim of a new temporal kernel.
