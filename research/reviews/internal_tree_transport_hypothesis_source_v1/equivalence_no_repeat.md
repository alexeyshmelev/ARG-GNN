**Raw internal-node overlap transport: exact reduction and no-repeat decision**

This proposal should not be implemented as a fresh operator. The implicit algorithm is useful, but its normalized kernel is a specialization of the clade-incidence kernel already derived in docs/lineage_kernel_operator_audit.md. The existing src/arggnn/models.py LineageCrossAttention also already sends all-node foreign context into tree computation using content, same-sample and descendant-Jaccard scores. The raw-overlap formula below is not numerically identical to that softmax/Jaccard implementation. It lies in the earlier explicitly proposed finite-feature structural-kernel family.

Let A_o[i,u] indicate that stable observed haplotype i descends from old node u, and A_n[i,v] do the same in the new tree. For fixed nonnegative source weights a_u, let

    W = A_n^T A_o diag(a),  d = W 1,
    K[v,u] = W[v,u]/d[v] when d[v]>0, otherwise 0.

Then m=K h is normalized descendant-overlap attention with constant content features. Its exact implicit action is:

1. Old root-to-leaf pass: z_i=sum_(u ancestor of i) a_u*h_u and c_i=sum_(u ancestor of i) a_u.
2. Join z,c by the observed haplotype identity, independent of old/new node indexes.
3. New leaf-to-root pass: Z_v=sum_(i descends from v) z_i, C_v=sum_(i descends from v) c_i, and m_v=Z_v/C_v on positive support.

There is no node-pair table or retained descendant-incidence matrix. Work is O((N_old+N_new+H)D), memory O((N_old+N_new+H)D), with extra autograd buffers dependent on implementation. Tree depth can still cause many GPU launch levels. These statements concern materialized trees, not compressed-ARG edit complexity.

K preserves constants on positive-support rows, but is not generally column-stochastic, invertible, an optimal transport plan or ancestry-preserving. Its rank is at most H. Uniform source weights overcount overlapping ancestral levels. Root inclusion guarantees support but adds global context, while excluding roots/proper-clade-free leaves requires explicit zero masks. Unary insertion changes the operator unless source mass is explicitly conserved across equivalent nodes. Cosmetic row normalization cannot repair those semantic choices.

For arbitrary old/new internal-node permutation matrices P_o,P_n and a common sample permutation S, the incidence arrays become S A_o P_o^T and S A_n P_n^T. Consequently K'=P_n K P_o^T and m'=P_n m when h'=P_o h. Thus it is independently node-permutation equivariant and invariant to common renaming of sample identities. Genomic old/new order and physical gaps must be supplied explicitly. Swapping trees changes the map, but this is not an evolutionary-time or ancestor-identity statement. A nonlinear recurrent update after every serialized hold has no general hold-subdivision theorem.

**Relationship to recorded experiments and primary art.** H039's reference leaf-sector witnesses condition a native-hold CRF and do not implement K h. H040 uses selected changed-clade endpoint trilinear messages, H043 uses nearest changed-clade paired states, and H045 uses exact-clade episodes. These are distinct sparse routes, but their distinction does not make all-node overlap attention new. H021 already uses sample-incidence lift/project exchange. The unimplemented age-indexed branch-overlap note in research/reviews/next_architecture_novelty_fresh_v1/ordered_collection_priority_note_v2.md is closer still. H029/H030 already put carrier memory inside whole-tree decoding.

[GMN §3.2](https://proceedings.mlr.press/v97/li19d.html) establishes foreign-node matching messages during graph computation. [TGN §3.1](https://arxiv.org/abs/2006.10637v3) establishes event memory with GRU/LSTM updates. [DeltaNet](https://arxiv.org/abs/2406.06484v6) and [Gated DeltaNet](https://arxiv.org/abs/2412.06464v3) establish matrix prediction-error updates and gated retention. K h is not algebraically their delta update. Appending their updater to this already-derived kernel would reuse known mechanisms, rather than establish a new interaction principle. The literature comparison is bounded, not an exhaustive priority certificate.

No raw-overlap module, GRU wrapper, cosmetic normalization variant or prototype was authored. The single subsequent utility hypothesis is in marker_grid_design.md. No scientific payload, checkpoint, truth, GPU, guard, admission, installation or Git operation was used.
