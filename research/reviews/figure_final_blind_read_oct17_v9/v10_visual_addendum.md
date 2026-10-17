# Visual addendum: v10

Only paper/figure_redesign_20261017_v10/TreeLoom_architecture_render.png was inspected for this addendum. The original v9 review is preserved.

**Input-route issue resolved:** a continuous enclosure now groups the local trees, g_(t,q) binary rows, [τ,0,0] internal-node feature, reference ancestry vectors, and unknown query label. Its visible exit connects the grouped inputs to the first MLP. This no longer reads as an ancestry-vector-only MLP input. Exact tensor packing and which feature belongs to which node remain caption details.

**No drawing blocker found:** the enclosure clears its contents, the b marker and the incoming graph-to-trees arrow. Its exit is traceable to the MLP and does not collide with a sample label or vector. I found no new literal overlap, clipped content, topology contradiction, or conflicting connection.

**Method and identity reconfirmed:** grouped observed inputs → MLP → b_v → Up/Down GNN; the output splits to Δ_str and same-sample cross-tree memory → Linear. These combine with the b_v residual at plus → second Up/Down GNN → embeddings e_tq → one-layer BiGRU receiving interval lengths → concatenation with the e_tq bypass → MLP → four-class probabilities → argmax ancestry for q1. Persistent sample IDs and fixed memory/embedding rows maintain identity through changed leaf order. Magenta q1 remains distinct throughout; the example ends in WHG across all three intervals. No added panel title or prose is present.

**Remaining caption definitions, not drawing blockers:** R; the meaning of g_(t,q) and τ/vector components; numeric treatment of the missing q1 label; Δ_str and the Sample memory update; the role of l_t inside BiGRU; ∥/concatenated quantities; whether θ indicates shared parameters; and the query/all-sample scope of e_tq and p_tq. The image still does not justify inferring attention, a specific memory algorithm, losses, or a training protocol.
