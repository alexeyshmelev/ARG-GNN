# Common-boundary mechanism study

The candidate and every control consume the same observed leaf feature bank,
normalized boundary response, query/reference partition, physical intervals and
fixed scoring coordinates. Raw parent-branch features are excluded. Integer
sample identities select correspondence and are never learned features. The
full source parent is checked for admissibility before windowing.

The model names describe independent family adaptations, not exact published
reproductions.

| Model | What the control measures |
| --- | --- |
| boundary_feature | Leaf features without graph edges or temporal context |
| boundary_local | Same boundary-response GNN without temporal context |
| boundary_kernel | Same GNN with exact scalar exponential interval smoothing |
| boundary_gru | Same GNN with a conventional per-query bidirectional GRU |
| boundary_transformer | Same GNN with temporal Transformer and coordinate attention |
| boundary_full | Exact changing-operator coupled query memory |
| boundary_diagonal | Same sources, generator diagonal, no off-diagonal memory exchange |
| boundary_gain_matched | Decoupled memory with full coupled static equilibrium |
| boundary_gain_leak | Decoupled memory matching full equilibrium and uniform-state leakage |
| boundary_bag | Same source and memory parameterization with width-averaged operator |
| boundary_equilibrium | Static coupled response at the target tree |
| boundary_diagonal_equilibrium | Static diagonal response at the target tree |

Raw diagonal removal changes both equilibrium and decay spectra. The gain
controls address this attribution problem. Their instantaneous source mixing
still uses the coupled equilibrium. They remove cross-query transfer of prior
state, rather than all instantaneous graph dependence. Diagonal systems cannot
match every coupled spatial mode while remaining permutation equivariant.

Full, diagonal and gain controls have the same learned parameters and same-size
heads. Their common initialization is reset from each declared seed. The static
equilibrium variants have smaller heads, and their parameter counts must be
reported. Ordinary GRU and Transformer record updates do not guarantee exact
interval refinement. The feature, local and exact scalar-kernel controls retain
the two refinement contracts under the same feature assumptions.

Parents are split before fixed windowing. Each tile gets its own bag
eigensystem, because averaging a whole-parent operator would add context outside
the tile. Train and validation-only exploratory pilots reserve all final test
data. Checkpoints are selected by minimum mean parent-validation NLL. The
runner records exact configuration, dataset, implementation and prediction
hashes, training time, memory and parameter count.

The primary comparison must report all predeclared controls and distinguish
exploratory validation from a separately frozen held-out study. Semigroup laws,
Kron reduction, exact affine flow and graph state-space modeling are established
building blocks. This protocol does not establish novelty or predictive gain.
