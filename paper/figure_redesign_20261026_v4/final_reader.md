# Final independent reader

Agent: `/root/v4_blind_reader_final`. Fresh context, images only. The following account is the reader's response, retained before the orchestrator's source comparison. No explanation of the architecture was supplied to the reader.

---

I inspected only the two figure images. My independent reading is:

1. **Each tree interval starts with node features and leaf observations.** The inferred trees supply topology and node ages. A node MLP encodes age and root/leaf indicators into 32 features per node. Separately, observed allele, frequency, and callability features pass through a site MLP; mean/max pooling over called sites produces a 32-dimensional representation for each leaf in each interval. A leaf MLP combines this with reference ancestry and sampling age. Query ancestry labels are masked. Its output is added only to the leaf rows of the node representation, producing B_t. Branch geometry and allele agreement also produce edge gates for the tree encoder.

2. **The first encoder round constructs within-tree summaries in both directions.** Upward message passing takes a gated mean over children, yielding U_t^1. Downward message passing incorporates the parent and other branches, yielding O_t^1, with the root’s outside state initialized to zero. Each result retains one 32-dimensional row per node. I read these as complementary summaries of the descendant side and outside of each node’s subtree.

3. **One encoder correction carries group context across trees through persistent leaf identity.** Groups are defined by their highest ancestor no older than 350 generations; an older leaf becomes a singleton. At each group ancestor, the upward and outside states are added and projected from 32 to 8 dimensions to form f_t,g. This context is copied to all leaves in that group. Stacking across trees produces an 8-dimensional sequence for each leaf, even when that leaf’s group membership changes.

   Forward and reverse memories smooth these sequences across trees. The visible recurrence mixes previous memory with current context using alpha_t = exp(-L_t/ell), where interval length L_t controls the weighting and ell is learned. Each direction starts from its current context; the directions are averaged. Returning to a particular tree, the method averages the resulting memories over the leaves of each current group, subtracts that group’s current context, and projects the difference from 8 to 32 dimensions. This correction, Delta_t, is placed only at group-ancestor rows.

4. **A second correction uses changes in clade presence across adjacent trees.** Clades are matched by descendant leaf sets. Their presence in the previous, current, and next tree determines gain, loss, and persistence flags. The examples distinguish newly appearing, disappearing, persistent, and isolated clades; persistence means presence in at least two consecutive trees.

   The current tree’s upward state u feeds learned nonlinear maps with dimensions 32→8→32. The flags combine their outputs as S_v = gG(u) - lL(u) + pP(u). This value is placed at the current clade’s node row. Here, neighboring trees contribute **clade-presence information**; the figure does not show neighboring hidden-state vectors being transferred into these maps.

5. **Both corrections modify the original node representation before a second tree pass.** The explicit update is B'_t = B_t + S_t + Delta_t. A second upward/downward round uses the same weights as the first. Its all-node outputs U_t^2, O_t^2 return to Figure 1b. Figure 1 calls these U_t, O_t; Figure 2’s return label makes the correspondence clear. Their leaf rows are selected and concatenated, then projected 64→32 with tanh to obtain E_t, one embedding per leaf in that interval.

6. **The final sequence model integrates interval embeddings for the same query haplotype.** For query leaf q, the corresponding row is stacked across trees, giving E_q with shape trees × 32. Three interval descriptors are appended: interval length and distances to the previous and next interval centers. These undergo the displayed logarithmic transformation and scaling; missing neighbors receive zero. The resulting trees × 35 sequence enters a bidirectional GRU, producing 64 contextual features per interval.

   A bypass retains the original 32-dimensional local embedding. Concatenating local embedding and GRU context gives 96 features, followed by 96→32, GELU, 32→4, and softmax. The output is one four-way ancestry distribution per query leaf and interval: Anatolian farmer, Caucasus hunter-gatherer, Western hunter-gatherer, and Eastern hunter-gatherer. Each sampled leaf represents one haplotype.

The aggregation axes are distinguishable: called sites within each leaf and interval; children within a tree; trees along each leaf’s context sequence; forward/reverse memory directions; leaves within a current group; and, finally, the ordered tree sequence for a query leaf. Upward/outside concatenation and appending interval descriptors operate along feature columns.

The figures therefore show **three cross-tree mechanisms**: group-context memory, adjacent-tree clade-presence flags, and the final bidirectional GRU. The tree encoder itself already exchanges information across trees before that final GRU.

I found no major disconnected conceptual step or visibly contradictory connection. Two points remain visually ambiguous or unspecified:

- In Figure 2a, the lines from U_t^1 and O_t^1 join a shared horizontal connection feeding both branches. The destination labels resolve the intended distinction—sum both states for group context, select only the upward state for clade corrections—but the shared line alone could suggest an earlier combination.
- Figure 2c does not specify how gain/loss flags treat a missing neighboring tree at the sequence boundaries. Figure 1’s “missing neighbor → 0” belongs to interval-distance features, so I cannot extend that convention to the clade flags.

Exact gate formulas and the detailed downward aggregation are not shown. I can explain their conceptual roles, but cannot reconstruct those numerical operations from these images alone.
