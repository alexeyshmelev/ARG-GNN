Blind figure-reading report

I inspected only TreeLoom_architecture.png and TreeLoom_encoder.png. This reading uses the labels, arrows, shapes, equations, and examples in those images. I did not consult a paper, caption, source file, prior review, or other project material.

The figures describe an ancestry-class predictor that combines observations on a sequence of local trees with two forms of cross-interval correction, two rounds of tree message passing, and a final bidirectional sequence model. The example has five leaves, nine nodes per displayed tree, eight branches, three genomic intervals, 32-dimensional tree states, and four output classes. Gold denotes learned layers, pale boxes fixed operations, teal embeddings or logits, gray observed data, and purple probabilities.

The main computation is reconstructible at the level of tensor shapes, data flow, message equations, aggregation, and major parameter-sharing scopes. The figures do not by themselves specify every observation-to-feature rule or every boundary convention. They support a substantially more precise reconstruction of the network than of the preprocessing.

The architecture figure can be read in this order:

1. In panel a, T2 spans 3000–7000 bp. Its leaf order is q, r1, r2, r3, r4; the other rows are a, b, c, root. The displayed topology is r1,r2 → b; q,b → c; r3,r4 → a; and c,a → root. Each node has three observed features: transformed age, a root indicator, and a leaf indicator. The age transform is log(1 + age) / log(10001). A learned Linear 3→32 followed by tanh creates N2 of shape 9×32. R is a separate 5×6 leaf table containing four known population indicators, an unknown indicator, and transformed sampling age. q has the unknown indicator, whereas r1, r2, r3, and r4 are respectively marked Ana, CHG, WHG, and EHG. The displayed sampling ages are zero.

2. Panel b constructs a site representation separately for a leaf and interval. At site 5000, the example q allele is 0, while r1 and r2 carry 1 and r3 and r4 carry 0. The allele lookup also receives R. Its displayed 19-channel result has eight log-frequency channels, eight availability channels, and three additional values: panel log probability, entropy, and called status. The eight-channel blocks are shown as two sampling-age bins by four reference classes. For this q example, the younger bin has log frequencies −1.4, −1.4, −0.3, −0.3 and availability ones; the older bin has zeros. The three additional values are −0.51, 0.67, and 1. Linear 19→32, GELU, Linear 32→32, and tanh produce site embeddings Z with shape S×32. The displayed interval contains S=1 called site.

3. Mean and maximum over called-site embeddings give 32 channels each. Three scalars labeled count′, empty, and span′ make a 67-channel pooled vector. Linear 67→32 and tanh compress it. Concatenation with the leaf's six-channel R row gives 38 channels; Linear 38→32 and tanh produce that leaf's 32-channel feature. Repeating over the five leaves gives F_leaf_t of shape 5×32. This is pooling over sites within a leaf and interval, not pooling over leaves or intervals.

4. Panel c puts the five leaf-feature rows into a 9×32 node matrix and fills the internal-node rows with zero. Adding the node embedding N produces B_t, also 9×32. Thus the initial state of an internal ancestor such as b receives the node embedding, while a leaf receives both node and observation-derived features.

5. Each branch also has nine observed features: child age, parent age, duration, leaf share, exact fit, closest fit, discrepancy, split count, and empty. The displayed example is b→c, one of eight branches. Two learned Linear 9→32 maps followed by sigmoid generate distinct upward and downward gate matrices G↑ and G↓, each 8×32. “2 maps shared ×8” explicitly indicates reuse over the branches. The encoder consumes B and these gates and returns U2 and O2, each 9×32.

6. Panel d selects the five leaf rows of U2 and O2, concatenates their two 32-channel states, and applies Linear 64→32 and tanh. This creates one 5×32 matrix E_t per interval, with the same projection for T1, T2, and T3. Selecting the q row from E1, E2, and E3 creates E_q of shape 3×32.

7. The three intervals have lengths 3000, 4000, and 3000 bp. The displayed interval centers are 1500, 5000, and 8500. A three-column matrix contains interval length L, the preceding-center distance d−, and the following-center distance d+. Missing endpoint distances are explicitly zero in the example. Applying log(1 + entry) / log(2000001) yields D of shape 3×3.

8. Panel e concatenates E_q and D to obtain X of shape 3×35. One forward and one reverse GRU, each one layer with 32-dimensional output, read the same X in opposite directions. Their aligned output rows concatenate to H of shape 3×64. Concatenating H with the original E_q gives 3×96. Linear 96→32, GELU, and Linear 32→4 yield class logits; softmax yields P of shape 3×4. The rows correspond to intervals and the columns to Ana, CHG, WHG, and EHG. The displayed output is for q.

The encoder figure makes the two rounds explicit. A first application of M to B gives U1 and O1. U1 supplies the clade correction S; paired U1 and O1 supply the memory correction Δ. Both corrections have shape 9×32. A second application of the same M takes B + S + Δ and the same gates, producing U2 and O2. The corrections are added to the initial node-state matrix, not directly to the first-round U and O outputs. The role of the input B in the detailed M diagram is taken by this corrected matrix in the second call.

Within M, upward propagation is bottom-up. For the displayed parent b, the child rows U(r1) and U(r2) form a 2×32 matrix. A shared Linear 32→32 produces transformed child rows Z. Elementwise multiplication by the corresponding upward gate rows and division by d=2 produce two messages m. Summing those rows gives A(b), a 32-channel aggregate. In this example, this is a mean of gated transformed child messages; there is no normalization by the sum of gate values.

The b update is:

U(b) = LayerNorm(B(b) + tanh(Linear64→32(concat(B(b), A(b))))).

The diagram assigns zero 32-channel A vectors to leaves. The upward processing levels are leaves, then a and b, then c, then root. All nine results are stacked into U in the stated node-row order.

Downward propagation starts with O(root)=0. For the displayed branch b→r1, it first forms:

C = O(b) + B(b) + A(b) − m(r1).

This subtracts the upward contribution of the receiving child. Linear 32→32 acts on C, the downward gate multiplies its result elementwise, and another Linear 32→32, tanh, and LayerNorm produce O(r1). The downward processing levels are root; a and c; q, b, r3, and r4; then r1 and r2. O has the same nine node rows as U. No additional residual addition is drawn on the downward update.

The title of panel b explicitly scopes these maps to all nodes and trees, distinguishes upward from downward maps, and says the same maps are used in both rounds. The two rounds are therefore repeated applications of one tree-message module, rather than independent encoders with unrelated weights.

The memory correction in panel c crosses intervals through leaf identities, even when those leaves belong to different groups. The cut is labeled “Groups at 350 generations.” Its membership table is:

| Leaf | T1 group | T2 group | T3 group |
|---|---|---|---|
| q | d | q | q |
| r1 | d | b | b |
| r2 | r2 | b | b |
| r3 | a | a | a |
| r4 | a | a | a |

For T2, the procedure gathers the q, b, and a rows of U1 and O1. It adds the paired 32-channel states and applies Linear 32→8, producing a 3×8 group-context matrix F2. Copying rows in the order q, b, b, a, a creates V2 of shape 5×8, one row per leaf. The same process is repeated for the other intervals.

Each leaf then follows its own three-interval sequence of eight-channel contexts. A learned scalar θ sets a length scale ℓ = 100 + 99,900 × sigmoid(θ). The figure uses ℓ=5000 as an example. With interval lengths L=(3000,4000,3000), α_t=exp(−L_t/ℓ) is approximately (0.55,0.45,0.55). These are interval-length decays; they are not the d− and d+ center distances used later in D.

For r1 specifically, f1=F1[d], f2=F2[b], and f3=F3[b]. Its complete forward recurrence is:

- m1→ = f1.
- m2→ = α2 m1→ + (1−α2) f2.
- m3→ = α3 m2→ + (1−α3) f3.

Its complete reverse recurrence is:

- m3← = f3.
- m2← = α2 m3← + (1−α2) f2.
- m1← = α1 m2← + (1−α1) f1.

The two directions are averaged at each interval. Repeating this for all five leaves gives a 5×8 matrix M_t. Group means then reduce M_t to the groups present in that interval. At T2, q is a singleton group, b averages the r1 and r2 rows, and a averages the r3 and r4 rows. The resulting J2 is 3×8. The local group context F2 is subtracted; Linear 8→32 acts on J2−F2. The result is scattered into the q, b, and a node rows of Δ2. The r1, r2, r3, r4, c, and root rows are explicitly zero in the displayed Δ2.

This shows exactly how r1 travels through all three memory intervals: it carries d's context from T1 into b's contexts at T2 and T3, and the reverse recurrence also carries later b contexts back toward T1. At T1, its averaged memory is pooled with q in d; at T2 and T3, it is pooled with r2 in b. The memory correction acts on those group representatives. It is not a direct scatter into the r1 leaf row. The same copying, recurrence, direction averaging, group averaging, and scatter procedure applies to q and the other leaves.

The clade correction in panel d is a second, different route for information about neighboring intervals. It repeats over proper clades and intervals and sums the scattered node rows into S. The example clade C={r1,r2} is absent in T1, present in T2, and present in T3. Its consecutive run is two, so the displayed repeat indicator e is one. At T2, prev=0 and next=1 imply g=1−prev=1 and l=1−next=0.

Taking u=U1(b), three learned Linear 32→8 maps create x, y, and z. The vector g(x⊙y)−l(x⊙z) passes through Linear 8→32. Three further Linear 32→8 maps produce x′, y′, and z′; x′⊙y′+z′ passes through another Linear 8→32 and is multiplied by e. The two 32-channel contributions are added and scattered to b. All other rows are zero for this particular clade contribution. Other eligible clades contribute their own scattered rows, and the figure specifies summation. The correction maps are shared across those clades and intervals.

The query q has two connected paths from observations to probabilities. Its own displayed alleles are 1, 0, and 0 at 1500, 5000, and 8500. Within each interval, its site features, site pooling, and unknown-marked R row produce its leaf feature and hence B(q). In T2, q is a leaf under c. Its upward aggregate is zero, so U1(q) is computed from its local input state. Its downward state O1(q) receives context through c, including the b subtree and context arriving from the root's other side. The subtraction rule removes q's own child message at its parent before forming the message back to q.

For the cross-interval memory path, q follows group d in T1 and singleton group q in T2 and T3. In T2 and T3 the group memory correction can therefore be scattered directly into q's input row. Its T1 group correction is instead scattered to d. The minimum clade size shown for S is two, so the displayed clade-scatter mechanism does not target the singleton q leaf. The second tree round propagates corrected input states. Its q rows U2(q) and O2(q) become E_t(q), and the final bidirectional GRU combines the three E_t(q) rows with positional features before producing the three four-class probability rows.

r1 is also an observed leaf, with displayed alleles 1, 1, and 0 and a known Ana row in R. The “repeat leaves” feature construction gives it a 32-channel leaf feature in each interval. It participates in the tree-message computation and the cross-interval memory described above. In the displayed T2 topology it contributes an upward message to b and receives a downward message from b that excludes its own upward contribution. The same leaf state also participates in the leaf readout E_t. The final classifier panel selects q; a separate final probability sequence for r1 is not displayed and should not be claimed as an output of this example.

For ancestor b in T2, the complete shown route is especially clear. Its observed node features have transformed age 0.582, root indicator zero, and leaf indicator zero. Its initial B(b) is the node embedding because the padded leaf-feature matrix is zero on internal nodes. The r1 and r2 child messages produce A(b), then U1(b). Its outside state O1(b) arrives from c; applying the displayed downward rule to c→b excludes b's own upward contribution and retains other context. U1(b) supplies the structural correction for clade {r1,r2}. U1(b)+O1(b) supplies b's eight-channel group context, copied to r1 and r2. Their three-interval recurrences are averaged back into b, producing Δ2(b). The second round therefore uses B(b)+S2(b)+Δ2(b). Its resulting U2(b) contributes upward toward c, while its state and outside context enter downward messages to r1 and r2. b is not one of the leaf rows read directly by the final q classifier.

The diagrams establish several distinct aggregation and reuse scopes:

- Sites: mean and max within a leaf and interval.
- Tree children: sum of degree-divided, gated child messages.
- Memory directions: arithmetic mean of forward and reverse states.
- Memory groups: mean over their member leaves at the current interval.
- Clade corrections: sum of scattered contributions.
- Leaf readout: concatenation of U2 and O2, not averaging.
- Final sequence model: concatenation of forward and reverse GRU outputs, followed by concatenation with E_q.

They explicitly show branch-gate reuse across eight branches, tree-message-map reuse over nodes and trees and between rounds, shared clade-correction maps over clades and intervals, and the same leaf readout projection across T1–T3. The GRU drawing is a one-layer recurrence unrolled over intervals, supporting recurrent weight reuse within each direction. “Same X” says the forward and reverse GRUs receive the same input sequence; it does not establish that their parameters are tied. Repeated use of a feature block is drawn, but additional parameter-tying claims beyond the stated scopes should not be invented.

There are three visibly distinct ways intervals interact. S uses clade presence in adjacent intervals and consecutive-run information. Δ moves learned group contexts through persistent leaf identities in both genomic directions, then aggregates them back into current groups. The final bidirectional GRU mixes the interval-level q embeddings after tree encoding. The first M round itself is tree-local; the second round uses input states already corrected with cross-interval information. The illustrations do not show direct tree edges connecting nodes in different intervals.

The material ambiguities and limits are:

- The 19-channel lookup values are shown, but the exact rule producing frequencies, panel log probability, entropy, and availability from the allele table is not specified. I cannot infer smoothing, self-reference handling, missing-data policies, or related preprocessing rules.
- The transformed count′ and span′ values are shown without their defining transforms. The empty flag is named, but behavior of mean/max pooling for an empty interval is not diagrammed.
- Several branch-feature definitions, including exact fit, closest fit, discrepancy, and split count, cannot be recovered from their names and one example.
- The 350-generation membership table specifies the example groups. It does not fully specify all cut, equality, or exceptional-tree cases.
- The clade correction gives its interior-interval example and a run condition, but not first/last-interval prev/next conventions. “Proper clades” and the displayed size filter need a precise eligibility rule for a full implementation; the size filter alone would not explain how the whole-tree clade is handled.
- d=2 is explicit for the two-child example. Any broader degree convention for trees outside that example is not separately defined.
- The complete topologies and observation-derived feature matrices for T1 and T3 are not drawn. Their group table and clade-presence example support the traces above but do not justify inventing every missing edge.
- GRU initial-state conventions, exact LayerNorm settings, and unstated bias or parameter-tying choices are outside the diagram's specification. The softmax placement and 3×4 output strongly indicate per-interval class normalization, although the axis is not explicitly labeled.
- Reused symbols require local interpretation: for example, F denotes leaf features in the architecture figure and eight-channel group contexts in the encoder figure; Z also names different intermediate representations. Their displayed dimensions distinguish them.

My judgment is that a reader can reconstruct the key computation and correctly distinguish tree propagation, clade corrections, leaf-based cross-interval memory, and the final sequence classifier. The q, r1, and b traces are supported by explicit arrows, dimensions, memberships, equations, and scatter rows. A complete numerical reproduction from observations would require additional definitions for the items above; filling those gaps by guessing would go beyond the figures.
