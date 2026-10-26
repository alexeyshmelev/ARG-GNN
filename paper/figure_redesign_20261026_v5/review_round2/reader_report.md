I inspected only the two specified PNGs, fetched into memory over direct SSH. No manuscript, captions, code, directory inventory, or prior review was consulted.

## Architecture reconstructed from the figures

This is a model for predicting a query leaf’s four ancestry probabilities in each genomic interval. It combines per-site allele evidence, local-tree message passing, two encoder corrections that use neighboring intervals, and a final bidirectional GRU over intervals.

The example contains five leaves, three genomic intervals, nine nodes in the illustrated middle tree, eight branches, and 32-dimensional main embeddings.

### 1. Observations become node, leaf, and branch features

**Tree/node features.** For \(T_2=3000\)–\(7000\) bp, the nine node rows are

\[
[q,r1,r2,r3,r4,a,b,c,\mathrm{root}].
\]

Each has three structural features: transformed age, root indicator, and leaf indicator. The age transform is explicitly

\[
\mathrm{age}'=\log(1+\mathrm{age})/\log(10001).
\]

Thus \(X_2:9\times3\) passes through Linear \(3\to32\) and tanh to produce \(N_2:9\times32\).

**Leaf metadata.** \(R:5\times6\) has four ancestry indicators—Ana, CHG, WHG, EHG—an unknown indicator, and transformed age. Query \(q\) has unknown \(=1\); \(r1,\ldots,r4\) have the four known ancestries, respectively. All illustrated leaf ages are zero.

**Allele evidence.** For each leaf and called site, an allele lookup uses the observed allele table and \(R\) to make 19 features:

- Eight log-frequency entries: four ancestry classes in each of two reference sampling-age bins, \(<350\) and \(\ge350\).
- Eight corresponding availability entries.
- Panel log probability, entropy, and a called indicator.

For \(q\) at position 5000, the shown vector is \(X_{5000,q}:19\). A network

\[
19\to32\;\text{Linear},\;\mathrm{GELU},\;
32\to32\;\text{Linear},\;\tanh
\]

produces site embeddings \(Z_{2q}:S\times32\), with \(S=1\) in this interval.

Mean and maximum over called sites contribute 32 channels each. Together with transformed count, an empty indicator, and transformed span, they make \(v_{2q}:67\). Linear \(67\to32\) plus tanh produces a pooled allele representation. Concatenating its 32 channels with the six metadata channels and applying Linear \(38\to32\) plus tanh gives a leaf embedding. Repeating for the five leaves produces \(F_t^{leaf}:5\times32\).

**Base node representation.** The leaf embeddings occupy the first five rows of a \(9\times32\) matrix, with zero rows for internal nodes. Adding this to \(N_t\) yields \(B_t:9\times32\).

**Branch gates.** Each of the eight branches has nine features. For \(b\to c\), these are labeled child age, parent age, duration, leaf share, exact fit, closest fit, discrepancy, split count, and empty. Two learned Linear \(9\to32\) maps followed by sigmoid produce

\[
G^\uparrow,G^\downarrow:8\times32.
\]

The figure explicitly says these are two maps shared across the eight branches.

### 2. The encoder performs two tree passes, with corrections between them

Figure 2 expands ENC as

\[
B
\xrightarrow{M}
(U^1,O^1)
\longrightarrow (S,\Delta)
\longrightarrow B+S+\Delta
\xrightarrow{\text{same }M}
(U^2,O^2).
\]

Every displayed node-state matrix is \(9\times32\). Both rounds use the same directional branch gates. The message-passing maps are shared across nodes and trees, with distinct upward and downward maps, and the same maps in both rounds.

**Upward pass.** At ancestor \(b\), children \(r1,r2\) supply \(U(r1,r2):2\times32\). A Linear \(32\to32\) map transforms the child states. Their branch gates multiply the transformed states elementwise. Each resulting message is divided by the number of children, \(d=2\), and the rows are summed:

\[
m_i=\frac{g_i^\uparrow\odot \operatorname{Linear}(U_i)}{d},
\qquad A(b)=\sum_i m_i.
\]

The node update is

\[
U(b)=\operatorname{LayerNorm}
\left[B(b)+\tanh\!\left(
\operatorname{Linear}([B(b),A(b)])
\right)\right].
\]

The concatenation has 64 channels and the learned output has 32. Leaves have \(A=0_{32}\). The illustrated traversal is leaves, then \(a,b\), then \(c\), then root.

**Downward pass.** The root outside state is zero. To send context from \(b\) to \(r1\), the diagram forms

\[
C=O(b)+B(b)+A(b)-m(r1).
\]

It applies a Linear \(32\to32\) map, multiplies by the downward branch gate, then applies the shown “Linear + tanh, LayerNorm” block to obtain \(O(r1)\). Subtracting \(m(r1)\) removes that child’s upward message from the parent’s aggregate. The traversal is root, then \(a,c\), then \(q,b,r3,r4\), then \(r1,r2\).

### 3. Two corrections use relationships across intervals

**Memory correction \(\Delta\).** At a cut of 350 generations, leaves are assigned to groups. The table explicitly gives:

| Interval | q | r1 | r2 | r3 | r4 |
|---|---|---|---|---|---|
| T1 | d | d | r2 | a | a |
| T2 | q | b | b | a | a |
| T3 | q | b | b | a | a |

For \(T_2\), the encoder gathers rows \([q,b,a]\) from \(U^1\) and \(O^1\), adds the two \(3\times32\) matrices, and applies Linear \(32\to8\). This gives group features \(F_2:3\times8\).

Those rows are copied to member leaves, in order \([q,b,b,a,a]\), producing \(V_2:5\times8\). This operation repeats for the three intervals. Each leaf then has a sequence of three eight-dimensional group features.

A learned length scale is shown as

\[
\ell=100+99{,}900\,\operatorname{sigmoid}(\theta),
\qquad
\alpha_t=\exp(-L_t/\ell).
\]

With example \(\ell=5000\) and interval lengths \(3000,4000,3000\), the displayed decays are approximately \(.55,.45,.55\).

Forward memory starts by copying the first feature and uses

\[
m_t^\rightarrow
=\alpha_t m_{t-1}^\rightarrow+(1-\alpha_t)f_t.
\]

Backward memory starts by copying the last feature and uses the analogous recurrence from the right. The two directions are averaged, yielding \(M_t:5\times8\).

The leaf memories are averaged within the current groups. For \(T_2\), that yields \(J_2:3\times8\). The difference \(J_2-F_2\) passes through Linear \(8\to32\) and is scattered into the corresponding group-node rows. Thus \(\Delta_2\) has potentially nonzero rows at \(q,b,a\), and zero rows at \(r1,r2,r3,r4,c,\mathrm{root}\).

**Clade correction \(S\).** This repeats over proper clades and intervals, using shared maps, then sums scattered node corrections.

For \(C=\{r1,r2\}\) at \(T_2\), clade presence across previous/current/next intervals is \(0,1,1\). The consecutive run has length two, so \(e=1\). The current clade embedding \(u=U^1(b)\) supplies two sets of three Linear \(32\to8\) maps.

The first set forms

\[
g(x\odot y)-l(x\odot z),
\qquad
g=1-\mathrm{prev},\quad l=1-\mathrm{next},
\]

then maps \(8\to32\). The second forms \(x'\odot y'+z'\), maps \(8\to32\), and multiplies by \(e\). Their sum is scattered to node \(b\).

For this example, \(g=1,l=0,e=1\). The correction therefore includes the displayed clade-entry term and persistence term.

### 4. Leaf outputs become ancestry probabilities

After the second tree pass, the five leaf rows from \(U_t^2\) and \(O_t^2\) are concatenated into \(5\times64\). Linear \(64\to32\) plus tanh gives \(E_t:5\times32\), with the same projection used across \(T_1,T_2,T_3\).

Selecting the \(q\) row from each interval yields \(E_q:3\times32\).

A geometry matrix \(D:3\times3\) contains interval length and the distances to previous/next interval midpoints. The raw example is

\[
\begin{bmatrix}
3000&0&3500\\
4000&3500&3500\\
3000&3500&0
\end{bmatrix},
\]

transformed entrywise by \(\log(1+x)/\log(2000001)\).

Concatenating \(E_q\) and \(D\) gives \(X:3\times35\). Forward and reverse GRU paths each produce 32 channels per interval. Aligning and concatenating their outputs gives \(H:3\times64\).

The model concatenates \(H\) with the original \(E_q\), producing \(3\times96\), then applies

\[
\operatorname{Linear}(96\to32)
\to\mathrm{GELU}
\to\operatorname{Linear}(32\to4)
\to\mathrm{softmax}.
\]

The result \(P:3\times4\) has one row per interval and columns Ana, CHG, WHG, EHG. Unknown is an input status, not a fifth output class.

## Requested traces

### Query q

The observed query alleles are \(1,0,0\) at positions \(1500,5000,8500\). Its metadata says unknown ancestry and age zero. Its structural row says age zero, not root, leaf.

For each interval, its called-site evidence is embedded and pooled, combined with metadata, and added to its structural embedding to form \(B_t(q)\).

In \(T_2\), the first upward query state uses its own base features and zero child aggregate. Its downward state supplies context from the rest of the local tree through parent \(c\).

Across intervals, q’s memory groups are \(d,q,q\). Consequently, its middle-interval memory is informed by the first interval’s shared \(d\) group and its third-interval singleton group. At \(T_2\), the memory correction is scattered directly to q. Proper-clade corrections can also change contextual information arriving through its ancestors and neighboring branches.

The second tree pass produces \(U^2(q),O^2(q)\), which are paired and projected to the q row of \(E_2\). The three q rows enter the final GRU together with genomic geometry; the GRU output and direct embedding skip connection feed the four-class probability head.

### Reference r1 across all three intervals

Its observed alleles are \(1,1,0\), and its metadata identifies Ana ancestry. It receives interval-specific leaf and structural embeddings by the same displayed feature path.

The memory trace is especially explicit:

\[
f_1=F_1[d],\qquad f_2=F_2[b],\qquad f_3=F_3[b].
\]

Thus r1 carries context from the \(q/r1\) group in \(T_1\) into the \(r1/r2\) group in later intervals. Both forward and backward recurrences operate on this sequence.

At \(T_2\), r1’s resulting memory is averaged with r2’s memory to form the correction for group \(b\). The correction is scattered to \(b\), not directly to the r1 row. The clade correction for \(\{r1,r2\}\) also lands at \(b\). The second downward pass then carries the revised group context to r1.

All five leaves obtain \(E_t\) rows. The final GRU/head is drawn only for q. Applying that head to r1 is not shown.

### Ancestor b through the shown stages

1. In \(T_2\), b is the parent of r1 and r2 and a child of c.
2. Its structural feature row is \([.582,0,0]\), projected to \(N_2(b):32\).
3. It receives no directly padded leaf-feature row, so its initial base row is structural.
4. Its branch features contribute directional gates, including the illustrated \(b\to c\) branch.
5. In the first upward pass, it aggregates gated, degree-scaled messages from r1 and r2 into \(A(b)\), combines that with \(B(b)\), and produces \(U^1(b)\).
6. Its first downward state comes from c by the same parent-to-child rule. Its own state then participates in messages to r1 and r2.
7. At the 350-generation cut, b represents group \(\{r1,r2\}\). Its \(U^1+O^1\) feature is compressed to eight channels and copied to both leaves.
8. Bidirectional leaf histories are averaged back into the current b group, producing \(\Delta_2(b)\).
9. The illustrated clade-presence computation produces \(S_2(b)\).
10. The second pass receives the corrected base row \(B_2(b)+S_2(b)+\Delta_2(b)\) and produces \(U^2(b),O^2(b)\).
11. b is not directly selected into \(E_t\), because that stage selects leaf rows. Its influence reaches the output through tree propagation to leaves.

## Repetition, aggregation, reuse, and cross-interval information

**Repetition axes:** called sites, five leaves, tree nodes, branches, three intervals, two tree-message rounds, current groups, proper clades, and two recurrence directions.

**Aggregations:** site mean/max; degree-scaled sum of child messages; mean of forward/backward memory; mean over current group members; sum of scattered clade corrections. Concatenations are used separately for node updates, leaf-state pairing, directional GRU outputs, geometry, and the final embedding skip connection.

**Explicit reuse:** two gate maps across branches; message maps across nodes and trees and both rounds; distinct upward/downward message maps; shared clade-correction maps across clades and intervals; the leaf-output projection across intervals.

**What crosses intervals:**

- Eight-dimensional group-derived features, transported along each leaf’s sequence by the memory recurrence.
- Clade presence in adjacent intervals and consecutive-run information, used to modify the current clade’s state.
- Thirty-two-dimensional query embeddings plus three geometry features, contextualized by the final bidirectional GRU.

The clade panel does not show neighboring interval embeddings entering its correction; it shows neighboring clade-presence information.

## What cannot be inferred or remains ambiguous

1. **The 19 allele features are organized clearly but not fully defined.** The figures do not explain the frequency estimator, smoothing, log base, panel probability, entropy formula, treatment of missing alleles, or whether a reference leaf is excluded from its own lookup.

2. **Several branch statistics remain names rather than definitions.** “Exact fit,” “closest fit,” “discrepancy,” and “split count” cannot be reconstructed from these pictures alone. Nor are all branch-feature normalizations specified.

3. **Pooling edge cases are absent.** The count and span transformations are not defined, and an empty indicator does not explain how mean/max are evaluated when there are no called sites.

4. **The 350-generation grouping construction is illustrated rather than fully specified.** The example memberships are unambiguous; boundary cases, general cut rules, and whether 350 is fixed or configurable are not established.

5. **The clade correction lacks boundary conventions.** The example explains a middle interval, but not how previous/next presence is set at the first or last interval, nor precisely how run lengths are determined in general.

6. **Some parameter details are implicit.** The number of GRU layers, cell internals, initial hidden states, and whether directional parameters are tied are absent. The memory path depicts a learned length scale but does not explicitly declare its parameter shape or scope.

7. **The downward update is compressed.** Its operations are visible, but the final block does not label every intermediate width or separately identify each map. It is less explicit than the upward update.

8. **The overview’s “pair” boxes are potentially vague in isolation.** The other panels resolve them: the memory uses gathered \(U^1+O^1\), while the output uses concatenated leaf rows \([U^2,O^2]\).

9. **T1 and T3 topology is only partially exposed.** Their group memberships and the illustrated clade-presence sequence can be traced, but their complete node-by-node message propagation cannot.

10. **Repeated colored grids are schematic.** They identify tensor roles and dimensions, but cannot establish equality of values or correspondence between particular channels.

A reader can recover the main computational turns, including both internal cross-interval corrections and the final sequence model. The remaining gaps mainly concern feature definitions, edge cases, and compact notation, rather than the overall information flow.
