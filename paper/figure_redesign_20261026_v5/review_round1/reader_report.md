I read only the two frozen PNGs. The overall computation is understandable, but several details needed for a complete, unambiguous reconstruction are missing.

1. **Inputs and local features**

The worked tree is \(T_2\), spanning 3000–7000 bp. Its nine node rows are ordered \(q,r1,r2,r3,r4,a,b,c,\text{root}\). There are five leaves and eight branches. The topology is:

- \(r1,r2\) descend from \(b\).
- \(q,b\) descend from \(c\).
- \(r3,r4\) descend from \(a\).
- \(a,c\) descend from the root.

\(X_2\) is \(9\times3\): rows are nodes; columns are normalized age, root indicator, and leaf indicator. The displayed age transformation is \(\log(1+\mathrm{age})/\log(10001)\). A learned linear \(3\to32\), followed by tanh, produces \(N_2\), a \(9\times32\) node embedding.

\(R\) is a separate \(5\times6\) leaf table. Its columns are Ana, CHG, WHG, EHG, unknown, and normalized age. Query \(q\) has row \([0,0,0,0,1,0]\); \(r1\) has \([1,0,0,0,0,0]\).

The allele array has site rows at 1000, 5000, and 8000 bp, and the five leaf columns. For \(q\), the displayed alleles are \(1,0,0\). For \(r1\), they are \(1,1,0\).

For the site at 5000 and query \(q\), allele lookup using \(R\) produces a 19-channel vector:

- Eight log-frequency entries: four population columns in each of two age bins.
- Eight availability entries with the same arrangement.
- Log \(p\), entropy, and called.

The displayed values are \([-1.4,-1.4,-0.3,-0.3]\) in the younger age bin, zeroes in the older bin; availability is four ones followed by four zeroes; the last three values are \(-0.51,0.67,1\).

The site vector passes through linear \(19\to32\), GELU, linear \(32\to32\), and tanh. This produces \(Z_{2q}\), \(S\times32\), with one row per site in the interval. Here \(S=1\).

Mean and maximum over site rows yield two 32-vectors. Concatenating them with count′, empty, and span′ produces \(v_{2q}\), length 67. The shown scalar values are \(0.060,0,0.572\). Linear \(67\to32\) plus tanh gives a pooled vector. Concatenating \(R[q]\) gives 38 channels; linear \(38\to32\) plus tanh produces \(q\)’s leaf feature. Repeating over five leaves produces \(F_t^{leaf}\), \(5\times32\).

The leaf features are extended to nine rows by adding zero rows for the four ancestors, then added to \(N_t\). Thus \(B_t\) is \(9\times32\), with one 32-vector per node. The ancestor base features have no direct contribution from this padded leaf-feature array.

Branch features are illustrated for \(b\to c\). The nine entries are child age, parent age, duration, leaf share, exact fit, closest fit, discrepancy, split count, and empty. Two learned linear \(9\to32\) maps followed by sigmoid produce \(G^\uparrow,G^\downarrow\), each \(8\times32\): one row per branch and one gate per embedding channel.

2. **The tree message operator**

The encoder applies the same learned tree-message operator \(M\) twice.

Within one application, upward processing proceeds through leaves, then \(a,b\), then \(c\), then root. At \(b\), the two child states \(U(r1),U(r2)\) form a \(2\times32\) array. A linear \(32\to32\) acts on its rows. The result is multiplied elementwise by the corresponding upward gates and divided by \(d=2\). These two message rows \(m(r1),m(r2)\) are summed to give \(A(b)\), length 32.

Then:

\[
U(b)=\operatorname{LayerNorm}
\left[B(b)+\tanh\!\left(\operatorname{Linear}_{64\to32}
([B(b),A(b)])\right)\right].
\]

All upward node results are stacked in the nine-node order to form \(U\), \(9\times32\).

Downward processing starts with root context zero. Its order is root; \(a,c\); \(q,b,r3,r4\); then \(r1,r2\). For the illustrated child \(r1\), the context before learned transformation is

\[
C=O(b)+B(b)+A(b)-m(r1).
\]

It passes through linear \(32\to32\), elementwise multiplication by the downward gate for that branch, and a further linear/tanh/LayerNorm block to produce \(O(r1)\), length 32. Subtracting \(m(r1)\) removes that child’s own upward contribution from the parent aggregate.

The same pattern gives \(O(b)\) from parent \(c\), and \(O(c)\) from the root. \(O\) is stacked into \(9\times32\), with the root row zero.

The drawings do not specify the upward initialization or empty-child handling at leaves. Consequently I cannot give an exact first upward operation for \(q\) or \(r1\).

3. **What crosses between trees in the memory path**

After round one, memory uses both \(U^1\) and \(O^1\).

At the 350-generation grouping threshold, the leaf-to-group table is:

| Interval | q | r1 | r2 | r3 | r4 |
|---|---|---|---|---|---|
| \(T_1\) | d | d | r2 | a | a |
| \(T_2\) | q | b | b | a | a |
| \(T_3\) | q | b | b | a | a |

For \(T_2\), gathering representative rows \([q,b,a]\) gives two \(3\times32\) arrays. Their sum passes through linear \(32\to8\), producing \(F_2\), \(3\times8\), with group rows \(q,b,a\). Copying rows according to \(q,b,b,a,a\) gives \(V_2\), \(5\times8\), with one row per leaf. \(V_1,V_3\) are also shown, implying the corresponding procedure for those intervals.

The explicitly illustrated track is \(r1\):

\[
f_1=F_1[d],\qquad f_2=F_2[b],\qquad f_3=F_3[b].
\]

Thus the identity carried across intervals is the leaf \(r1\), even when its group changes from \(d\) to \(b\). The carried states have eight channels.

A learned \(\theta\) defines

\[
\ell=100+999900\,\sigma(\theta),\qquad
\alpha_t=\exp(-L_t/\ell).
\]

The example has \(\ell=5000\) bp and lengths \(3000,4000,3000\), giving approximately \(0.55,0.45,0.55\).

For \(r1\), the complete displayed forward recurrence is

\[
m_1^\rightarrow=f_1,\quad
m_2^\rightarrow=\alpha_2m_1^\rightarrow+(1-\alpha_2)f_2,\quad
m_3^\rightarrow=\alpha_3m_2^\rightarrow+(1-\alpha_3)f_3.
\]

The reverse recurrence is

\[
m_3^\leftarrow=f_3,\quad
m_2^\leftarrow=\alpha_2m_3^\leftarrow+(1-\alpha_2)f_2,\quad
m_1^\leftarrow=\alpha_1m_2^\leftarrow+(1-\alpha_1)f_1.
\]

This is repeated for all five leaves. At \(T_2\), averaging the two directions gives \(M_2\), \(5\times8\). Group averaging then produces \(R_2\), \(3\times8\): \(q\) alone, \(r1/r2\) averaged into \(b\), and \(r3/r4\) averaged into \(a\).

The group difference \(R_2-F_2\) passes through linear \(8\to32\). It is scattered to the nine-node array \(\Delta\), with nonzero rows at \(q,b,a\) and zero rows at \(r1,r2,r3,r4,c,\text{root}\). In particular, the correction derived from \(r1\)’s history is returned to its current group node \(b\), not directly to the \(r1\) row.

4. **Structural correction and ancestor \(b\)**

The structural example tracks the proper clade \(C=\{r1,r2\}\). It is absent in \(T_1\), represented by \(b\) in \(T_2\), and represented by \(b\) in \(T_3\). The shown flags are gain \(g=1\), loss \(l=0\), and persistence \(e=1\).

The input is \(u=U^1(b)\), length 32. One set of three linear \(32\to8\) maps gives \(x,y,z\), each length eight. It computes

\[
g(x\odot y)-l(x\odot z),
\]

then maps \(8\to32\).

A second set of three maps gives \(x',y',z'\), computes \(x'\odot y'+z'\), maps \(8\to32\), and multiplies by \(e\). The two 32-vectors are added and scattered to row \(b\), with other rows zero for this example. The instruction is to repeat this for proper clades \(2\le |C|<5\), summing scattered rows to form \(S\).

The complete visible role of \(b\) is therefore:

- Its structural input row is \([\mathrm{age}'=0.582,0,0]\), which becomes \(N(b)\), hence its initial \(B(b)\).
- Its incoming child branches and outgoing \(b\to c\) branch have upward/downward gates.
- In round one, it aggregates \(r1,r2\) into \(U^1(b)\), which contributes upward through \(c\) to the root.
- It receives \(O^1(b)\) through \(c\), then provides downward context to \(r1,r2\).
- \(U^1(b)+O^1(b)\) supplies group feature \(F_2[b]\), copied to both \(r1,r2\).
- Their smoothed histories are averaged back into a correction \(\Delta(b)\).
- \(U^1(b)\) also supplies the illustrated structural correction \(S(b)\).
- Its second-round input is \(B(b)+S(b)+\Delta(b)\).
- Round two repeats the upward and downward operations with the same \(M\) weights and the same gate arrays, producing \(U^2(b),O^2(b)\).

The final readout takes leaf rows, so \(b\)’s final row is not itself a prediction row. Its influence reaches predictions through the tree computations.

The structural panel does not determine precisely whether gain/loss/persistence flags are global clade statistics or interval-specific events, or which flags apply to \(b\) separately at \(T_2\) and \(T_3\).

5. **Completing the query \(q\) path**

For \(q\), the site-processing path above supplies its leaf feature, combined with \(N(q)\) to form \(B(q)\). Round-one upward processing includes \(q\)’s contribution to \(c\); its downward context arrives from \(c\), using the parent context and the aggregate with \(q\)’s own message removed.

Its memory group sequence is \(d,q,q\), as established by the grouping table. Memory returns a correction directly to row \(q\) in \(T_2\). The structural correction excludes singleton clades, so the illustrated structural construction has no direct leaf-\(q\) scatter. It can affect \(q\) through corrected ancestors during round two.

After round two, selecting the five leaf rows from \(U_t^2,O_t^2\) and concatenating corresponding rows yields \(5\times64\). Linear \(64\to32\) and tanh produce a per-tree leaf embedding \(E_t\), \(5\times32\).

Selecting \(q\)’s row from each of \(E_1,E_2,E_3\) yields \(E_q\), \(3\times32\): interval rows and feature columns.

The interval lengths are \(3000,4000,3000\). The centers are \(1500,5000,8500\); neighboring-center distances are 3500, with zero at missing boundaries. The three columns \(L,d_-,d_+\) are transformed by \(\log(1+x)/\log(2000001)\), producing \(D\), \(3\times3\).

Concatenating \(E_q\) with \(D\) gives \(X\), \(3\times35\). Forward and reverse GRUs each produce a 32-vector per interval. Stacking rows and concatenating directions gives \(H\), \(3\times64\).

Concatenating the original query embeddings with \(H\) gives 96 channels per interval. Linear \(96\to32\), GELU, and linear \(32\to4\) produce four logits per interval. Softmax produces \(P\), \(3\times4\), with columns labeled A, C, W, E. These appear to correspond to the four named reference populations, although the abbreviation mapping is not explicitly stated.

This final bidirectional recurrence is another cross-tree pathway, distinct from the eight-channel memory inside the encoder.

6. **Repetition and sharing**

The visible repetition axes are sites within an interval, five leaves, eight branches, nine tree nodes in dependency order, three genomic intervals, all proper clades, and exactly two applications of \(M\).

The drawing explicitly states that the two \(M\) applications use the same weights. Both rounds receive the same \(G^\uparrow,G^\downarrow\). The title also calls \(M\) a shared tree message operator.

Other sharing relationships are not explicitly defined: for example, whether separate direction-specific linear maps share parameters, whether the two gate maps do, or precisely which correction maps are reused across intervals/clades. I would not assert those relationships from the drawings alone.

7. **Unresolved visual and definitional issues**

- **Leaf upward initialization is missing.** The upward schedule begins at leaves, but the equation is illustrated only for a two-child ancestor. No leaf \(U\), empty aggregate, or zero-child normalization rule is shown.
- **The top-level encoder wiring does not explicitly group both outputs.** The arrow from the first \(M\) enters the \(U^1\) graphic, while the correction branch leaves beside \(O^1\). The second \(M\) similarly points into \(U^2\), while the leaf-output arrow leaves beside \(O^2\). Panels b–d establish that both arrays are involved, but the top-row connectors alone do not.
- **The \(B\) array above \(B+S+\Delta\) has no visible connecting arrow.** The text in the addition block conveys its intended role.
- **Per-tree replication is partly implicit.** The feature example is \(T_2\), but the readout arrow in Figure 1d leads directly into \(E_1\). \(E_2,E_3\) are shown without separate incoming computation arrows or an explicit “repeat trees” instruction.
- **Only \(T_2\)’s group gathering and correction are expanded.** \(V_1,V_3\) are supplied to recurrence, but their generation and corresponding output corrections are inferred by repetition.
- **The actual definitions of several input features are absent:** log-frequency estimation, log \(p\), entropy, availability, count′, span′, exact fit, closest fit, discrepancy, and split count. Their numerical values and dimensions are visible; their exact construction is not.
- **Structural flag scope is unclear**, particularly how gain/loss/persistence are assigned separately to occurrences in different intervals.
- **Several tensors lack explicit dimensions at the point of use.** \(S\) and \(\Delta\) must match \(B\)’s \(9\times32\) shape for addition, and the final classifier input must be \(3\times96\), but these dimensions are not printed there.
- **The reference table \(R\) and memory group mean \(R_2\) reuse the letter \(R\)** for different quantities; context and shapes distinguish them, but the distinction is not introduced.
- **No major text occlusion or unreadable overlap stood out.** The principal visual issues are connectivity and scope rather than collisions.