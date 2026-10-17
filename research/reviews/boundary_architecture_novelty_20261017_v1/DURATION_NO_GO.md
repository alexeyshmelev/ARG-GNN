# Age-conditioned tract durations: no-go as a new mathematical architecture

17 October2026. This assessed alternative is biologically motivated, but its proposed inference family is established. It does not supersede REPORT.md's ordered-evidence candidate.

Let e[b,q,c] be emissions from the same complete-tree context and ordered marker evidence, a[q] the observed sample age, and g[b] a supplied genetic coordinate (or disclosed constant-rate approximation). For class-c tract(i,j], use

    F(c,i,j)=sum_(b=i+1..j)e[b,q,c]
             +D_c(g[j]-g[i],a[q],context_i,context_j).
    p(segments|O) proportional to exp(sum_segments F + sum_boundaries A).

Disallow adjacent equal-class segments. The exact recursion is

    alpha[j,c]=logsumexp_(i<j,c'!=c){alpha[i,c']+A(c',c,i)+F(c,i,j)},

with an initial term for i=0. This is precisely a neural semi-Markov CRF. A whole-tree encoder supplies specialized covariates. It does not change the inference principle. With B positions,K classes,Q queries and maximum duration L, the generic cost is O(QBLK^2), quadratic in B if L=B, plus encoder/segment-feature costs. Neural duration scores are not automatically demographic likelihoods. Chromosome endpoints require censoring/entry treatment.

A latent arrival-phase mixture also has an exact old-family reduction. Let a class-c tract choose phase r with pi[c,r], then have exponential length rate lambda[c,r]. Its survival is S_c(l)=sum_r pi[c,r]exp(-lambda[c,r]l). For a zero-diagonal row-stochastic class-switch matrix A, the expanded-state Markov generator is

    Q[(c,r),(c',r')]=lambda[c,r]A[c,c']pi[c',r'], c'!=c,
    Q[(c,r),(c,r)]=-lambda[c,r].

Marginalizing r gives exactly that duration law. Continuous propagation is exp(delta Q); a discrete observation implementation must preserve its hidden-event semantics. Finite discrete durations admit residual-duration state expansion. Covariate-conditioned rates remain conditioned/inhomogeneous state models.

Verified primary sources:

- [Kong, Dyer & Smith2015, Segmental Recurrent Neural Networks](https://arxiv.org/html/1511.06018v1), §§2–4 and Eqs1–3: neural segment representations, labels/durations, joint global normalization, observed/latent boundary training and dynamic programming are explicit.
- [Wiseman, Shieber & Rush2018, Learning Neural Templates for Text Generation](https://arxiv.org/html/1808.10122v1), §§4–5: neural HSMM transition/duration/emission parameterization is explicit. It is a generation task, not the identical ancestry model.
- [Gravel2012, Population Genetics Models of Local Ancestry](https://doi.org/10.1534/genetics.112.139808), “General incoming migration in the absence of drift”, Eq2 and Appendix2: under its tractable Wright–Fisher approximation, it restores the Markov property using **ancestry population and time of arrival**, then computes tract-length excursions across states of one ancestry. This already handles ancestry durations beyond one exponential. Its task is demographic tract statistics, not a neural conditional decoder.
- [Price etal.2009, HAPMIX](https://doi.org/10.1371/journal.pgen.1000519), Methods “Overview of haplotype-based inference” and ancestry model: frequent donor-copy changes and rarer ancestry switches are distinguished. Source redraw can retain the same ancestry, so event rate and visible class-change rate differ.

Sample age modifies elapsed recombination time under a known single-pulse model, but multiple pulses, drift, phasing/inference errors and unknown demography defeat an unqualified one-rate interpretation. Exact simulator event times would be additional privileged demographic information if supplied at prediction, not something inferred by the GNN. The tract decoder cannot recover marker arrangement already discarded by pooling.

A duration model might have useful boundary effects, but utility must compare the same ordered emissions and observations against a capacity-matched class-pair Markov decoder. A win over hold-pooled TreeLoom confounds new marker information with duration dependence. Deleting true short tracts fails the intended endpoint. No utility run, new registration or new-operator claim follows here.

Source/primary-text reasoning only. No scientific arrays or numerical model execution.
