# Next architecture direction after H050

**Verdict: no new operator is qualified by this audit. Do not implement or register this candidate.** The biologically concrete direction is uncertainty in a query lineage's inferred attachment, rather than another context adapter on a fixed genealogy. Its obvious readout is already anticipated by ARGweaver and phylogenetic placement.

## Residual evidence and no-repeat boundary

The current whole-tree model pools SNP features by mean/max within each native hold, uses full-tree sweeps and a genomic BiGRU, then classifies native holds. H050 changes internal cut-history context while retaining that readout.

The H038 diagnostic gives precision10.71%, recall17.99% and F113.42%, against a class-agnostic hold-edge recall ceiling72.70%. Of3289misses,2195had a nearby admissible edge. Predicted changes outnumbered true changes1.68-fold. False internal islands<=10kb accounted for35.46%of extra boundaries. This concerns H038, not an unobserved H050 result. H045's development accuracy is96.8270%, with F118.1558%.

More output resolution alone is already an answered idea. H044 gained0.0053pp accuracy but reduced F1from18.6750%to15.5374%. H046flank context, H047cohort risk and H048graph-conditioned reset chains failed their utility conditions. H043/H045/H050fixed-tree context must not be repeated under another name. The separately owned donor-identity/remix hypothesis is also different.

## One explicit candidate, rejected as architectural novelty

For each focal sampled query q, remove q from an observed local genealogy and retain a dated scaffold of all other observed haplotypes. Candidate states are a scaffold branch e and a compatible attachment time tau. Weights pi(e,tau) would come from a declared conditional sequence likelihood, mutation/error model and branch/time prior. Query ancestry targets cannot enter them.

A candidate readout is

p(a|X) = sum_(e,tau) pi(e,tau|X,T_without_q) * softmax(f_theta(q,e,tau,X))[a].

Different observed SNP blocks could favor different attachments inside an unchanged native hold. A point-mass control retains the original inferred attachment. This asks whether uncertain geometry helps, rather than adding another embedding average, shared switch or chain smoother.

However, ARGweaver already removes/rethreads a sequence with branch/time HMM states, sequence emissions and recombination transitions [1]. ARGweaver-D adds population paths [2]. Pplacer already computes likelihood and Bayesian placement weights over edges, integrates attachment and pendant lengths, and caches tree traversals [3]. DEPP learns sequence embeddings for phylogenetic placement on a different biological task [6]. Attaching a PyG embedding and ancestry head produces an ordinary conditional mixture, with no new primitive or demonstrated expressivity distinction.

Even a changed focal parent is not a certified focal recombination. ARGweaver's Figure3 discussion explicitly says its attachment can move because of old as well as new recombinations. Keep the earlier counterexample: pruning a from((a,b),(c,d)) and attaching beside c creates clades{a,c}and{a,c,d}containing unrecombined c/d. Stable sampled-haplotype identity is valid. Changed clades are not event-descendant labels.

## A falsifiable utility question, without registration

If a later scientific qualification justifies this direction, hold the full-tree encoder, SNP support, query/reference split, ancestry readout, parameter budget, optimization and selection fixed. Compare attachment-likelihood averaging against the original-attachment point mass and a candidate-weight permutation within frozen observation-defined strata. The sham should retain candidate sets and weight histograms while disrupting their alignment. A restricted candidate set must be called a candidate-conditioned approximation.

The discriminating observation is improved correctly typed boundary recall with fewer false islands, preserved real short-tract recall and no physical-span accuracy loss, beyond both controls. Beating only an old hold decoder would confound placement uncertainty with extra SNP resolution. Truth-selected candidate branches would be leakage. Preserve the fixed1kb scorer, report NLL/Brier and parent-paired changes, and use independent parents for confirmation. This memo chooses no new threshold, cohort, protocol or experiment.

## Scientific and cost objections

A qualified pi does not currently exist. Retained-variable-site ascertainment, missing calls, ancient sampling, imputation/phase error and branch-age uncertainty must enter its conditional likelihood. Uniform compatibility weights are arbitrary. Learned attention cannot be renamed a biological posterior. Source review provides no empirical evidence that attachment error causes the current misses.

Naively rebuilding a whole-tree encoder after removing every q costs O(Q*sum_t N_t*d^2), before marker/time-state likelihoods. A fixed K-candidate readout adds about O(S*Q*K) state evaluations plus time integration. PyG Batch/edge_index and Torch scatter can batch this work but do not remove its algorithmic cost. Reusing a full-panel embedding generally retains q's information, so it is not an exact leave-q-out cavity. An exact directional-message factorization would need justification. Pplacer's likelihood caching is prior art, as are Ripple's fixed-weight incremental GNN updates [9]. Training changes invalidate unrestricted learned-state caches. No O(edge edits) guarantee follows.

RFMix's random forests plus CRF [4], LAMP's window/HMM framework [5], neural BP [7] and robust higher-order auxiliaries [8] also block novelty claims based on marker continuity, unrolling or group consistency.

The useful conclusion is narrower than a model proposal: uncertain genealogy is a distinct potential biological bottleneck. The obvious architecture is known, computationally nontrivial and currently lacks a qualified likelihood. It should not consume a new training run or support an A* novelty claim. H050 and current benchmark work remain untouched.

## Primary references

1. ARGweaver,2014. DOI10.1371/journal.pgen.1004342. Threading and Figure3.
2. ARGweaver-D,2020. DOI10.1371/journal.pgen.1008895. Branch/time/population-path states.
3. Pplacer,2010. DOI10.1186/1471-2105-11-538. Overview and Eq1–3.
4. RFMix,2013. DOI10.1016/j.ajhg.2013.06.020. Material and Methods/Theory.
5. LAMP-LD/HAP,2012. DOI10.1093/bioinformatics/bts144. Window/HMM methods.
6. DEPP. DOI10.1093/sysbio/syac031. Figure1 and methods.
7. Neural Enhanced BP. arXiv2003.01998v5. Sections2.2and4.
8. Robust higher-order potentials,2009. DOI10.1007/s11263-008-0202-0. Eq17onward.
9. Ripple,2025. DOI10.1109/ICDCS63083.2025.00088. Sections2.1and4.1.

Exact text references, URLs and quote locators are in primary_evidence.v1.json. Search results do not prove novelty absence. No scientific payload, model, fixture, process probe, GPU, guard, admission or Git operation was used.
