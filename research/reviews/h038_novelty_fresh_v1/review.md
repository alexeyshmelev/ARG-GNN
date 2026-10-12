# Independent H038 novelty and architecture assessment

Date: 2026-10-12. Reviewer scope: source inspection and primary-source literature comparison. No model execution, biological array/target reads, training, package installation, source edits or Git operations. This is one fresh-context review using the repository-required Summary / Major concerns / Minor concerns / Verdict structure, adapted from the paper-review skill to the existing reviewer delegation. No other reviewers' outputs were read.

## Summary

**H038 is a defensible observation-only whole-tree model, but the inspected implementation does not establish a new general GNN architecture.** Its potential contribution is the biological representation and empirical value of observed allele/clade concordance, plus a careful full-panel ancestry benchmark. A positive H038 feature ablation would support that feature contribution. It would not establish priority of its gated tree sweeps or genomic recurrence.

The directed clade-exchange proposal is an interesting test of topology-change evidence. It is currently a hand-engineered boundary feature followed by an established linear-chain CRF. Calling it a new neural architecture would be too strong, even if accuracy improves.

## Actual implemented computation

The H038 wrapper only selects aligned versus zero-first-three branch-support fields (full_panel_pyg_model lines 7–26). The PyG classes expressly inherit the H036 learned modules and replace aggregation with parameter-free MessagePassing (pyg_gnn lines 22–80, 83–99). Upward messages are linear transformed child states, multiplied by sigmoid edge gates and divided by parent child-degree. Downward messages subtract the recipient's stored upward message from parent base + sibling sum + outside context (lines 100–156). Complete-tree sample contexts then enter one shared width-32 bidirectional GRU, with physical-span features, before a four-class classifier (whole_tree_gnn lines 252–279, 343–425).

This correctly keeps persistent sampled haplotype columns while leaving ancestors anonymous. There is no learned ancestor-ID feature and no cross-tree internal-node correspondence. There is also no explicit cross-boundary tree-edit operator: all genomic coupling occurs after tree compression, independently along each sampled column.

The five support fields are exact allele-split support, nearest split support, mean discrepancy, callable split count and empty flag. They are calculated from observed descendant allele-count sweeps (tree_observations lines 262–312), not verified ancestral mutations. Both biallelic orientations are counted and ancestral polarization is absent. H038 removes the first three fields together and retains count, missingness and geometry.

## Closest prior operators and legitimate distinctions

| Primary source inspected | Relevant overlap | What H038 genuinely changes |
| --- | --- | --- |
| Gilmer et al., MPNNs, arXiv:1704.01212, §2 and §5.1 | Learned edge-conditioned messages, sum aggregation, vertex updates and permutation symmetry are established. | Genealogy-specific geometry and allele/clade gates, complete-tree scheduled sweeps. No new generic message-passing principle demonstrated. |
| Tai et al., Tree-LSTM, arXiv:1503.00075, abstract | Learned aggregation on variable-branching tree topology predates H038. | H038 is not a Tree-LSTM reproduction. Its outside sweep adds complementary tree context. |
| Bonet et al., ARGformer, btag438, §2.1 and §3.1 | Genealogical representation learning and local ancestry from reference embeddings already exist. The primary text explicitly tokenizes shared ancestral node identifiers along leaf-to-root paths. | H038 is a feature-based whole-tree classifier that can transfer across unrelated ARG namespaces and avoids ancestor-ID semantics. This is a useful inductive distinction, not proof of new GNN machinery. |
| Shanks et al., ARGMix, 2026.04.05.714076, primary API abstract plus released author source | Ancient reference-guided marginal-tree ancestry prediction already exists. Current source uses TMRCA-relative graph attention and a genomic bidirectional GRU (tmrca_attn.py, joint_support.py). | H038 uses full native tree edges, observed SNP inputs and mutation/clade gates. The April paper full text was HTTP 403, so current-release GRU details are attributed to the source, not asserted as April-v1 paper details. |
| Irving-Pease et al., Nature 625 (2024), s41586-023-06705-1, Supplementary Note 3, pp.30–32 | AncestralPaths traverses focal paths, collects newly encountered reference-group proportions excluding previously visited leaves, and supervises a classifier on inferred trees. Inferred internal population labels are explicitly unknown. | H038 learns full-tree aggregation instead of using five fixed informative ancestry-shell rows. Whole-tree coverage and learned routing are useful differences. Recipient exclusion and anonymous inferred ancestors are not new biological concepts. |
| Escrig Molina et al., GST, arXiv:2606.05116v1, §3.1, equations 1–12 | Repeated local propagation, cross-graph context and gated reinjection are already an explicit architecture contribution. | H038 processes an ordered, sample-aligned sequence rather than a permutation-invariant graph set. It does not currently interleave genomic feedback into tree sweeps. Adding feedback alone would face GST and dynamic-GNN prior art. |
| Maples et al., RFMix, PMC3738819, Theory / LAI | Local ancestry classifiers plus linear-chain CRF transition potentials, MAP or forward-backward smoothing are established. | Proposed directed triplet witnesses could supply a new useful genealogical covariate. CRF, four-state inference, soft marginals and a recurrent emission model are not the novelty. |

An accessible EvolveGCN abstract (arXiv:1902.10191) also states that recurrent temporal node embeddings and changing node sets were established issues. Stable sampled identities plus ephemeral inferred ancestors are scientifically appropriate, but not sufficient on their own for an architecture-priority claim.

## Major concerns

1. **Feature value and architecture novelty are different questions.** The zero-concordance arm tests joint availability of three informative statistics. It does not isolate their alignment on the tree, routing versus emission use, or a new message algebra. A claim that mutation-aware gates help requires a same-information feature-only readout or placement control. A new-operator claim requires comparison with an ordinary tree MPNN receiving exactly the same inputs. No H038 accuracy result is asserted here.

2. **The proposed directed-versus-destination-removed comparison does not isolate a transition interaction.** A CRF boundary log-potential can contain terms r_i(a)+c_i(b) that are exactly absorbable into endpoint unary emissions. The proposed sham preserves source-row witness totals but changes destination evidence. Its advantage could therefore be better local source/destination classification, not coupling between ancestry states. Give the same W/D, g/u, masks and span information to the ordinary emission/GRU control. Separately remove only the non-additive boundary interaction while preserving the unary components.

3. **A rooted-triplet flip is evidence of changing donor affinity, not a verified ancestry switch.** Incomplete lineage sorting, noisy inferred topologies and donor-age/panel composition can produce these flips with unchanged ancestry. The document acknowledges this, correctly. Availability normalization does not fix donor representativeness. Eight sentinel haplotypes is a sparse measurement choice, not a mathematical sufficiency result. Report resolved-pair coverage by query age/source, witness frequency and sensitivity to the predeclared sentinel rule. These are feature diagnostics, not new training runs requested here.

4. **The mechanism needs more precise scope.** The outside sweep excludes the recipient's learned upward contribution algebraically. It is not an exact belief-propagation cavity that excludes all information from that sample: branch gates are whole-panel allele-split summaries and sibling site inputs include shared panel frequencies. Do not claim a conditional-independence theorem or exact probabilistic ancestry messages. Genomic left-to-right source/destination is coordinate direction, not ancestral-time direction.

## Minor concerns

- H038 requires sampled leaves and rejects sampled internal nodes (whole_tree_gnn lines 229–234). The proposal's treatment of sampled-ancestor triplets as unresolved does not by itself extend the retained encoder to sampled ancestors.
- Clarify whether two distinct observed individuals per source means one haplotype from each. K=8 and the stated 48 ordered cross-source pairs require eight haplotype sentinels. Retaining both homologues of eight diploids gives K=16.
- The proposed sigmoid exit rule and repeated emissions are not invariant to subdividing an identical hold. Neither current H038 nor the proposal should claim resolution invariance. Physical center distance is an empirical covariate, not a coalescent/recombination transition derivation.
- Twenty added scalars and uniform transition initialization are consistent with the written formula. At initialization, all transition entries are 1/4 and CRF marginals equal emission softmax with a uniform initial state. End-to-end fitting and reused-VAL checkpoint choice still make a short continuation a development probe.
- The Nature DOI contains 2023 but the publication is 2024. Its biobank-scale painting should not be attributed entirely to AncestralPaths: the main paper describes an adapted GLOBETROTTER pipeline, while AncestralPaths is Supplementary Note 3.

## Concrete next design direction

**First turn the directed-triplet proposal into an identifiable pairwise-interaction probe.** Keep all complete-tree observations and stable sample joins. Build a 4×4 witness score F_i(q), with off-diagonal directed D and zero diagonal. Set C=I-11^T/4, J=C F C, A=F-J. A has only row/column effects and is absorbable into endpoint emissions. J is the residual source-by-destination interaction.

Use CRF log-potentials B_i + gamma(A_i+J_i) versus B_i + gamma A_i, with identical generic boundary model B, emissions, learned parameter count, initialization and budget. Log-potentials need not be row-stochastic probabilities in a CRF. Also give all witness information to the ordinary H038 emission/GRU feature control. This separates biological feature value, smoothing value and irreducible pairwise interaction. It is a proposal, not an experiment authorization.

If the literal deliverable must be a new GNN architecture, this CRF probe does not satisfy it. A more targeted architecture hypothesis is to retain individual persistent query–donor witness identities as learned relation states and let boundary relation messages condition the next complete-tree cavity sweep. Only sample-pair state persists; every internal state is rebuilt from the current tree. The decisive comparison is the same witness tensors supplied to ordinary MPNN+GRU, plus a blocked-order control. This remains a hypothesis requiring targeted temporal/higher-order-GNN prior-art checking. Generic recurrence, interleaving, learned gating or anonymous ancestors must not be advertised as its invention. Do not reopen prior negative carrier/Gram/operator-order cells merely by renaming them.

## Verdict

**Current H038: architecture novelty not established, potentially useful biological feature/modeling contribution. Directed clade-exchange CRF: worth a narrowly attributed feature/interaction probe, insufficient as a new GNN claim.** A successful common full-panel comparison can make a strong application paper. A general ML architecture paper needs a distinct relation operator, a precise property beyond standard composition, and evidence against same-information ordinary architectures.

## Primary-source access and limits

Freshly read ARGformer European PMC XML, GST arXiv full HTML, RFMix PMC full HTML, ARGMix bioRxiv API metadata/abstract, and current pinned author ARGMix source. Read the cached Nature primary XML and its 38 MB supplementary PDF through existing pdftotext to stdout. ARGMix DOI, full HTML, JATS and Highwire syndication returned 403. Tree-LSTM/MPNN/GGNN/EvolveGCN arXiv abstracts were accessible, and MPNN full HTML was accessible. This is a bounded comparison, not an exhaustive priority search. No inaccessible paper-specific details or superiority figures were inferred from abstracts.
