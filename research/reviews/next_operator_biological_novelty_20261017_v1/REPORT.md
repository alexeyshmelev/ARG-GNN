# One biological hypothesis, rejected as a new operator

17 October 2026. **NO-GO for a distinct ML contribution.** A precise next hypothesis is recipient-excluded exchange between marker ancestry evidence and genealogy. It can represent switches inside inferred holds, but its defining operation is exactly established sum-product inference on a factor graph. Do not register another experiment or implement this construction for novelty. No distinct operator justification survived this audit.

The biological motivation is specific: a short observed allele run can support a source change even when inferred topology stays constant. Conversely, relatives can stabilize weak markers. Sending a query's own belief through the genealogy and back risks reinforcing its initial error. Cavity messages exclude that immediate return. They do **not** make correlated genotypes independent or guarantee calibration.

Current evidence warrants caution. H050's selected memory condition reaches 97.077844±0.567395% modern span accuracy versus 97.000028±0.497841% continuation, but boundary F1 falls to 18.380657% versus 19.048847%. SD spans eight reused development simulations, one optimizer seed, 960 additional updates per arm. H044 localization and H046 foreign context failed utility, H048 persistence and H050 higher-order memory failed their rules. H057's TRAIN-only hold/grid oracle gap is 0.1044431±0.0098382 accuracy points, not an achievable gain or boundary explanation. H058 has no trained result. H059–H061 supply no new LAI efficacy evidence in the inspected decision snapshot.

## Exact computational definition

Retain the complete existing two-sweep/structural/sample-memory/genomic-BiGRU backbone. At every original observed SNP s and query q, the existing proposed marker head supplies positive unary weights φ_sq(c)=exp(ℓ_t(s),q(c)+η_sq(c)). It sees the same observed source/age/allele features, calls, missing masks and coordinates as the finer-marker design. No new supervision is introduced.

On the complete native forest at s introduce C-state surrogate variables Y_sv. Clamp known pure-reference sample nodes to their original source labels. Use edge factors ψ_se(c,d)=exp(κ_e·1[c=d]), where κ_e=softplus(θ₀+θ₁ log1p(max(parent_age−child_age,0))) uses two shared scalars. Internal states are discriminative auxiliaries, **not** inferred ancestral populations. Preserve internal samples, disconnected components, roots and empty reference groups. Only actual sampled query identities connect across sites. For consecutive observed SNPs set χ_sq(c,d)=exp(β_cd,0+β_cd,1 log1p(Δx_s/L)), with chromosome length L and 2C² shared scalars. These are discriminative scores, not demographic rates.

Define the conditional model

    P(Y|O) ∝ ∏_(s,q) φ_sq(Y_sq)
               ∏_(s,e) ψ_se(Y_su,Y_sv)
               ∏_(s,q) χ_sq(Y_sq,Y_(s+1)q),

with reference clamps and internal-variable marginalization. This is a tree-plus-chain conditional random field.

Let T_s(w)_q be the exact marginal at q from full-tree sum-product with query inputs w, reference clamps and ψ. Let C_q(w)_s be exact forward/backward chain marginals with emissions w and χ. Initialize a=b=1 and perform exactly two exchange rounds:

    a_sq ← normalize[T_s({φ_sr b_sr})_q /(φ_sq b_sq)]
    b_sq ← normalize[C_q({φ_uq a_uq})_s /(φ_sq a_sq)]
    p_sq = normalize(φ_sq a_sq b_sq).

Use simultaneous updates within each displayed module and the new a in the following chain update. Division is componentwise and well-defined for positive query inputs. Log-domain subtraction implements it. The ratios remove the recipient input exactly; they are ordinary factor-to-variable cavity messages. Full tree and chain subsolves are exact, while the combined loopy inference is approximate.

Dense generic work is O(Σ_s N_t(s) C²+SQC²) per exchange round, in addition to the unchanged neural backbone. Marker-by-query interfaces and backward state require storage/recomputation. No sparse H058 path identity accelerates this nonlinear inference automatically, and no fit or speed claim follows.

## Exact original endpoint and falsification

Use the already specified observation-only SNP cells, midpoint ties to the right, clipped separately to each original half-open hold. A zero-SNP hold keeps the original coarse predictor. European original 100 bp centers, clipped bin masses, 686 query identities and seven cohorts remain unchanged. Aggregate original external TRAIN masses by their fixed marker-anchor map, then compute the identical span-weighted CE. American original 10,000 focal records retain positions, order, identities and their original external labels; aggregate counts only by the fixed map. No truth chooses coordinates, cells, factors or graph structure. Sparse American focals cannot validate dense boundaries. Any later comparison must retain the original full schedules and complete paired prediction barrier before held-out truth.

The primary utility control uses identical marker unaries, tree/chain factors, capacity, initialization and budget, but sends tree→chain messages once without a return. The equally informed finer-marker head is a separate practical reference whose added capacity must be matched explicitly. Generic BP with these same factors is an **identity control**, requiring no numerical campaign. A falsifiable utility prediction is ≥1.00 pp modern boundary-F1 gain over both learned references, ≤0.05 pp accuracy loss, no NLL regression and no loss of uncensored true-short-tract recall. A one-way tie rejects feedback utility. This is an explicit decision criterion, not a proposed additional run.

## Decisive prior art

[Yedidia, Freeman & Weiss 2005](https://doi.org/10.1109/TIT.2005.850085), IEEE TIT 51(7):2282–2312, primary §II Eqs 4–6, defines variable messages as products excluding the recipient, factor messages as marginalized products, and beliefs as incoming-message products. Substituting the marginalized genealogy factor and whole-chain factor gives exactly the ratios above. Its introduction explicitly relates forward/backward and turbo decoding. Exact subsolvers grouped into modules do not create a new principle. [Kschischang, Frey & Loeliger 2001](https://doi.org/10.1109/18.910572) is the foundational factor-graph reference; its bibliography was independently verified.

Internally, H030 already supplies log-domain recipient-excluded tree cavities. H033 already tests interleaved tree/genomic correction against blocked ordering, with no convincing gain. Ordered learning v2 already supplies matched marker-feedback placement; finer-marker readout already supplies within-hold localization and exact support. Random-frontier heat, conditional-mean memory and age-duration/HSMM no-gos remain intact. A learned tree potential changes the application, not this inference algebra.

**Decision:** retain the biological error question and existing ordered-marker work. The assessed alternative offers no defensible new operator. All analysis is source-only; no scientific payload, model import, numerical run, resource/job query, installation, registration or Git operation occurred.
