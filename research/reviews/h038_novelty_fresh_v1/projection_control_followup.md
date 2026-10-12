# Precise pairwise versus unary control

2026-10-12. Follow-up to root's request for a practical 20-parameter comparator. Proposal only. No implementation or experiment action.

## Exact log-score projection

For boundary i between holds i-1 and i, query q, and the same 20 parameters theta, compute:

- B = log P_theta(D_destination_removed), retaining the original exit features, availability, beta and gamma.
- L = log P_theta(D_directed).
- E = L - B.
- C = I - 11^T/4.
- J = C E C.
- A = E - J.

Equivalently r_a = mean_b E_ab, c_b = mean_a E_ab, mu = mean_ab E_ab, so A_ab = r_a + c_b - mu. J has zero row and column sums. A preserves every row/column sum of E. These are arithmetic LOG-score sums, not probability marginals.

Full CRF potential is B+A+J=L. Control CRF potential is B+A. Both compute all directed features and use the same 20 learned scalars. Gamma still affects A unless the observed feature configuration happens to have vanishing additive component. Do not pretend identical nominal parameter count guarantees identical effective capacity in that degenerate case.

The control is exactly a same-W unary/readout plus generic-smoothing model. Add r_a to the left hold's emission and c_b-mu to the right hold's emission, then run the original generic B CRF. This demonstrates algebraically that all feature-induced endpoint evidence remains available while the non-additive source-by-destination interaction is removed. Log potentials need not be row stochastic in a CRF. Renormalizing B+A by row would add a source-unary term and alter the stated control.

Use stable logsigmoid/logsoftmax to construct P's log scores. No probability floors are required. Both arms start with the same uniform transition matrix, for which E=A=J=0. Separately fitted gates and emissions may diverge. Equality is a within-model construction, not an assertion that independently trained arms have identical fitted rates.

A pure head-only diagnostic can project all of L to its additive component and absorb it into emissions without any chain interaction. It uses the same W and 20 parameters but also removes generic persistence. It cannot replace B+A for attributing a directed interaction gain.

## If literal probability-marginal preservation is required

Let M=exp L, K=exp B. Find positive row/column scalings u,v such that

    M_control = diag(u) K diag(v)
    M_control 1 = M 1
    1^T M_control = 1^T M

Then use the control log-potential B_ab + log u_a + log v_b. This is a unary-only change relative to B and preserves row/column potential-mass marginals. For positive full-support matrices, iterative proportional fitting/Sinkhorn scaling provides the constrained projection. A finite implementation must certify its numerical tolerance rather than claim exact convergence. Use log-domain scaling and differentiate through the solve if end-to-end training is intended.

This uses the same 20 learned theta and no added learned scalars, but adds numerical work. It preserves per-boundary unnormalized potential-mass marginals, not global CRF posterior marginals, expected switch counts, or exact fitted exit rates. With row-stochastic M, row mass is already one, while columns capture destination mass. Fixing diagonals as well changes the estimand and generally prevents the correction from being wholly unary on the full 4x4 table.

I recommend the double-centered residual as the simplest precise architecture-attribution control. If exact mass marginals are the scientific nuisance to remove, choose the scaling projection prospectively and record its tolerance/runtime. Do not describe either as preserving all quantities simultaneously.

## Contribution supported by possible outcomes

Directed features beating a destination-erased sham alone support useful destination information, which could be unary. Full interaction beating B+A supports useful pair coupling beyond all feature-induced unary evidence. Same-W ordinary emission/GRU controls test whether a specialized chain is needed.

These findings could make a valuable domain-specific ancestry representation or structural-feature contribution on a convincing common full-panel benchmark. They do not establish new generic GNN/CRF machinery. RFMix already has feature-classifier emissions plus ancestry CRF transitions. MPNNs, recurrent graph models and GST already cover generic learned propagation, recurrence and interleaved feedback.
