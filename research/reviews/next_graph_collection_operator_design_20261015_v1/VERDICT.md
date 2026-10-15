# Architecture verdict and next useful experiment

**Hold the one graph-collection candidate.** REPORT.md specifies a globally constrained mixture over SPR-compatible ancestor-role correspondences, including its actual node-message formula, feedback and fixed-evidence matched control. Its primitives are standard structured graph matching and BP/CRF-RNN. Dating semantics, nontrivial admissible support and complete-boundary cost are unresolved; no measured failure has been linked to that latent uncertainty. No distinct new general operator is established. Do not implement it or begin a broad validation campaign now.

**A boundary-focused TRAIN-loss arm is justified as a utility experiment.** NEXT_EXPERIMENT.md fixes Lnew=.75Lall+.25Lwithin1kbTRAINboundary on the unchanged H044 localized model. H044's tiny accuracy gain accompanied a3.14pp boundaryF1 loss. This loss condition addresses that measured failure directly. Reuse the completed baseline and run only one new960-update arm, with original inputs/support/initialization, unchanged selection and no alpha/radius sweep. It is separate from H047's cohort-risk allocation and carries no architecture/novelty claim.

No new hypothesis was registered, model implemented or experiment run. H048 bytes remain frozen. Root owns any registration and execution.
