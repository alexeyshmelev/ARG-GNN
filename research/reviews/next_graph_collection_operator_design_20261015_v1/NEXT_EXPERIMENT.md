# One justified next utility experiment: boundary-focused TRAIN risk

**Recommendation:** one new loss-condition arm on the unchanged H044 localized model, with the completed H044 localized run as the frozen comparator. This is a concrete ancestry-tract utility experiment. It claims no new GNN/operator and needs no broad prerequisite campaign.

**Measured reason.** H038's selected classifier has1kb boundaryF1=13.42±1.47%, precision10.71±1.21% and recall17.99±2.01% across8VAL parents. Although27.30% of true changes lack a nearby native-hold edge,66.74% of observed misses still have a candidate edge. H044 then exposes marker-local variation but changes Present182 accuracy only+0.005259530pp versus pooled evidence and drops boundaryF1 from18.674979208% to15.537415736%. Thus representation resolution alone did not supply tract fidelity. A loss that spends a fixed fraction of its risk on true TRAIN boundary neighborhoods is directly motivated. It can also fail by increasing false switches, which the existing precision/recall/F1 diagnostics will reveal.

**Exact intervention.** Retain the existing H044 localized model source,31,428 parameters, all H042 weights trainable, original952/686 observations and original complete100bp scoring bins. Use the identical original H042 pass30 initialization and exact H044 localized head initialization/seed, verified against its recorded initial-state identity. Retain32originalTRAIN/8VAL,30passes/960complete-parentFP32updates, AdamW and six checkpoints. Add no features, head, architecture or context mechanism.

For each TRAIN query i, construct R_i as the union of1kb neighborhoods on both sides of every actual **C4 ancestry change** in its original truth intervals, clipped to the original scored coverage. Union overlapping neighborhoods. Derive this only in the external TRAIN target/loss stage, after the observation barrier. No boundary flag or target enters model inputs, observation preparation, predicted rates or inference.

Let M[p,i,c] be the unchanged exact physical target mass, and

    MB[p,i,c] = length(bin_p ∩ R_i ∩ {x:Y_i(x)=c}).

Compute MB by interval intersection; do not replace it with a center-label mask, assume one event per bin, count one boundary per ancestral source change before the existing C4 collapse, or discard ambiguous/mixed bins. The ordinary loss is

    Lall = −sum M*logsoftmax(logits) / sum M.
    LB   = −sum MB*logsoftmax(logits) / sum MB.
    Lnew = 0.75*Lall + 0.25*LB.

If a complete parent has zero near-boundary query-bp, set Lnew=Lall for that parent by a prespecified rule and record it. All original bins/queries retain positive baseline risk. Alpha=.25 and radius1kb are fixed once, not swept. They are a modest declared allocation and the scorer's existing tolerance scale, not a learned biological tract scale.

This changes supervised risk using the original TRAIN truth, with no new annotation source. It is not a claim that both objectives contain identical supervision or that a positive outcome demonstrates a graph-architecture mechanism. MB adds boundary-location supervision derived from the same truth intervals. One parent target is constructed once/shared or released under the existing bounded target lifetime; no32-parent truth cache is assumed.

**Non-repetition and matched comparison.** H047 changed Present182/ancient504 cohort risk and was negative. This condition changes physical distance-to-TRAIN-boundary risk across all original queries; the ledger/source search found no prior registered boundary-neighborhood loss condition. The search was bounded and this note does not register a new ID.

Do not refit the completed unchanged H044 localized baseline. Reuse its actual checkpoint set, initialization/config/observations and frozen selected checkpoint. If original dense probabilities were RAM-only, regenerate its48validation products from those fixed checkpoints for the new common seal matrix, without optimization. The sole new fit gets the same960updates/seed/init/support, with only the loss above changed. This is one new condition, not a hyperparameter or architecture sweep.

Complete all six new checkpoints and seal all48new+48frozen-comparator VAL products before any new external VAL scoring. The original8VAL have already been used in development and motivated this experiment; a fresh seal does not make them independent. Retain the existing H044 accuracy/NLL/earlier-pass selection unchanged. Do not select on boundaryF1, alter decoder thresholds or sweep the loss weight after seeing VAL.

**Falsifiable endpoint.** Prespecify boundary utility as at least+1.00percentagepoint equal-parent mean Present1821kb boundaryF1 versus the frozen localized baseline, with no more than0.05pp mean span-accuracy decrease. Report paired effects and sampleSD across8parents, original precision/recall and short-tract metrics, NLL/calibration and all cohorts. A precision collapse, accuracy loss beyond that allowance, or failure to reach the boundary criterion rejects this condition. Accuracy and all-record support remain visible. H045/global-best comparison is reported separately.

This single objective experiment has enough measured motivation for prospective registration. Its efficacy is unestablished. It should run only after root's ongoing H046/H048/native work and actual resource scheduling permit; it does not alter their source, budget, completion criteria or interpretation. No new code, target read, fit, model, GPU action or registration occurred in this task.
