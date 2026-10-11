# H032 completed discovery conclusion
All three newly fitted static conditions completed 160 FP32 updates with seed 101. The prespecified minimum-NLL checkpoint is epoch 10 for every arm. Tree Gram gives Present64 accuracy 85.9378459% and NLL 0.374591. The independence control gives 85.9385491% and 0.374581. Smoothing gives 85.9397166% and 0.389136.

The tested donor-dependence condition has no measured advantage. Its improvement over smoothing is in likelihood and calibration. Do not repeat this unchanged condition or claim a novel donor-dependence mechanism. Higher last-epoch or earlier-epoch accuracy must not replace prespecified checkpoint selection.

Class recall remains weakest for WHG at about 77.22%. Tree-Gram boundary predictions are highly fragmented. Existing selected records contain 976/752 Present64 boundaries for the two parents, of which 33/27 match within the prespecified tolerance. These observations motivate testing whether an ordered graph layer can use local emitter confidence and suppress unsupported transitions. They do not demonstrate the cause of errors or the benefit of a future layer.

This is a two-TRAIN/two-reused-VAL-parent development pilot with no independent TEST. The frozen emitter uses Relate and added graph context uses tsinfer/tsdate. H030's recovered final static first-moment comparison remains outstanding.

Evidence: experiments/results/reference_dependence_full_storage_v1_summary_root/summary.json.
