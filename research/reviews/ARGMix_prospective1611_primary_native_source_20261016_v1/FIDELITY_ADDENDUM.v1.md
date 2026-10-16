# Additive correction to per-focal donor-policy wording

This supersedes the HANDOFF.md phrase "Same observed panels, donor/query identity policy and observed native Relate topology are shared" only as regards per-focal candidate eligibility. The sealed source, contract, old sensitivity and original author code are unchanged.

The same GLOBAL observed sample identities and declared donor/query roles are used. This is not a claim of identical per-focal candidate reference lists between all methods.

The untouched current-author FastSubgraphEngine does not exclude the focal haplotype or its diploid homologue. It ranks all594 declared donors, then applies native TRAIN nearest-drop1 or VAL/inference0. NF has318 donor/query haplotypes and Yam has26, for344 dual-role haplotypes. Both focal haplotypes remain eligible for these queries before nearest-drop augmentation. Post-Neolithic and Present have no donor-role overlap, so this particular candidate eligibility difference does not affect their available reference sets.

The generator's separate eligible_donor_indices helper excludes both haplotypes sharing the focal sample_id. The primary native backend does not call that helper. Code in prepare_observed_parent writes every observed metadata row and supplies native reference/predict populations, preserving current-author behavior.

The root must disclose whether and where our own route calls that helper and report native per-focal policy separately. The main no-overlap Present endpoint can be reported separately from dual-role NF/Yam endpoints. A focal-diploid-excluded ARGMix sensitivity may be added later under a different method label, but has not been implemented or executed as part of this primary packet.

No accuracy metrics, native builds, training or inference have been run by this source worker. The native inference route remains separate pending source work and actual root qualification.
