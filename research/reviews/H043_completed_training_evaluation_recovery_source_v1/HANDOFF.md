# H043 recovery handoff

Original actual direct wait is available; use the existing frozen route. No new proof adapter is needed for this lineage gap.

Root-only admission command from /workspace-SR003.nfs2/ARG-GNN, with root's eligible GPU choice substituted:

    .runtime/env/bin/python -B research/reviews/H043_GPU45_transport_root_source_v1/root_driver.py --admit evaluate --gpu <ROOT_CHOSEN_4_OR_5> --train-completion experiments/results/H043_GPU5_transport_train_seed101_v1/completed.json --expected-train-completion-sha256 0f2c5ff032d0245472cdb8beef7c1feee2e98feac7bfd32bf6f8a3dda5b710c6 --train-launch research/reviews/H043_GPU45_transport_root_source_v1/admitted_train/launch.json --expected-train-launch-sha256 95603cfd63e28985c9dc1d2ec95ad1f8e681e0f64392bc8b1b25ae2c640a2aed --train-actual-wait research/reviews/H043_GPU45_transport_root_source_v1/admitted_train/actual_wait.json --expected-train-actual-wait-sha256 de2fa623f6019b9048c41ea2ab0d5ca9894da1a92342621fe8f353bf32b4b419

After successful original admission:

    .runtime/env/bin/python -B research/reviews/H043_GPU45_transport_root_source_v1/root_driver.py --launch evaluate

These commands were not executed by this source reviewer. Root has since reported consuming tool5619 terminal0 and running original evaluation admission on GPU5; do not replay admission solely from this handoff.

Frozen root-driver SHA256: 5e91ea5d43779e0ce360230ec9bdadb2e11dac3f2cb9756aeaa0c56c696b7813
Frozen root-common SHA256: 04b25ad067653ea67c9b8a00820d852e2e567a8d2c21f35704cb6fb3aba6057c

See FINDINGS.md and source_metadata_review.json for exact refs, comparisons, provenance, and scope.
