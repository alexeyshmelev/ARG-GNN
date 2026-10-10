# Common-input pipeline audit

Status: training/validation engineering protocol, no final test data generated.
Reviewer: independent architecture subagent, code/data audit with existing project context, not a fresh-context manuscript peer review.

The reviewer inspected the fixed reference adapter and the root-owned study registry, preparation/training scripts and frozen v2 protocol. The initial audit found a real parent-binding gap: the loader hashed a raw tensor named by the manifest but independently loaded labels from a directory. A read-only altered manifest could pair one parent's features with another parent's labels. No generated dataset or experiment was altered by the audit.

The repaired loader binds directory, tensor and metadata paths, verifies all file hashes, verifies parent replicate and seeds, checks the adapter configuration against the frozen protocol, and requires exact original-feature-prefix and graph/query/coordinate equality between raw and prepared sequences. Regression tests reproduce both mixed-directory and mixed-prepared-parent failures. The reviewer independently confirmed that correct inputs load and altered parent identities reject.

The reviewer found no remaining blocker in the audited paths. Prepared bundles contain only GraphSequence inputs and adapter provenance. Query reference columns are zero, census nodes/population truth are removed before packing, and every family receives the same 48-column all-node input. Checkpoint selection uses minimum mean parent-validation NLL. The runner requires test count zero.

Caveats retained: GraphSSM has its own native diffusion encoder. This pilot was generated before the integration commit, so its manifest explicitly records dirty-generation status and source hashes. This audit does not establish architectural novelty, held-out generalization, or manuscript acceptance.
