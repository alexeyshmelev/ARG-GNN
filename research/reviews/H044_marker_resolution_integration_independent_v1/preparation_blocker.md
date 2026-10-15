# H044 integration preliminary blocker

Source-only finding, 2026-10-15. Worker review awaits its final author SHA. No corpus, targets/checkpoints, actual runtime/GPU/accounting state, imports, execution, admissions or Git operations were used.

**[P1] Preparation's executable path cannot accept the original registry.** `src/arggnn/prepare_marker_resolution_observations_v1.py` SHA d849533b827f82a44e943f5da9ef2496d1509bcca9f193f202add13fdbdad952 passes ctx["registry"] to read_full_cache_index_v1 at lines102–103. The fast reader delegates to the original cache reader, whose line85 calls validate_registry_v1 and requires the complete original48-row registry. The prep then constructs rows from every ctx registry entry at line104 and requires set(rows)==PARENTS40 at105. No registry can satisfy both contracts. This fails before the first original parent is prepared.

The intended fix is already expressed by selected_rows_v1 at lines40–47: retain the original48 registry for cache checks, use selected_rows_v1(ctx["registry"]) for local TRAIN/VAL loading. The helper currently is not called by the executable path. Root owns the correction and source/manifest closure; this review did not edit any frozen source. Context ce46a83c529b1751da4e24254712e3bc3a49b5f87f72e97aa0c6bc7bdd9cefed passes generation["parent_registry"] unchanged at121, consistent with retaining original48.

Driver8700c63ba515c5ae2e9d37c5efc1188b8596a1f37a20ca8ff8e507f6c0836318 uses module-based CPU and GPU worker commands at199–207. Its live admission state was not inspected.
