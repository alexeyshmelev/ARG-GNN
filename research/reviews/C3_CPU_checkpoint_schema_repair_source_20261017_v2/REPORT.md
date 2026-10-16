# Minimal namespace and directive integration repair

V1's checkpoint metadata correction remains valid, but its builder did not match two unchanged contract requirements. It emitted packet/output namespaces beginning checkpoint_schema_, while both validate_request and stage_source_v1 require actual_plan_lineage_. Its new directive also differed from the contract's exact original directive. Independent review found these failures before any v1 admission or execution.

V2 preserves v1 and reuses its immutable repaired prediction bridge. The builder now emits actual_plan_lineage_checkpoint_schema_ for both packet and output stems, remaining fresh while satisfying every unchanged prefix predicate. Its DIRECTIVE now exactly matches the unchanged acceptance contract. The producer, request, matrix, score and numerical science gates are unchanged.

One focused source-only execution passed 36 checks. It runs the actual builder path-assignment ASTs and complete unchanged validate_request, stage_source_v1, actual_stage_v1 and matrix/score evidence function ASTs against an explicitly synthetic in-memory VFS. All four phases accept corrected paths/directive. Each prior packet, output and directive mismatch rejects at its actual primitive gate. Matrix and score caller paths reject incompatible producer prefixes. The all5 closed-plan and all30 pretruth barriers remain enforced.

The prior 98 genuine-checkpoint metadata checks are retained by exact receipt and were not repeated. No checkpoint, model, scientific data, truth, guard, accounting or prediction was opened/executed during v2 checks. The original six-checkpoint metadata fix, full scientific support and existing VAL000–002 jobs remain unchanged.

This is an inactive source successor. Actual capacity/admission/execution still belong to Root after a complete independent review and a fresh explicit directive.
