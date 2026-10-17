# Strict JSON parse memo independent component review

BLOCKED: computation reuse adds a RecursionError for valid original JSON.

The exact original parser accepts a 1,201-byte JSON value containing 600 nested arrays at recursion limit 1000. The full component constructor accepts its explicit frozen RAM provenance. Through exact original source-derived guard readers, namespace handle, transaction state, path, SHA and raw object ownership, the first memoized read then fails at parse_memo_v1.py:198 in deepcopy. Original parser success has become component failure; the raw and parsed caps do not prevent it.

The minimal repair is an uncachable original-parser fallback when deepcopy raises RecursionError. Evict or avoid retaining the canonical value, preserve fresh mutable returns, and keep every original byte/ref/read/check, exclusion, cap, lock and final-validation boundary. Preserve v1 and review a successor. Do not narrow valid original inputs or raise the recursion limit.

Seventeen focused positive checks and five rejection checks passed, alongside this confirmed semantic regression. Genuine historical storage agrees on ordinary reads and copies remain isolated. Error cleanup restores all functions and clears state; final uncached validation and the old snapshot frontier are preserved in RAM.

No operational module import, actual guard/lock/state/process/GPU/accounting operation, clock read, source mutation or live adoption occurred. All original jobs, sources and metadata remain unchanged. This review establishes no complete future route/control exclusions, speedup or runtime fit.
