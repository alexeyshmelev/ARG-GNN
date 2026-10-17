# Inactive v2 deepcopy recursion repair

V1 remains BLOCKED and unchanged. Its independent review confirmed that the original strict parser accepts a1201byte array nested600levels at recursion limit1000, while the component's deepcopy raised RecursionError. V2 is a minimal immutable successor for that concern only.

On deepcopy RecursionError, v2 removes the private parsed value and marks the path uncacheable for the current original mutex transaction. An initial-copy failure returns its just-computed original-parser value. A later cache-hit failure calls the original parser afresh. Subsequent calls for that path also use the original parser, so no canonical parsed object is retained or shared. Other exception behavior is unchanged.

The uncacheable marker retains only immutable raw bytes and SHA. It keeps the original raw charge and participates in the same final original uncached byte comparison. Parsed charge is released. The original reference, byte/name, current-control, policy/schema and scope checks, lock, required guard calls, caps, restoration and original memo final snapshot remain. Both maps are destroyed before that snapshot frontier.

Twenty narrow author checks reproduce original-accept/v1-fail/v2-pass through the actual component constructor, exact old namespace/state ownership and original Census reader frames. The fixture SHA is250f8f81de72fe62fb3c3202430524b7a6938c02969b8be425e8b265d6899eea. The checks also cover a later hit copy failure from a deeper caller, independent returned objects, unchanged MemoryError propagation, uncached fallback final-byte drift rejection and cleanup. Original lock, IO and final snapshot frontiers use disclosed RAM providers. The independent scaffold is reused only for this narrow regression and source-derived setup.

One own fixture diagnostic is preserved: import-stripped AST loading changed two CPython3.11.12 adapter bytecodes relative to full-source compilation. The constructor correctly rejected. The test fixture now compiles the full original inert adapter source and substitutes only its RAM lifecycle provider. No component change or live operation resolved that fixture issue.

Independent v2 component review and a complete future actual source/control/runtime join remain necessary. This packet has no launch or adoption authority. No existing job, source, scope, clock, charge or failure was changed. No speedup, fit or training result is claimed.
