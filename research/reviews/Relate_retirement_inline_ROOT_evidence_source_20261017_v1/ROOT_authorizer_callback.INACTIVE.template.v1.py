"""INACTIVE ROOT callback contract. No observation, authority or write is implemented."""
def authorize_inline_ROOT_retirement_v1(*,guard,context,finite):
    """ROOT must supply separately reviewed actual source in its fresh preparation.

    Called exactly once, within the original mutex/deadline, after original fresh
    state. context contains deep copies of original_reservation, expected_before,
    bindings, root_instruction_ref, fresh_metadata_paths and permanent_revocation_ref.
    ROOT must genuinely observe ALL original known PID/start/groups, then exclusively
    write NEW original-schema observation and original operation_draft authority
    binding SAME context state and new observation SHA. Return ONLY authority ref.
    Preserve all unknown history. No state replay, lock, accounting mutation,
    measurement, old-file mutation, deadline reset, retry or journal operation here.
    This stub has no active authority and must never be relabelled runnable.
    """
    raise RuntimeError("INACTIVE: actual ROOT callback, whole-join review and fresh finite instruction required")
