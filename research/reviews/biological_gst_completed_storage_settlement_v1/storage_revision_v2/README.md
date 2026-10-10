# Completed GST storage settlement repair v2

Root preserved the original inspect-only source and481.8MB inventory. No ledger reduction was applied. Independent review identified silent os.walk errors and lack of under-lock scope revalidation.

This separate revision raises every walk error. It hash-binds the previous root inspection, verifies full completed output/log membership, conservative sizes and file SHA256 values twice under scheduler.lock, and requires exact agreement with that original inspection. Terminal known worker/descendant identities are checked again. Only the finished GST reservation can change, from23.521GB to a permanent2GB charge. Every file, other reservation, historical baseline and130GB external reserve remains represented and unchanged.

The output is contractually frozen after successful worker termination. This does not impose an OS quota or exclude a noncooperating writer. Any new writes to the finished scope require renewed accounting before being authorized. The complete measured allocated/apparent scope is a targeted bound, not a global storage measurement. Source review is required before --apply. Default inspect mode cannot mutate the ledger.
