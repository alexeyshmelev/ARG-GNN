# Independent project budget v3 source review

Verdict: FAIL for 80bf620cbc6fc8b9de5e61a5d285ce039e3fbb0751bc012c48ea7d4faa6b4567.

V3 resolves all four original v2 findings.32 guarded handmade tests ran across the main suite and strengthened cardinality reproduction, with30 passes,2 failures demonstrating the same source defect,0 errors and0 guard events. No production ledger/product was accessed.

current rejects a supplemental reservation map with more than4096 entries, but reserve bounds only the incoming batch and not the resulting permanent map. A qualifying4096-entry ledger accepts one fresh request, writes4097 entries atomically and returns success, after which every current/check_reservation fails closed.

Reject len(current reservations)+len(incoming)>4096 before proposal construction or any mutation. Retain atomic rejection and verify4095+1 and4096+1 edges.

The initial strengthened harness interruption and corrected bounded runner are preserved. Previous v2 and GST FAIL receipt hashes remain unchanged.
