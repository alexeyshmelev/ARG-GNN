# H022 GST payload-copy V2: denied-write correction

V2 corrects the sealed V1 independent rejection. V1, its configuration, author bundle, reviewer receipt and counterexample remain unchanged. V1's exception handler could append a151-byte `copy_failed` event after the exact authority/reservation guard had already rejected further writes. Its final receipt was refused, but its failure event bypassed that guard.

V2 guards every progress/failure event before writing it. An explicit guard rejection sets a sticky per-scope flag. Subsequent progress events, direct writes, file creation (including the internal unguarded creation option), directory creation and final-receipt writes reject immediately. The caller cannot resume a scope by returning to a now-successful guard. The production CLI has the same sticky denial across its initial copy and preflight-failure fallback, so an early denial cannot lead to another owned failure-output attempt. Closing existing descriptors remains permitted.

Existing files and progress remain intact. Failure-event and final-receipt refusals do not replace the original error. Inherited caller-owned stdout retains the original failure, authority-refusal diagnostics, real total written count and completed/failed member prefix metadata. It opens no additional destination or diagnostic file. A valid authority may still preserve a bounded failure event/receipt after an ordinary CRC, size or I/O failure. Once denied, that scope receives no further writes.

All member-byte and native-data behavior remains V1's behavior: the same fixed3,446,572-byte opaque ZIP SHA256, the same frozen inventory SHA256 `68c01a23e8b8aebfc9158f296f162df78107a7226171068fac6cb48c44f7139f`, all five member names/order including the ancillary TSV, the same12,710,870-byte payload, raw-deflate/stored size/CRC/SHA checks,65,536-byte chunks, exclusive no-follow output and16,000,000-byte scope cap. Final-receipt and progress caps remain1,000,000 and256,000 bytes. The original v4 implementation is unchanged. No text/row parsing, newline conversion, permission restoration, model/scoring/GPU route or native parser change is added. The payload manifest schema stays `gst_uspto_payload_manifest_v1`, with a truthful authority-denial diagnostic field.

The source delta proof verifies unchanged ASTs for raw member decompression and validation, frozen inventory matching, containment/output descriptor creation, directory traversal, final receipt serialization, close, and the normalized original root admission logic. V2's configuration binds the exact V1 source/config/author receipt and independent REJECT/counterexample. It retains every original bound, raw/inventory/protocol identity and source-only permission.

Author qualification has94 finite stdlib handmade checks. The62 adapted V1 byte-copy tests still pass. The32 correction cases freshly reproduce V1's151-byte append and show V2 appending0 bytes, with exact existing payload retained and later members/final receipt refused. They also deny specifically the failure event after a CRC error, verify the original CRC cause survives, test every scope writer after a one-time rejection and guard recovery, and exercise CLI initial/preflight denials using explicit handmade admission/opaque/budget hook stubs. No genuine authority, budget operation or acquired archive is used. Actual raw payload/member hashes and CRCs remain unqualified.

Default `python -B -m arggnn.gst_uspto_payload_copy_v2` reports source metadata only. Actual copying still needs a fresh separately registered root payload-only run, exact independent V2 source/config PASS, new destination, complete unchanged-v4 reservation and first-read ledger pins. The rejected V1 capability cannot activate V2. Root templates and exact commands are in `research/reviews/gst_uspto_payload_copy_v2_author/` and remain inactive.

```sh
cd /workspace-SR003.nfs2/ARG-GNN
source scripts/project_env.sh
PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" -B \
  -m arggnn.gst_uspto_payload_copy_v2 --execute \
  --admission ROOT_FINAL_V2_COPY_ADMISSION_JSON \
  --expected-admission-sha256 ROOT_FINAL_V2_COPY_ADMISSION_SHA256
```

Root alone activates that command after the separate review and resource binding. The existing parent's destination must be fresh. A failed destination is never resumed or overwritten. Later genuine native parsing remains the separate phase in `docs/gst_uspto_native_data_protocol_v1.md`, preserving original `open(path,"r")`, raw row order/IDs, retained limits, partial-token handling and fatal parser exceptions. V2 grants no actual payload, parser, model, profile or empirical acceptance.
