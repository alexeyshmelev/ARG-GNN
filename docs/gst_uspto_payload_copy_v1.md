# H022 GST USPTO member-byte copy

This source stage copies every frozen ZIP member as exact binary bytes, verifies independent uncompressed size/CRC32/SHA256, and preserves archive order, original member names, ancillary content and native split paths. It implements no dataset text/row parser, chemistry dependency, author model, scoring, GPU route, installation or new budget code. Default execution reports source metadata only. Author evidence uses handmade ZIPs and no actual ZIP payload.

The fixed opaque archive is `data/paper_native_raw_acquisition_v1/gst_native_snapshots/USPTO15K_data.zip`,3,446,572 bytes, SHA256 `66b4f5cb6b5de4bd7c3d72bdb7fa0c5073c4b97984109f0dae0b10997a9aa723`. The completed v2 inventory is `experiments/results/gst_uspto_archive_inventory_v2/gst-uspto-inventory-v2-r002/inventory.json`, SHA256 `68c01a23e8b8aebfc9158f296f162df78107a7226171068fac6cb48c44f7139f`. It declares12,710,870 uncompressed bytes. Its member hashes and payload CRCs remain unknown until a separately admitted actual copy.

The exact frozen order is:

1. `data/` (directory)
2. `data/connor_original.tsv` (7,792,561 bytes)
3. `data/test.txt` (987,653 bytes)
4. `data/train.txt` (3,443,361 bytes)
5. `data/valid.txt` (487,295 bytes)

Logical split mappings remain `train.txt -> data/train.txt`, `valid.txt -> data/valid.txt`, `test.txt -> data/test.txt`. All five members are retained. No flattening, renaming, deduplication, row decoding, newline conversion, split reassignment or archive-derived permissions occurs. New regular files use0600 and new directories0700. A later genuine native parser can receive `<destination>/data` without moving or changing any member bytes. This copy does not establish that the acquired upstream snapshot equals the original GST authors' used snapshot.

## Fixed bounds and authority

The one fresh destination reservation is16,000,000 bytes. Its basis is the complete12,710,870-byte inventory payload, a1,000,000-byte final receipt cap, a256,000-byte append-only progress cap and explicit filesystem-entry headroom. Opaque archive memory is bounded to the fixed3,446,572 bytes. Member decompression/copy chunks are at most65,536 bytes, with ordinary stdlib/zlib working buffers and bounded metadata. No entire uncompressed member is held in memory.

The unchanged `gst_uspto_archive_inventory_v2` revalidates complete central-directory metadata, path/type/encoding/extra-field protections and exact inventory order before any payload copy. Raw immutable bytes come from its original checked opaque reader, which verifies raw size, SHA256, singly linked regular type, no-follow opening and stable stat identity. Member metadata must equal the frozen inventory exactly.

Actual execution requires a fresh exact root copy admission containing implementation/config/raw/inventory digests, a separately registered H022 copy-only run, a source/handmade independent PASS, an exclusive exact destination, a complete reviewed `project_budget_v4` reservation and fresh first-read ledger pins. Authority dates are timezone-aware and bounded to24 hours. No row/model/GPU permission can enter this narrow authority. The existing source-only registration is bound in the author receipt and grants no payload execution.

Root uses one16 MB v4 reservation with no auxiliary scopes. Its consumer bindings include the final producer source/config, independent review and frozen inventory JSON. The original v4 implementation and configuration remain unchanged. The parent of the new destination must already exist; this consumer does not create unreserved ancestor directories. An example fresh destination is `experiments/results/gst_uspto_payload_copy_v1-r001`, whose parent already exists. Root may choose a different exact contained destination and freeze it in its authority/reservation.

The first v4 check binds complete ledger pins before opaque archive reading. Further admitted boundaries cover member/directory creation and planned member writes, then the final receipt. Admission bytes and freshness are rechecked at those boundaries. The copy also enforces its own byte/count limits during every actual write. These are the existing contractual v4 boundaries rather than an OS quota or new scheduler.

## Streaming and partial failures

`copy_payload_v1` is the finite byte-copy primitive used by the admitted CLI and handmade fixtures. It creates a fresh destination exclusively. Directory and file opens traverse relative directory descriptors with `O_DIRECTORY/O_NOFOLLOW`; regular files use `O_EXCL/O_NOFOLLOW`. Existing output is never adopted, overwritten or resumed. Every member retains the exact archive spelling and location.

`ZipFile.open` validates local names and overlapping member boundaries. Its declared-size-truncating payload reader is not used. Stored bytes or raw-deflate streams are independently consumed under fixed chunk bounds. Local method/flags/name and ordinary local CRC/size metadata must agree with the central directory. Deflate end-of-stream, trailing compressed bytes, full uncompressed count and CRC32 are checked. SHA256 comes from exactly the bytes successfully written, including actual short-write prefixes. Any mismatch fails the copy.

`progress.jsonl` records starts, actual written prefix counts/SHA256, member completions and failures. It is bounded and fsynced after events. `payload_manifest.json` records every frozen member in original order, completed content hashes/CRCs, failed prefixes and unreached members, plus exact source/admission/inventory/storage bindings. A successful receipt marks complete byte coverage and explicitly leaves parsing/model/scoring unperformed and dataset parsing unadmitted.

A failure keeps named files and completed/prefix evidence. Partial extraction is reported as partial, with real counts and prefix SHA/CRC, and is never marked a complete payload copy. Later members remain unreached with no invented hashes. CRC rejection can preserve a fully written but unaccepted member. If authority, namespace or storage prevents the final receipt, the bounded progress log and existing files remain and the caller reports that no final receipt was written. No additional write is permitted under stale authority. An early opaque/preflight failure can receive a bounded failure receipt only while the same root authority and scope remain valid. New attempts always require a fresh destination.

## Author evidence and next execution

The author suite has62 finite stdlib handmade checks. It verifies stored/deflated exact bytes, binary content and original line endings, member order, all member SHA/CRC values, raw/inventory/member/order mismatches, CRC corruption, full-stream size overrun/deficit,5000 bytes actually written before injected I/O failure, unsafe names/collisions, refusal to overwrite and stale authority preserving progress while refusing final receipt. A hard audit guard forbids opening the actual acquired ZIP and recorded zero attempts. Handmade output evidence is under1 MB per successful suite execution. These are author checks, not an independent verdict or actual dataset result.

Root preparation templates are under `research/reviews/gst_uspto_payload_copy_v1_author/`. All placeholders remain inactive. Root first completes the independent source review and registers the explicit unfinished H022 member-byte condition. Root then stages exact fresh admission/reservation records through unchanged v4, binds their hashes and obtains fresh first-read ledger pins. The actual next consumer command is:

```sh
cd /workspace-SR003.nfs2/ARG-GNN
source scripts/project_env.sh
PYTHONDONTWRITEBYTECODE=1 "$ARGGNN_PYTHON" -B \
  -m arggnn.gst_uspto_payload_copy_v1 --execute \
  --admission ROOT_FINAL_COPY_ADMISSION_JSON \
  --expected-admission-sha256 ROOT_FINAL_COPY_ADMISSION_SHA256
```

The default command without `--execute` is source-report only and accesses no ZIP or live budget/lock. The consumer CLI has no parse/model/scoring mode.

After successful actual copying, root must bind every resulting member hash and receipt before a separate original-parser implementation/admission. Preserve the unchanged native `open(path,"r")` encoding/universal-newline semantics, raw row enumeration, native retained-limit behavior, partial bond-token omissions and fatal H-edit exceptions described in `docs/gst_uspto_native_data_protocol_v1.md`. No row counts, parser outputs, native model results or empirical claims are supplied by this stage.
