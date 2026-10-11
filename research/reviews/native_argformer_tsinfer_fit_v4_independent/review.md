# Independent source review: REJECT

Source SHA256: 0cc1a9d47d69e1533a00de4ce8a8cefb2db2a8b342cba112072e64e24cc40457.
Config SHA256: 04aa7e623575c97b527ebbd45e3d68e5fdcbd6db16d71b41b9bf6ba3a4813061.
Author receipt SHA256: 8970d517d2d641ea8b552af6ce1d5043e2df16f5da6878b6e0bce24d658cce86.

V4 repairs the rejected v3 entrypoint case for safely proven native child
outputs. Real main() and real policy() now finalize a private configuration
byte mismatch and a missing private configuration. An injected early policy
error also produces bounded receipt.json and exit_status1. Scope and64KiB
failure limit come from the source-pinned ENTRY_OWNERSHIP contract, with no
rejected-configuration path or limit used. Missing-config observed digest is
null, while reviewed and observed digests remain separate.

The179 fresh source/delta checks include27 entrypoint cases. Root and marker
must bind the exact stage/run/source/config/scope/TMPDIR, and the root must bind
an exact independent PASS. Wrong/missing/symlink marker, root, source and review,
wrong/missing config arguments or a wrong config reference, bad TMPDIR, missing
native/execute flags, nonexistent output, symlink output and either regular
terminal file are denied without creating or amending output. Early
diagnostics create no directory and never authorize science.

One P2 terminal-preservation blocker remains:

**A dangling exit_status symlink passes terminal-presence guards and allows
a new receipt.** existing_entry_ownership at line893 and terminal_receipt at
line778 use Path.exists(), which follows symlinks. A dangling exit_status is an
existing directory entry but exists() reports false. The fresh real-main/
real-policy counterexample establishes otherwise exact fixture ownership,
places that existing dangling terminal entry, and causes a configuration byte
mismatch. Ownership is accepted. terminal_receipt creates receipt.json at
line806, then its exclusive open of exit_status at line807 raises
FileExistsError. The existing terminal entry therefore fails to prevent
output amendment. A dangling receipt entry is also accepted as owned, although
exclusive receipt creation stops that case before any amendment.

This is within the explicitly requested denial of amendment when either
terminal file exists. Treat all preexisting terminal directory entries as
present without following symlinks, consistently at ownership and terminal
write boundaries. Retain the exclusive-create protection and exact ownership.
The sealed v4 source/config and preceding v3 REJECT must remain unchanged.

The complete executor remainder equals v3 after only declared source/config/
version bindings and exclusion of the new source contract, two ownership
helpers and main. The entire configuration likewise equals v3 after its
enumerated version/result/TMP/reservation/inventory-kind bindings. All30 native
scientific ASTs and all five scientific wrapper ASTs remain exact. Strict
inventory parsing, full actual-child package/version/origin/import-byte
comparison, exact helper loading and executable-shadow denial remain exact
ASTs. The bound134-check independent v3 evidence remains preserved, including
its11,520-pair full-reference seal on280 handmade leaves with all96 reference
and all120 query paths. Its old entrypoint failure is specifically repaired.

A new tiny native retrieval kernel exercise searches every one of8 handmade
reference rows for3 query rows,24 exhaustive pairs, across chunks. It verifies
exact-ID ties, the zero-query case, full reference inclusion and raw k1/k5
votes without a probability floor. It contains no fit success wiring, model,
actual prepared arrays or biological results. Massive4096/large retrieval
fixtures were not repeated.

The original scientific recipe remains inactive: MLM5 epochs/logical512/
micro64/seed17 and contrastive1 epoch/full4096/seed36, the released tokenizer,
full-path dedup/source corpus, original encoder/model/optimizer/scheduler/loss,
native primary full-reference k1 and sensitivity k5 remain unchanged.
No query truth, scientific fallback or paper numerical-reproduction claim is
granted. The four fresh20GB parent reservations, short TMP namespaces,
unchanged budgetv4 companion and original measured500GB GPU launcher guard
remain as documented.

After a successor earns exact independent PASS, root still needs a genuine
read-only CPU capture in the pinned native interpreter, with CUDA hidden,
-B/no bytecode writes, no PYTHONHOME/user site/PYTHONPATH, the exact root helper,
complete scientific imports/distributions/origins and full prefix import-byte
manifest. Capture constructs no model and is not a fit or GPU qualification.
The retained v4 capture command is a proposal bound in the check report, not
an executed capture. A successor must provide newly version-bound proposals.

GPU work remains a separate root step: fresh stage/runtime/registration/storage
authority, exact same-parent prepared bindings, valid logical title and lease,
a genuinely free eligible device under the two-device ceiling, then the actual
unchanged4096 optimizer-memory probe. Fit requires its same-device PASS.
Retrieval requires completed same-ARG fit and its own CPU authority. No source
verdict, fake inventory, fixture receipt or storage PASS supplies these gates.

No native model, actual arrays, GPU inventory/context, live ledgers/locks,
genuine native runtime capture, installation, native-prefix mutation, Git
operation or authority activation occurred. Review writes are confined to this
independent review directory. Original source/config/author receipt and v3
REJECT retain their exact hashes.
