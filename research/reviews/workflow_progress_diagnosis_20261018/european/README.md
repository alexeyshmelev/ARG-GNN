# European metadata cap repair

All nine existing coordinate manifests passed the original JSON loader with a 256 MiB cap. Exact schema and every coordinate row passed validation. Each contains 1,336,748–1,342,573 SNPs and occupies 156,470,864–157,161,859 bytes. The old 128 MiB rejection was reproduced for every parent.

Original manifest hashes matched before and after parsing. Recipe, seeds, parent identities and all observed input hash bindings match the successful generation products. Parsing was sequential, with one full coordinate object in memory. No truth arrays, model or inference command was executed.

Reuse the nine successful simulation products. Preserve the nine failed inference scopes and start the required 80 downstream European and empirical tasks in fresh v3 scopes. Activation belongs to ROOT and should preserve the 139 registered successes. This review requests no separate controller.

The 256 MiB result establishes the input coordinate budget. It does not qualify future inference receipts. The frozen receipt writer can produce a site-join component up to 299,393,838 bytes at the largest observed site count, so a 512 MiB ceiling is recommended for downstream receipt readers. This changes an IO bound only.

[Repair specification](repair_specification.json) lists the required tasks, fresh scope scheme and unchanged input bindings. [Reviewed smoke](reviewed_coordinate_cap_smoke.json) points to an immutable failure snapshot. The raw smoke's hash of live controller state is retained as a read-time fingerprint, with its nonreplayable nature stated explicitly.
