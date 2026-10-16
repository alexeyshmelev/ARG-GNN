"""Inactive root-side usage sketch. No executable statements.

Root obtains actual JSON ref/raw records with its existing guarded reader.
The reference manifest lists saved six-record bundles for chr21 and22.

    packet = construct_native_chr21_chr22_v2(
        actual_bundle, actual_runtime_identity, original_bindings,
        chromosome=actual_chromosome, target=root_selected_target,
        max_chunks=root_selected_finite_J,
        created_at_utc=actual_issue, expires_at_utc=actual_absolute_expiry)

Root alone chooses target names/lifetime, materializes encoded_documents,
recounts consumer_bindings and obtains original guard admission. Only then
may root derive active input from input.inactive.json, fill actual storage_ref,
hash it and launch the reviewed nativev2. No helper reads,writes or launches.
"""
