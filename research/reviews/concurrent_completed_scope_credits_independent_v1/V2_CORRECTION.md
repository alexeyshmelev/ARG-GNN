# Follow-up source correction

At root's request, this reviewer authored the inactive v2 correction. The independent verdict in AUDIT.md remains an audit of the preserved v1 SHA, not an independent audit of this follow-up.

Corrected source: research/reviews/concurrent_completed_scope_credits_v2/completed_scope_credits_v2.py

SHA256: f1b5141247244acbf74c60f2b0c2f93048736e9d417b2f7b634b609c41fe4ec3

The saved reference now uses SHA256 of the serialized data, with post-write regular-file/link/inode/size checks. Source path, fresh journal/mode and inactive templates use v2. The data schemas and public producer API names remain compatible.

A fake-I/O regression of the extracted save method returned the correct digest for a fabricated 33,554,433-byte serialized product and rejected inode replacement. No scientific product was written and no live guard was imported, configured or replayed. See the v2 source_receipt.json and synthetic_checks.json. Root still owns audit integration and activation; no amount of available credit was measured.
