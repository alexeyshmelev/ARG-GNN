# H049 bounded-v5: fixed directory allowance only

Actual v4 launcher returned1 and fully drained; its full terminal guard returned and256MiB original reservation remained retained. The exact wait is research/reviews/H049_bounded_actual_metadata_allocation_repair_20261015_v1/admitted_prepare/actual_wait.json SHA72fb580bc5fb3b36b6413a09d47708c959233cada78ca69c9be08eb9866d05fc.

The log records38/40 prepared parents (all32TRAIN plus VAL000–005). This is a preparation diagnostic, not a scientific result. VAL006's NPZ and JSON were published before the post-write check failed; their exact paths were checked by metadata only. VAL007 had not started. Every old partial artifact remains preserved and unadopted. No ancestry truth, cut array or checkpoint payload was opened.

The new primary error was Directory metadata cap at bounded_phase_writer_v1.py:105, called after receipt publication at line166. The directory had st_size12288 and st_blocks24, so counted charge=max(12288,24*512)+8192=20480 > old16384. The failed.json attempt then correctly failed because the phase was poisoned. failed.json and completed.json are absent.

## Minimal successor

Only two layout constants change: the allowance is65536 for each of the three fixed directories (output,aux,aux/ro). Writer and lifecycle bytes are identical. Driver/context changes only names, paths and the operational version. Scientific prepare_v2, codec_v2, cut mathematics, original40parents, six-member archives, file allocation/slack rules, scratch accounting and original guard/lifecycle behavior stay unchanged. No fixtures, probes, data jobs or broad framework were added.

Counted peak:261332992 +3*(65536-16384)=261480448 bytes. Full cap stays268435456 (256MiB), leaving6955008 bytes. The exact observed20480-byte directory charge fits. The unchanged runtime checks still fail-stop on actual growth; this is not a universal filesystem allocation guarantee.

Fresh source packet: research/reviews/H049_bounded_directory_metadata_repair_20261015_v1
Fresh job: H049-order2-prepare-seed101-bounded-v5-CPU
Fresh output: experiments/results/H049_CPU_order2_cut_memory_prepare_bounded_v5
Fresh admission/input: research/reviews/H049_bounded_directory_metadata_repair_20261015_v1/admitted_prepare/input.json

All proposals are inactive. Before root admission, BOTH scientific config execution_enabled and root_resource_caps_reviewed must be true in a fresh active config, and the request must have status=active plus that exact new config_ref. Both CLI calls require --execute. ROOT_COMMANDS.txt supplies the exact flags and request ref; root must fill the actual fresh input SHA returned by admission before launch. Expected active refs are computed metadata only; no active file was written here.

operational_directory_allowance.diff is the complete source change. source_static_receipt.v1.json records syntax/AST checks without importing or running project modules. source_manifest.json binds new sources and preserves old pins; stale inherited operational refs were updated to the new explicit namespace/version5. root_activation_requirements.inactive.json gives all activation fields/paths.

Original failed scopes/charges, v4 wait1,38 logged complete recipes and VAL006 partial publication remain untouched. No credit, close, replay, resume, admission or execution occurred.
