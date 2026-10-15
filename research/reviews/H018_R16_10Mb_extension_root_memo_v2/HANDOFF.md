# H018 10 Mb operational memo successor

This is the same frozen H018 20–30 Mb acquisition. Scientific worker prepare_h018_r16_10mb_extension_v1.py remains unchanged at 4c97c629…. The condition, config/runtime refs, 20 subjects, R16/Q24 roles, region, QC/AP rules, 500 MB network cap and 200 MB scientific output cap are unchanged. No acquisition was run.

The new packet is H018_R16_10Mb_extension_root_memo_v2 and output is data/H018_R16_20_30Mb_extension_memo_v2. The operational job/reservation key gains memo_v2. The old never-launched reservation remains fully charged, with no close/credit/release action in this driver.

Memo helper v2 is explicitly installed only in the future worker and launcher read/check processes, before configure/first budget check. Root admission, inspect/reserve/apply use guard() with memo_consumer false. Helper v2 and original v4 source are direct reserved bindings and manifest pins. All strict planned-write boundaries remain unthrottled.

Root supplies an active bound old-never-launched/revocation certificate using the inactive template, plus experiment_decisions_snapshot.json in this new folder. Admission requires exact old admitted/admission/storage refs, old driver/source identity, absent old launch.json/data folder, revoked old launch/resume/write capabilities, and unchanged fully charged old allocation. The same barrier is rechecked by the future worker and launcher.

Future root invocation: --admit --old-neverlaunched-ref PATH --old-neverlaunched-sha256 SHA, then --launch. The default is inert. No live guard/admission/data/runtime was executed or read by this source task.

The existing process control is byte-identical after excluding the added memo identity fields in launch.json: 2400-second deadline, immediate SIGKILL of the owned worker group on timeout, bounded 16 MiB RAM stdout, actual direct wait, owned-group drain and strict post-drain log writes. Reservation remains 200 MB+16MiB+65536 bytes as before. No new accounting framework or fixture campaign was added.
