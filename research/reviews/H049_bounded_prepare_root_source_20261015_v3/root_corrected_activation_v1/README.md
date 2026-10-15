# Corrected H049 v3 activation metadata

The failed root request referenced the source-only config. Its scientific protocol and registration already matched. Only the two operational activation flags caused the failure: execution_enabled=false, then root_resource_caps_reviewed=false.

This packet changes those two flags and the documented config status. The active request differs from the failed active request only in config_ref. The sealed v3 driver, source manifest, corrected prepare_v2, codec_v2, all scientific fields and numeric caps remain identical. Both original pure metadata validators were actually called and passed; original failed flag combinations were reproduced. All128 source pins were hashed. Counts and the original layout functions give241945331 output forecast and242184192 bounded peak within unchanged242221056 full charge. No guard/accounting/data/job action ran.

Root must explicitly approve these exact config/request refs and reviewed caps, then use the unchanged v3 --admit prepare --execute route. The failed old request, trace, failure receipt and sealed source files are immutable. The driver retains the fresh v3 admission/output paths, which still do not exist. No driver or policy successor is needed.
