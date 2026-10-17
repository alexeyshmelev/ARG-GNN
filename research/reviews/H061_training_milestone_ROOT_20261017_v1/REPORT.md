# H061 selected validation training milestone

Snapshot:2026-10-17T00:00:21.629186+00:00. Four completed fits at seeds101/202
verify3200of4800planned optimizer updates. Each completed fit retains40epochs and
800updates. Seed303 was not included; this snapshot does not await completion.
Overall job completion is unaccepted pending genuine wait/drain and consumed outer
terminal. TEST remains withheld. These are selected validation count forecasts,
not TEST results,ancestry accuracy or a final memory-utility comparison.

| Seed | Arm | Selected epoch | VAL MAE,cases | VAL RMSE,cases |
|---|---|---:|---:|---:|
|101|Full|22|16.034358|25.807364|
|101|NoMem|12|16.113024|26.235912|
|202|Full|37|15.958219|25.296648|
|202|NoMem|23|16.092824|25.830167|

Across2of3planned initialization seeds,Full selected VAL MAE is
15.996289±0.053838cases and RMSE25.552006±0.361131cases. NoMem selected VAL MAE is
16.102924±0.014284cases and RMSE26.033039±0.286905cases. Values are mean±sampleSD
across completed initialization seeds at each fit's own selected epoch. These
partial,validation-selected summaries do not establish an unbiased final effect.

Support remains78VALIDforecast weeks/all20counties/52pastweeks,TRAINtargets52:365,
VALIDtargets365:443 and scalerfit0:365. AdamWlr0.001,weight_decay0.0001,40epochs/
800updates perfit;6fits/4800updates planned. CPU223,one thread,FP32. Complete40epoch
history and earliest minimum VALMAE selection were checked in immediate selection
JSON only. Exact sourcepaths/SHArefs are in milestone.root_v1.json. No weights,
prediction/target/truth arrays,resource/process paths or model were read.

scripts/update_current_scoreboard.py now reads runtime_status.root_latest.json
jobs.H061.summary and jobs.H061.training_milestone_ref. It authenticates only the
compact milestone JSON and adds H061_Chickenpox_training to generated JSON,a
selected-validation section to current_scoreboard.md,and H061rows/prose to the
generated READMEblock. Existing reporting is preserved. The generator compiled;
it was not imported/run,and no refresh/Git occurred. README,decisions,runtime
snapshot,progress_current and generated scoreboards were not edited. ROOT should
merge ROOT_reporting_snippets.v1.json fields while preserving all other records,
then perform its own normal refresh/integration. No job completion or TESTrelease
is authorized by these reporting artifacts.
