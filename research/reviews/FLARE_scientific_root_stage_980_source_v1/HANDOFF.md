# Root FLARE 980 GB scientific stage IO successor

Source only, ready for root execution. The TRAIN000 role is
TRAIN_full30Mb_inference_canary_development_not_heldout.
No heldout run or ancestry accuracy is asserted. Scoring remains pending.
Root owns actual admissions, launch, completion accounting, Git and publishing.

The new driver and context route actual IO through the separately pinned active
prospective_980gb_v1 guard and root policy. Original v3 sources and policies,
the original stage packet and build helper, all 19 frozen scientific/helper
sources, all templates, actual completed build, target metadata and scientific
options remain byte unchanged. No frozen helper is monkeypatched.
The qualifier reads only sources, bounded JSON metadata and a pure byte formula.
Its future conversion/native predecessor fixtures are synthetic in memory and
are never written as actual completion evidence.

## Pins and static evidence

Driver: root_stage_980_driver.py
cd120bd6332ddc575d2adc1fd1af06104b3c96d2f4e6a81559958d83c0789e24

Context: root_context_980_v1.py
c53ecb9b159aa344383061044591631c60b5834ecabe7215d9fdb7dff9b41e04

Guard: research/reviews/prospective_980GB_capacity_source_v1/prospective_980gb_v1.py
fc71030865e0a3d5da31120d4daa2caa0b2d39ec7dd212b8c46c476b5a866d56

Policy: research/reviews/prospective_980GB_capacity_source_v1/activation_policy.root_active_v1.json
eb3408004a508136aa434c0a2e02a9d14a5598a96b8483c264c93026f9326018

Repair: research/reviews/H039_missing_marker_repair_v1/repair_authority_20261015.json
f6f2d628199e8be73db8b5d90d428570d8afdefeab9dc5cd94bb5eafb39247db

source_static_receipt.json records 53 checks, full ordered 27-source closure,
and prospective binding counts convert 46/native 54/seal 51. The preview roster
comes from existing actual980 admission metadata. Root admission obtains fresh
live state and enforces the 64-binding limit again.

Whole-context AST comparison permits only exact IO constants, admission schema,
storage mode and explanatory text changes. WaitWriter, target_v1, conversion_v1
and config_v1 ASTs match the frozen originals. Actual completed build/provenance
and real outer/direct wait0 metadata were validated without reading the JAR.
No genotype, focal, tree or query-truth payload was opened.

## Preserved scientific and resource settings

The actual target has 226185 observed sites, 600 reference plus 1000 query
haplotypes, 30 Mb and 10000 focal records. All supported sites/alleles and the same
focal registry are preserved. Author native options remain:
probs=true em=true update-p=false array=false min-maf=.005 min-mac=50 gen=10
panel-probs=false nthreads=4 seed=-99999.
Original native source commit ec0ad295aeebfacf10d8cbfb6486ab78981de1b1,
version flare version 0.6.0 [04ad73fd0 31-Oct-2025], 16 GiB Java heap,
14400-second native timeout and all four native output products remain exact.
Sealing uses original AN1/AN2 calls and preserves raw rounded ANP. Only the
already-disabled optional derived marker tract TSV stays disabled.

Convert/native/seal caps remain 950000000/9000000000/32000000 bytes, including
the original 524288-byte outer auxiliary allowance. RSS limits remain 4/24/4 GiB,
deadlines 3600/14400/3600 seconds and CPU threads 4. RSS enforcement is a polling
soft bound, with between-poll process limitations explicitly recorded.

The outer worker blocks on an inherited empty pipe before admission loading,
guard configuration or science. The parent seeds actual PID/start/PGID/session,
samples aggregate worker plus supervisor RSS and fsyncs launch_monitor_seed.json
before releasing the admission SHA ticket. Actual direct wait is exclusively
fsynced before postwait group checks, guard checks or logs. Original native
direct-wait persistence is unchanged.

## Root commands

Run from /workspace-SR003.nfs2/ARG-GNN on anogena.
These fixed scopes must be fresh. If any exists, root must choose a fresh suffix
consistently. The example expiry is 2026-10-17T18:00:00Z. Root may choose a closer
future absolute UTC expiry with sufficient native runtime. Nothing below has
been admitted or launched by this agent.

~~~bash
CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B -m research.reviews.FLARE_scientific_root_stage_980_source_v1.root_stage_980_driver   --admit --stage convert   --target research/reviews/FLARE_scientific_root_stage_source_v1/train000_target.json   --expected-target-sha256 a944251c7e5b0d7cc311192b2e953aac4f2d193235f4e8834c5f7745406e954b   --packet research/reviews/FLARE_train000_convert_980_root_v1   --output data/FLARE_train000_convert_980_root_v1   --job-id FLARE/TRAIN000-convert-980-root-v1   --expires-at-utc 2026-10-17T18:00:00Z
~~~

Use the actual returned admission SHA for launch. For the fixed scope above:

~~~bash
CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B -m research.reviews.FLARE_scientific_root_stage_980_source_v1.root_stage_980_driver   --launch --admission research/reviews/FLARE_train000_convert_980_root_v1/admission.json   --expected-admission-sha256 "$(sha256sum research/reviews/FLARE_train000_convert_980_root_v1/admission.json | cut -d ' ' -f 1)"
~~~

Only after conversion genuinely succeeds and its real outer/direct wait0
receipts are preserved, admit native inference using that actual conversion.
The already successful installed-JDK build is the pinned default. No rebuild
or installed-Java probe is requested by this source handoff.

~~~bash
CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B -m research.reviews.FLARE_scientific_root_stage_980_source_v1.root_stage_980_driver   --admit --stage native   --target research/reviews/FLARE_scientific_root_stage_source_v1/train000_target.json   --expected-target-sha256 a944251c7e5b0d7cc311192b2e953aac4f2d193235f4e8834c5f7745406e954b   --conversion data/FLARE_train000_convert_980_root_v1/completed.json   --expected-conversion-sha256 "$(sha256sum data/FLARE_train000_convert_980_root_v1/completed.json | cut -d ' ' -f 1)"   --packet research/reviews/FLARE_train000_native_980_root_v1   --output data/FLARE_train000_native_980_root_v1   --job-id FLARE/TRAIN000-native-980-root-v1   --expires-at-utc 2026-10-17T18:00:00Z

CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B -m research.reviews.FLARE_scientific_root_stage_980_source_v1.root_stage_980_driver   --launch --admission research/reviews/FLARE_train000_native_980_root_v1/admission.json   --expected-admission-sha256 "$(sha256sum research/reviews/FLARE_train000_native_980_root_v1/admission.json | cut -d ' ' -f 1)"
~~~

Only after actual successful native completion/direct native wait0 and actual
outer wait0, seal its original products:

~~~bash
CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B -m research.reviews.FLARE_scientific_root_stage_980_source_v1.root_stage_980_driver   --admit --stage seal   --target research/reviews/FLARE_scientific_root_stage_source_v1/train000_target.json   --expected-target-sha256 a944251c7e5b0d7cc311192b2e953aac4f2d193235f4e8834c5f7745406e954b   --native-run data/FLARE_train000_native_980_root_v1/completed.json   --expected-native-run-sha256 "$(sha256sum data/FLARE_train000_native_980_root_v1/completed.json | cut -d ' ' -f 1)"   --packet research/reviews/FLARE_train000_seal_980_root_v1   --output data/FLARE_train000_seal_980_root_v1   --job-id FLARE/TRAIN000-seal-980-root-v1   --expires-at-utc 2026-10-17T18:00:00Z

CUDA_VISIBLE_DEVICES='' .runtime/env/bin/python -B -m research.reviews.FLARE_scientific_root_stage_980_source_v1.root_stage_980_driver   --launch --admission research/reviews/FLARE_train000_seal_980_root_v1/admission.json   --expected-admission-sha256 "$(sha256sum research/reviews/FLARE_train000_seal_980_root_v1/admission.json | cut -d ' ' -f 1)"
~~~

For every actual run preserve admission/storage/authority/reserve metadata,
launch_monitor_seed.json, actual_wait_minimal.json, outer_wait.json and unchanged
producer receipts. No automatic completed-scope credit is applied. Root decides
any later exact-evidence completion accounting and registered heldout/scoring
work. This handoff exposes no score admission mode.
