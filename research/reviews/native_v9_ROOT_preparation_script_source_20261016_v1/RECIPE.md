# ROOT preparation script draft

Default and --help are inert. Only ROOT may run --prepare-root-v9 after inspecting the script.

Run from the repository root with the pinned .runtime/env/bin/python -B -s interpreter:

    .runtime/env/bin/python -B -s -m research.reviews.native_v9_ROOT_preparation_script_source_20261016_v1.prepare_root_v1 \
      --prepare-root-v9 \
      --review-path research/reviews/native_VAL000_full_state_resume_independent_review_20261016_v9/review.v9.json \
      --review-sha256 0635aa2bfe723a1839487d4ad70178fd9358b823cf0f39c8287b4e83fe7c430c \
      --root-directive 'ROOT authorizes one exact reviewed v9 attempt and permanently revokes old launch and write capabilities'

The review path and hash are explicit actual CLI inputs. They are validated against the actual bytes and exact v9 manifest. Additional known old ROOT launcher metadata can be supplied using repeated --additional-known-source-ref exact JSON refs.

The active path preserves the genuine v8 qualification/bundle/revocation/failure and all old hashes and timestamps. It writes three new immutable compact ROOT records under the original charged AUX using the unchanged original storage validation and strict per-write boundary. It does not create current absence observations, replace historical records, access checkpoint contents, normalize the old alias, acquire a CUDA lease, sample GPU capacity or launch training.

No preliminary capacity call exists. The separately invoked reviewed v9 executor installs its real capacity-evidence sink before attach/capacity calls and performs actual GPU0/controller/capacity/lease checks. Supervisor and worker each obtain their own genuine fresh known-target evidence at the reviewed entry points.

The script validates the active v9 public contract, original full charge/prospective census and finite registry after writing the new authority. Failure blocks continuation. Existing outputs are never overwritten and no automatic retry is implemented. ROOT must inspect any actual failed preparation and retain its metadata. No scientific completion or new optimizer updates are implied.

This is a source-only draft. The author did not run the active path, original guards, accounting, resource probes, authority writes or any launch.
