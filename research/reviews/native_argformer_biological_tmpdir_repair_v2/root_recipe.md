# Root retry recipe

1. Review the bootstrap-only diff against original/source040 and freeze runner
   7bba1a16c57710ebab491bdd42c1ed444421cc4631a8841bdef57e5fa1224e16.
2. Reuse the existing prepared parent directory and its prepared_manifest SHA256;
   do not prepare paths again. Keep policy84 unchanged.
3. Source scripts/project_env.sh and launch through the existing GPU scheduler
   with a unique job and fresh probe2 output:

    .runtime/env/bin/python scripts/launch_gpu.py --gpu <eligible-free-gpu> --job <unique-probe2-job> \
      arggnn.train_official_argformer_biological --action probe-memory \
      --prepared-dir <existing-prepared-parent-directory> \
      --expected-prepared-sha256 <existing-prepared-manifest-sha256> \
      --expected-config-sha256 84eaf8b6638f01f37b6519193b8e9d85880880123cdede002a17de80129f098e \
      --output-dir <fresh-probe2-directory>

Verify exec_admission.json TMPDIR and actual full4096 forward/backward/optimizer
step receipts. Preserve any new failure; it does not authorize source/model/loader
changes. Same-code actual cache parity remains mandatory before a cache fit.

CPU IPC smoke already passed. Its standalone recipe is cpu_ipc_smoke.py; rerunning
that exact smoke identity will reject the existing test TMPDIR rather than overwrite.
