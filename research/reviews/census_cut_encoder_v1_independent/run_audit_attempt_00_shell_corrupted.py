from pathlib import Path
import datetime, hashlib, json, os, subprocess, sys, time
ROOT=Path(/workspace-SR003.nfs2/ARG-GNN)
OUT=ROOT/research/reviews/census_cut_encoder_v1_independent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
expected={
 src/arggnn/census_cut_exchange_v1.py:2819e7cef7ded38968e019ec015f0b2d1d264502a0762bfb3cf795e90c183b62,
 src/arggnn/census_tree_encoder_v1.py:09351221d2d91f72c8b1df12bc3799ed72e538597c31a55737ed1bec111f886e}
for p,h in expected.items(): assert sha(ROOT/p)==h, p
binding_checks={}
for namespace,bname in [(census_cut_exchange_v1_author,core_binding.json),(census_tree_encoder_v1_author,encoder_binding.json)]:
 bpath=ROOT/research/reviews/namespace/bname
 b=json.loads(bpath.read_text())
 hashes=dict(b[artifact_sha256])
 hashes[research/reviews/+namespace+/receipt.json]=b[receipt_sha256]
 check={p:{expected:h,actual:sha(ROOT/p)} for p,h in hashes.items()}
 binding_checks[namespace]={binding_sha256:sha(bpath),all_match:all(v[actual]==v[expected] for v in check.values()),artifacts:check}
 assert binding_checks[namespace][all_match]
(OUT/binding_checks.json).write_text(json.dumps(binding_checks,indent=2)+n)
paths=set(expected)
for b in binding_checks.values(): paths.update(b[artifacts])
paths.update([AGENTS.md,scripts/project_env.sh,docs/biological_kernel_design_v2.md,src/arggnn/anchor_spectral_compact_v1.py,src/arggnn/anchor_kernel.py])
before={p:sha(ROOT/p) for p in sorted(paths)}
(OUT/source_hashes_before.json).write_text(json.dumps(before,indent=2)+n)
for label,args in [(author_pytest,[-m,pytest,tests/test_census_cut_exchange_v1.py,tests/test_census_tree_encoder_v1.py,-q,-p,no:cacheprovider,--basetemp=+str(OUT/pytest_tmp)]),(independent_checks,[str(OUT/independent_checks.py)])]:
 log=OUT/(label+_attempt_01.txt); receipt=OUT/(label+_attempt_01.json)
 if log.exists() or receipt.exists(): raise RuntimeError(Refusing overwrite of +label)
 command=[sys.executable,*args];start=time.monotonic()
 with log.open(w) as handle: p=subprocess.run(command,cwd=ROOT,env=os.environ.copy(),stdout=handle,stderr=subprocess.STDOUT)
 record={started_at_utc:datetime.datetime.now(datetime.timezone.utc).isoformat(),command:command,seconds:time.monotonic()-start,exit_code:p.returncode,output_sha256:sha(log),environment:{k:os.environ.get(k) for k in [CUDA_VISIBLE_DEVICES,PYTHONDONTWRITEBYTECODE,OMP_NUM_THREADS,MKL_NUM_THREADS,TMPDIR]}}
 receipt.write_text(json.dumps(record,indent=2)+n)
 print(label,record,flush=True)
after={p:sha(ROOT/p) for p in sorted(paths)}
(OUT/source_hashes_after.json).write_text(json.dumps(after,indent=2)+n)
assert before==after, Sealed or author artifact changed during audit
