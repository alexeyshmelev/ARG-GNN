#!/usr/bin/env python3
"""Root supervisor for fresh native CPU-only preparation. Preserve terminal evidence."""
import datetime as dt,hashlib,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
os.chdir(ROOT)
OUT=ROOT/"experiments/results/official_argformer_tsinfer_parent_val001_v3_root"
admission=OUT/"admission_launch.json"
expected="c0a9b2c6d7185f231756641d0eeed5a902f87657978f902581a4645b6cb7984b"
assert hashlib.sha256(admission.read_bytes()).hexdigest()==expected
log=ROOT/"research/reviews/official_argformer_tsinfer_parent_v3_author/bio_paths_pilot_val_001.launch.log"
args=[str(ROOT/".runtime/conda/envs/argformer_native_v1/bin/python"),"scripts/prepare_official_argformer_tsinfer_parent_v3.py","--execute","--parent","bio_paths_pilot_val_001","--expected-source-sha256","3b767b7e5e0d86291173f80f2df1feff4122592753262de8950aed9a724b1bd3","--expected-config-sha256","686356c3ceb0e15ed7b500f00d97bcae4fe2d91d13b3c8317362392f09414616","--admission",str(admission.relative_to(ROOT)),"--expected-admission-sha256",expected]
env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1")
receipt={"args":args,"log":str(log.relative_to(ROOT)),"created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"GPU_requested":False,"scientific_fallback":False,"external_walltime_seconds":3600}
with log.open("xb") as out:
 child=subprocess.Popen(args,stdout=out,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 procstat=Path("/proc")/str(child.pid)/"stat"
 receipt["pid"]=child.pid
 receipt["start_ticks"]=int(procstat.read_text().rsplit(")",1)[1].split()[19])
 with (OUT/"launch.json").open("x") as f:json.dump(receipt,f,indent=2);f.write("\n")
 print(json.dumps({"pid":child.pid,"start_ticks":receipt["start_ticks"],"log":receipt["log"]}),flush=True)
 try:exit_code=child.wait(timeout=3600)
 except subprocess.TimeoutExpired:
  receipt["watchdog_timeout"]=True
  if procstat.exists() and int(procstat.read_text().rsplit(")",1)[1].split()[19])==receipt["start_ticks"]:
   os.killpg(child.pid,signal.SIGTERM)
   try:exit_code=child.wait(timeout=30)
   except subprocess.TimeoutExpired:
    if procstat.exists() and int(procstat.read_text().rsplit(")",1)[1].split()[19])==receipt["start_ticks"]:os.killpg(child.pid,signal.SIGKILL)
    exit_code=child.wait()
  else:exit_code=child.wait()
receipt.update(exit_code=exit_code,completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
with (OUT/"exit.json").open("x") as f:json.dump(receipt,f,indent=2);f.write("\n")
print(json.dumps({"exit_code":exit_code,"receipt":str((OUT/"exit.json").relative_to(ROOT))}),flush=True)
