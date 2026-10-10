"""Root CPU canary supervisor. It launches only the hash-bound configured canary."""
from pathlib import Path
import os, json, hashlib, datetime, subprocess, sys
ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
OUT=ROOT/'experiments/results/biological_observed_tsinfer_v2_root_canary_admission'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p): return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(name,data):
    p=OUT/name
    if p.exists(): raise RuntimeError('one-shot supervisor receipt already exists')
    p.write_text(json.dumps(data,indent=2)+'\n')
def proc_start(pid):
    text=(Path('/proc')/str(pid)/'stat').read_text()
    return int(text.rsplit(')',1)[1].split()[19])
def main():
    if len(sys.argv)!=3: raise SystemExit('usage: supervisor.py ADMISSION_RELATIVE_PATH ADMISSION_SHA256')
    path=(ROOT/sys.argv[1])
    if path.is_symlink() or path.resolve()!=path or ROOT not in path.parents:raise RuntimeError('canonical repo admission required')
    if sha(path)!=sys.argv[2]:raise RuntimeError('admission changed before supervisor')
    a=json.loads(path.read_text())
    if a.get('authorizer')!='root_agent' or a.get('mode')!='canary' or a.get('template_inactive_not_authorization') is not False:raise RuntimeError('fresh active canary receipt required')
    if a.get('scope_parents')!=['bio_paths_pilot_train_000']:raise RuntimeError('single TRAIN000 canary only')
    for key in ['implementation','config','independent_source_review','runtime','storage','experiment_registration']:
        p=ROOT/a[key]['path']
        if p.is_symlink() or p.resolve()!=p or ROOT not in p.parents or sha(p)!=a[key]['sha256']:raise RuntimeError('admitted receipt/source changed: '+key)
    review=json.loads((ROOT/a['independent_source_review']['path']).read_text())
    if a['independent_source_review_status']!=review.get('status'):raise RuntimeError('review status changed')
    if a.get('independent_review_verdict')!='PASS':raise RuntimeError('root-assessed independent PASS required')
    cfg=json.loads((ROOT/a['config']['path']).read_text())
    if (ROOT/cfg['output_root']).exists():raise RuntimeError('fresh absent canary output required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('hide CUDA explicitly')
    log=ROOT/cfg['phase_launch_logs']['canary']
    if log.exists():raise RuntimeError('fresh canary log required')
    argv=[str(ROOT/'.runtime/envs/biological_tsinfer_v1/bin/python'),a['implementation']['path'],'--mode','canary','--expected-config-sha256',a['config']['sha256'],'--expected-source-sha256',a['implementation']['sha256'],'--admission',str(path.relative_to(ROOT)),'--expected-admission-sha256',sys.argv[2]]
    launch=dict(kind='root_observed_canary1000_process',started_utc=utc(),admission=ref(path),supervisor_pid=os.getpid(),supervisor_start_ticks=proc_start(os.getpid()),argv=argv,log=str(log.relative_to(ROOT)),truth_or_oracle_scope=False,CUDA_VISIBLE_DEVICES='')
    with log.open('xb') as handle:
        child=subprocess.Popen(argv,cwd=ROOT,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        launch.update(child_pid=child.pid,child_start_ticks=proc_start(child.pid))
        write('started.json',launch)
        code=child.wait()
    result={**launch,'finished_utc':utc(),'exit_code':code,'log_sha256':sha(log),'log_bytes':log.stat().st_size}
    manifest=ROOT/cfg['output_root']/'canary'/'manifest.json'
    if manifest.exists():
        result['manifest']=ref(manifest)
        result['native_status']=json.loads(manifest.read_text()).get('status')
    else:result['native_status']='no_terminal_native_manifest'
    write('finished.json',result)
    print(json.dumps({k:result.get(k) for k in ['exit_code','native_status','child_pid','finished_utc']}))
if __name__=='__main__':main()
