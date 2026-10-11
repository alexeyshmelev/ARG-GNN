from pathlib import Path
import subprocess,json,datetime
R=Path('/workspace-SR003.nfs2/ARG-GNN')
D=R/'research/reviews/census_tsinfer_actual32_numerical_v1_root'
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
result=subprocess.run(['du','-s','-B1',str(R)],cwd=R,capture_output=True,text=True,timeout=1800)
(D/'storage_du.stdout').write_text(result.stdout)
(D/'storage_du.stderr').write_text(result.stderr)
receipt={'schema':'census_native_root_storage_snapshot_v1','status':'complete' if result.returncode==0 else 'failed','returncode':result.returncode,'project_root':str(R),'command':['du','-s','-B1',str(R)],'started_utc':start,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'used_project_bytes':int(result.stdout.split()[0]) if result.returncode==0 else None}
(D/'storage_snapshot.json').write_text(json.dumps(receipt,indent=2)+'\n')
