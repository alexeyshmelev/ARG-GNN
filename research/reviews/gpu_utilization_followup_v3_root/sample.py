from pathlib import Path
import subprocess,json,time,datetime
root=Path("/workspace-SR003.nfs2/ARG-GNN");base=root/"research/reviews/gpu_utilization_followup_v3_root";records=[]
for i in range(45):
 a=subprocess.run(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True).stdout
 rows=[[int(v.strip()) for v in l.split(",")] for l in a.splitlines()]
 b=subprocess.run(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name,used_memory","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True).stdout
 records.append({"UTC":datetime.datetime.now(datetime.timezone.utc).isoformat(),"GPUs":[{"index":x,"utilization_percent":y,"memory_MiB":z} for x,y,z in rows if x in [2,3]],"compute_processes":b})
 time.sleep(1)
mean={str(g):sum(next(x["utilization_percent"] for x in r["GPUs"] if x["index"]==g) for r in records)/len(records) for g in [2,3]}
with (base/"sample.json").open("x") as f:json.dump({"scope":"45-second current-workload sample, not whole-run throughput comparison","GPU_mean_percent":mean,"samples":records},f,indent=2);f.write("\n")
