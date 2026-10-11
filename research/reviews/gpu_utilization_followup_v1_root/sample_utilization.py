from pathlib import Path
import subprocess,time,json,datetime
base=Path("/workspace-SR003.nfs2/ARG-GNN/research/reviews/gpu_utilization_followup_v1_root")
records=[]
for i in range(45):
 p=subprocess.run(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True)
 rows=[tuple(int(v.strip()) for v in line.split(",")) for line in p.stdout.splitlines()]
 records.append({"UTC":datetime.datetime.now(datetime.timezone.utc).isoformat(),"GPUs":[{"index":a,"utilization_percent":b,"memory_mib":c} for a,b,c in rows if a in (2,3)]})
 time.sleep(1)
result={"scope":"45 second diagnostic sample, not full-training utilization","samples":records,"GPU_mean_percent":{str(g):sum(next(x["utilization_percent"] for x in r["GPUs"] if x["index"]==g) for r in records)/len(records) for g in (2,3)}}
with (base/"utilization_sample.json").open("x") as f:json.dump(result,f,indent=2);f.write("\n")
