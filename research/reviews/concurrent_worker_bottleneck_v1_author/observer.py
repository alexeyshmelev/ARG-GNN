import os,json,time,subprocess,collections,statistics,hashlib
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).parent
PIDS=(1230680,1317405,1317433)
ROOT=Path.cwd()
HZ=os.sysconf("SC_CLK_TCK")
RESULTS={
"uncached202":Path("results/biological_mechanism_controls_v1_seed202"),
"cached303":Path("results/biological_mechanism_controls_v1_seed303_cached_gpu3_v2"),
"census101":Path("experiments/results/census_biological_fit_v2_gpu3_seed101_r002"),
}
LOG_ROOTS=(Path(".runtime/gpup/gpu3_fits_v2/jobs"),Path("experiments/results/gpu3_parallel_fits_v2_root"))
def utc():return datetime.now(timezone.utc).isoformat()
def readtext(p):
    with open(p) as f:return f.read()
def stat(p):
    s=readtext(p);a=s[s.rfind(")")+2:].split()
    return {"state":a[0],"utime_ticks":int(a[11]),"stime_ticks":int(a[12]),"starttime_ticks":int(a[19]),"processor":int(a[36])}
def process(pid):
    base=f"/proc/{pid}";d={"pid":pid}
    try:
        d.update(stat(base+"/stat"))
        d["io"]={x.split(":")[0]:int(x.split(":")[1]) for x in readtext(base+"/io").splitlines()}
        keep=("State:","Threads:","VmRSS:","voluntary_ctxt_switches:","nonvoluntary_ctxt_switches:")
        d["status"]={x.split(":")[0]:x.split(":",1)[1].strip() for x in readtext(base+"/status").splitlines() if x.startswith(keep)}
        threads=[]
        for tid in sorted(os.listdir(base+"/task"),key=int):
            try:
                t=stat(base+"/task/"+tid+"/stat");t["tid"]=int(tid)
                t["wchan"]=readtext(base+"/task/"+tid+"/wchan").strip()
                q=readtext(base+"/task/"+tid+"/schedstat").split()
                t["sched_cpu_ns"]=int(q[0]);t["sched_runqueue_wait_ns"]=int(q[1]);t["sched_slices"]=int(q[2])
                threads.append(t)
            except OSError:pass
        d["threads"]=threads
        files=[]
        for fd in os.listdir(base+"/fd"):
            try:target=os.readlink(base+"/fd/"+fd)
            except OSError:continue
            if target.startswith(str(ROOT)) and not target.startswith(str(ROOT)+"/.runtime/locks/"):
                files.append({"fd":int(fd),"path":target})
        d["project_open_fd_paths"]=sorted(files,key=lambda x:x["fd"])
    except OSError as e:d["error"]=str(e)
    return d
def metadata():
    result={}
    for name,p in RESULTS.items():
        entries=[]
        try:
            for e in os.scandir(p):
                s=e.stat(follow_symlinks=False)
                entries.append({"name":e.name,"bytes":s.st_size,"mtime_ns":s.st_mtime_ns,"is_dir":e.is_dir(follow_symlinks=False)})
            result[name]={"file_count":len(entries),"entries":sorted(entries,key=lambda x:x["name"])}
        except OSError as e:result[name]={"error":str(e)}
    logs=[]
    for base in LOG_ROOTS:
        try:
            for path,dirs,files in os.walk(base):
                if len(Path(path).relative_to(base).parts)>=2:dirs[:]=[]
                for name in files:
                    if name.endswith((".log",".stdout",".stderr")):
                        p=Path(path)/name;s=p.stat()
                        logs.append({"path":str(p),"bytes":s.st_size,"mtime_ns":s.st_mtime_ns})
        except OSError:pass
    result["log_metadata"]=logs
    return result
def query(args):
    before=time.monotonic()
    try:
        p=subprocess.run(["nvidia-smi"]+args,capture_output=True,text=True,timeout=5)
        return {"returncode":p.returncode,"stdout":p.stdout.strip(),"stderr":p.stderr.strip(),"duration_s":time.monotonic()-before}
    except (OSError,subprocess.TimeoutExpired) as e:return {"error":str(e),"duration_s":time.monotonic()-before}
record={"kind":"concurrent_worker_bottleneck_readonly_sample_v1","started_utc":utc(),"scope":"source, metadata, /proc counters and nvidia-smi only; no payload decoding, attachment or CUDA context","clock_ticks_per_second":HZ,"pids":PIDS,"metadata_before":metadata(),"samples":[]}
start=time.monotonic()
print(json.dumps({"stage":"observer_started","utc":record["started_utc"],"duration_requested_s":45}),flush=True)
for i in range(24):
    sample={"i":i,"utc":utc(),"elapsed_s":time.monotonic()-start,"processes":[process(pid) for pid in PIDS],"gpu":query(["-i","2,3","--query-gpu=index,uuid,pci.bus_id,utilization.gpu,utilization.memory,memory.used,memory.total","--format=csv,noheader,nounits"])}
    if i in (0,23):sample["compute_apps"]=query(["--query-compute-apps=pid,gpu_uuid,used_gpu_memory","--format=csv,noheader,nounits"])
    record["samples"].append(sample)
    if i<23:time.sleep(max(0,start+(i+1)*2-time.monotonic()))
record["wall_elapsed_s"]=time.monotonic()-start;record["finished_utc"]=utc();record["metadata_after"]=metadata()
summary={"elapsed_between_first_last_s":record["samples"][-1]["elapsed_s"]-record["samples"][0]["elapsed_s"],"sample_count":len(record["samples"]),"processes":{},"gpu":{}}
elapsed=summary["elapsed_between_first_last_s"]
for pid in PIDS:
    seq=[s["processes"][PIDS.index(pid)] for s in record["samples"]]
    first,last=seq[0],seq[-1]
    if "error" in first or "error" in last or first["starttime_ticks"]!=last["starttime_ticks"]:
        summary["processes"][str(pid)]={"identity_invalid":True};continue
    cpu=(last["utime_ticks"]+last["stime_ticks"]-first["utime_ticks"]-first["stime_ticks"])/HZ
    io={k:last["io"][k]-first["io"][k] for k in first["io"]}
    tb={t["tid"]:t for t in first["threads"]};td=[]
    for t in last["threads"]:
        if t["tid"] not in tb:continue
        a=tb[t["tid"]]
        td.append({"tid":t["tid"],"cpu_s":(t["utime_ticks"]+t["stime_ticks"]-a["utime_ticks"]-a["stime_ticks"])/HZ,"sched_cpu_s":(t["sched_cpu_ns"]-a["sched_cpu_ns"])/1e9,"runqueue_wait_s":(t["sched_runqueue_wait_ns"]-a["sched_runqueue_wait_ns"])/1e9})
    states=collections.Counter(p["state"] for p in seq if "state" in p)
    summary["processes"][str(pid)]={"cpu_s":cpu,"cpu_percent_one_core":100*cpu/elapsed,"io_deltas":io,"rchar_MiB_s":io["rchar"]/elapsed/1048576,"read_bytes_MiB_s":io["read_bytes"]/elapsed/1048576,"states":dict(states),"thread_deltas":sorted(td,key=lambda x:x["cpu_s"],reverse=True),"starttime_ticks":first["starttime_ticks"],"all_24_alive_same_starttime":all(p.get("starttime_ticks")==first["starttime_ticks"] for p in seq)}
for index in ("2","3"):
    rows=[]
    for s in record["samples"]:
        for line in s["gpu"].get("stdout","").splitlines():
            parts=[p.strip() for p in line.split(",")]
            if len(parts)==7 and parts[0]==index:
                rows.append([float(x) for x in parts[3:]])
    if rows:summary["gpu"][index]={"samples":len(rows),"gpu_util_mean_percent":statistics.mean(x[0] for x in rows),"gpu_util_min_percent":min(x[0] for x in rows),"gpu_util_max_percent":max(x[0] for x in rows),"memory_util_mean_percent":statistics.mean(x[1] for x in rows),"memory_used_min_MiB":min(x[2] for x in rows),"memory_used_max_MiB":max(x[2] for x in rows)}
record["summary"]=summary
path=OUT/"sample.json";path.write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
print(json.dumps({"stage":"observer_finished","artifact":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"summary":summary},sort_keys=True),flush=True)
