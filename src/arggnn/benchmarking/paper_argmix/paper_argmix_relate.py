"""Bounded, readable RelateParallel stage equivalent with retained logs.

Chunks run sequentially; BuildTopology and InferBranchLengths sections use one
shared bounded worker pool. Every native process has a real consumed wait.
No log removal, background shell expressions, encoded source or auto retry.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import re
import subprocess
import threading
import time
from . import paper_argmix_generator as g


def run_parallel(config_path,native_work,output,*,guard):
    config=g.metadata(config_path)
    commands_path=g.confined(native_work)/"commands.json"
    commands=g.metadata(commands_path)
    g.need(commands["config_ref"]==g.file_ref(config_path),"Native recipe differs")
    install=g.metadata(g.verify_ref(config["input_refs"]["relate_installation"]))
    binary=str(g.verify_ref(install["binaries"]["Relate"]))
    converter=str(g.verify_ref(install["binaries"]["RelateFileFormats"]))
    source_wrapper=g.verify_ref(install["scripts"]["RelateParallel"])
    for reference in commands["input_refs"].values():
        g.verify_ref(reference)
    argv=commands["argv"][0]
    def option(flag):
        return argv[argv.index(flag)+1]
    g.need(argv[0]==binary and commands["argv"][1][0]==converter and option("--mode")=="All",
           "Recovered direct command required")
    workers=config["relate_execution"]["workers"]
    g.need(1<=workers<=16,"Finite worker bound1-16 required")
    output=g._fresh_directory(output,guard)
    cwd=g.confined(commands["cwd"])
    g.need(cwd==g.confined(native_work),"Native cwd differs")
    prefix=option("-o")
    g.need(prefix=="inferred" and not (cwd/prefix).exists(),"Fresh native prefix required")
    started=time.monotonic()
    lock=threading.Lock()
    records=[]
    def run(name,args):
        guard(0)
        exact=[binary]+args
        print(json.dumps(dict(stage=name,argv=exact,cwd=str(cwd))),flush=True)
        begin=time.monotonic()
        with (output/(name+".log")).open("xb") as log:
            process=subprocess.run(exact,cwd=cwd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                   timeout=max(1,config["bounds"]["native_timeout_seconds"]-(time.monotonic()-started)),check=False)
        record=dict(stage=name,argv=exact,returncode=process.returncode,actual_wait_consumed=True,
                    seconds=time.monotonic()-begin,log_ref=g.file_ref(output/(name+".log")))
        with lock:
            records.append(record)
            # An append-only receipt preserves completed waits even if a later stage fails.
            with (output/"actual_native_waits.jsonl").open("a") as stream:
                stream.write(json.dumps(record,sort_keys=True)+"\n")
        guard(0)
        g.need(process.returncode==0,"Relate stage failed: "+name)
    def parallel_sections(chunk,sections,mode):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures=[]
            for section in range(sections):
                common=["--mode",mode,"--chunk_index",str(chunk),"--first_section",str(section),"--last_section",str(section)]
                if mode=="BuildTopology":
                    common += ["--painting","0.001,1","-N",option("-N")]
                else:
                    common += ["-m",option("-m"),"-N",option("-N")]
                common += ["--sample_ages",option("--sample_ages"),"--seed",option("--seed"),"-o",prefix]
                futures.append(executor.submit(run,f"chunk{chunk}_{mode}_section{section}",common))
            for future in as_completed(futures):
                future.result()
    run("MakeChunks",["--mode","MakeChunks","--haps",option("--haps"),"--sample",option("--sample"),
                      "--map",option("--map"),"--memory",option("--memory"),"-o",prefix])
    chunks=sorted(int(re.fullmatch(r"parameters_c(\d+)\.bin",path.name).group(1))
                  for path in (cwd/prefix).glob("parameters_c*.bin"))
    g.need(chunks==list(range(len(chunks))) and len(chunks)>0,"Native chunk numbering differs")
    for chunk in chunks:
        run(f"chunk{chunk}_Paint",["--mode","Paint","--painting","0.001,1","-o",prefix,"--chunk_index",str(chunk)])
        sections=len(list((cwd/prefix/f"chunk_{chunk}"/"paint").glob("*bin")))
        g.need(sections>0,"No native painting sections")
        parallel_sections(chunk,sections,"BuildTopology")
        run(f"chunk{chunk}_FindEquivalentBranches",["--mode","FindEquivalentBranches","--chunk_index",str(chunk),"-o",prefix])
        parallel_sections(chunk,sections,"InferBranchLengths")
        run(f"chunk{chunk}_CombineSections",["--mode","CombineSections","-N",option("-N"),"--chunk_index",str(chunk),
                                             "--sample_ages",option("--sample_ages"),"-o",prefix])
    run("Finalize",["--mode","Finalize","--sample_ages",option("--sample_ages"),"-o",prefix])
    exact=commands["argv"][1]
    with (output/"ConvertToTreeSequence.log").open("xb") as log:
        process=subprocess.run(exact,cwd=cwd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                               timeout=max(1,config["bounds"]["native_timeout_seconds"]-(time.monotonic()-started)),check=False)
    record=dict(stage="ConvertToTreeSequence",argv=exact,returncode=process.returncode,actual_wait_consumed=True,
                log_ref=g.file_ref(output/"ConvertToTreeSequence.log"))
    records.append(record)
    guard(0)
    receipt=dict(schema="paper_argmix_native_terminal_v1",commands_ref=g.file_ref(commands_path),actual_wait_consumed=True,
                 returncodes=[0,process.returncode],returncode_semantics="all native staged waits succeeded; actual converter return code",
                 native_stage_waits=records,source_ref=g.file_ref(__file__),upstream_wrapper_ref=g.file_ref(source_wrapper),
                 execution_mode="bounded_staged",workers=workers,chunks=len(chunks),successful=process.returncode==0,
                 seconds=time.monotonic()-started,logs_removed=False)
    g.Writes(4*1024*1024,guard).json(output/"terminal.json",receipt)
    g.need(process.returncode==0,"Native conversion failed; preserve outputs")
    return receipt
