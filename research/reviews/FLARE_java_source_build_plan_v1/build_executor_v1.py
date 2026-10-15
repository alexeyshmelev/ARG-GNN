"""Inactive, root-admitted one-build FLARE executor; no acquisition on import."""
from pathlib import Path,PurePosixPath
import argparse,datetime,errno,gzip,hashlib,io,json,os,posixpath,resource,selectors,shutil,signal,stat,struct,subprocess,sys,tarfile,time,zipfile
from urllib.request import Request,build_opener,HTTPRedirectHandler
from urllib.parse import urlsplit,urlunsplit
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/FLARE_java_source_build_plan_v1/build_executor_v1.py"
PLAN_REF={'path': 'research/reviews/FLARE_java_source_build_plan_v1/build_plan.inactive.json', 'sha256': '4dcadac82692081431dfbb79b66e3cb92eee0c237107678f928de0310dded600'}
SOURCES_REF={'path': 'research/reviews/FLARE_java_source_build_plan_v1/official_source_blob_manifest.json', 'sha256': 'fa6dd0fdd5f5524bd7f89437d5e29290eb6304729a6b4c1e6a3bcf72b63e6b85'}
TOOLCHAIN_REF={'path': 'research/reviews/FLARE_java_source_build_plan_v1/toolchain_plan.inactive.json', 'sha256': 'df618cc4a46d3829805f1ea9edea428c7f0c221742b23f3baf053c6afd79f007'}
CAP=1000000000
NET_CAP=200000000
JDK_CAP=600000000
MEMBERS=20000
SCOPE=".runtime/FLARE_java_source_build_jdk11_v1"
GUARD_REF=dict(path="research/reviews/concurrent_completed_scope_credits_v3/completed_scope_credits_v3.py",sha256="ea14c12c41b9385a86182aa9c24058f7b1276107fc5706c70e47828e8b2c31ef")
VERSION="flare version 0.6.0 [04ad73fd0 31-Oct-2025]"
COMMIT="ec0ad295aeebfacf10d8cbfb6486ab78981de1b1"
def require(ok,msg):
    if not ok:raise ValueError(msg)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(path,kind="sha256"):
    h=hashlib.new(kind)
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()
def ref(path):
    p=Path(path);p=p if p.is_absolute() else ROOT/p
    require(p.resolve()==p and p.is_relative_to(ROOT) and p.is_file(),"Canonical project file")
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=digest(p))
def bound(r):
    require(isinstance(r,dict) and set(r)=={"path","sha256"} and ref(r["path"])==r,"Exact project SHA binding")
    return ROOT/r["path"]
def read(r):
    p=bound(r);require(p.stat().st_size<=65536 and p.suffix==".json","Compact metadata only")
    return json.loads(p.read_text())
def safe_name(value):
    p=PurePosixPath(value.rstrip("/"))
    require(value and len(value)<=1024 and not p.is_absolute() and ".." not in p.parts
        and p.as_posix()==value.rstrip("/") and all(x not in ("",".") for x in p.parts),"Canonical relative member path")
    return p.as_posix()
def safe_link(name,value,symbolic,top):
    require(value and not PurePosixPath(value).is_absolute() and "\\" not in value,"Relative internal link required")
    result=posixpath.normpath(posixpath.join(posixpath.dirname(name),value) if symbolic else value)
    require(result==top or result.startswith(top+"/"),"Link escapes JDK subtree")
    safe_name(result);return result
def single_pass_bytes(path):
    total=0;count=0
    for base,dirs,files in os.walk(path,followlinks=False):
        for name in dirs+files:
            count+=1;require(count<=50000,"Bounded owned native-product count")
            p=Path(base)/name
            try:s=p.lstat()
            except FileNotFoundError:continue
            require(not stat.S_ISLNK(s.st_mode) and (stat.S_ISDIR(s.st_mode) or stat.S_ISREG(s.st_mode)),"Owned products must be ordinary")
            total+=max(s.st_size,s.st_blocks*512)
    return total
class Redirects(HTTPRedirectHandler):
    def __init__(self,state):self.state=state
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        self.state.request(newurl)
        return super().redirect_request(req,fp,code,msg,headers,newurl)
class Executor:
    def __init__(self,ad,guard):
        self.ad,self.guard=ad,guard;self.path=ROOT/SCOPE;self.started=time.monotonic()
        self.network=0;self.requests=0;self.http=[];self.waits=[];self.stage="initialize";self.metadata_used=0;self.log_bytes=0
        self.opener=build_opener(Redirects(self))
    def limits(self):
        require(time.monotonic()-self.started<=3600,"Root build elapsed deadline")
        require(datetime.datetime.now(datetime.timezone.utc)<datetime.datetime.fromisoformat(self.ad["expires_at_utc"].replace("Z","+00:00")),"Root build admission expired")
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=2000000000,"Build supervisor RSS allowance")
    def check(self,growth=0):
        self.limits()
        result=self.guard(boundary=True,required_write_bytes=int(growth))
        require(isinstance(result,dict) and result["reservation"]==self.storage["reservation"],"Actual root guard must cover this exact scope")
        self.limits()
    def file(self,name):
        name=safe_name(name);p=self.path/name
        require(p.resolve()==p and p.is_relative_to(self.path),"Owned file without links")
        p.parent.mkdir(parents=True,exist_ok=True);return p
    def write(self,name,data):
        require(self.metadata_used+len(data)<=8000000,"Retained build metadata cap")
        self.check(len(data)+8192)
        self.metadata_used+=len(data)
        p=self.file(name)
        with p.open("xb") as f:f.write(data);f.flush();os.fsync(f.fileno())
        return ref(p)
    def json(self,name,value):
        b=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()
        require(len(b)<=65536,"Compact execution receipt bound")
        return self.write(name,b)
    def request(self,url):
        u=urlsplit(url)
        require(u.scheme=="https" and u.hostname in ("github.com","release-assets.githubusercontent.com","objects.githubusercontent.com","raw.githubusercontent.com")
            and u.username is None and u.password is None,"Official HTTPS download hosts only")
        self.requests+=1;require(self.requests<=128,"HTTP response/request bound");self.limits()
    def fetch(self,url,name,n,sha256=None,blob_sha1=None):
        self.request(url);require(self.network+n+1<=NET_CAP,"Exact aggregate download body bound")
        p=self.file(name);h=hashlib.sha256();blob=hashlib.sha1(b"blob "+str(n).encode()+b"\0");got=0
        with self.opener.open(Request(url,headers={"User-Agent":"ARG-GNN-FLARE-root-build/1","Accept-Encoding":"identity"}),timeout=30) as r:
            require(r.status==200 and r.headers.get("Content-Encoding") in (None,"identity"),"Identity HTTP200 required")
            require(r.headers.get("Content-Length") in (None,str(n)),"Published exact response length")
            u=urlsplit(r.url);require(u.scheme=="https" and u.hostname in ("github.com","release-assets.githubusercontent.com","objects.githubusercontent.com","raw.githubusercontent.com"),"Final official download host")
            with p.open("xb") as out:
                while got<n:
                    self.limits();b=r.read(min(1048576,n-got))
                    require(b,"Truncated pinned download");out.write(b);h.update(b);blob.update(b);got+=len(b);self.network+=len(b)
                extra=r.read(1);self.network+=len(extra)
                require(not extra,"Pinned download exceeds exact maximum")
                out.flush();os.fsync(out.fileno())
        require(got==n and (sha256 is None or h.hexdigest()==sha256)
            and (blob_sha1 is None or blob.hexdigest()==blob_sha1),"Pinned package/source digest mismatch; preserve partial")
        self.http.append(dict(URL=url,final_URL=urlunsplit((u.scheme,u.netloc,u.path,"","")),bytes=got,sha256=h.hexdigest(),git_blob_sha1=blob.hexdigest() if blob_sha1 else None))
        return p
    def tar(self,path):
        class Limited:
            def __init__(self,f):self.f=f;self.total=0
            def read(inner,n=-1):
                require(n>=0 and n<=1048576,"Bounded tar read")
                data=inner.f.read(n);inner.total+=len(data)
                require(inner.total<=650000000,"Expanded JDK tar stream cap");self.limits();return data
        compressed=gzip.open(path,"rb")
        return compressed,tarfile.open(fileobj=Limited(compressed),mode="r|")
    def unpack_jdk(self,archive):
        self.stage="inspect_JDK";entries={};top=None
        compressed,tar=self.tar(archive)
        try:
            for m in tar:
                name=safe_name(m.name);first=name.split("/")[0]
                top=top or first;require(first==top and len(entries)<MEMBERS and name not in entries,"Single bounded unique JDK root")
                require(m.isdir() or m.isfile() or m.issym() or m.islnk(),"No special archive members")
                require(0<=m.size<=JDK_CAP,"Bounded archive member")
                entries[name]=dict(kind="dir" if m.isdir() else "file" if m.isfile() else "link",
                    size=m.size,mode=m.mode&0o777,target=safe_link(name,m.linkname,m.issym(),top) if m.issym() or m.islnk() else None,
                    original_linkname=m.linkname if m.issym() or m.islnk() else None,symbolic=m.issym())
        finally:tar.close();compressed.close()
        require(entries and top in entries and entries[top]["kind"]=="dir","Explicit unique JDK directory")
        for name in list(entries):
            parent=posixpath.dirname(name)
            while parent:
                require(parent in entries and entries[parent]["kind"]=="dir","No extraction through link or implicit unsafe parent")
                parent=posixpath.dirname(parent)
        children={}
        for name in entries:
            if name!=top:children.setdefault(posixpath.dirname(name),[]).append(name)
        expanded={};links=[]
        def expand(source,dest,stack):
            require(source in entries and source not in stack,"Internal link target exists without cycles")
            e=entries[source]
            if e["kind"]=="link":
                links.append(dict(path=dest,source_member=source,original_linkname=e["original_linkname"],
                    symbolic=e["symbolic"],resolved_target=e["target"],materialized_as_ordinary=True))
                return expand(e["target"],dest,stack+[source])
            require(dest not in expanded and len(expanded)<MEMBERS,"Bounded unique materialized paths")
            expanded[dest]=dict(e,source_member=source)
            if e["kind"]=="dir":
                for child in sorted(children.get(source,[])):expand(child,dest+"/"+child.rsplit("/",1)[-1],stack+[source])
        expand(top,top,[])
        planned=sum(((e["size"]+4095)//4096)*4096+8192 if e["kind"]=="file" else 8192 for e in expanded.values())
        require(planned<=JDK_CAP,"JDK extraction plus duplicated internal-link content exceeds600MB")
        self.check(planned+8192);self.stage="extract_JDK"
        for name,e in sorted(expanded.items(),key=lambda row:(row[0].count("/"),row[0])):
            if e["kind"]=="dir":self.file("jdk/"+name).mkdir(exist_ok=False)
        originals={};compressed,tar=self.tar(archive)
        try:
            for m in tar:
                name=safe_name(m.name)
                if not m.isfile():continue
                e=entries[name];p=self.file("jdk/"+name);h=hashlib.sha256();left=m.size
                with tar.extractfile(m) as source,p.open("xb") as out:
                    while left:
                        b=source.read(min(1048576,left));require(b,"Truncated qualified archive member")
                        out.write(b);h.update(b);left-=len(b);self.limits()
                    out.flush();os.fsync(out.fileno())
                p.chmod(e["mode"]);originals[name]=dict(size=e["size"],sha256=h.hexdigest(),mode=e["mode"])
        finally:tar.close();compressed.close()
        for name,e in expanded.items():
            if e["kind"]!="file" or name==e["source_member"]:continue
            source=self.path/"jdk"/e["source_member"];p=self.file("jdk/"+name);expected=originals[e["source_member"]]
            h=hashlib.sha256();n=0
            with source.open("rb") as src,p.open("xb") as out:
                for b in iter(lambda:src.read(1048576),b""):out.write(b);h.update(b);n+=len(b)
                out.flush();os.fsync(out.fileno())
            require(n==expected["size"] and h.hexdigest()==expected["sha256"],"Materialized internal link content differs")
            p.chmod(expected["mode"]);self.limits()
        for name,e in expanded.items():
            if e["kind"]=="dir":(self.path/"jdk"/name).chmod(e["mode"]|0o700)
        manifest=[]
        for name,e in sorted(expanded.items()):
            if e["kind"]=="file":
                expected=originals[e["source_member"]];p=self.path/"jdk"/name
                require(p.is_file() and p.stat().st_nlink==1 and digest(p)==expected["sha256"],"Extracted file content changed")
                manifest.append(json.dumps(dict(path=name,source_member=e["source_member"],**expected),sort_keys=True))
        self.write("receipts/JDK_contents.jsonl",("\n".join(manifest)+"\n").encode())
        link_ref=self.write("receipts/JDK_archive_links.jsonl",("\n".join(json.dumps(r,sort_keys=True) for r in links)+"\n").encode())
        self.json("receipts/JDK_archive_links.json",dict(normalization="validated_internal_links_materialized_to_nlink1_files",link_count=len(links),link_records_ref=link_ref,expanded_count_bound_bytes=planned))
        self.check();return self.path/"jdk"/top
    def run(self,label,argv,timeout):
        self.stage=label;start=time.monotonic();output=bytearray();error=None;proc=None;selector=None
        env=os.environ.copy()
        for key in ("JAVA_TOOL_OPTIONS","_JAVA_OPTIONS","JDK_JAVA_OPTIONS"):env.pop(key,None)
        env.update(CUDA_VISIBLE_DEVICES="",TMPDIR=str(self.path/"tmp"),TMP=str(self.path/"tmp"),TEMP=str(self.path/"tmp"),
            XDG_CACHE_HOME=str(self.path/"cache"),LANG="C",LC_ALL="C",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
        def process_limits():
            resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_FSIZE,(16000000,16000000))
        def terminate(sig):
            try:os.killpg(proc.pid,sig)
            except ProcessLookupError:pass
        try:
            proc=subprocess.Popen(argv,cwd=self.path,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                start_new_session=True,preexec_fn=process_limits)
            selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ)
            while selector.get_map() or proc.poll() is None:
                self.limits()
                require(time.monotonic()-start<=timeout,"Native tool walltime cap")
                # Own-child observations only; no storage guard/mutex while child writes.
                status=Path("/proc")/str(proc.pid)/"status"
                try:
                    fields={a:b.strip() for a,_,b in (line.partition(":") for line in status.read_text().splitlines())}
                    require(int(fields.get("VmRSS","0 kB").split()[0])*1024<=2000000000,"Owned native tool RSS cap")
                except FileNotFoundError:pass
                require(single_pass_bytes(self.path/"tmp")<=16000000,"Owned JVM tmp cap")
                require(single_pass_bytes(self.path/"logs")+len(output)<=4000000,"All JVM/tool logs cap")
                require(sum(p.stat().st_size for p in (self.path/"flare/src").rglob("*.class"))<=16000000,"Generated classes cap")
                require(not (self.path/"flare.jar").exists() or (self.path/"flare.jar").stat().st_size<=16000000,"JAR cap")
                for key,_ in selector.select(.25):
                    b=os.read(key.fileobj.fileno(),65536)
                    if not b:selector.unregister(key.fileobj);continue
                    require(len(output)+len(b)<=1048576 and self.log_bytes+len(output)+len(b)<=4000000,"Per-tool/aggregate stdout cap");output.extend(b)
        except BaseException as exc:
            error=dict(type=type(exc).__name__,message=str(exc)[:2048])
        finally:
            if proc is not None:
                if error:terminate(signal.SIGTERM)
                try:rc=proc.wait(timeout=30 if error else 5)
                except BaseException as exc:
                    error=error or dict(type=type(exc).__name__,message=str(exc)[:2048])
                    terminate(signal.SIGKILL);rc=proc.wait()
                group_absent=False
                try:
                    for _ in range(100):
                        try:os.killpg(proc.pid,0)
                        except ProcessLookupError:group_absent=True;break
                        terminate(signal.SIGKILL);time.sleep(.1)
                except BaseException as exc:error=error or dict(type=type(exc).__name__,message=str(exc)[:2048])
                if not group_absent:error=error or dict(type="GroupNotAbsent",message="No boundary or further write permitted until root quiescence")
            else:rc=None;group_absent=True
            try:
                if selector is not None:selector.close()
                if proc is not None:proc.stdout.close()
            except BaseException as exc:error=error or dict(type=type(exc).__name__,message=str(exc)[:2048])
        event=dict(label=label,argv=argv,returncode=rc,actual_direct_wait=proc is not None,
            owned_group_ESRCH=group_absent,error=error,stdout_sha256=hashlib.sha256(output).hexdigest(),elapsed_seconds=time.monotonic()-start)
        self.waits.append(event)
        try:print(json.dumps(dict(tool_actual_wait=event)),flush=True)
        except BaseException:pass
        require(group_absent,"Owned group unresolved; no disk write or guard boundary")
        # The caller precharged all tool/source/log growth. This write occurs only after ESRCH.
        require(self.log_bytes+len(output)<=4000000,"Aggregate retained tool log cap")
        p=self.file("logs/"+label+".txt")
        with p.open("xb") as out:out.write(output);out.flush();os.fsync(out.fileno())
        self.log_bytes+=len(output);event["stdout_ref"]=ref(p)
        require(error is None and rc==0,"Qualified native tool failed; preserve partials, no fallback")
        return bytes(output)
def execute_v1(admission_ref,*,guard):
    require(callable(guard),"Root's configured current v3 guard callback is mandatory")
    ad=read(admission_ref);storage=read(ad["storage_ref"])
    require(ad["schema"]=="FLARE_source_build_root_admission_v1" and ad["root_authorizer"]=="root_agent"
        and ad["execution_authorized"] is True and ad["GPU_authorized"] is False
        and ad["source_ref"]==ref(SOURCE) and ad["plan_ref"]==PLAN_REF and ad["source_manifest_ref"]==SOURCES_REF
        and ad["toolchain_plan_ref"]==TOOLCHAIN_REF and ad["output_directory"]==SCOPE
        and ad["output_cap_bytes"]==CAP and ad["network_byte_cap"]==NET_CAP and ad["CPU_threads"]==4
        and ad["max_host_RSS_bytes"]==2000000000 and ad["outer_process_group_timeout_seconds"]==3600
        and ad["root_host_resource_enforcement_active"] is True,"Fresh exact root CPU build authority required")
    require(storage["active"] is True and storage["consumer_uses_v9_guard"] is True
        and storage["accounting_mode"]=="concurrent_completed_scope_credits_v3"
        and storage["budget_components"]["consumer_adapter"]==GUARD_REF
        and storage["output_directory"]==SCOPE and storage["reserved_new_bytes"]==CAP
        and storage["reservation"]["output"]==str(ROOT/SCOPE) and storage["reservation"]["auxiliary_scopes"]==[],
        "One exact fresh1GB current-guard scope required")
    for r in (ad["source_ref"],PLAN_REF,SOURCES_REF,TOOLCHAIN_REF,ad["runtime_ref"]):
        bound(r);require(r in storage["reservation"]["consumer_bindings"],"Actual executor/plan/source/toolchain/runtime must be reserved")
    plan,sources,toolchain=read(PLAN_REF),read(SOURCES_REF),read(TOOLCHAIN_REF)
    require(sources["source_commit"]==COMMIT and sources["source_count"]==96
        and sum(r["bytes"] for r in sources["sources"])==898745,"Exact official96-source plan")
    require(not (ROOT/SCOPE).exists(),"Never overwrite/resume a failed build scope")
    x=Executor(ad,guard);x.storage=storage;x.check(32768);x.path.mkdir()
    for name in ("downloads","jdk","flare/src","tmp","cache","home","logs","receipts"):(x.path/name).mkdir(parents=True,exist_ok=True)
    try:
        x.stage="download_JDK";x.check(toolchain["exact_archive_bytes"]+8192)
        archive=x.fetch(toolchain["exact_archive_URL"],"downloads/temurin11.tar.gz",toolchain["exact_archive_bytes"],sha256=toolchain["exact_archive_sha256"])
        x.check();jdk=x.unpack_jdk(archive)
        release=jdk/"release";require(release.is_file() and 'JAVA_VERSION="11.0.32.1"' in release.read_text(),"Exact extracted JDK version metadata")
        for name in ("java","javac","jar"):require((jdk/"bin"/name).is_file() and os.access(jdk/"bin"/name,os.X_OK),"JDK executable present")
        x.stage="download_official_sources";x.check(3000000)
        for r in sources["sources"]:x.fetch(r["URL"],"flare/"+r["path"],r["bytes"],blob_sha1=r["git_blob_sha1"])
        for name,record in plan["official_docs"].items():
            data=bound(record["existing_project_ref"]).read_bytes()
            require(len(data)==record["bytes"] and hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()==record["git_blob_sha1"],"Pinned official documentation blob")
            with x.file("flare/"+name).open("xb") as out:out.write(data)
        def verify_sources():
            for r in sources["sources"]:
                p=x.path/"flare"/r["path"];raw=p.read_bytes()
                require(len(raw)==r["bytes"] and hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()==r["git_blob_sha1"],"Original Java sources changed")
        verify_sources();x.check()
        # Entire child-growth bound is checked once before this six-command sequence.
        x.check(76000000)
        common=["-XX:-UsePerfData","-XX:ActiveProcessorCount=4","-Djava.io.tmpdir="+str(x.path/"tmp"),"-Duser.home="+str(x.path/"home"),
            "-XX:ErrorFile="+str(x.path/"logs/hs_err_pid%p.log")]
        javacommon=["-J"+v for v in common]
        java_version=x.run("java_version",[str(jdk/"bin/java"),*common,"-version"],20)
        x.run("javac_version",[str(jdk/"bin/javac"),*javacommon,"-version"],20)
        x.run("jar_version",[str(jdk/"bin/jar"),*javacommon,"--version"],20)
        x.run("javac_build",[str(jdk/"bin/javac"),*javacommon,"-J-Xmx512m","-J-XX:ActiveProcessorCount=4",
            "-cp",str(x.path/"flare/src")+"/",str(x.path/"flare/src/admix/AdmixMain.java")],600)
        x.run("jar_build",[str(jdk/"bin/jar"),*javacommon,"-J-Xmx256m","-J-XX:ActiveProcessorCount=4",
            "cfe",str(x.path/"flare.jar"),"admix/AdmixMain","-C",str(x.path/"flare/src")+"/","./"],60)
        native_help=x.run("native_help",[str(jdk/"bin/java"),*common,"-Xmx128m","-XX:ActiveProcessorCount=4","-jar",str(x.path/"flare.jar")],20)
        require(b'11.0.32.1' in java_version and VERSION.encode() in native_help,"Actual JDK/native version differs")
        verify_sources()
        with zipfile.ZipFile(x.path/"flare.jar") as jar:
            names=jar.namelist();require(len(names)==len(set(names)) and all(safe_name(n)==n.rstrip("/") for n in names),"Safe unique JAR members")
            main=jar.read("META-INF/MANIFEST.MF").decode()
            require("Main-Class: admix/AdmixMain" in main or "Main-Class: admix.AdmixMain" in main,"Official main class")
            klass=jar.read("admix/AdmixMain.class")
            require(klass[:4]==b"\xca\xfe\xba\xbe" and struct.unpack(">H",klass[6:8])[0]==55,"JDK11 class target55")
            require({n for n in names if n.endswith(".java")}=={r["path"][4:] for r in sources["sources"]},"Complete original packaged Java set")
            for r in sources["sources"]:
                raw=jar.read(r["path"][4:])
                require(hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()==r["git_blob_sha1"],"Packaged source modified")
        x.check()
        java_record=dict(schema="root_qualified_project_JDK_for_FLARE_v1",status="qualified_actual_toolchain",
            root_authorizer="root_agent",toolchain_plan_ref=TOOLCHAIN_REF,archive_ref=ref(archive),
            extracted_release_ref=ref(release),java_executable=str(jdk/"bin/java"),java_executable_sha256=digest(jdk/"bin/java"),
            java_version_output_ref=ref(x.path/"logs/java_version.txt"),java_version_output_sha256=hashlib.sha256(java_version).hexdigest(),
            version_probe_argv=x.waits[0]["argv"],JDK_contents_ref=ref(x.path/"receipts/JDK_contents.jsonl"),
            link_normalization_ref=ref(x.path/"receipts/JDK_archive_links.json"),
            javac_executable_ref=ref(jdk/"bin/javac"),jar_tool_executable_ref=ref(jdk/"bin/jar"),actual_tool_exit_codes={r["label"]:r["returncode"] for r in x.waits})
        java_ref=x.json("receipts/java_qualification.json",java_record)
        http_ref=x.write("receipts/HTTP_downloads.jsonl",("\n".join(json.dumps(r,sort_keys=True) for r in x.http)+"\n").encode())
        build_record=dict(schema="root_actual_FLARE_source_build_v1",root_authorizer="root_agent",source_commit=COMMIT,
            source_manifest_ref=SOURCES_REF,plan_ref=PLAN_REF,toolchain_plan_ref=TOOLCHAIN_REF,
            official96Java_verified_before_after_and_inside_JAR=True,native_sources_modified=False,
            JVM_operational_flags_disclosed=common,CPU_threads=4,tool_waits=x.waits,
            HTTP_receipts_ref=http_ref,HTTP_request_count=x.requests,JAR_ref=ref(x.path/"flare.jar"),java_qualification_ref=java_ref,
            jar_reproducible_bytes_or_paper_binary_equivalence_claimed=False,completed_utc=utc())
        build_ref=x.json("receipts/build_record.json",build_record)
        binary=x.json("receipts/binary_provenance.json",dict(schema="root_qualified_FLARE_binary_v1",status="qualified_actual_source_build",
            root_authorizer="root_agent",repository="https://github.com/browning-lab/flare",tag="v0.6.0",source_commit=COMMIT,
            native_sources_modified=False,source_commit_build_verified=True,reported_version=VERSION,jar_ref=ref(x.path/"flare.jar"),
            build_record_ref=build_ref,java_qualification_ref=java_ref,licence="Apache-2.0",
            author_Table1_exact_binary_provenance_established=False))
        result=x.json("completed.json",dict(schema="FLARE_actual_source_build_completed_v1",status="completed_verified_official_source_build",
            binary_provenance_ref=binary,java_qualification_ref=java_ref,build_record_ref=build_ref,
            jar_ref=ref(x.path/"flare.jar"),download_body_bytes=x.network,HTTP_requests_including_redirects=x.requests,
            root_admission_ref=admission_ref,source_ref=ad["source_ref"],GPU_used=False,native_data_inference_performed=False))
        x.check();return result
    except BaseException as error:
        record=dict(schema="FLARE_source_build_failed_v1",status="failed_partial_preserved_no_fallback",stage=x.stage,
            error_type=type(error).__name__,error=str(error)[:4096],actual_tool_waits=x.waits,
            network_body_bytes=x.network,HTTP_requests_including_redirects=x.requests,root_admission_ref=admission_ref,
            native_sources_modified=False,scientific_inference_claimed=False)
        try:print(json.dumps(dict(build_failure=record)),flush=True)
        except BaseException:pass
        # If a child group is unresolved, no product boundary/write is safe.
        if all(r["owned_group_ESRCH"] for r in x.waits):
            try:x.json("failed.json",record)
            except BaseException:pass
        raise
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source-report",action="store_true");a=p.parse_args()
    if not a.source_report:p.error("Root must call execute_v1 with its exact admission and configured current guard callback")
    print(json.dumps(dict(status="source_only_no_acquisition",source_ref=ref(SOURCE),scope=SCOPE,cap_bytes=CAP,CPU_threads=4,
        plan_ref=PLAN_REF,source_manifest_ref=SOURCES_REF,toolchain_plan_ref=TOOLCHAIN_REF)))
