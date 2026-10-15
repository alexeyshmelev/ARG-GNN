"""Inactive installed-JDK FLARE build successor. Reuses pinned original executor."""
from pathlib import Path
import hashlib,importlib.util,json,os,struct,zipfile
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/FLARE_existing_JDK_source_v1/build_v1.py"
OLD_SOURCE="research/reviews/FLARE_java_source_build_plan_v1/build_executor_v1.py"
OLD_SHA256="ac0cc9aa9d9a035d24b22234d90bcdfdffb3af1749ce41db7be773c29f152ac9"
SOURCES_REF=dict(path="research/reviews/FLARE_java_source_build_plan_v1/official_source_blob_manifest.json",
    sha256="fa6dd0fdd5f5524bd7f89437d5e29290eb6304729a6b4c1e6a3bcf72b63e6b85")
COMMIT="ec0ad295aeebfacf10d8cbfb6486ab78981de1b1"
VERSION="flare version 0.6.0 [04ad73fd0 31-Oct-2025]"
CAP=19000000
def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def project_ref(path):
    p=Path(path);p=p if p.is_absolute() else ROOT/p
    need(p.resolve()==p and p.is_relative_to(ROOT),"Canonical project file")
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p))
def load_original():
    need(sha(ROOT/OLD_SOURCE)==OLD_SHA256,"Frozen original build executor")
    spec=importlib.util.spec_from_file_location("_FLARE_frozen_original_build",ROOT/OLD_SOURCE)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def execute_v1(ad,storage,*,guard):
    """Root supplies a configured current-v3 budget callback. No JDK acquisition."""
    need(ad["stage"]=="build" and ad["execution_authorized"] is True
         and ad["root_host_resource_enforcement_active"] is True
         and storage["reserved_new_bytes"]==CAP,"Root-admitted exact19MB installed-JDK source build")
    original=load_original()
    source_manifest=original.read(SOURCES_REF)
    need(source_manifest["source_commit"]==COMMIT and source_manifest["source_count"]==96
         and sum(r["bytes"] for r in source_manifest["sources"])==898745,"Original official96 source blobs")
    qualification=original.read(ad["installed_JDK_qualification_ref"])
    need(qualification["schema"]=="FLARE_existing_installed_JDK_root_qualification_v1"
         and qualification["installed_environment_modified"] is False
         and qualification["no_JDK_install_required"] is True,"Genuine existing installed JDK qualification")
    tools={r["tool"]:r for r in qualification["records"]}
    need(set(tools)=={"java","javac","jar"},"Complete qualified installed JDK toolchain")
    prefix=Path("/home/jovyan/miniconda3/envs/busco/lib/jvm")
    for name,row in tools.items():
        path=prefix/"bin"/name
        need(row["actual_subprocess_run_returncode"]==0 and Path(row["executable"])==path
             and path.resolve()==path and os.access(path,os.X_OK) and sha(path)==row["executable_sha256"],
             "Exact read-only root-qualified installed JDK executable")
    allowed={r["URL"] for r in source_manifest["sources"]}
    class InstalledExecutor(original.Executor):
        def __init__(self):
            super().__init__(ad,guard);self.path=ROOT/ad["output_directory"];self.storage=storage
        def fetch(self,url,name,n,sha256=None,blob_sha1=None):
            need(url in allowed and name.startswith("flare/src/") and self.network+n<=898745,
                 "Only exact official Java source blobs; never JDK/package download")
            return super().fetch(url,name,n,sha256=sha256,blob_sha1=blob_sha1)
        def write(self,name,data):
            need(self.metadata_used+len(data)<=1000000,"Compact build receipt cap")
            return super().write(name,data)
        def limits(self):
            super().limits()
            if not self.path.exists():return
            classes=list((self.path/"flare/src").rglob("*.class"))
            need(len(classes)<=384 and sum(p.stat().st_size for p in classes)<=4000000,
                 "Count-bound complete generated class cap")
            jar=self.path/"flare.jar"
            need(not jar.exists() or jar.stat().st_size<=4000000,"Compact native JAR cap")
            for folder,cap in (("tmp",1000000),("logs",1000000)):
                need(original.single_pass_bytes(self.path/folder)<=cap,"Installed-JDK scratch/log cap")
    x=InstalledExecutor();need(not x.path.exists(),"Fresh exclusive build scope, no replay")
    x.check(18000000);x.path.mkdir()
    for name in ("flare/src","tmp","cache","home","logs","receipts"):x.file(name).mkdir(exist_ok=True)
    try:
        for row in source_manifest["sources"]:
            x.fetch(row["URL"],"flare/"+row["path"],row["bytes"],blob_sha1=row["git_blob_sha1"])
        oldplan=original.read(original.PLAN_REF)
        for name,row in oldplan["official_docs"].items():
            data=original.bound(row["existing_project_ref"]).read_bytes()
            need(len(data)==row["bytes"] and hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()==row["git_blob_sha1"],
                 "Original pinned README/licence")
            x.write("flare/"+name,data)
        def verify_sources():
            for row in source_manifest["sources"]:
                raw=(x.path/"flare"/row["path"]).read_bytes()
                need(len(raw)==row["bytes"] and hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()==row["git_blob_sha1"],
                     "Unmodified official Java bytes")
        verify_sources();x.check(14000000)
        flags=["-XX:-UsePerfData","-XX:ActiveProcessorCount=4","-Djava.io.tmpdir="+str(x.path/"tmp"),
            "-Duser.home="+str(x.path/"home"),"-XX:ErrorFile="+str(x.path/"logs/hs_err_pid%p.log")]
        jflags=["-J"+v for v in flags]
        java_version=x.run("java_version",[str(prefix/"bin/java"),*flags,"-version"],20)
        need(hashlib.sha256(java_version).hexdigest()==tools["java"]["version_output_sha256"],
             "Exact root-qualified Java version bytes")
        x.run("javac_build",[str(prefix/"bin/javac"),*jflags,"-J-Xmx512m","--release","11",
            "-cp",str(x.path/"flare/src")+"/",str(x.path/"flare/src/admix/AdmixMain.java")],600)
        x.run("jar_build",[str(prefix/"bin/jar"),*jflags,"-J-Xmx256m","cfe",str(x.path/"flare.jar"),
            "admix/AdmixMain","-C",str(x.path/"flare/src")+"/","./"],60)
        help_output=x.run("native_help",[str(prefix/"bin/java"),*flags,"-Xmx128m","-jar",str(x.path/"flare.jar")],20)
        need(VERSION.encode() in help_output,"Pinned official native help/version")
        verify_sources()
        with zipfile.ZipFile(x.path/"flare.jar") as jar:
            names=jar.namelist()
            need(len(names)==len(set(names)) and all(original.safe_name(n)==n.rstrip("/") for n in names),
                 "Safe unique native JAR members")
            need("Main-Class: admix/AdmixMain" in jar.read("META-INF/MANIFEST.MF").decode()
                 or "Main-Class: admix.AdmixMain" in jar.read("META-INF/MANIFEST.MF").decode(),"Original JAR main")
            klass=jar.read("admix/AdmixMain.class")
            need(klass[:4]==b"\xca\xfe\xba\xbe" and struct.unpack(">H",klass[6:8])[0]==55,"Explicit Java11 target55")
            need({n for n in names if n.endswith(".java")}=={r["path"][4:] for r in source_manifest["sources"]},
                 "All96 original packaged Java files")
            for row in source_manifest["sources"]:
                raw=jar.read(row["path"][4:])
                need(hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()==row["git_blob_sha1"],
                     "Every packaged source blob unchanged")
        java_ref=x.json("receipts/java_qualification.json",dict(schema="root_qualified_installed_JDK_for_FLARE_v1",
            root_authorizer="root_agent",installed_JDK_qualification_ref=ad["installed_JDK_qualification_ref"],
            java_executable=tools["java"]["executable"],java_executable_sha256=tools["java"]["executable_sha256"],
            java_version_output_sha256=hashlib.sha256(java_version).hexdigest(),version_probe_argv=x.waits[0]["argv"],
            installed_prefix_read_only=True,new_JDK_downloaded=False,tool_waits=x.waits))
        http_ref=x.write("receipts/HTTP_sources.jsonl",("\n".join(json.dumps(v,sort_keys=True) for v in x.http)+"\n").encode())
        build_ref=x.json("receipts/build_record.json",dict(schema="root_actual_FLARE_source_build_v1",
            root_authorizer="root_agent",source_commit=COMMIT,source_manifest_ref=SOURCES_REF,
            official96Java_verified_before_after_and_inside_JAR=True,native_sources_modified=False,
            compiler_release_target=11,class_major_version=55,installed_JDK_read_only=True,
            tool_waits=x.waits,HTTP_receipts_ref=http_ref,JAR_ref=project_ref(x.path/"flare.jar"),
            java_qualification_ref=java_ref,jar_reproducible_bytes_or_paper_binary_equivalence_claimed=False))
        binary=x.json("receipts/binary_provenance.json",dict(schema="root_qualified_FLARE_binary_v1",
            status="qualified_actual_source_build",root_authorizer="root_agent",repository="https://github.com/browning-lab/flare",
            tag="v0.6.0",source_commit=COMMIT,native_sources_modified=False,source_commit_build_verified=True,
            reported_version=VERSION,jar_ref=project_ref(x.path/"flare.jar"),build_record_ref=build_ref,
            java_qualification_ref=java_ref,licence="Apache-2.0",author_Table1_exact_binary_provenance_established=False))
        return x.json("completed.json",dict(schema="FLARE_actual_source_build_completed_v1",
            status="completed_verified_official_source_build",binary_provenance_ref=binary,
            java_qualification_ref=java_ref,build_record_ref=build_ref,jar_ref=project_ref(x.path/"flare.jar"),
            source_ref=project_ref(SOURCE),GPU_used=False,native_data_inference_performed=False))
    except BaseException as error:
        failure=dict(status="failed_partial_preserved_no_fallback",error_type=type(error).__name__,
            error=str(error)[:4096],actual_tool_waits=x.waits,source_ref=project_ref(SOURCE))
        print(json.dumps(failure),flush=True)
        if all(row["owned_group_ESRCH"] for row in x.waits):
            try:x.json("failed.json",failure)
            except BaseException:pass
        raise
