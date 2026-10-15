"""Bounded phase output engine. Standard library only. No global guard imports.

Activation is exclusively through a pinned root context AFTER the original full
launch validation. The immutable full reservation remains charged until root's
actual wait/drain, phase-lease release and original terminal/close guard.
"""
from __future__ import annotations
import fcntl, hashlib, io, json, os, stat, threading, time, zipfile
from datetime import datetime
from pathlib import Path

OVERHEAD=8192
ROUND=4096
NPZ_MEMBERS=("representatives_u32","episode_groups_u16","episode_of_hold_u32",
             "group_offsets_u32","above_horizon_bits","root_saturated_bits")

def require(ok,message):
    if not ok:raise ValueError(message)

def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()

def digest(value):return hashlib.sha256(encoded(value)).hexdigest()
def charge_cap(payload):return ((payload+ROUND-1)//ROUND)*ROUND+OVERHEAD
def actual_charge(s):return max(s.st_size,s.st_blocks*512)+OVERHEAD
def signature(s):return (s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_blocks,s.st_mtime_ns,s.st_ctime_ns)

def canonical(root,value):
    root=Path(root);p=Path(value);p=p if p.is_absolute() else root/p
    require(root==root.resolve() and p==p.resolve() and p.is_relative_to(root)
        and ".." not in p.parts,"Canonical contained path without symlinks")
    return p

def read_bound(root,record,maximum=32*1024**2):
    require(isinstance(record,dict) and set(record)=={"path","sha256"},"Exact bound path/hash")
    p=canonical(root,record["path"])
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(fd,"rb") as stream:
        before=os.fstat(stream.fileno());require(stat.S_ISREG(before.st_mode) and before.st_nlink==1,"Bound regular single-link file")
        data=stream.read(maximum+1);after=os.fstat(stream.fileno())
    require(len(data)<=maximum and signature(before)==signature(after)
        and signature(p.lstat())==signature(after)
        and hashlib.sha256(data).hexdigest()==record["sha256"],"Bound bytes/identity differ")
    return data

def identity(pid):
    p=Path("/proc")/str(pid);raw=(p/"stat").read_text()
    fields=raw[raw.rfind(")")+2:].split()
    return {"pid":pid,"uid":p.stat().st_uid,"start_ticks":int(fields[19]),"state":fields[0]}

def publish_exclusive_v1(directory_fd,temp_name,final_name):
    # NFS rejects renameat2 NOREPLACE. Reserve only a FRESH final name atomically,
    # then rename over this own empty placeholder under the admitted sole-writer
    # lock. Closed-reader phase: complete bytes are guaranteed when save returns.
    placeholder=os.open(final_name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=directory_fd)
    try:
        held=os.fstat(placeholder)
        named=os.stat(final_name,dir_fd=directory_fd,follow_symlinks=False)
        require(stat.S_ISREG(held.st_mode) and held.st_nlink==1 and held.st_size==0
            and actual_charge(held)<=OVERHEAD
            and signature(held)==signature(named),"Exact fresh empty owned placeholder")
        os.rename(temp_name,final_name,src_dir_fd=directory_fd,dst_dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:os.close(placeholder)

def validate_npz_members_v1(data,members):
    require(tuple(members)==NPZ_MEMBERS,"Exact H049 six-member contract")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        rows=archive.infolist();names=[r.filename for r in rows]
        require(len(rows)==6 and len(set(names))==6
            and set(names)=={name+".npy" for name in members},"Exact NPZ members; no extras/duplicates")
        require(all(not r.is_dir() and not r.flag_bits&1 and r.compress_type==zipfile.ZIP_DEFLATED for r in rows),"Regular unencrypted compressed NPY members")

class LocalWriteEngineV1:
    """Single-process local bounds. Only PhaseOutputV1 admits scientific use.

    The small fixture exercises this engine directly, without pretending that
    a fixture has a production reservation or a global-guard receipt.
    """
    def __init__(self,*,root,plan,auxiliary_existing,check_local):
        self.root=Path(root);self.plan=plan;self.check_local=check_local
        self.path=canonical(self.root,plan["output"]);self.aux=canonical(self.root,plan["auxiliary"])
        require(not self.path.exists() and self.path.parent.is_dir() and self.aux.is_dir(),"Fresh output; existing exclusive aux")
        self.allowed=plan["files"];self.records={};self.poisoned=False;self.lock=threading.Lock()
        self.aux_existing=auxiliary_existing;self.aux_signatures={}
        require(plan["per_file_overhead_bytes"]==OVERHEAD and plan["allocation_round_bytes"]==ROUND,"Original conservative overhead plus allocation rounding")
        require(plan["peak_phase_charge_bound_bytes"]<=plan["reservation_bytes"],"Whole phase peak fits fully charged reservation")
        self.check_local();self.path.mkdir(mode=0o700)
        self.fd=os.open(self.path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        self.directory_identity=(self.path.stat().st_dev,self.path.stat().st_ino)
        self._scan()

    def _scan(self):
        self.check_local()
        require((self.path.stat().st_dev,self.path.stat().st_ino)==self.directory_identity,"Owned output directory changed")
        total=0
        for folder in [self.path,self.aux,*[self.aux/x for x in self.plan["readonly_aux_directories"]]]:
            s=folder.lstat()
            require(stat.S_ISDIR(s.st_mode) and folder==folder.resolve(),"Exact regular scope directory")
            require(actual_charge(s)<=self.plan["per_directory_charge_cap_bytes"],"Directory metadata cap")
            total+=actual_charge(s)
        with os.scandir(self.path) as entries:names={e.name for e in entries}
        require(names==set(self.records),"Unknown/partial/replaced output blocks local writer")
        groups={}
        for name,record in self.records.items():
            s=(self.path/name).lstat()
            require(signature(s)==record["signature"] and stat.S_ISREG(s.st_mode) and s.st_nlink==1,"Owned file identity/size/blocks changed")
            total+=actual_charge(s);group=self.allowed[name]["group"]
            groups[group]=groups.get(group,0)+actual_charge(s)
        expected=set(self.aux_existing)|set(self.plan["readonly_aux_directories"])
        with os.scandir(self.aux) as entries:require({e.name for e in entries}==expected,"Unaccounted auxiliary growth")
        for name,record in self.aux_existing.items():
            p=self.aux/name;s=p.lstat()
            require(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and actual_charge(s)<=record["charge_cap_bytes"],"Exact bounded aux regular file")
            if name not in self.aux_signatures:
                require(hashlib.sha256(p.read_bytes()).hexdigest()==record["sha256"],"Bound aux contents")
                self.aux_signatures[name]=signature(s)
            require(signature(s)==self.aux_signatures[name],"Concurrent auxiliary writer is forbidden")
            total+=actual_charge(s)
        for name in self.plan["readonly_aux_directories"]:
            folder=self.aux/name
            require(not folder.stat().st_mode&0o222,"TMP/cache directory must have no write mode bits")
            with os.scandir(folder) as entries:require(not list(entries),"TMP/cache growth is forbidden")
        require(all(used<=self.plan["group_charge_caps"][g] for g,used in groups.items()),"Aggregate output group cap")
        require(total+self.plan["reserved_terminal_tail_bytes"]<=self.plan["reservation_bytes"],"Output+TMP+aux+terminal tail cap")
        self.used=total;self.groups=groups
        return total

    def write_bytes(self,name,data,kind):
        with self.lock:
            require(not self.poisoned,"Poisoned phase is never resumed")
            require(type(data) is bytes and name in self.allowed and name not in self.records
                and Path(name).name==name,"Fresh exact allowed output name")
            rule=self.allowed[name]
            require(kind==rule["kind"] and len(data)<=rule["max_payload_bytes"],"Exact kind and per-file payload cap")
            if kind=="npz":validate_npz_members_v1(data,rule["members"])
            self._scan()
            cap=charge_cap(len(data));group=rule["group"]
            require(self.groups.get(group,0)+cap<=self.plan["group_charge_caps"][group],"Prospective group cap")
            # Current temp and final share one payload inode plus one empty owned placeholder. Its payload
            # is charged below; scratch reserves the empty placeholder counter.
            # Never subtract an active file or create reservation credit.
            require(self.used+cap+self.plan["scratch_charge_bound_bytes"]+self.plan["reserved_terminal_tail_bytes"]
                <=self.plan["reservation_bytes"],"Prospective whole phase cap including scratch/TMP/aux/log")
            v=os.statvfs(self.path)
            require(cap+self.plan["scratch_charge_bound_bytes"]<=v.f_bavail*v.f_frsize,"Actual filesystem free bytes")
            temp="."+self.plan["nonce"]+".write_%03d.tmp"%len(self.records)
            fd=None
            try:
                fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=self.fd)
                with os.fdopen(fd,"wb") as stream:
                    fd=None;stream.write(data);stream.flush();os.fsync(stream.fileno());s=os.fstat(stream.fileno())
                    require(s.st_size==len(data) and actual_charge(s)<=cap and s.st_nlink==1,"Actual file allocation cap")
                    written_inode=(s.st_dev,s.st_ino)
                self.check_local()
                publish_exclusive_v1(self.fd,temp,name)
                s=(self.path/name).lstat()
                require(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_size==len(data)
                    and (s.st_dev,s.st_ino)==written_inode and actual_charge(s)<=cap,"Published exact fsynced complete single-link inode cap")
                self.records[name]={"signature":signature(s),"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data),"charge":actual_charge(s)}
                self._scan()
            except BaseException:
                self.poisoned=True
                if fd is not None:os.close(fd)
                # Preserve exact partial scratch/final artifacts. Root handles
                # any incomplete scratch/final state only after genuine owned drain.
                raise
            return {"path":str((self.path/name).relative_to(self.root)),"sha256":self.records[name]["sha256"]}

    def summary(self):
        self._scan()
        return {"schema":"bounded_phase_local_counter_v1","files":len(self.records),"scope_charge_bytes":self.used,
            "reserved_terminal_tail_bytes":self.plan["reserved_terminal_tail_bytes"],"scratch_charge_bound_bytes":self.plan["scratch_charge_bound_bytes"],
            "full_reservation_bytes":self.plan["reservation_bytes"],"active_scope_credit":False,"global_history_replayed_per_write":False}

    def close(self):os.close(self.fd)

class PhaseOutputV1:
    """H049 prepare adapter; no GPU/native mode and no global mutation API."""
    def __init__(self,*,root,plan_ref,spec,storage,full_launch_result,reader,expected_parents,immutable_metadata_refs,phase_lease_ref,environ=None):
        self.root=Path(root);self.reader=reader;self.spec=spec;self.storage=storage
        self.plan_ref=plan_ref;self.plan=json.loads(read_bound(self.root,plan_ref));p=self.plan
        require(p["schema"]=="H049_prepare_bounded_phase_plan_v1" and p["status"]=="active"
            and p["phase"]==spec["phase"]=="prepare" and spec["device"]=="cpu" and spec["physical_gpu"] is None,"Only prospective CPU preparation")
        require(p["job_id"]==storage["job_id"]==spec["job_id"] and p["output"]==spec["output_directory"]
            and p["auxiliary"]==spec["worker_auxiliary_scope"]
            and p["reservation_bytes"]==storage["reserved_new_bytes"]==storage["reservation"]["reserved_bytes"],"Exact immutable original reservation/output/aux")
        require(full_launch_result["reservation"]==storage["reservation"]
            and full_launch_result["scope_measurement"] is not None,"Direct original full launch result must be supplied by pinned context")
        require(plan_ref in storage["reservation"]["consumer_bindings"],"Phase plan was fully reserved before launch")
        require(p["reservation_key"]==storage["reservation"]["key"]
            and {str(canonical(self.root,x)) for x in storage["reservation"]["auxiliary_scopes"]}=={str(canonical(self.root,p["auxiliary"]))},"Exact reservation key and entire sole auxiliary scope")
        expire=lambda value:datetime.fromisoformat(value.replace("Z","+00:00")).timestamp()
        require(p["expires_unix"]<=min(expire(spec["expires_at_utc"]),expire(storage["expires_at_utc"])),"Local deadline cannot exceed original input/storage expiry")
        expected_files={p+".cut_recipe."+ext for p in expected_parents for ext in ["npz","json"]}|{"started.json","completed.json","failed.json"}
        require(len(expected_parents)==40 and set(p["parents"])==set(expected_parents)
            and set(p["files"])==expected_files
            and p["normal_output_sequence"]==["started.json"]+[q for parent in expected_parents for q in [parent+".cut_recipe.npz",parent+".cut_recipe.json"]]+["completed.json"],"Exact original40 pair filenames and normal sequence")
        self.env=os.environ if environ is None else environ
        require(self.env.get("CUDA_VISIBLE_DEVICES")=="" and self.env.get("PYTHONDONTWRITEBYTECODE")=="1","CPU and no bytecode cache")
        self.owner=identity(os.getpid());self.lease_fd=int(self.env["ARGGNN_BOUNDED_PHASE_LEASE_FD"])
        self.lease_path=canonical(self.root,phase_lease_ref["path"])
        require(self.lease_path==canonical(self.root,p["auxiliary"])/p["phase_lease_name"],"Exact reserved auxiliary phase lease path")
        self.lease_bytes=read_bound(self.root,phase_lease_ref);self.lease=json.loads(self.lease_bytes)
        require(self.lease["plan_ref"]==plan_ref and self.lease["reservation_digest"]==digest(storage["reservation"])
            and self.lease["job_id"]==p["job_id"],"Exact inherited root phase lease")
        # Sources/config/runtime are small immutable bindings. Bulk scientific
        # observations still use the original authenticated ObservationReader.
        require(p["source_sha256"]==spec["source_sha256"],"Complete original and additive source closure")
        self.immutable_refs=[{"path":k,"sha256":v} for k,v in p["source_sha256"].items()]
        self.immutable_refs+=immutable_metadata_refs
        own_source=Path(__file__).resolve().relative_to(self.root).as_posix()
        layout_source=str(Path(own_source).with_name("H049_prepare_layout_v1.py"))
        lifecycle_source=str(Path(own_source).with_name("root_phase_lifecycle_v1.py"))
        required=[{"path":k,"sha256":p["source_sha256"][k]} for k in [own_source,layout_source,spec["root_context_module_ref"]["path"],lifecycle_source]]
        require(p["required_reserved_refs"]==required
            and all(r in storage["reservation"]["consumer_bindings"] for r in required),"Exact reserved additive writer/layout/context closure")
        require(p["peak_phase_charge_bound_bytes"]<=p["reservation_bytes"],"Count-derived peak fits full original reservation")
        self.output_sequence=p["normal_output_sequence"];self.next=0;self.terminal=False
        aux={self.lease_path.name:{"sha256":phase_lease_ref["sha256"],"charge_cap_bytes":p["phase_lease_charge_cap_bytes"]}}
        self.engine=LocalWriteEngineV1(root=self.root,plan=p,auxiliary_existing=aux,check_local=self.local_check)
        self.path=self.engine.path

    def local_check(self):
        require(all(identity(os.getpid())[k]==self.owner[k] for k in ["pid","uid","start_ticks"]),"Same exclusive writer process; no fork/replay")
        require(time.time()<self.plan["expires_unix"],"Original phase/source authority expired")
        require(time.time()<self.lease["expires_unix"],"Phase lease expired")
        root_identity=identity(self.lease["root_pid"])
        require(root_identity["uid"]==self.lease["root_uid"] and root_identity["start_ticks"]==self.lease["root_start_ticks"]
            and root_identity["state"] not in {"Z","X"},"Root-owned writer lifecycle must still be live")
        disk=self.lease_path.lstat();held=os.fstat(self.lease_fd)
        require(stat.S_ISREG(disk.st_mode) and disk.st_nlink==1 and (disk.st_dev,disk.st_ino)==(held.st_dev,held.st_ino)
            and os.pread(self.lease_fd,65536,0)==self.lease_bytes,"Inherited immutable phase inode/metadata")
        with self.lease_path.open("r+b") as independent:
            try:fcntl.flock(independent,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:pass
            else:raise ValueError("Original root/worker phase lifetime lock is not held")
        for record in self.immutable_refs:read_bound(self.root,record)
        ro=str(canonical(self.root,self.plan["auxiliary"])/self.plan["readonly_aux_directories"][0])
        require(all(self.env.get(k)==ro for k in self.plan["readonly_environment_keys"]),"Every named TMP/cache route is fixed to empty read-only aux")
        # This checks sources, expiry and one bounded namespace. It never
        # reads global history, acquires the budget mutex, or frees a charge.

    def save(self,name,value,kind="json"):
        require(not self.terminal,"Terminal phase is never resumed")
        if name=="failed.json":
            require(self.next>=1,"Started receipt precedes failure")
        else:
            require(self.next<len(self.output_sequence) and name==self.output_sequence[self.next],"Exact started/40pairs/completed output sequence")
        rule=self.plan["files"].get(name);require(rule is not None,"Exact allowed filename")
        if kind=="npz":
            import numpy as np
            require(isinstance(value,dict) and set(value)==set(NPZ_MEMBERS),"Exact six compact arrays")
            require(all(a.flags.c_contiguous and not a.dtype.hasobject for a in value.values()),"Native C numeric arrays only")
            require(all(value[k].dtype.name==("uint16" if k=="episode_groups_u16" else "uint32" if k.endswith("_u32") else "uint8") for k in NPZ_MEMBERS),"Exact original compact member dtypes")
            parent=name[:-len(".cut_recipe.npz")];limits=self.plan["parents"][parent]
            n,t=limits["native_nodes"],limits["native_holds"];k=value["representatives_u32"].size;g=value["episode_groups_u16"].shape[0]
            require(0<k<=min(n,t*952) and 0<g<=t
                and value["representatives_u32"].shape==(k,) and value["episode_groups_u16"].shape==(g,952)
                and value["episode_of_hold_u32"].shape==(t,) and value["group_offsets_u32"].shape==(t+1,)
                and value["above_horizon_bits"].shape==((952+7)//8,)
                and value["root_saturated_bits"].shape==((k+7)//8,),"Original count-derived compact shapes")
            buffer=io.BytesIO();np.savez_compressed(buffer,**value);data=buffer.getvalue()
        elif kind=="json":
            from arggnn.census_tsinfer_fit_v1 import _jsonable_v1
            require(isinstance(value,dict) and value.get("schema")==rule["schema"],"Exact original receipt schema")
            if name.endswith(".cut_recipe.json"):
                parent=name[:-len(".cut_recipe.json")]
                require(value["parent_id"]==parent and value["observations_only"] is True and value["targets_read"] is False
                    and value["arrays_ref"]=={"path":str((self.path/(parent+".cut_recipe.npz")).relative_to(self.root)),"sha256":self.engine.records[parent+".cut_recipe.npz"]["sha256"]},"Receipt binds the newly written exact parent NPZ")
            data=encoded(_jsonable_v1(value))+b"\n"
        else:raise ValueError("Preparation permits only JSON and NPZ")
        result=self.engine.write_bytes(name,data,kind)
        if name in {"completed.json","failed.json"}:self.terminal=True
        else:self.next+=1
        return result

    def discard(self,record):raise ValueError("No deletion, replacement or active-scope credit")
    def local_summary_v1(self):return self.engine.summary()

def terminal_lease_gate_v1(root,phase_lease_ref):
    """Root-only prerequisite BEFORE the unchanged terminal/close guard.

    Call only after actual worker wait and complete owned writer drain. Root
    must first close its inherited lifetime handle. Returned independent held
    FD stays open throughout original full terminal validation and close.
    """
    path=canonical(Path(root),phase_lease_ref["path"]);expected=read_bound(Path(root),phase_lease_ref)
    fd=os.open(path,os.O_RDWR|os.O_NOFOLLOW)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(os.pread(fd,65536,0)==expected,"Exact ended phase lease")
        return fd
    except BaseException:os.close(fd);raise
