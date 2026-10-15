"""Exact root lifecycle helpers. Source only; no global guard imports or calls.

Caller retains the original reserve, full prewrite/launch and terminal/close
calls. This module only creates the admitted tiny exclusive lease/readonly
runtime namespace, and gates the existing terminal call after actual drain.
"""
from pathlib import Path
import fcntl,hashlib,os
from bounded_phase_writer_v1 import canonical,read_bound,encoded,digest,identity,require,actual_charge,terminal_lease_gate_v1

def establish_after_original_full_guard_v1(*,root,plan_ref,spec,storage,full_prewrite_result):
    root=Path(root);p=__import__("json").loads(read_bound(root,plan_ref))
    require(p["status"]=="active" and p["phase"]==spec["phase"]=="prepare"
        and p["job_id"]==spec["job_id"]==storage["job_id"],"Actual fresh admitted prepare")
    require(full_prewrite_result["reservation"]==storage["reservation"]
        and full_prewrite_result["scope_measurement"] is not None
        and full_prewrite_result["required_write_bytes"]>=2*p["per_directory_charge_cap_bytes"]+p["phase_lease_charge_cap_bytes"],"Caller performed original full prewrite guard for exact tiny aux growth")
    aux=canonical(root,p["auxiliary"])
    require(not aux.exists() and not canonical(root,p["output"]).exists(),"Fresh exclusive output and whole aux")
    aux.mkdir(mode=0o700);ro=aux/p["readonly_aux_directories"][0];ro.mkdir(mode=0o555);ro.chmod(0o555)
    lease=aux/p["phase_lease_name"]
    fd=os.open(lease,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);owner=identity(os.getpid())
        value=dict(schema="bounded_phase_root_lifetime_lease_v1",job_id=p["job_id"],phase="prepare",
            root_pid=owner["pid"],root_uid=owner["uid"],root_start_ticks=owner["start_ticks"],
            nonce=p["nonce"],plan_ref=plan_ref,reservation_digest=digest(storage["reservation"]),expires_unix=p["expires_unix"])
        data=encoded(value)+b"\n";require(len(data)<=8192,"Exact tiny lease cap")
        written=0
        while written<len(data):
            amount=os.write(fd,data[written:]);require(amount>0,"Progress in tiny lease write");written+=amount
        os.fsync(fd);require(actual_charge(os.fstat(fd))<=p["phase_lease_charge_cap_bytes"],"Actual lease allocation cap")
        os.set_inheritable(fd,True)
        lease_ref={"path":lease.relative_to(root).as_posix(),"sha256":hashlib.sha256(data).hexdigest()}
        env={k:str(ro) for k in p["readonly_environment_keys"]}
        env.update(CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1",ARGGNN_BOUNDED_PHASE_LEASE_FD=str(fd))
        return fd,lease_ref,env
    except BaseException:os.close(fd);raise

def gate_after_actual_wait_and_drain_v1(*,root,phase_lease_ref,lifetime_fd,actual_wait):
    require(actual_wait["actual_direct_launcher_wait"] is True
        and actual_wait["owned_group_drained"] is True
        and type(actual_wait["returncode"]) is int,"Original genuine wait and complete owned writer drain")
    path=canonical(Path(root),phase_lease_ref["path"]);named=path.lstat();held=os.fstat(lifetime_fd)
    require((named.st_dev,named.st_ino)==(held.st_dev,held.st_ino),"Close only own exact lifetime handle")
    os.close(lifetime_fd)
    # Keep returned independent EX handle across original full terminal/close.
    # A surviving inherited holder blocks here. No blanket process skip.
    return terminal_lease_gate_v1(root,phase_lease_ref)
