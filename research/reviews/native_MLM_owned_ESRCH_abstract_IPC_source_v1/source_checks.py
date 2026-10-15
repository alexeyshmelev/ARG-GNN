"""Bounded stdlib-only source checks and one real native-interpreter fork toy."""
from pathlib import Path
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import multiprocessing as mp
import multiprocessing.connection as connection
from multiprocessing.reduction import DupFd
import multiprocessing.resource_sharer as resource_sharer
import os
import signal
import stat
import sys
import tempfile
import traceback
from unittest.mock import patch

ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
HERE=ROOT/"research/reviews/native_MLM_owned_ESRCH_abstract_IPC_source_v1"
OLD=ROOT/"research/reviews/American_C3_native_MLM_amendment_v11"
sys.path[:0]=[str(HERE),str(OLD),str(ROOT/"research/reviews/American_C3_native_MLM_full_fit_v8"),str(ROOT/"research/reviews/American_C3_native_MLM_full_source_v5")]
import owned_lifecycle_esrch_v1 as owned
import abstract_unix_bootstrap_v1 as bootstrap
passed=[]
def check(name,condition):
    if not condition: raise AssertionError(name)
    passed.append(name)
def raises(kind,fn):
    try:fn()
    except kind:return True
    return False

def child_work(pipe,payload):
    try:
        item=pipe.recv();fd=item.detach()
        data=os.read(fd,4096);os.close(fd)
        fd=os.open(payload,os.O_RDONLY)
        duplicate=DupFd(fd);os.close(fd)
        pipe.send({"data":data,"duplicate":duplicate,"abstract":duplicate._id[0],
                   "factory_inherited":connection.arbitrary_address.__module__==bootstrap.__name__})
        assert pipe.recv()=="done"
        resource_sharer.stop(timeout=5)
        pipe.send({"stopped":True})
    finally:
        pipe.close()

def main():
    original=(OLD/"owned_lifecycle_v2.py").read_text()
    actual=(HERE/"owned_lifecycle_esrch_v1.py").read_text()
    expected=original.replace("except FileNotFoundError:","except (FileNotFoundError, ProcessLookupError):")
    check("exact_three_ESRCH_exception_extensions_only",actual==expected and original.count("except FileNotFoundError:")==3)
    for error in (FileNotFoundError,ProcessLookupError):
        with patch.object(Path,"read_text",side_effect=error()):
            check(error.__name__+"_stat_is_absent",owned._stat(987654) is None)
        with patch.object(Path,"iterdir",side_effect=error()):
            check(error.__name__+"_outer_children_is_absent",owned._children(987654)==[])
        with patch.object(Path,"iterdir",return_value=iter([Path("/proc/987654/task/987654")])),patch.object(Path,"read_text",side_effect=error()):
            check(error.__name__+"_inner_task_children_is_absent",owned._children(987654)==[])
    with patch.object(Path,"read_text",side_effect=PermissionError()):
        check("stat_PermissionError_still_raises",raises(PermissionError,lambda:owned._stat(987654)))
    with patch.object(Path,"iterdir",side_effect=PermissionError()):
        check("children_PermissionError_still_raises",raises(PermissionError,lambda:owned._children(987654)))
    with patch.object(Path,"read_text",return_value="malformed"):
        check("malformed_proc_record_still_raises",raises(IndexError,lambda:owned._stat(987654)))
    expected_identity={"pid":999999,"start":1}
    opens=[]
    check("reused_PID_never_opens_pidfd",not owned.signal_identity(expected_identity,0,
          read_stat=lambda pid:{"pid":pid,"start":2},open_fd=lambda *args:opens.append(args)) and not opens)
    reads=iter([expected_identity,{"pid":999999,"start":2}]);closed=[];sent=[]
    check("reuse_after_pidfd_open_never_signals",not owned.signal_identity(expected_identity,0,
          read_stat=lambda pid:next(reads),open_fd=lambda *args:123,
          send_fd=lambda *args:sent.append(args),close_fd=closed.append) and not sent and closed==[123])
    check("pidfd_PermissionError_still_raises",raises(PermissionError,lambda:owned.signal_identity(expected_identity,0,
          read_stat=lambda pid:expected_identity,open_fd=lambda *args:(_ for _ in ()).throw(PermissionError()))))
    leader={"pid":999999,"start":1,"ppid":123,"group":999999,"session":999999,"state":"S"}
    state={999999:leader.copy()}
    tree=owned.OwnedTree(999999,owner=123,read_stat=lambda pid:state.get(pid),read_children=lambda pid:[])
    state[999999]=dict(leader,start=2)
    tree.refresh()
    check("OwnedTree_reused_identity_excluded",tree.reused=={999999} and tree.known[999999]["start"]==1 and tree.identities_absent())

    auth_before=bytes(mp.current_process().authkey)
    Listener_before=connection.Listener;Client_before=connection.Client
    receipt=bootstrap.install_abstract_unix_v1("source-toy")
    check("native_exact_stdlib_source_pin",hashlib.sha256(Path(connection.__file__).read_bytes()).hexdigest()==bootstrap.EXPECTED_CONNECTION_SHA256)
    check("default_fork_unchanged",mp.get_context().get_start_method()=="fork")
    check("AF_INET_passthrough",connection.arbitrary_address("AF_INET")==("localhost",0))
    check("unknown_family_original_error",raises(ValueError,lambda:connection.arbitrary_address("UNKNOWN")))
    addresses=[connection.arbitrary_address("AF_UNIX") for _ in range(32)]
    check("short_unique_abstract_addresses",len(set(addresses))==32 and all(a.startswith(b"\0") and len(a)<=107 for a in addresses))
    check("Listener_Client_authkey_unchanged",connection.Listener is Listener_before and connection.Client is Client_before and bytes(mp.current_process().authkey)==auth_before)
    check("second_install_rejected",raises(ValueError,lambda:bootstrap.install_abstract_unix_v1("again")))
    work=HERE/"toy_runtime";work.mkdir(exist_ok=False)
    temp=work/"tmp";temp.mkdir()
    os.environ.update(TMPDIR=str(temp),TMP=str(temp),TEMP=str(temp))
    tempfile.tempdir=None
    payload=work/"payload.bin";payload.write_bytes(b"arggnn-authenticated-resource-sharer-toy\n")
    parent,child=mp.get_context("fork").Pipe()
    proc=mp.get_context("fork").Process(target=child_work,args=(child,str(payload)))
    proc.start();child.close()
    fd=os.open(payload,os.O_RDONLY);wrapped=DupFd(fd);os.close(fd)
    parent_address=wrapped._id[0]
    parent.send(wrapped)
    check("toy_child_responds",parent.poll(15))
    reply=parent.recv()
    check("authenticated_parent_to_child_FD_transfer",reply["data"]==payload.read_bytes())
    check("fork_bootstrap_inherited",reply["factory_inherited"])
    check("actual_parent_child_abstract_resource_sharers",parent_address.startswith(b"\0") and reply["abstract"].startswith(b"\0"))
    fd=reply["duplicate"].detach()
    check("authenticated_child_to_parent_FD_transfer",os.read(fd,4096)==payload.read_bytes());os.close(fd)
    # Bad auth is rejected by the original Listener/Client, not a replacement.
    expected_auth_stderr=io.StringIO()
    with contextlib.redirect_stderr(expected_auth_stderr):
        check("wrong_authkey_rejected",raises(mp.AuthenticationError,lambda:connection.Client(parent_address,authkey=b"intentionally-wrong-toy-key")))
        text=(ROOT/"src/arggnn/project_budget_v4.py").read_text();module=ast.parse(text)
        nodes=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in ("canonical","measure_scopes")]
        ns={"Path":Path,"os":os,"stat":stat}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),"original_v4_stat_measure_for_own_toy","exec"),ns)
        measurement=ns["measure_scopes"](ROOT,[work])
        check("original_scope_measure_passes_while_both_IPC_listeners_active",measurement["conservative_scope_bytes"]>0)
        check("no_filesystem_socket_nodes_during_real_IPC",not any(stat.S_ISSOCK(p.lstat().st_mode) for p in work.rglob("*")))
        parent.send("done")
        check("toy_child_resource_sharer_stops",parent.poll(15) and parent.recv()=={"stopped":True})
        proc.join(15)
        check("single_real_fork_child_wait0",proc.exitcode==0)
        resource_sharer.stop(timeout=5)
    parent.close();proc.close()
    names=["@"+a[1:].decode("ascii") for a in [parent_address,reply["abstract"]]]
    matching=[line for line in Path("/proc/net/unix").read_text().splitlines() if any(line.endswith(" "+n) for n in names)]
    check("only_toy_abstract_kernel_endpoints_gone_after_stop",matching==[])
    check("original_scope_measure_still_passes_after_clean_stop",ns["measure_scopes"](ROOT,[work])["entries"]>=3)
    check("source_v11_unchanged",hashlib.sha256((OLD/"owned_lifecycle_v2.py").read_bytes()).hexdigest()==hashlib.sha256(original.encode()).hexdigest())
    report={"schema":"owned_ESRCH_abstract_IPC_source_qualification_v1","status":"PASS",
      "checks_passed":len(passed),"checks":passed,"bootstrap_receipt":receipt,
      "native_interpreter":sys.executable,"python_version":sys.version,
      "real_toy":{"fork_children":1,"actual_wait":0,"bidirectional_authenticated_FD_transfer":True,
                  "wrong_authkey_rejected":True,"filesystem_sockets_created":0,
                  "kernel_endpoints_absent_after_close":True,"original_measure_scopes_unchanged":True,
                  "scope":str(work),"expected_wrong_auth_stderr":expected_auth_stderr.getvalue()[-2000:]},
      "scientific_payload_or_CUDA_access":False,"native_training_executed":False,
      "live_or_frozen_source_modified":False,
      "source_refs":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
          [HERE/"owned_lifecycle_esrch_v1.py",HERE/"abstract_unix_bootstrap_v1.py",HERE/"source_checks.py"]}}
    (HERE/"source_checks_result.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"PASS","checks":len(passed),"real_fork_children":1,"wait":0,"filesystem_sockets":0}))

if __name__=="__main__":main()
