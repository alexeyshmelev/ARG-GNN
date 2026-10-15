"""ROOT-only failed-v8 preservation helper; inert until explicit authorized call."""
from pathlib import Path
import json,os,stat
from terminal_namespace_v9 import stamp,socket_candidates,kernel_listeners,clean_dead_sockets
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
OLD_OUTPUT=ROOT/"experiments/results/American_C3_native_MLM_full30mb_mlm_fit_v8"
OLD_TMP=ROOT/".runtime/tmp/c3full8/f"
def require(value,message):
    if not value:raise ValueError(message)
def snapshot(root):
    rows=[];stack=[Path(root)]
    while stack:
        p=stack.pop();s=p.lstat()
        if stat.S_ISDIR(s.st_mode):kind="directory";stack.extend(p.iterdir())
        elif stat.S_ISREG(s.st_mode) and s.st_nlink==1:kind="regular"
        elif stat.S_ISSOCK(s.st_mode) and s.st_nlink==1:kind="socket"
        else:raise ValueError("Unexpected failed-scope type/hardlink; no archival mutation")
        rows.append(dict(path=str(p),relative=str(p.relative_to(root)),kind=kind,lstat=stamp(s)))
    return sorted(rows,key=lambda r:r["relative"])
def admin_write(path,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    require(len(raw)<=1048576,"Root administrative metadata bounded")
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        view=memoryview(raw)
        while view:
            n=os.write(fd,view);require(n>0,"Complete archive administrative receipt");view=view[n:]
        os.fsync(fd)
    finally:os.close(fd)
def preserve_failed_v8(archive,*,authority,prove_old_writers_absent):
    """Called only by ROOT after explicit archive/IPC authority and true absence proof."""
    archive=Path(archive)
    require(authority["root_authorizer"]=="root_agent" and authority["active"] is True
        and authority["archive_failed_v8_by_same_filesystem_rename"] is True
        and authority["preserve_all_scientific_and_regular_files"] is True
        and authority["narrow_dead_socket_cleanup_after_metadata"] is True
        and authority["archive_allowance_bytes"]==5242880
        and authority["old_output"]==str(OLD_OUTPUT) and authority["old_tmp"]==str(OLD_TMP)
        and authority["archive_root"]==str(archive),"Exact separately authorized root archive <=5MiB")
    require(archive.is_relative_to(ROOT) and archive.resolve()==archive and not archive.exists()
        and not archive.is_relative_to(OLD_OUTPUT) and not archive.is_relative_to(OLD_TMP),"Fresh own archive outside original scopes")
    require(prove_old_writers_absent(),"Root's complete actual old-writer absence before any mutation")
    sources=[OLD_OUTPUT,OLD_TMP];before=[snapshot(root) for root in sources]
    sockets=socket_candidates(OLD_TMP)
    require(not kernel_listeners([r["path"] for r in sockets],Path("/proc/net/unix").read_text()),"No original own listener remains live")
    estimate=sum(max(4096,r["lstat"][4]) for rows in before for r in rows)
    require(estimate+524288<=5242880,"Retained payload/namespace plus bounded admin headroom <=5MiB")
    archive.mkdir()
    require(all(root.lstat().st_dev==archive.lstat().st_dev for root in sources),"Same filesystem archive only")
    admin_write(archive/"before_rename.json",dict(scopes=before,dead_socket_metadata=sockets,
        original_kernel_listener_matches=[],all_old_writers_actually_absent=True,
        native_wait_exit="UNKNOWN",actual_optimizer_updates="UNKNOWN"))
    moves=[]
    for root,name,rows in zip(sources,["original_output","original_tmp"],before):
        require(prove_old_writers_absent() and snapshot(root)==rows,"Actual writers absent and complete old inode tree unchanged")
        dest=archive/name
        os.rename(root,dest)
        after=snapshot(dest)
        require([(r["relative"],r["kind"],r["lstat"][:5]) for r in after]==
            [(r["relative"],r["kind"],r["lstat"][:5]) for r in rows],"All old inodes/types/sizes preserved by rename")
        moves.append(dict(original=str(root),archived=str(dest),files=after))
    count=[0]
    def preserve(value):
        count[0]+=1;admin_write(archive/("dead_ipc_metadata_%02d.json"%count[0]),value)
    clean_dead_sockets(archive/"original_tmp",prove_drained=prove_old_writers_absent,preserve=preserve)
    measured=sum(max(4096,r["lstat"][4]) for r in snapshot(archive))
    require(measured+65536<=5242880,"Actual regular archive/admin bytes <=5MiB")
    receipt=dict(schema="native_MLM_failed_v8_root_archive_receipt_v9",root_authorizer="root_agent",active=True,
        old_writer_absence_ref=authority["old_writer_absence_ref"],archive_root=str(archive),
        all_scientific_and_regular_files_preserved=True,all_socket_metadata_preserved_before_narrow_cleanup=True,
        archive_regular_namespace_confirmed=True,same_filesystem_rename=True,
        retained_archive_and_metadata_bytes=measured+65536,moves=moves)
    require(all(r["kind"] in ("directory","regular") for r in snapshot(archive)),"Archived namespace regular after preserved dead IPC cleanup")
    admin_write(archive/"archive_receipt.json",receipt)
    return receipt
