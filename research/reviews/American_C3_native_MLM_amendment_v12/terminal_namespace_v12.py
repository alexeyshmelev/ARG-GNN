"""Post-drain narrow owned namespace normalization; no import-time actions."""
from pathlib import Path
import hashlib,json,os,stat
def require(value,message):
    if not value:raise ValueError(message)
def stamp(s):return [s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def own_ancestors(path,root):
    path=Path(path);root=Path(root)
    require(path.is_relative_to(root) and path!=root,"Within exact own namespace")
    p=path.parent
    while True:
        require(stat.S_ISDIR(p.lstat().st_mode),"No-follow own directory ancestry")
        if p==root:break
        p=p.parent
def socket_candidates(tmp):
    tmp=Path(tmp);rows=[];stack=[tmp]
    if not tmp.exists():return []
    while stack:
        p=stack.pop();s=p.lstat()
        if stat.S_ISDIR(s.st_mode):stack.extend(p.iterdir());continue
        if stat.S_ISREG(s.st_mode):
            require(s.st_nlink==1,"Do not normalize hardlinks");continue
        require(stat.S_ISSOCK(s.st_mode) and p.parent.name.startswith("pymp-")
            and p.name.startswith("listener-") and s.st_nlink==1,"Only expected own multiprocessing listener sockets")
        own_ancestors(p,tmp);rows.append(dict(path=str(p),lstat=stamp(s)))
    require(len(rows)<=256,"Bounded complete own IPC node list")
    return sorted(rows,key=lambda r:r["path"])
def kernel_listeners(paths,text):
    # Do not return/store unrelated kernel process/socket details.
    wanted=set(paths);return sorted({row.split(maxsplit=7)[7] for row in text.splitlines()[1:]
        if len(row.split(maxsplit=7))==8 and row.split(maxsplit=7)[7] in wanted})
def clean_dead_sockets(tmp,*,prove_drained,preserve):
    require(prove_drained(),"Actual complete owned writer drain required")
    rows=socket_candidates(tmp)
    paths=[r["path"] for r in rows]
    require(not kernel_listeners(paths,Path("/proc/net/unix").read_text()),"No live kernel endpoint on exact owned socket paths")
    preserve(dict(schema="native_MLM_dead_IPC_preserved_metadata_v9",nodes=rows,
        exact_kernel_listener_matches=[],all_owned_writers_actually_drained=True))
    # Root authority narrowly permits dead socket unlink after durable metadata.
    for row in rows:
        p=Path(row["path"]);own_ancestors(p,Path(tmp))
        require(prove_drained() and stamp(p.lstat())==row["lstat"],"Dead own socket identity/drain unchanged")
        require(not kernel_listeners([str(p)],Path("/proc/net/unix").read_text()),"No live exact socket immediately before cleanup")
        p.unlink()
    return rows
def normalize_latest(directory,*,prove_drained,preserve,max_bytes=2950918528):
    require(prove_drained(),"Actual checkpoint writer drain required")
    directory=Path(directory);alias=directory/"latest-rank0.pt"
    try:s=alias.lstat()
    except FileNotFoundError:return None
    require(stat.S_ISLNK(s.st_mode) and s.st_nlink==1,"Only Composer own latest symlink may be normalized")
    target_text=os.readlink(alias);rel=Path(target_text)
    require(not rel.is_absolute() and rel.parent==Path(".") and rel.name not in ("",".",".."),"Relative single-component own checkpoint target")
    target=directory/rel;own_ancestors(target,directory);ts=target.lstat()
    require(stat.S_ISREG(ts.st_mode) and ts.st_nlink==1 and 0<ts.st_size<=max_bytes,"Bounded own regular checkpoint target")
    regular=[p for p in directory.iterdir() if stat.S_ISREG(p.lstat().st_mode) and p.suffix==".pt"]
    require(regular==[target] or len(regular)==1 and regular[0]==target,"One retained checkpoint before independent latest copy")
    metadata=dict(schema="native_MLM_Composer_latest_alias_metadata_v9",alias=str(alias),lstat=stamp(s),
        target_text=target_text,target=str(target),target_lstat=stamp(ts))
    preserve(metadata)  # Metadata durable before touching the alias.
    temp=directory/".latest-rank0.pt.regularizing-v9"
    src=os.open(target,os.O_RDONLY|os.O_NOFOLLOW)
    dst=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    total=0;h=hashlib.sha256()
    try:
        require(stamp(os.fstat(src))==stamp(ts),"Opened exact own target")
        while True:
            block=os.read(src,8388608)
            if not block:break
            total+=len(block);require(total<=max_bytes,"Independent alias copy bounded")
            h.update(block)
            view=memoryview(block)
            while view:
                n=os.write(dst,view);require(n>0,"Complete own alias copy");view=view[n:]
        os.fsync(dst);ds=os.fstat(dst)
        require(total==ts.st_size and stat.S_ISREG(ds.st_mode) and ds.st_nlink==1 and ds.st_size==total,"Independent regular alias byte count")
        require(prove_drained() and stamp(target.lstat())==stamp(ts) and stamp(os.fstat(src))==stamp(ts)
            and stamp(alias.lstat())==stamp(s) and os.readlink(alias)==target_text,"Checkpoint/alias identities unchanged after actual drain")
    finally:os.close(src);os.close(dst)
    # Verify copied bytes independently; no scientific decoding.
    copied=hashlib.sha256();copyfd=os.open(temp,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        require(stamp(os.fstat(copyfd))==stamp(ds) and ds.st_ino!=ts.st_ino,"Exact independent copied inode")
        while True:
            block=os.read(copyfd,8388608)
            if not block:break
            copied.update(block)
        require(stamp(os.fstat(copyfd))==stamp(ds),"Copy identity stable during independent verification")
    finally:os.close(copyfd)
    require(copied.hexdigest()==h.hexdigest() and stamp(temp.lstat())==stamp(ds)
        and stamp(target.lstat())==stamp(ts) and stamp(alias.lstat())==stamp(s)
        and os.readlink(alias)==target_text and prove_drained(),"Independent latest bytes/identities/drain unchanged")
    os.replace(temp,alias)
    dfd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(dfd)
    finally:os.close(dfd)
    result=dict(metadata,normalized_regular_bytes=total,sha256=h.hexdigest(),
        author_checkpoint_interval_or_keep_modified=False)
    preserve(result);return result
