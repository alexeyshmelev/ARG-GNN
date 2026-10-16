"""Read-only retained-node metadata census; never opens file payloads or follows links."""
from pathlib import Path
import hashlib,json,os,stat

MAX_ENTRIES=1000000
MAX_DEPTH=64
PER_NODE_METADATA_CHARGE=4096

def require(ok,message):
    if not ok:raise ValueError(message)

def fields_v1(s):
    return (s.st_dev,s.st_ino,s.st_mode,s.st_uid,s.st_gid,s.st_nlink,
        s.st_size,s.st_blocks,s.st_mtime_ns,s.st_ctime_ns)

def _anchor(s):
    return (s.st_dev,s.st_ino,s.st_mode,s.st_uid,s.st_gid)

def _relative(root,path):
    p=Path(path)
    require(p.is_absolute() and p!=root and root in p.parents and ".." not in p.parts
        and p.as_posix()==str(p),"Exact lexical absolute contained path")
    return p.relative_to(root).parts

def _open_directory(rootfd,parts,device):
    fd=os.dup(rootfd)
    try:
        for name in parts:
            before=os.stat(name,dir_fd=fd,follow_symlinks=False)
            require(stat.S_ISDIR(before.st_mode) and before.st_dev==device,"No linked/non-directory/cross-device ancestor")
            nextfd=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            try:require(_anchor(os.fstat(nextfd))==_anchor(before),"Ancestor changed during no-follow open")
            except BaseException:os.close(nextfd);raise
            os.close(fd);fd=nextfd
        return fd
    except BaseException:os.close(fd);raise

def _leaf_metadata(rootfd,parts,device):
    parent=_open_directory(rootfd,parts[:-1],device)
    try:
        before=os.stat(parts[-1],dir_fd=parent,follow_symlinks=False);link=None
        if stat.S_ISLNK(before.st_mode):
            target=os.fsencode(os.readlink(parts[-1],dir_fd=parent))
            require(len(target)==before.st_size and len(target)<=65536,"Bounded immutable stored link text")
            link=hashlib.sha256(target).hexdigest()
        require(fields_v1(os.stat(parts[-1],dir_fd=parent,follow_symlinks=False))==fields_v1(before),"Leaf changed during metadata read")
        return before,link
    finally:os.close(parent)

def lstat_no_follow_v1(root,path):
    root=Path(root);parts=_relative(root,path)
    rootfd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:return _leaf_metadata(rootfd,parts,os.fstat(rootfd).st_dev)[0]
    finally:os.close(rootfd)

def measure_retained_nodes_v1(root,paths,*,expected_paths,socket_is_absent):
    """Cooperative stopped-retention boundary, not an atomic snapshot or OS quota.

    Counts every directory, singly linked regular file, stored symlink and inactive
    socket. The link target is never opened or resolved. Other node types fail.
    """
    root=Path(root);paths=[Path(p) for p in paths]
    require([str(p) for p in paths]==list(expected_paths) and len(paths)==2,"Both exact charged parents, in original order")
    require(callable(socket_is_absent),"Actual targeted kernel endpoint check required for retained socket nodes")
    parts=[_relative(root,p) for p in paths]
    require(not any(a==b or a in b.parents or b in a.parents for i,a in enumerate(paths) for b in paths[i+1:]),"Disjoint charged roots")
    rootfd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    anchor=os.fstat(rootfd);device=anchor.st_dev

    def snapshot():
        rows=[];seen=set();per_scope=[]
        def add(relative,s,link=None):
            require(s.st_dev==device and type(s.st_size) is int and type(s.st_blocks) is int
                and s.st_size>=0 and s.st_blocks>=0,"Safe bounded same-device node metadata")
            identity=(s.st_dev,s.st_ino)
            require(identity not in seen,"Aliased node/hardlink/cycle in charged parents")
            seen.add(identity)
            kind=("directory" if stat.S_ISDIR(s.st_mode) else "regular" if stat.S_ISREG(s.st_mode)
                else "symlink" if stat.S_ISLNK(s.st_mode) else "socket" if stat.S_ISSOCK(s.st_mode) else None)
            require(kind is not None and (kind=="directory" or s.st_nlink==1),"Unsafe node or multiply linked non-directory")
            if kind=="socket":require(socket_is_absent(str(root/relative)) is True,"Retained filesystem socket still has a kernel endpoint")
            charge=max(s.st_size,s.st_blocks*512)+PER_NODE_METADATA_CHARGE
            rows.append((relative,fields_v1(s),kind,link,charge))
            require(len(rows)<=MAX_ENTRIES,"Bounded retained-node inventory")
        def scan(fd,relative,depth):
            require(depth<=MAX_DEPTH,"Bounded charged directory depth")
            before=os.fstat(fd);require(stat.S_ISDIR(before.st_mode),"Scope directory changed type")
            add(relative,before)
            with os.scandir(fd) as entries:
                names=sorted(e.name for e in entries)
            require(len(names)<=MAX_ENTRIES,"Bounded directory entries")
            for name in names:
                require(name not in (".","..") and "/" not in name and all(ord(c)>=32 and ord(c)!=127 for c in name),"Canonical unambiguous directory entry")
                s=os.stat(name,dir_fd=fd,follow_symlinks=False);child=relative+"/"+name
                if stat.S_ISDIR(s.st_mode):
                    childfd=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
                    try:
                        require(fields_v1(os.fstat(childfd))==fields_v1(s),"Directory changed before no-follow descent")
                        scan(childfd,child,depth+1)
                    finally:os.close(childfd)
                else:
                    link=None
                    if stat.S_ISLNK(s.st_mode):
                        text=os.fsencode(os.readlink(name,dir_fd=fd))
                        require(len(text)==s.st_size and len(text)<=65536,"Bounded genuine stored link metadata")
                        link=hashlib.sha256(text).hexdigest()
                    add(child,s,link)
                require(fields_v1(os.stat(name,dir_fd=fd,follow_symlinks=False))==fields_v1(s),"Child changed during scan")
            require(fields_v1(os.fstat(fd))==fields_v1(before),"Directory membership changed during scan")
        for path,relative_parts in zip(paths,parts):
            fd=_open_directory(rootfd,relative_parts,device);start=len(rows)
            try:scan(fd,"/".join(relative_parts),0)
            finally:os.close(fd)
            own=rows[start:];counts={kind:sum(r[2]==kind for r in own) for kind in ("directory","regular","symlink","socket")}
            per_scope.append(dict(path=str(path),entries=len(own),conservative_scope_bytes=sum(r[4] for r in own),node_types=counts))
        return sorted(rows),per_scope

    try:
        first,scopes=snapshot();second,again=snapshot()
        require(first==second and scopes==again,"Retained metadata or membership changed between complete passes")
        for relative,old,kind,link,charge in second:
            s,current_link=_leaf_metadata(rootfd,tuple(relative.split("/")),device)
            require(fields_v1(s)==old and current_link==link,"Final descriptor-relative retained-node revalidation failed")
            if kind=="socket":require(socket_is_absent(str(root/relative)) is True,"Retained kernel endpoint reappeared")
        require(_anchor(os.fstat(rootfd))==_anchor(anchor) and _anchor(root.lstat())==_anchor(anchor),"Project anchor changed")
        digest=hashlib.sha256(json.dumps(second,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
        return dict(schema="two_key_exact_stopped_retained_node_measurement_v1",
            conservative_scope_bytes=sum(r[4] for r in second),entries=len(second),scope_paths=[str(p) for p in paths],
            stat_only=True,per_scope=scopes,inventory_sha256=digest,two_complete_passes_and_final_no_follow_revalidation=True,
            per_node_additional_metadata_charge_bytes=PER_NODE_METADATA_CHARGE,
            regular_payloads_opened=False,symlink_targets_followed=False,unsafe_nodes_rejected=True,
            node_metadata_preserved=True,namespace_normalization_performed=False,
            cooperative_snapshot_not_atomic_quota=True)
    finally:os.close(rootfd)
