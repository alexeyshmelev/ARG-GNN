"""Metadata-only normalization of one exact stopped Composer latest alias.

Pure stdlib, no import-time filesystem actions. The reviewed v12 no-follow
ancestry/stamp checks are retained. Full checkpoint copying is deliberately absent.
Only an explicitly bound symlink can be unlinked, after durable metadata, genuine
writer drain and unchanged metadata. Every regular file is retained.
"""
from pathlib import Path
import hashlib,json,os,stat

MAX_RECEIPT_BYTES=65536

def require(value,message):
    if not value:raise ValueError(message)

def stamp(s):
    return [s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns]

def own_ancestors(path,root):
    path=Path(path);root=Path(root)
    require(path.is_relative_to(root) and path!=root,"Within exact own namespace")
    p=path.parent
    while True:
        require(stat.S_ISDIR(p.lstat().st_mode),"No-follow own directory ancestry")
        if p==root:break
        p=p.parent

def identity(s):
    return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,link_count=s.st_nlink,
        mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)

def canonical_directory(root,directory):
    root=Path(root);directory=Path(directory)
    require(root.is_absolute() and root.resolve()==root and directory.is_absolute()
        and directory!=root and root in directory.parents and ".." not in directory.parts
        and directory.resolve()==directory,"Exact canonical contained checkpoint directory")
    own_ancestors(directory,root)
    require(stat.S_ISDIR(directory.lstat().st_mode),"Real checkpoint directory")
    return directory

def regular_snapshot(directory):
    rows=[]
    for p in sorted(Path(directory).iterdir()):
        s=p.lstat()
        if stat.S_ISREG(s.st_mode):
            require(s.st_nlink==1,"Every regular checkpoint has one link")
            rows.append(dict(name=p.name,lstat=stamp(s),allocated_bytes=s.st_blocks*512))
        else:
            require(p.name=="latest-rank0.pt" and stat.S_ISLNK(s.st_mode)
                and s.st_nlink==1,"Only one exact Composer alias in checkpoint directory")
    return rows

def inspect_latest_v7(root,directory,*,max_bytes,max_regular_files,expected_target_ref=None,
        expected_target_identity=None):
    directory=canonical_directory(root,directory);alias=directory/"latest-rank0.pt"
    s=alias.lstat()
    require(stat.S_ISLNK(s.st_mode) and s.st_nlink==1 and 0<s.st_size<=256,
        "Only Composer own latest symlink may be normalized")
    target_text=os.readlink(alias);rel=Path(target_text)
    require(not rel.is_absolute() and rel.parent==Path(".")
        and rel.name not in ("",".","..") and rel.suffix==".pt"
        and rel.name!="latest-rank0.pt","Relative single-component own checkpoint target")
    target=directory/rel;own_ancestors(target,directory);ts=target.lstat()
    require(stat.S_ISREG(ts.st_mode) and ts.st_nlink==1 and 0<ts.st_size<=max_bytes,
        "Bounded own regular checkpoint target")
    regular=regular_snapshot(directory)
    require(0<len(regular)<=max_regular_files and any(r["name"]==rel.name for r in regular),
        "Exact bounded retained regular checkpoint roster")
    relative=str(target.relative_to(root))
    if expected_target_ref is not None:
        require(set(expected_target_ref)=={"path","sha256"} and relative==expected_target_ref["path"]
            and len(expected_target_ref["sha256"])==64,
            "Fixed original SHA-qualified checkpoint target only")
    if expected_target_identity is not None:
        require(identity(ts)==expected_target_identity,"Exact ROOT-qualified original target identity")
    return dict(schema="native_VAL000_exact_Composer_alias_before_unlink_metadata_v7",
        alias=str(alias.relative_to(root)),lstat=stamp(s),alias_logical_bytes=s.st_size,
        target_text=target_text,target=str(target.relative_to(root)),target_lstat=stamp(ts),
        target_identity=identity(ts),target_allocated_bytes=ts.st_blocks*512,
        target_ref=expected_target_ref,
        target_SHA_provenance="authenticated_ROOT_full_state_verifier" if expected_target_ref else
            "UNKNOWN_not_read_for_metadata_only_alias_operation",
        regular_checkpoint_metadata_before=regular,
        regular_checkpoint_payload_read_or_copied=False,regular_checkpoint_delete_authorized=False,
        original_guard_canonical_or_measure_scopes_modified=False)

def verify_receipt_v7(root,record,value):
    require(isinstance(record,dict) and set(record)=={"path","sha256"},
        "Durable metadata receipt needs exact real path and digest")
    path=Path(root)/record["path"]
    require(path!=Path(root) and Path(root) in path.parents and path.resolve()==path
        and ".." not in path.parts,"Canonical contained durable receipt")
    s=path.lstat()
    require(stat.S_ISREG(s.st_mode) and s.st_nlink==1 and 0<s.st_size<=MAX_RECEIPT_BYTES,
        "Bounded real durable receipt")
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        require(stamp(os.fstat(fd))==stamp(s),"Exact receipt descriptor identity")
        data=os.read(fd,MAX_RECEIPT_BYTES+1)
        require(len(data)==s.st_size and stamp(os.fstat(fd))==stamp(s),
            "Stable complete receipt bytes")
    finally:os.close(fd)
    require(hashlib.sha256(data).hexdigest()==record["sha256"] and json.loads(data)==value,
        "Exact durable metadata bytes before alias unlink")
    return record

def normalize_latest_v7(root,directory,*,prove_drained,preserve,revalidate_pre_unlink,
        strict_boundary,max_bytes,max_regular_files,expected_target_ref=None,
        expected_target_identity=None,context=None):
    require(prove_drained(),"Actual complete checkpoint writer drain required")
    directory=canonical_directory(root,directory)
    metadata=inspect_latest_v7(root,directory,max_bytes=max_bytes,max_regular_files=max_regular_files,
        expected_target_ref=expected_target_ref,expected_target_identity=expected_target_identity)
    value=dict(metadata,context=context,actual_complete_owned_writer_drain_verified=True,
        schema="native_VAL000_durable_ROOT_Composer_alias_before_unlink_v7")
    require(len((json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode())<=MAX_RECEIPT_BYTES,
        "Bounded exact alias metadata before any write")
    record=preserve(value)
    verify_receipt_v7(root,record,value)
    revalidate_pre_unlink()
    require(prove_drained(),"Genuine drain immediately before exact alias removal")
    fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        ds=os.fstat(fd)
        require((ds.st_dev,ds.st_ino)==(directory.lstat().st_dev,directory.lstat().st_ino),
            "Unchanged exact checkpoint directory descriptor")
        before=inspect_latest_v7(root,directory,max_bytes=max_bytes,max_regular_files=max_regular_files,
            expected_target_ref=expected_target_ref,expected_target_identity=expected_target_identity)
        require(before==metadata,"Exact alias, target and every retained regular unchanged before unlink")
        verify_receipt_v7(root,record,value)
        require(prove_drained(),"Complete owned drain remains true at unlink")
        os.unlink("latest-rank0.pt",dir_fd=fd)
        os.fsync(fd)
        try:os.stat("latest-rank0.pt",dir_fd=fd,follow_symlinks=False)
        except FileNotFoundError:pass
        else:raise ValueError("Exact Composer alias remains after unlink")
        require((ds.st_dev,ds.st_ino)==(directory.lstat().st_dev,directory.lstat().st_ino)
            and regular_snapshot(directory)==metadata["regular_checkpoint_metadata_before"],
            "Every regular checkpoint identity/metadata unchanged after alias removal")
    finally:os.close(fd)
    # No canonical bypass. The actual original strict boundary is mandatory now.
    boundary=strict_boundary()
    return dict(schema="native_VAL000_exact_Composer_alias_removed_v7",before_unlink_metadata_ref=record,
        alias=metadata["alias"],alias_absent=True,all_regular_checkpoint_metadata_unchanged=True,
        regular_checkpoint_payload_read_copied_or_deleted=False,
        strict_original_boundary_after_unlink=boundary,storage_close_or_credit_authorized=False)
