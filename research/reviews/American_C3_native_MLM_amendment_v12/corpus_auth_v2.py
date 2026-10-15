
"""One quiescent streaming authentication of exact full-MLM inputs, without decoding/copying."""
from pathlib import Path
import hashlib,os,stat
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
def _stamp(s):
    return (s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
def verify_stream(record,reader,before,after,bytecap):
    if not stat.S_ISREG(before.st_mode) or before.st_nlink!=1 or not 0<=before.st_size<=bytecap:
        raise ValueError("Bounded single-link regular corpus file required")
    h=hashlib.sha256();total=0
    while True:
        block=reader(8*1024*1024)
        if not block:break
        total+=len(block)
        if total>bytecap:raise ValueError("Corpus stream exceeded source serialization bound")
        h.update(block)
    fd_after,path_after=after()
    if _stamp(before)!=_stamp(fd_after) or _stamp(before)!=_stamp(path_after) or total!=before.st_size:
        raise ValueError("Corpus identity/size/mtime/ctime changed during streaming read")
    if h.hexdigest()!=record["sha256"]:raise ValueError("Actual corpus SHA differs from completed immutable v3 ref")
    return dict(ref=dict(record),bytes=total,identity=list(_stamp(before)),sha256_verified=True,stable_read_verified=True)
def authenticate_file(record,expected,bytecap):
    path=ROOT/record["path"]
    if path!=expected or path.resolve()!=path or not path.is_relative_to(ROOT) or ".." in path.parts:
        raise ValueError("Exact canonical bound corpus input path")
    before=path.lstat()
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        opened=os.fstat(fd)
        if _stamp(before)!=_stamp(opened):raise ValueError("Corpus pathname changed before open")
        return verify_stream(record,lambda n:os.read(fd,n),opened,lambda:(os.fstat(fd),path.lstat()),bytecap)
    finally:os.close(fd)
def authenticate_inputs(prep,prepared_dir):
    maximum=prep["native_serialization_bound_bytes"]
    if type(maximum) is not int or maximum<=0:raise ValueError("Actual completed corpus serialization bound required")
    refs={"mlm/train":prep["native_dataset_refs"]["mlm/train"],"mlm/val":prep["native_dataset_refs"]["mlm/val"],"vocab":prep["vocab_ref"]}
    entries={}
    for name,record in refs.items():
        expected=prepared_dir/"vocab.pkl" if name=="vocab" else prepared_dir/name/("raw_"+name.split("/")[1]+"_sequences_and_vocab.pkl")
        entries[name]=authenticate_file(record,expected,maximum)
    result=dict(schema="american_c3_native_MLM_corpus_authentication_v2",status="authenticated_full_MLM_inputs",
        inputs=entries,all_consumed_MLM_raw_files_authenticated=True,standalone_vocab_authenticated=True,
        row_order_sidecars_authenticated=False,row_order_sidecars_used=False,
        other_retained_view_payloads_authenticated=False,stream_chunk_bytes=8*1024*1024,
        corpus_copied=False,decoded=False,verification_count_per_input=1)
    confirm_unchanged(result,prepared_dir)
    return result
def confirm_unchanged(result,prepared_dir):
    for name,row in result["inputs"].items():
        path=ROOT/row["ref"]["path"]
        if path.resolve()!=path or list(_stamp(path.lstat()))!=row["identity"]:
            raise ValueError("Authenticated corpus pathname changed before native loader")
    for split in ("train","val"):
        if (prepared_dir/"mlm"/split/("tokenized_"+split+"_sequences_and_vocab.pkl")).exists():
            raise ValueError("A shadow tokenized input appeared before native loading")
