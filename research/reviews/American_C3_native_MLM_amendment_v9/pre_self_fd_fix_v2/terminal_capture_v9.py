"""Precharged fixed terminal evidence slots; no import-time I/O or actions."""
from pathlib import Path
import hashlib,json,os,stat
HEADER=128
WAIT_BYTES=1048576
ROLES=("native","launcher","outer")
TAIL_CAPS={"native":4194304,"outer":16777216}
def require(value,message):
    if not value:raise ValueError(message)
def _identity(s):return [s.st_dev,s.st_ino,s.st_size]
def _regular(s):return stat.S_ISREG(s.st_mode) and s.st_nlink==1
def _pwrite_all(fd,data,offset):
    while data:
        n=os.pwrite(fd,data,offset)
        require(n>0,"Complete bounded administrative pwrite")
        data=data[n:];offset+=n
def prepare(directory):
    """ROOT calls only after original guard precharge; creates23MiB regular files."""
    directory=Path(directory);directory.mkdir()
    rows={}
    sizes={**{role+".wait":WAIT_BYTES for role in ROLES},**{role+".tail":cap for role,cap in TAIL_CAPS.items()}}
    for name,size in sizes.items():
        path=directory/name;fd=os.open(path,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        try:
            zero=bytes(min(size,1048576))
            for offset in range(0,size,len(zero)):_pwrite_all(fd,zero[:min(len(zero),size-offset)],offset)
            os.fsync(fd);s=os.fstat(fd);require(_regular(s) and s.st_size==size,"Precharged capture identity")
            rows[name]=_identity(s)
        finally:os.close(fd)
    return dict(schema="native_MLM_precharged_terminal_capture_v9",directory=str(directory),files=rows,
        total_bytes=sum(sizes.values()),no_file_growth_after_launch=True)
class Capture:
    def __init__(self,role,manifest):
        require(role in ROLES and manifest["schema"]=="native_MLM_precharged_terminal_capture_v9","Known capture role")
        self.role=role;self.manifest=manifest;self.fds={};self.committed=set()
        for suffix in (["wait","tail"] if role in TAIL_CAPS else ["wait"]):
            name=role+"."+suffix;path=Path(manifest["directory"])/name
            fd=os.open(path,os.O_RDWR|os.O_NOFOLLOW)
            s=os.fstat(fd);named=path.lstat()
            require(_regular(s) and _regular(named) and _identity(s)==manifest["files"][name]
                and _identity(named)==_identity(s),"Exact precharged capture regular inode/length")
            self.fds[suffix]=fd
    def event(self,index,value):
        require(index in (0,1) and index not in self.committed,"Single-use actual event slot")
        raw=json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
        size=WAIT_BYTES//2;require(len(raw)<=size-HEADER,"Bounded complete genuine wait/drain record")
        fd=self.fds["wait"];offset=index*size
        require(_regular(os.fstat(fd)) and _identity(os.fstat(fd))==self.manifest["files"][self.role+".wait"],"Capture inode unchanged")
        # Payload first, fsync, then checksummed commit header. No torn payload is accepted.
        _pwrite_all(fd,raw,offset+HEADER);os.fsync(fd)
        header=("C3WAIT9 %d %s\n"%(len(raw),hashlib.sha256(raw).hexdigest())).encode()
        _pwrite_all(fd,header.ljust(HEADER,b" "),offset);os.fsync(fd);self.committed.add(index)
    def tail(self,logs):
        if "tail" not in self.fds:return None
        raw=b"".join(logs.parts);require(len(raw)<=TAIL_CAPS[self.role],"Actual bounded raw stdout tail")
        fd=self.fds["tail"];s=os.fstat(fd)
        require(_regular(s) and _identity(s)==self.manifest["files"][self.role+".tail"],"Tail capture unchanged")
        _pwrite_all(fd,raw,0);os.fsync(fd)
        return dict(retained_raw_bytes=len(raw),total_pipe_bytes=logs.total,discarded_earlier_bytes=logs.total-len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),raw_bytes_preserved_without_UTF8_or_JSON_expansion=True)
    def close(self):
        for fd in self.fds.values():os.close(fd)
        self.fds.clear()
def decode_slot(raw,index):
    size=WAIT_BYTES//2;header=raw[index*size:index*size+HEADER].decode()
    parts=header.split()
    require(len(parts)==3 and parts[0]=="C3WAIT9","Committed terminal evidence header")
    length=int(parts[1]);require(0<length<=size-HEADER,"Bounded payload length")
    payload=raw[index*size+HEADER:index*size+HEADER+length]
    require(hashlib.sha256(payload).hexdigest()==parts[2],"Terminal evidence payload checksum")
    return json.loads(payload)


class WaitRelay:
    """Forward only bounded authentic own-pipe terminal notices to ROOT stdout."""
    def __init__(self,emit=None):
        self.pending=b"";self.count=0;self.emit=emit or (lambda line:print(line,flush=True))
    def feed(self,data):
        self.pending+=data
        while b"\n" in self.pending:
            line,self.pending=self.pending.split(b"\n",1)
            if self.count<8 and len(line)<=8192 and any(line.startswith(prefix) for prefix in (
                b'{"event":"actual_direct_native_wait_v9",',
                b'{"event":"actual_direct_entry_wait_v9",')):
                self.emit(line.decode("utf-8",errors="replace"));self.count+=1
        if len(self.pending)>8192:self.pending=self.pending[-8192:]
