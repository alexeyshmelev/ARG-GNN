"""Source bindings, single-writer bounded outputs, and owned-family accounting."""
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from .dataset import ROOT, require
BASE=ROOT/'research/joint_evaluation_20261029/european_four'
RESULTS=BASE/'outputs'
MODEL_ID='treeloom_joint_european_direct_four_no_baa_v1'
CAP=128*2**20
RAM_CAP=64*2**30
GPUS={3:'GPU-df11de7f-7a15-ebe2-085d-9c7d907f3786',4:'GPU-6cf64ff3-d8c2-9201-b2b8-683029625d71',5:'GPU-3db85505-d061-e45d-d154-e7e8c1cad497'}


def ref(path):
    path=Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(reference):
    path=ROOT/reference['path'];require(ref(path)==reference,'Authenticated reference: '+reference['path'])
    return json.loads(path.read_bytes())


def tree_bytes(path):return sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) if path.exists() else 0


def check_ram():
    import psutil
    own=psutil.Process();total=0
    for p in [own]+own.children(recursive=True):
        try:total+=p.memory_info().rss
        except psutil.NoSuchProcess:pass
    require(total<=RAM_CAP,'Owned process family exceeds64GiB')
    return total


def check_disk(additional=0):
    require(tree_bytes(BASE)+additional<=CAP,'European direct4 allocation128MiB')
    roots=[ROOT/'research/joint_evaluation_20261029',ROOT/'experiments/results/paper_ancestralpaths_four_20261027']
    require(sum(tree_bytes(p) for p in roots)+additional<=2**30,'Shared evaluation ceiling1GiB')


def write_bytes(path,payload,replace=False):
    path=Path(path).resolve();require(path.is_relative_to(BASE),'European4 output containment')
    require(not path.exists() or replace,'Never overwrite prior attempt')
    check_ram();check_disk(len(payload));path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(payload);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)
    return ref(path)


def write(path,value,replace=False):return write_bytes(path,(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode(),replace)


def save_torch(path,state,replace=False):
    stream=BytesIO();torch.save(state,stream);return write_bytes(path,stream.getvalue(),replace)


def freeze(path=BASE/'freeze_v1.json'):
    frozen=json.loads(Path(path).read_bytes())
    require(frozen['schema']=='european_direct4_freeze_v1','Direct4 source freeze')
    for r in frozen['source_refs']:require(ref(ROOT/r['path'])==r,'Frozen source changed: '+r['path'])
    protocol=read(frozen['protocol_ref'])
    require(protocol['model']==dict(width=64,layers=3,classes=4) and protocol['epochs']==50,'Frozen direct4 architecture and final50epochs')
    smoke=read(frozen['smoke_ref'])
    require(smoke['passed'] and smoke['source_refs']==frozen['source_refs'] and smoke['protocol_ref']==frozen['protocol_ref'],'CPU integration gate binds exact source and protocol')
    return frozen,protocol,ref(path)


def device(gpu):
    import setproctitle
    setproctitle.setproctitle('AI Auto Research Pipeline Test')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==GPUS[gpu],'Expose exactly authorized GPU UUID')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'One assigned GPU')
    actual=str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
    require(actual==GPUS[gpu].removeprefix('GPU-'),'Assigned physical GPU identity')
    torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    return torch.device('cuda:0')
