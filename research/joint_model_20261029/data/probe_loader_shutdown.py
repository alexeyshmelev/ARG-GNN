"""CPU-only owned DataLoader shutdown reproduction with repo-local temp dirs."""
import argparse
import json
from multiprocessing import util
import os
from pathlib import Path
import time
import torch
from torch.utils.data import Dataset,DataLoader
from .loader_shutdown import register_worker_resource_cleanup,shutdown_loaders

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')


class Fixture(Dataset):
    def __len__(self):return 128
    def __getitem__(self,index):
        folder=util.get_temp_dir()
        if not Path(folder).resolve().is_relative_to(ROOT):raise ValueError('Temp dir must stay in project')
        return dict(tensor=torch.arange(4096,dtype=torch.float32)+index,pid=os.getpid(),folder=folder,index=index)


def initialize_worker(_):
    torch.set_num_threads(1)
    register_worker_resource_cleanup()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['baseline','complete','early'],required=True);args=parser.parse_args()
    torch.set_num_threads(1);started=time.perf_counter()
    loader=DataLoader(Fixture(),batch_size=4,num_workers=2,persistent_workers=True,
        multiprocessing_context='spawn',prefetch_factor=2,
        worker_init_fn=None if args.case=='baseline' else initialize_worker)
    folders=set();count=0;worker_pids=[]
    iterator=iter(loader);worker_pids=[w.pid for w in iterator._workers]
    for batch in iterator:
        count+=len(batch['index']);folders.update(batch['folder'])
        if args.case=='early' and count>=8:break
    result=shutdown_loaders(loader)
    print(json.dumps(dict(case=args.case,torch_version=torch.__version__,workers=worker_pids,rows=count,
        shutdown=result,temp_folders=sorted(folders),remaining_temp_folders=[p for p in sorted(folders) if Path(p).exists()],
        seconds=time.perf_counter()-started)),flush=True)


if __name__=='__main__':main()
