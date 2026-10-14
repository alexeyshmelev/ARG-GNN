"""Common source-only C3 I/O. Root supplies existing storage/runtime authority."""
from pathlib import Path
import argparse,hashlib,importlib.util,importlib.metadata,json,os,platform,sys

ROOT=Path('/workspace-SR003.nfs2/ARG-GNN')
CLASSES=('EUR','EAS','AFR')


def require(ok,message):
    if not ok:raise ValueError(message)


def local(value):
    p=Path(value);p=p if p.is_absolute() else ROOT/p
    require('..' not in p.parts and p.resolve()==p,'Canonical nonsymlink project path required')
    p.relative_to(ROOT);return p


def ref(value):
    p=local(value);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=h.hexdigest())


def bound(record):
    require(isinstance(record,dict) and set(record)=={'path','sha256'} and ref(record['path'])==record,'Exact file SHA binding required')
    return local(record['path'])


def read(record):return json.loads(bound(record).read_text())


def source_closure(source,dependencies=()):
    return [ref(p) for p in dict.fromkeys(('src/arggnn/american_c3_common_v1.py',source,*dependencies))]


def runtime_report():
    packages={}
    for name in ('numpy','tskit','msprime','stdpopsim','tsinfer','tsdate','torch','torch_geometric'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    return dict(schema='american_c3_cpu_runtime_metadata_v1',prefix=str(Path(sys.prefix).resolve()),
        executable=str(Path(sys.executable).resolve()),python=platform.python_version(),
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),packages=packages,
        metadata_only=True,scientific_arrays_created=False)


def cli(source,defaults,execute,dependencies=()):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true');mode.add_argument('--runtime-report',action='store_true')
    parser.add_argument('--admission');parser.add_argument('--expected-admission-sha256');args=parser.parse_args()
    closure=source_closure(source,dependencies)
    if args.runtime_report:
        print(json.dumps(dict(status='metadata_only_runtime_report',source_closure_refs=closure,
            runtime=runtime_report()),indent=2));return
    if not args.execute:
        print(json.dumps(dict(status='source_only_inactive',source_ref=ref(source),
            source_closure_refs=closure,config_draft=defaults),indent=2));return
    require(args.admission and args.expected_admission_sha256,'Exact activated root admission required')
    spec=read(dict(path=args.admission,sha256=args.expected_admission_sha256));cfg=read(spec['config_ref'])
    require(Path.cwd().resolve()==ROOT and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Explicit CPU-only repository execution required')
    require(spec.get('root_authorizer')=='root_agent' and spec.get('execution_authorized') is True
        and spec.get('GPU_authorized') is False and spec['source_ref']==ref(source)
        and cfg.get('execution_enabled') is True,'Actual root CPU/source/config authority required')
    require(spec.get('source_closure_refs')==closure,'Exact common/producer/helper source closure required')
    bindings=spec['storage']['reservation']['consumer_bindings']
    require(all(r in bindings for r in [*closure,spec['config_ref'],spec['runtime_ref'],spec['storage_adapter_ref']]),
        'Actual common/producer/helpers/config/runtime/adapter must be reserved')
    runtime=read(spec['runtime_ref'])
    require(runtime.get('prefix')==str(Path(sys.prefix).resolve()) and runtime.get('CUDA_VISIBLE_DEVICES')=='',
            'Exact root-qualified interpreter/CPU runtime required')
    if 'packages' in runtime:
        actual=runtime_report()['packages']
        require(all(actual.get(k)==v for k,v in runtime['packages'].items()),'Root-qualified package metadata changed')
    module_spec=importlib.util.spec_from_file_location('american_c3_existing_storage',bound(spec['storage_adapter_ref']))
    adapter=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(adapter)
    if hasattr(adapter,'configure_v2'):adapter.configure_v2(spec['marker_repair_authority_ref'])
    elif hasattr(adapter,'configure_repair_v1'):adapter.configure_repair_v1(spec['marker_repair_authority_ref'])
    require(spec['storage']['budget_components']['consumer_adapter']==spec['storage_adapter_ref'],'Actual existing adapter binding required')
    adapter.validate_storage_v9(spec,cfg,spec['storage'],source_ref=spec['source_ref'],
        config_ref=spec['config_ref'],runtime_ref=spec['runtime_ref'])
    return execute(cfg,Writer(spec,adapter))


class Writer:
    """Exclusive producer scope; existing guard checks each planned actual write."""
    def __init__(self,spec,adapter):
        self.storage,self.adapter=spec['storage'],adapter;self.path=local(spec['output_directory'])
        require(not self.path.exists(),'Fresh exclusive stage output required')
        adapter.budget_check_v9(self.storage,boundary=True,required_write_bytes=8192);self.path.mkdir()
    def prepare(self,name,budget_bytes):
        p=self.path/name;local(p);require(not p.exists(),'No overwrite/replay')
        if not p.parent.exists():
            self.adapter.budget_check_v9(self.storage,boundary=True,required_write_bytes=8192)
            p.parent.mkdir(parents=True)
        self.adapter.budget_check_v9(self.storage,boundary=True,required_write_bytes=int(budget_bytes)+8192)
        return p
    def finish(self,p,planned):
        require(p.stat().st_size<=planned,'Written bytes exceed serialization/count bound; preserve partial stage')
        self.adapter.budget_check_v9(self.storage,boundary=True)
        return ref(p)
    def json(self,name,value):
        b=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();p=self.prepare(name,len(b))
        with p.open('xb') as f:f.write(b)
        return self.finish(p,len(b))
    def npy(self,name,array):
        import numpy as np
        require(not array.dtype.hasobject,'No object arrays')
        size=array.nbytes+4096;p=self.prepare(name,size)
        with p.open('xb') as f:np.save(f,array,allow_pickle=False)
        return self.finish(p,size)
    def npz(self,name,**arrays):
        import numpy as np
        require(all(not a.dtype.hasobject for a in arrays.values()),'No object arrays')
        size=sum(a.nbytes for a in arrays.values())+65536;p=self.prepare(name,size)
        with p.open('xb') as f:np.savez_compressed(f,**arrays)
        return self.finish(p,size)
    def pickle(self,name,value,planned):
        import pickle
        p=self.prepare(name,planned)
        with p.open('xb') as f:pickle.dump(value,f,protocol=pickle.HIGHEST_PROTOCOL)
        return self.finish(p,planned)
