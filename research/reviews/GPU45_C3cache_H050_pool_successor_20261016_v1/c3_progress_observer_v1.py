"""Operational progress observer for the new cache-C3 TRAIN worker only.

Wrap one exact frozen trainer module's global helper AFTER module load. Delegate
once with unchanged arguments, return object and original exceptions. Observe
only a successful whole_parent_optimizer_updates integer of1; never inspect
parameters, RNG, model/optimizer state, or Torch methods.
"""
from pathlib import Path
import functools,hashlib,importlib.machinery,json,os,sys
ROOT=Path("/workspace-SR003.nfs2/ARG-GNN")
SOURCE="research/reviews/GPU45_C3cache_H050_pool_successor_20261016_v1/c3_progress_observer_v1.py"
TRAINER_REF=dict(path="research/reviews/C3_GNN_prefetch_isolated980_root_20261015_v2/trainer_prefetch_v1.py",
 sha256="b2f08a3e84bd5fb65c5f3d95566ab082fb71a2dd0a826180a10aa518aa35025c")
JOB="C3-GNN-cachev3-shared-TRAIN-v1"


def require(ok,message):
    if not ok:raise ValueError(message)


def wrap_trainer_v1(module,*,pid=None,job=JOB,emit=None):
    require(Path(module.__file__).resolve()==ROOT/TRAINER_REF["path"],"Exact frozen trainer module path")
    require(hashlib.sha256((ROOT/TRAINER_REF["path"]).read_bytes()).hexdigest()==TRAINER_REF["sha256"],
        "Exact frozen trainer source bytes")
    require(job==JOB,"New cache-C3 job only")
    if getattr(module,"_C3_operational_progress_observer_v1",False):return
    original=module.checkpointed_complete_parent_step_v1
    require(callable(original),"Exact module global parent-step binding")
    process_pid=os.getpid() if pid is None else pid
    emitter=emit or (lambda value:print(json.dumps(value,sort_keys=True,separators=(",",":")),flush=True))
    count=0
    @functools.wraps(original)
    def observed(*args,**kwargs):
        nonlocal count
        result=original(*args,**kwargs)  # Exactly one original call, unchanged exceptions.
        if isinstance(result,dict) and type(result.get("whole_parent_optimizer_updates")) is int and result["whole_parent_optimizer_updates"]==1:
            count+=1
            if count<=600 and (count==1 or count%10==0):
                try:emitter(dict(event="C3_shared_complete_parent_progress_v1",pid=process_pid,job=job,
                    successful_whole_parent_optimizer_updates=count))
                except BaseException:
                    pass  # Operational output must never change the scientific return.
        return result
    module.checkpointed_complete_parent_step_v1=observed
    module._C3_operational_progress_observer_v1=True


def install_progress_observer_v1():
    require(os.environ.get("ARGGNN_LOGICAL_JOB_NAME")==JOB,"Exact fresh shared C3 logical identity")
    loader=importlib.machinery.SourceFileLoader;current=loader.exec_module
    if getattr(current,"_C3_operational_progress_observer_v1",False):return
    target=ROOT/TRAINER_REF["path"]
    for module in tuple(sys.modules.values()):
        name=getattr(module,"__file__",None)
        require(name is None or Path(name).resolve()!=target,"Install observer before trainer import")
    def exact_trainer_exec(loader_instance,module):
        name=Path(loader_instance.path)
        current(loader_instance,module)
        if name.resolve()==target:
            require(name==target,"Exact canonical trainer loader path")
            wrap_trainer_v1(module)
    exact_trainer_exec._C3_operational_progress_observer_v1=True
    loader.exec_module=exact_trainer_exec
