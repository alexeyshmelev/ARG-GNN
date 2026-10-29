"""Close only this DataLoader's processes and their own NFS listener sockets.

Python 3.11 SocketListener unlinks at priority 0 without first closing its
socket if the resource-sharer thread still owns the listener. On NFS that
leaves an open .nfs inode when temp-dir removal runs at priority -100.
Register inside worker_init_fn, and call shutdown_loaders from trainer finally.
"""
from multiprocessing import resource_sharer,util
from torch.utils.data import DataLoader,get_worker_info

_WORKER_FINALIZER=None


def register_worker_resource_cleanup():
    """Stop only the current worker's descriptor-sharing thread before unlink."""
    global _WORKER_FINALIZER
    if get_worker_info() is None:
        raise RuntimeError('Register resource cleanup only inside DataLoader worker_init_fn')
    if _WORKER_FINALIZER is None:
        _WORKER_FINALIZER=util.Finalize(None,resource_sharer.stop,
            kwargs={'timeout':5.0},exitpriority=10)


def shutdown_loaders(*loaders):
    """Ask the supplied loaders to close pin thread/queues, then join own workers.

    PyTorch 2.6.0's private iterator shutdown is version-qualified by the probe.
    It targets exactly the iterator's process objects. No global process scan,
    signal, unlink, rmtree, environment change or unrelated loader is involved.
    """
    result=[];seen=set()
    for loader in loaders:
        if loader is None:continue
        if not isinstance(loader,DataLoader):raise TypeError('Expected owned DataLoader or None')
        iterator=getattr(loader,'_iterator',None)
        if iterator is None or id(iterator) in seen:continue
        seen.add(id(iterator));workers=list(getattr(iterator,'_workers',()))
        close=getattr(iterator,'_shutdown_workers',None)
        if close is not None:close()
        for worker in workers:
            worker.join(timeout=5.0)
            if worker.is_alive():raise RuntimeError(f'Owned loader worker {worker.pid} did not stop')
        result.append({'worker_pids':[worker.pid for worker in workers],
            'worker_exitcodes':[worker.exitcode for worker in workers],
            'iterator_shutdown':bool(getattr(iterator,'_shutdown',True)),
            'pin_thread_alive':bool(getattr(iterator,'_pin_memory_thread',None) and iterator._pin_memory_thread.is_alive())})
        loader._iterator=None
    return result
