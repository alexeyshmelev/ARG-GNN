"""Copy complete observed CPU blocks to private storages, retaining view layout."""
from __future__ import annotations
import copy

def require(ok,message):
    if not ok:raise ValueError(message)

def clone_block_private_v1(block,*,tensor_fields,expected_digest,expected_size,identity,torch_api):
    """No IPC allocator, worker, RNG, device transfer, observation or model change."""
    torch=torch_api
    require(type(expected_size) is int and expected_size>0 and callable(identity),"Explicit original logical size/digest validator")
    require(all(k in block for k in tensor_fields),"Every complete original tensor-bank field")
    storage_sources={};tensor_count=0
    def gather(value):
        nonlocal tensor_count
        if torch.is_tensor(value):
            require(value.device.type=="cpu" and value.layout==torch.strided and not value.requires_grad
                and value.grad_fn is None,"Observed CPU strided tensors without learned/autograd state")
            storage=value.untyped_storage();key=int(storage._cdata)
            storage_sources.setdefault(key,storage);tensor_count+=1
        elif isinstance(value,(tuple,list)):
            for item in value:gather(item)
        elif isinstance(value,dict):
            for item in value.values():gather(item)
    for key,value in block.items():gather(value)
    physical=sum(int(s.nbytes()) for s in storage_sources.values())
    # Existing logical cap must continue to bound actual retained tensor storage.
    # Over-large backing views fail before allocation; never silently compact/restride.
    require(0<physical<=expected_size,"Unique backing bytes must fit original logical cache accounting")
    private={};shared_count=0
    for key,storage in storage_sources.items():
        size=int(storage.nbytes());shared_count+=int(storage.is_shared())
        raw=torch.empty(0,dtype=torch.uint8,device="cpu").set_(storage,0,(size,),(1,))
        owner=raw.clone()
        require(not owner.is_shared(),"Clone must use private CPU allocation")
        private[key]=owner.untyped_storage()
    memo={}
    def cloned(value):
        if torch.is_tensor(value):
            token=id(value)
            if token not in memo:
                source=value.untyped_storage();storage=private[int(source._cdata)]
                out=torch.empty(0,dtype=value.dtype,device="cpu").set_(storage,value.storage_offset(),tuple(value.shape),tuple(value.stride()))
                require(out.dtype==value.dtype and out.layout==value.layout and out.shape==value.shape
                    and out.stride()==value.stride() and out.storage_offset()==value.storage_offset() and not out.is_shared(),"Exact dtype/shape/stride/offset in private storage")
                require(not source.nbytes() or out.untyped_storage()._cdata!=source._cdata,"No retained incoming tensor storage alias")
                memo[token]=out
            return memo[token]
        if isinstance(value,tuple):return tuple(cloned(v) for v in value)
        if isinstance(value,list):return [cloned(v) for v in value]
        if isinstance(value,dict):return {copy.deepcopy(k):cloned(v) for k,v in value.items()}
        return copy.deepcopy(value)
    result=copy.copy(block)
    require(result is not block,"Independent block container")
    for key,value in block.items():result[key]=cloned(value)
    actual,size=identity(result)
    require(actual==expected_digest and size==expected_size,"Byte-identical original feature digest and logical tensor bytes")
    # Return scalar metadata only: no incoming block/storage/tensor reference escapes.
    return result,dict(tensor_occurrences=tensor_count,private_storage_count=len(private),
        private_storage_bytes=physical,incoming_shared_storage_count=shared_count,
        incoming_storage_aliases_retained=False,shape_stride_storage_offset_preserved=True,
        private_view_alias_graph_preserved=True,original_feature_digest_verified=True)
