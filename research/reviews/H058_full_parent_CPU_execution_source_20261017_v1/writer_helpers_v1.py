"""Bind metadata serialization to actual local modules, never frozen lazy globals.

ROOT supplies this callable to its existing admitted bounded writer. Checkpoint
serialization remains that writer's bounded local torch.save path.
"""
def make_jsonable_v1(torch_module,numpy_module):
    if getattr(torch_module,"__name__",None)!="torch"or getattr(numpy_module,"__name__",None)!="numpy":
        raise TypeError("Actual already-imported local Torch/NumPy modules required")
    def jsonable(value):
        if isinstance(value,dict):return {str(key):jsonable(item)for key,item in value.items()}
        if isinstance(value,(list,tuple)):return [jsonable(item)for item in value]
        if isinstance(value,torch_module.Tensor):return value.detach().cpu().tolist()
        if isinstance(value,numpy_module.ndarray):return value.tolist()
        if isinstance(value,numpy_module.generic):return value.item()
        if value is None or isinstance(value,(str,bool,int,float)):return value
        raise TypeError("Unsupported bounded metadata value: "+type(value).__name__)
    return jsonable
