"""Immutable observation-block reuse for H044 evaluation only.

Inert on import. Torch and frozen scientific code load only inside admitted
callers. No readers, targets, guards, launchers, model state or file writers.
"""
from dataclasses import dataclass,fields,FrozenInstanceError
import hashlib,json

SOURCE="src/arggnn/marker_resolution_input_cache_v2.py"
SCHEMA="H044_immutable_marker_query_blocks_v2"
VAL_IDS=tuple("ap_expanded_val_%03d"%i for i in range(8))
MAX_PARENTS=8
MAX_PAIR_FEATURE_BYTES=21_073_920_000
QUERY_BLOCK=64

def _require(value,message):
    if not value:raise ValueError(message)

def _runtime_v2():
    import torch
    from . import marker_resolution_readout_v1 as core
    return torch,core

def _stamp_v2(tensor):
    return (id(tensor),tensor.data_ptr(),tensor._version,tuple(tensor.shape),tensor.dtype,str(tensor.device))

def pair_feature_bytes_v2(centers,queries=686):
    _require(type(centers) is int and 1<=centers<=20000 and queries==686,
        "Original complete H044 scoring dimensions required")
    return 2*centers*queries*4*6*4

def _binding_digest_v2(markers):
    _require(markers.original_binding is not None,"Original identity-bound observation recipe required")
    return hashlib.sha256(json.dumps(markers.original_binding,sort_keys=True,
        separators=(",",":"),allow_nan=False).encode()).hexdigest()

@dataclass(frozen=True)
class ImmutableMarkerBlocksV2:
    parent_id: str
    source_bundle: object
    binding_sha256: str
    ranges: tuple
    localized: tuple
    tree_pooled: tuple
    localized_stamps: tuple
    pooled_stamps: tuple
    feature_bytes: int

    def validate(self):
        self.source_bundle.validate()
        _require(_binding_digest_v2(self.source_bundle.markers)==self.binding_sha256,
            "Original marker binding changed after cache construction")
        _require(tuple(_stamp_v2(x) for x in self.localized)==self.localized_stamps
            and tuple(_stamp_v2(x) for x in self.tree_pooled)==self.pooled_stamps,
            "Immutable cached marker block was mutated or replaced")
        return self

    def raw(self,start,end):
        self.validate()
        _require((start,end) in self.ranges,"Only original contiguous query ranges are cached")
        return self.localized[self.ranges.index((start,end))]

    def pooled(self,feature):
        self.validate()
        matches=[i for i,x in enumerate(self.localized) if x is feature]
        _require(len(matches)==1,"Pooling cache accepts only its exact original localized block")
        return self.tree_pooled[matches[0]]

    def receipt(self):
        return dict(schema=SCHEMA,parent_id=self.parent_id,binding_sha256=self.binding_sha256,
            query_block=QUERY_BLOCK,query_ranges=[list(x) for x in self.ranges],
            feature_bytes=self.feature_bytes,localized_blocks=len(self.localized),
            pooled_blocks=len(self.tree_pooled),dtype="float32",device="cpu",
            original_contiguous_block_layout=True,original_arithmetic_reused=True,
            learned_values_cached=False,targets_used=False,arrays_durable=False)

def build_marker_blocks_v2(parent_id,bundle,*,remaining_feature_bytes):
    torch,core=_runtime_v2()
    _require(parent_id in VAL_IDS and type(bundle) is core.H044BoundMarkerBankV1,
        "Evaluation-only exact original H044 bundle required")
    bank,markers=bundle.validate()
    _require(bank is bundle.source_bank and type(markers) is core.MarkerEvidenceBankV1,
        "Build once from the original CPU observation bundle, never a cache view")
    _require(markers.original_binding.get("parent_id")==parent_id
        and len(markers.query_column)==686 and not torch.is_inference_mode_enabled(),
        "Same-parent original queries and normal CPU tensor construction required")
    p=len(markers.centers_bp);needed=pair_feature_bytes_v2(p)
    _require(type(remaining_feature_bytes) is int and needed<=remaining_feature_bytes,
        "Preflight immutable feature cache byte cap")
    ranges=tuple((start,min(start+QUERY_BLOCK,686)) for start in range(0,686,QUERY_BLOCK))
    raw_blocks=[];pooled_blocks=[]
    with torch.no_grad():
        for start,end in ranges:
            # Keep exactly the original operations, casts, query order and layout.
            raw=core.MarkerEvidenceBankV1.features_for_queries(markers,start,end)
            raw_stamp=_stamp_v2(raw)
            pooled=core.MarkerEvidenceBankV1.pool_within_trees(markers,raw)
            _require(_stamp_v2(raw)==raw_stamp,"Original pooling mutated localized inputs")
            for value in (raw,pooled):
                _require(isinstance(value,torch.Tensor) and value.device.type=="cpu"
                    and value.dtype==torch.float32 and value.shape==(p,end-start,4,6)
                    and value.is_contiguous() and not value.requires_grad
                    and not torch.is_inference(value),
                    "Original normal contiguous FP32 observation block required")
            raw_blocks.append(raw);pooled_blocks.append(pooled)
    bundle.validate()
    count=sum(x.numel()*x.element_size() for x in (*raw_blocks,*pooled_blocks))
    _require(count==needed,"Exact original feature-pair byte account")
    result=ImmutableMarkerBlocksV2(parent_id,bundle,_binding_digest_v2(markers),ranges,
        tuple(raw_blocks),tuple(pooled_blocks),tuple(_stamp_v2(x) for x in raw_blocks),
        tuple(_stamp_v2(x) for x in pooled_blocks),count)
    return result.validate()

def bind_cached_bundle_v2(bundle,blocks):
    """Return a typed observation view; reuse frozen model forward unchanged."""
    _,core=_runtime_v2()
    _require(type(bundle) is core.H044BoundMarkerBankV1
        and type(blocks) is ImmutableMarkerBlocksV2,"Exact typed original pairing/cache required")
    bundle.validate();blocks.validate()
    _require(bundle.source_bank is blocks.source_bundle.source_bank
        and bundle.markers is blocks.source_bundle.markers,"Cross-parent/cache pairing denied")

    class CachedMarkerEvidenceViewV2(core.MarkerEvidenceBankV1):
        __slots__=("_immutable_blocks_v2",)
        def __setattr__(self,name,value):
            raise FrozenInstanceError("Immutable cached marker view")
        def __delattr__(self,name):
            raise FrozenInstanceError("Immutable cached marker view")
        def features_for_queries(self,start,end):
            return self._immutable_blocks_v2.raw(start,end)
        def pool_within_trees(self,feature):
            return self._immutable_blocks_v2.pooled(feature)

    original=bundle.markers
    view=CachedMarkerEvidenceViewV2(**{f.name:getattr(original,f.name) for f in fields(original)})
    object.__setattr__(view,"_immutable_blocks_v2",blocks)
    # Original tensor stamps cover every borrowed observation. The separate
    # blocks.validate() covers all cached tensors, which are not core fields.
    paired=core.H044BoundMarkerBankV1(bundle.bank,bundle.source_bank,view,
        bundle.source_stamps,bundle.device_stamps,core._tensor_stamps_v1(view))
    paired.validate()
    return paired

class EvaluationMarkerBlockStoreV2:
    """At most eight original VAL parents; explicit bytes and parent release."""
    def __init__(self,*,maximum_feature_bytes=MAX_PAIR_FEATURE_BYTES,maximum_parents=MAX_PARENTS):
        _require(type(maximum_feature_bytes) is int and 0<maximum_feature_bytes<=MAX_PAIR_FEATURE_BYTES
            and type(maximum_parents) is int and 1<=maximum_parents<=MAX_PARENTS,
            "Bounded evaluation-only immutable cache policy required")
        self.maximum_feature_bytes=maximum_feature_bytes;self.maximum_parents=maximum_parents
        self._values={};self.bytes=0;self.peak_bytes=0;self.builds=0;self.reuses=0;self._receipts={}

    def get(self,parent_id,bundle):
        _require(parent_id in VAL_IDS,"Only original eight VAL parents may enter this cache")
        if parent_id in self._values:
            value=self._values[parent_id]
            _require(value.source_bundle is bundle,"Cache hit requires the exact original CPU bundle")
            self.reuses+=1;return value.validate()
        _require(len(self._values)<self.maximum_parents,"Immutable parent cache count cap")
        value=build_marker_blocks_v2(parent_id,bundle,
            remaining_feature_bytes=self.maximum_feature_bytes-self.bytes)
        _require(parent_id not in self._receipts,"Released parent cannot be silently rebuilt")
        self._values[parent_id]=value;self._receipts[parent_id]=value.receipt()
        self.bytes+=value.feature_bytes;self.peak_bytes=max(self.peak_bytes,self.bytes);self.builds+=1
        return value

    def release(self,parent_id):
        _require(parent_id in self._values,"Only a retained parent can be released")
        value=self._values.pop(parent_id);value.validate();self.bytes-=value.feature_bytes

    def receipt(self):
        return dict(schema="H044_evaluation_marker_block_store_v2",maximum_parents=self.maximum_parents,
            maximum_feature_bytes=self.maximum_feature_bytes,retained_feature_bytes=self.bytes,
            peak_feature_bytes=self.peak_bytes,builds=self.builds,reuses=self.reuses,
            parents={k:dict(v) for k,v in self._receipts.items()},
            release_after_parent=True,targets_used=False,learned_values_cached=False,arrays_durable=False)
