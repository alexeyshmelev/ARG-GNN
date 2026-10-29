"""Reversible tensor layout packing, then ordinary lossless compression.

Floating-point values are only byte shuffled. Integer deltas are exact and
restored to their original dtype. Sparse CSR components retain every entry.
"""
import json
import struct
import numpy as np
import torch
from .common import require
from .compression_v2 import compress, decompress

MAGIC = b'AMJOINT2'


def pack(state):
    chunks = []
    def encode(value):
        if isinstance(value, torch.Tensor):
            if value.layout == torch.sparse_csr:
                return dict(kind='csr', shape=list(value.shape), crow=encode(value.crow_indices()),
                            col=encode(value.col_indices()), values=encode(value.values()))
            require(value.layout == torch.strided and value.device.type == 'cpu', 'CPU dense tensors or CSR only')
            array = value.detach().numpy()
            original_dtype, shape = array.dtype.str, list(array.shape)
            flattened = array.reshape(-1)
            delta = False
            if array.dtype.kind in 'iu' and flattened.size:
                require(np.max(np.abs(flattened.astype(np.float64))) < 2**60, 'Safe exact integer delta bound')
                flattened = np.diff(flattened.astype(np.int64), prepend=np.int64(0))
                delta = True
                low, high = int(flattened.min()), int(flattened.max())
                for dtype in (np.int8, np.int16, np.int32, np.int64):
                    bounds = np.iinfo(dtype)
                    if low >= bounds.min and high <= bounds.max:
                        flattened = flattened.astype(dtype)
                        break
            encoded_dtype = flattened.dtype.str
            shuffled = flattened.copy().view(np.uint8).reshape(-1, flattened.dtype.itemsize).T.copy().tobytes()
            index = len(chunks)
            chunks.append(shuffled)
            return dict(kind='tensor', shape=shape, dtype=original_dtype, storage_dtype=encoded_dtype,
                        delta=delta, chunk=index, bytes=len(shuffled))
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [encode(item) for item in value]
        return value
    manifest = json.dumps(encode(state), separators=(',', ':'), sort_keys=True, allow_nan=False).encode()
    return struct.pack('<Q', len(manifest)) + manifest + b''.join(chunks)


def unpack(raw):
    require(8 <= len(raw) <= 128 * 2**20, 'Bounded state encoding')
    manifest_length = struct.unpack('<Q', raw[:8])[0]
    require(manifest_length <= 2**20, 'Bounded tensor manifest')
    manifest = json.loads(raw[8:8 + manifest_length])
    leaves = []
    def collect(value):
        if isinstance(value, dict):
            if value.get('kind') == 'tensor':
                leaves.append(value)
            else:
                for item in value.values():
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
    collect(manifest)
    leaves.sort(key=lambda item: item['chunk'])
    require([item['chunk'] for item in leaves] == list(range(len(leaves))), 'Exact contiguous tensor chunk IDs')
    arrays = []
    cursor = 8 + manifest_length
    for item in leaves:
        dtype = np.dtype(item['storage_dtype'])
        size = item['bytes']
        data = raw[cursor:cursor + size]
        require(len(data) == size, 'Complete tensor data')
        cursor += size
        array = np.frombuffer(data, dtype=np.uint8).reshape(dtype.itemsize, -1).T.copy().reshape(-1).view(dtype)
        if item['delta']:
            array = np.cumsum(array, dtype=np.int64)
        array = array.astype(np.dtype(item['dtype']), copy=False).reshape(item['shape'])
        arrays.append(torch.from_numpy(array.copy()))
    require(cursor == len(raw), 'No trailing state bytes')
    def decode(value):
        if isinstance(value, dict):
            if value.get('kind') == 'tensor':
                return arrays[value['chunk']]
            if value.get('kind') == 'csr':
                return torch.sparse_csr_tensor(decode(value['crow']), decode(value['col']), decode(value['values']), size=value['shape'])
            return {key: decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [decode(item) for item in value]
        return value
    return decode(manifest)


def encode(state, level=3):
    raw = pack(state)
    return MAGIC + struct.pack('<Q', len(raw)) + compress(raw, level=level)


def decode(payload):
    require(payload[:8] == MAGIC, 'Exact prepared-state cache magic')
    size = struct.unpack('<Q', payload[8:16])[0]
    return unpack(decompress(payload[16:], size))
