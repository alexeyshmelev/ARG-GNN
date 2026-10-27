"""Bounded exact-signature evidence; canonical data hashing, never executable payloads.

All scalar/structural/provenance fields participate. Large plain structures are
streamed through SHA256 after the measured preparation interval. Arrays retain
the immutable exact_signature digests, which already bind dtype/shape/bytes.
"""
import hashlib
import json

SCHEMA = 'TreeLoom_complete_preparation_signature_sha256_v2'
ENCODING = dict(format='JSON', charset='ASCII', ensure_ascii=True, sort_keys=True,
                separators=[',', ':'], allow_nan=False, digest='SHA256',
                array_contract='original resident_pool_v2.exact_signature arrays unchanged')
SECTIONS = ('arrays', 'structures', 'outer', 'metadata')


def canonical_digest(value, *, resource_check):
    if not callable(resource_check):
        raise ValueError('Active resource callback required during untimed signature hashing')
    encoder = json.JSONEncoder(ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False)
    digest = hashlib.sha256()
    size = chunks = 0
    checked_bytes = 0
    resource_check('canonical_signature_start')
    for text in encoder.iterencode(value):
        for first in range(0, len(text), 65536):
            data = text[first:first+65536].encode('ascii')
            digest.update(data)
            size += len(data)
            chunks += 1
            if size - checked_bytes >= 2**20:
                resource_check('canonical_signature_stream')
                checked_bytes = size
    resource_check('canonical_signature_finish')
    return dict(sha256=digest.hexdigest(), canonical_bytes=size)


def array_layout(bundle):
    from research.reviews.workflow_progress_diagnosis_20261018.treeloom import resident_pool_v2 as original
    arrays, structures = original.codec.flatten(bundle.bank, bundle.episodes, bundle.cut)
    for name in original.OUTER_TENSORS:
        arrays['outer.' + name] = getattr(bundle, name).numpy()
    return {key: dict(shape=list(value.shape), dtype=value.dtype.str, count=int(value.size))
            for key, value in arrays.items()}


def bounded_signature(signature, *, array_metadata, resource_check):
    if type(signature) is not dict or set(signature) != set(SECTIONS):
        raise ValueError('Every original exact signature section is required')
    arrays = signature['arrays']
    if type(arrays) is not dict or not arrays or len(arrays) > 1024:
        raise ValueError('Finite nonempty original array hash map required')
    if any(type(key) is not str or not key or type(value) is not str
           or len(value) != 64 or any(c not in '0123456789abcdef' for c in value)
           for key, value in arrays.items()):
        raise ValueError('Original named SHA256 array digests required')
    if type(array_metadata) is not dict or set(array_metadata) != set(arrays):
        raise ValueError('Exact array shape/dtype/count metadata required')
    receipt = dict(schema=SCHEMA, encoding=dict(ENCODING), arrays=dict(arrays), array_metadata=array_metadata,
        section_digests={key: canonical_digest(signature[key], resource_check=resource_check) for key in SECTIONS},
        complete_signature_digest=canonical_digest(signature, resource_check=resource_check),
        all_original_sections_included=True, sampling_or_truncation=False)
    validate_signature(receipt)
    if len(json.dumps(receipt, sort_keys=True, allow_nan=False).encode()) > 512*2**10:
        raise ValueError('Bounded signature receipt exceeds 512 KiB')
    return receipt


def validate_signature(receipt):
    if (type(receipt) is not dict or receipt.get('schema') != SCHEMA
            or receipt.get('encoding') != ENCODING
            or receipt.get('all_original_sections_included') is not True
            or receipt.get('sampling_or_truncation') is not False
            or set(receipt.get('section_digests', {})) != set(SECTIONS)):
        raise ValueError('Complete bounded signature evidence required')
    arrays = receipt.get('arrays')
    if type(arrays) is not dict or not arrays or len(arrays) > 1024:
        raise ValueError('Complete original array digests required')
    layouts = receipt.get('array_metadata')
    if type(layouts) is not dict or set(layouts) != set(arrays):
        raise ValueError('Array layout keys differ from original hash map')
    import math
    for item in layouts.values():
        if (type(item) is not dict or set(item) != {'shape','dtype','count'}
                or type(item['shape']) is not list or any(type(n) is not int or n < 0 for n in item['shape'])
                or type(item['dtype']) is not str or not item['dtype']
                or type(item['count']) is not int or item['count'] != math.prod(item['shape'])):
            raise ValueError('Exact shape/dtype/count metadata required')
    if canonical_digest(arrays, resource_check=lambda stage: None) != receipt['section_digests']['arrays']:
        raise ValueError('Array digest map binding mismatch')
    for item in [*receipt['section_digests'].values(), receipt.get('complete_signature_digest')]:
        if (type(item) is not dict or set(item) != {'sha256', 'canonical_bytes'}
                or type(item['canonical_bytes']) is not int or item['canonical_bytes'] <= 0
                or type(item['sha256']) is not str or len(item['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in item['sha256'])):
            raise ValueError('Positive exact canonical size and SHA256 required')
    return receipt
