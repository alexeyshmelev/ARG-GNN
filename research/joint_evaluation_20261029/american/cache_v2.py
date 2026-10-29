"""Bounded immutable TRAIN tensor cache, authenticated by exact input identity."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
from .common_v2 import CACHE, CACHE_CAP, require, ref, json_data, tree_bytes

MAGIC = b'AMCACHE2'
MAX_FILE = 16 * 2**20
COUNTER = CACHE / 'reserved_bytes_v2.txt'
LOCK = CACHE / 'allocation_v2.lock'


def initialize():
    CACHE.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not COUNTER.exists():
            current = tree_bytes(CACHE) + 1024 * 2**10
            require(current <= CACHE_CAP, 'Existing probe data fits total63GiB cache allocation')
            with COUNTER.open('x') as destination:
                destination.write(f'{current:020d}\n')
                destination.flush()
                os.fsync(destination.fileno())
        reserved = int(COUNTER.read_text())
        require(0 <= reserved <= CACHE_CAP, 'Cache accounting valid')
        return reserved


def identity(namespace, parent, item, site):
    raw = json.dumps(json_data(dict(namespace=namespace, parent=parent, inputs=item, site=site)),
                     sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).digest()


def location(namespace, parent, site):
    require(len(namespace) == 64 and all(c in '0123456789abcdef' for c in namespace), 'Exact SHA256 cache namespace')
    require(parent.startswith('argformer_american_train_') and '/' not in parent, 'TRAIN-only cache parent')
    return CACHE / 'prepared_v2' / namespace / parent / f"{site['window_index']:05d}.amc"


def read_cache(namespace, parent, item, site):
    destination = location(namespace, parent, site)
    if not destination.exists():
        return None
    require(72 < destination.stat().st_size <= MAX_FILE, 'Bounded cache file before read')
    payload = destination.read_bytes()
    require(payload[:8] == MAGIC and payload[8:40] == identity(namespace, parent, item, site), 'Exact observed-input/window cache identity')
    require(payload[40:72] == hashlib.sha256(payload[72:]).digest(), 'Cached tensor payload checksum')
    return payload[72:]


def write_cache(namespace, parent, item, site, payload):
    destination = location(namespace, parent, site)
    content = MAGIC + identity(namespace, parent, item, site) + hashlib.sha256(payload).digest() + payload
    require(len(content) <= MAX_FILE, 'Bounded single prepared window')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        require(not destination.exists(), 'No overwrite or concurrent duplicate prepared window')
        reserved = int(COUNTER.read_text())
        # Conservatively include per-file directory/metadata overhead in the reservation.
        reservation = len(content) + 4096
        require(reserved + reservation <= CACHE_CAP, 'Cache63GiB cap reached before write; no target truncation')
        with COUNTER.open('r+') as counter:
            counter.write(f'{reserved + reservation:020d}\n')
            counter.flush()
            os.fsync(counter.fileno())
        with destination.open('xb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    return len(content)
