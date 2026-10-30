"""Publish complete controller bytes atomically; immutable destinations never overwrite."""
import json
import os
from pathlib import Path
import uuid
from research.joint_evaluation_20261029.argmix.recovery_20261030_queue import controller_v1 as original


def publish_bytes(filename, content, folder, *, immutable=True, max_bytes=original.RECORD_CAP):
    filename, folder = Path(filename), Path(folder)
    original.require(filename.is_relative_to(folder), 'Publication remains inside its registered scope')
    original.require(isinstance(content, bytes), 'Explicit serialized bytes are required')
    original.require(original.tree_bytes(folder) + len(content) <= max_bytes, 'Registered publication byte cap')
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_name('.' + filename.name + '.pending.' + str(os.getpid()) + '.' + uuid.uuid4().hex)
    try:
        with temporary.open('xb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            # A complete same-filesystem inode appears in one operation; EEXIST preserves prior evidence.
            os.link(temporary, filename, follow_symlinks=False)
            temporary.unlink()
        else:
            os.replace(temporary, filename)
        descriptor = os.open(filename.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
    return original.ref(filename)


def atomic_record(filename, value, folder, *, immutable=False, max_bytes=original.RECORD_CAP):
    content = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    return publish_bytes(filename, content, folder, immutable=immutable, max_bytes=max_bytes)
