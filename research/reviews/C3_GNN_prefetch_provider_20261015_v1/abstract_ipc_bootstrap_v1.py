"""Fresh pre-Torch Linux abstract IPC bootstrap for the optional C3 spawn pool.

Adapted source mechanism from reviewed native-MLM v12 helper; no amendment,
allocation, launch, cleanup, device or accounting authority is inherited.
"""
from pathlib import Path
import hashlib
import multiprocessing as mp
import multiprocessing.connection as connection
import os
import re
import secrets
import sys

EXPECTED_CONNECTION_SHA256 = "4fd122a7695ffa1429461e3e8f241cbac6361dc9aebf06bc94483545e3bf73d5"
PROVENANCE = dict(path="research/reviews/American_C3_native_MLM_amendment_v12/abstract_unix_bootstrap_v1.py",
                  sha256="4550ead44f36eeb0fda8a817a2002082c3d067cf1a84fd817301532577d8b014")
NAMESPACE = "c3-gnn-prefetch-v1"
_WRAPPER, _RECEIPT = None, None


def ensure_abstract_ipc_before_torch_v1(namespace=NAMESPACE):
    global _WRAPPER, _RECEIPT
    if _WRAPPER is not None:
        if namespace != _RECEIPT["namespace"] or connection.arbitrary_address is not _WRAPPER:
            raise ValueError("Exact interpreter-local optional C3 IPC bootstrap required")
        return dict(_RECEIPT)
    if sys.platform != "linux" or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", namespace):
        raise ValueError("Linux and explicit bounded fresh namespace required")
    if any(name == "torch" or name.startswith(("torch.", "torch_geometric")) for name in sys.modules):
        raise ValueError("Install optional C3 abstract IPC before every Torch/PyG import")
    source = Path(connection.__file__)
    if hashlib.sha256(source.read_bytes()).hexdigest() != EXPECTED_CONNECTION_SHA256:
        raise ValueError("Exact reviewed Python 3.11 connection source required")
    sharer = sys.modules.get("multiprocessing.resource_sharer")
    if sharer is not None and getattr(getattr(sharer, "_resource_sharer", None), "_listener", None) is not None:
        raise ValueError("No existing resource-sharer listener may precede bootstrap")
    original = connection.arbitrary_address
    if original.__module__ != "multiprocessing.connection" or original.__name__ != "arbitrary_address":
        raise ValueError("Original unwrapped stdlib arbitrary_address required")
    prefix = b"\0arggnn-" + namespace.encode("ascii") + b"-"
    def abstract_address(family):
        if family != "AF_UNIX":
            return original(family)
        address = prefix + str(os.getpid()).encode("ascii") + b"-" + secrets.token_hex(16).encode("ascii")
        if len(address) > 107:
            raise ValueError("Bounded Linux abstract AF_UNIX address required")
        return address
    _WRAPPER = abstract_address
    connection.arbitrary_address = abstract_address
    _RECEIPT = dict(schema="C3_prefetch_pre_torch_abstract_IPC_v1", namespace=namespace,
        connection_source_sha256=EXPECTED_CONNECTION_SHA256,
        leading_NUL_abstract=True, filesystem_listener_fallback=False,
        authkey_and_Listener_Client_unchanged=True, other_families_delegate_unchanged=True,
        required_pool_start_method="explicit_spawn", installed_before_Torch_PyG=True,
        no_MLM_amendment_or_execution_authority=True, mechanism_provenance=dict(PROVENANCE))
    return dict(_RECEIPT)
