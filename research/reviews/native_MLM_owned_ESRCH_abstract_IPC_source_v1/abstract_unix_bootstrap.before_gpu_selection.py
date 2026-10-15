"""Prospective native-child-only Linux abstract AF_UNIX address bootstrap.

Install once before torch/author imports and their unchanged default fork workers.
No installed file, guard, author function, authkey or scientific setting is changed.
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
_ORIGINAL = None

def install_abstract_unix_v1(namespace):
    global _ORIGINAL
    if _ORIGINAL is not None:
        raise ValueError("Install exactly once before native multiprocessing")
    if sys.platform != "linux" or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", namespace):
        raise ValueError("Linux and bounded explicit attempt namespace required")
    if mp.get_start_method(allow_none=True) not in (None, "fork"):
        raise ValueError("Only the unchanged Linux default fork path is qualified")
    source = Path(connection.__file__)
    if hashlib.sha256(source.read_bytes()).hexdigest() != EXPECTED_CONNECTION_SHA256:
        raise ValueError("Exact qualified native Python3.11 connection source required")
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
    _ORIGINAL = original
    connection.arbitrary_address = abstract_address
    return {"schema": "native_abstract_AF_UNIX_bootstrap_v1", "platform": sys.platform,
            "namespace": namespace, "connection_source_sha256": EXPECTED_CONNECTION_SHA256,
            "address_family": "AF_UNIX", "leading_NUL_abstract": True,
            "filesystem_listener_fallback": False, "other_families_delegate_unchanged": True,
            "authkey_and_Listener_Client_unchanged": True,
            "qualified_start_method": "unchanged default fork",
            "scope": "This interpreter and inherited fork descendants only"}
