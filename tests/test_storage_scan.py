"""Transient metadata races may retry, while persistent failures stay closed."""
from subprocess import CompletedProcess
from unittest.mock import patch
import pytest
from arggnn.resources import storage_bytes


def test_storage_retry_requires_a_complete_successful_scan():
    results=[CompletedProcess([],1,"123456 /partial","cannot access temporary file"),
             CompletedProcess([],0,"987654 /complete\n","")]
    with patch("arggnn.resources.subprocess.run",side_effect=results) as run,patch("arggnn.resources.time.sleep"):
        assert storage_bytes()==987654
        assert run.call_count==2


def test_persistent_storage_failure_never_accepts_partial_output():
    failure=CompletedProcess([],1,"123456 /partial","permission denied")
    with patch("arggnn.resources.subprocess.run",return_value=failure) as run,patch("arggnn.resources.time.sleep"):
        with pytest.raises(RuntimeError,match="four attempts.*permission denied"):
            storage_bytes()
        assert run.call_count==4
