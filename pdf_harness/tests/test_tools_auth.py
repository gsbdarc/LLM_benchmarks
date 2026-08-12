import pytest

from pdf_harness.auth import SharedPasswordAuth
from pdf_harness.tools import invoke, validate_allowlist


def test_password_auth():
    auth = SharedPasswordAuth("correct horse")
    assert auth.authenticate("correct horse")
    assert not auth.authenticate("wrong")


def test_tool_allowlist_blocks_unapproved_execution():
    with pytest.raises(PermissionError):
        invoke("normalize_text", [], value=" hi ")
    assert invoke("normalize_text", ["normalize_text"], value=" hi ") == "hi"
    assert validate_allowlist(["not_real"], "extraction") == ["unknown built-in tool: not_real"]
