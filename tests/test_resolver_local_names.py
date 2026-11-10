"""A bare model name counts as a local path only when it names something inside the working directory."""

import os

import pytest

from turing.models.resolver import ModelResolver


def test_bare_name_in_working_directory_is_local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mymodel").mkdir()
    spec = ModelResolver.parse("mymodel")
    assert spec.is_local_path and spec.provider == "local"


def test_bare_name_resolving_outside_working_directory_is_not_local(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    try:
        os.symlink(outside, cwd / "escape", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks need extra privileges on this platform")
    monkeypatch.chdir(cwd)
    assert not ModelResolver.parse("escape").is_local_path


def test_missing_bare_name_is_not_local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert not ModelResolver.parse("not-there").is_local_path
