"""Unit tests for scripts/check_docker_latest.py (#2195).

Ensures that the `latest` Docker tag is only applied when releasing the highest
semver tag of the repository, preventing security patch releases on older lines
(such as v0.76.2 when v0.77.0 is current) from displacing `latest`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure scripts/ directory is on sys.path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from check_docker_latest import is_latest_release, main, parse_semver  # noqa: E402


def test_parse_semver():
    assert parse_semver("v0.77.0") == (0, 77, 0)
    assert parse_semver("v1.2.3") == (1, 2, 3)
    assert parse_semver("0.76.2") == (0, 76, 2)
    assert parse_semver("  v2.0.1  ") == (2, 0, 1)

    assert parse_semver("v0.77.0-rc1") is None
    assert parse_semver("v0.77.0.1") is None
    assert parse_semver("main") is None
    assert parse_semver("") is None
    assert parse_semver("alpha") is None


def test_is_latest_release_newest_tag():
    tags = ["v0.77.0", "v0.76.2", "v0.76.1", "v0.75.0"]
    assert is_latest_release("v0.77.0", tags) is True


def test_is_latest_release_higher_than_all_existing():
    tags = ["v0.77.0", "v0.76.2", "v0.75.0"]
    assert is_latest_release("v0.78.0", tags) is True
    assert is_latest_release("v0.77.1", tags) is True


def test_is_latest_release_older_line_patch_refused():
    """Security patch release on older line must not be marked as latest."""
    tags = ["v0.77.0", "v0.76.1", "v0.75.0"]
    assert is_latest_release("v0.76.2", tags) is False
    assert is_latest_release("v0.75.1", tags) is False


def test_is_latest_release_ignores_non_semver_tags():
    tags = ["v0.77.0", "v0.78.0-rc1", "exp-v0.52.467", "release-2026", "invalid"]
    assert is_latest_release("v0.77.0", tags) is True
    assert is_latest_release("v0.76.0", tags) is False


def test_is_latest_release_empty_repo_tags():
    assert is_latest_release("v0.77.0", []) is True


def test_is_latest_release_invalid_candidate_raises():
    with pytest.raises(ValueError, match="not a valid vX.Y.Z semver tag"):
        is_latest_release("invalid-tag", ["v0.77.0"])


def test_cli_latest_tag_writes_github_output(tmp_path, monkeypatch, capsys):
    output_file = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    with patch("check_docker_latest.get_repo_tags", return_value=["v0.77.0", "v0.76.2"]):
        with patch.object(sys, "argv", ["check_docker_latest.py", "v0.77.0"]):
            rc = main()
            assert rc == 0

    assert "is_latest=true" in output_file.read_text(encoding="utf-8")
    captured = capsys.readouterr()
    assert "Tag: v0.77.0 | is_latest: true" in captured.out


def test_cli_older_tag_writes_github_output(tmp_path, monkeypatch, capsys):
    output_file = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))

    with patch("check_docker_latest.get_repo_tags", return_value=["v0.77.0", "v0.76.1"]):
        with patch.object(sys, "argv", ["check_docker_latest.py", "v0.76.2"]):
            rc = main()
            assert rc == 0

    assert "is_latest=false" in output_file.read_text(encoding="utf-8")
    captured = capsys.readouterr()
    assert "Tag: v0.76.2 | is_latest: false" in captured.out


def test_cli_env_var_fallback(tmp_path, monkeypatch):
    output_file = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output_file))
    monkeypatch.setenv("REF_NAME", "v0.77.0")

    with patch("check_docker_latest.get_repo_tags", return_value=["v0.77.0", "v0.76.0"]):
        with patch.object(sys, "argv", ["check_docker_latest.py"]):
            rc = main()
            assert rc == 0

    assert "is_latest=true" in output_file.read_text(encoding="utf-8")


def test_cli_missing_arg_exits_1(capsys):
    with patch.dict(os.environ, {}, clear=True):
        with patch.object(sys, "argv", ["check_docker_latest.py"]):
            rc = main()
            assert rc == 1
    captured = capsys.readouterr()
    assert "Usage: check_docker_latest.py <tag>" in captured.err
