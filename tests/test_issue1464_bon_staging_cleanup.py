"""Regression tests for Windows best-of-n offline staging cleanup (#1464)."""

from __future__ import annotations

import json
import os
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from soup_cli.utils.best_of_n_stream import stage_offline_datasets


def test_stage_offline_datasets_file_in_place_of_emit_pairs_dir_cleans_up(tmp_path, monkeypatch):
    """A file in place of emit-pairs dir must raise original error without leftovers (#1464)."""
    monkeypatch.chdir(tmp_path)

    sft_out = str(tmp_path / "sft.jsonl")
    bad_parent = tmp_path / "bad_dir_file"
    bad_parent.write_text("not a directory", encoding="utf-8")
    dpo_out = str(bad_parent / "dpo.jsonl")

    index = Mock()
    index.iter_rows = Mock(return_value=[])

    with pytest.raises(OSError) as excinfo:
        stage_offline_datasets(index, sft_out, dpo_out)

    # Must raise the original filesystem error (e.g. FileExistsError or NotADirectoryError),
    # never a masked Windows PermissionError [WinError 32].
    assert not isinstance(excinfo.value, PermissionError), (
        f"Expected filesystem error, got PermissionError: {excinfo.value}"
    )
    if os.name == "nt":
        # On Windows, os.makedirs over a file raises FileExistsError ([WinError 183])
        assert isinstance(excinfo.value, (FileExistsError, NotADirectoryError))
        assert "183" in str(excinfo.value) or "already exists" in str(excinfo.value).lower()

    leftovers = [str(p) for p in tmp_path.glob(".soup.group.*.tmp")]
    assert not leftovers, f"Leftover staging files were not cleaned up: {leftovers}"


def test_stage_offline_datasets_stream_failure_removes_both_staging_files(tmp_path, monkeypatch):
    """When row iteration fails, both SFT and DPO temporary files are unlinked (#1464)."""
    monkeypatch.chdir(tmp_path)

    sft_out = str(tmp_path / "sft.jsonl")
    dpo_out = str(tmp_path / "dpo.jsonl")

    index = Mock()

    def _fail_iter():
        raise RuntimeError("simulated stream failure")
        yield  # make it a generator

    index.iter_rows = _fail_iter

    with pytest.raises(RuntimeError, match="simulated stream failure"):
        stage_offline_datasets(index, sft_out, dpo_out)

    leftovers = [str(p) for p in tmp_path.glob(".soup.group.*.tmp")]
    assert not leftovers, f"Leftover staging files were not cleaned up: {leftovers}"


def test_cli_best_of_n_offline_bad_emit_pairs_reports_real_error_and_cleans_up(
    tmp_path, monkeypatch
):
    """CLI run with blocked emit-pairs path shows real error and leaves no tmp files (#1464)."""
    from soup_cli.commands.data import app

    monkeypatch.chdir(tmp_path)

    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(json.dumps({"prompt": "hello world"}) + "\n", encoding="utf-8")

    calls = []

    def factory(*_args, **_kwargs):
        def generate(prompt):
            calls.append(prompt)
            return f"candidate-{len(calls)}"

        return generate

    monkeypatch.setattr("soup_cli.utils.magpie.make_magpie_generate_fn", factory)
    artifact = tmp_path / "candidates.jsonl"
    export_res = CliRunner().invoke(
        app,
        [
            "best-of-n",
            "--provider",
            "ollama",
            "--model",
            "sampler-model",
            "--base-url",
            "http://localhost:11434",
            "--prompts",
            str(prompt_path),
            "--n",
            "2",
            "--export-candidates",
            str(artifact),
        ],
    )
    assert export_res.exit_code == 0, export_res.output

    # Write judgment file
    groups = [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines()][1:]
    judgments = tmp_path / "judgments.jsonl"
    judgments.write_text(
        "".join(
            json.dumps(
                {
                    "prompt_id": g["prompt_id"],
                    "group_digest": g["group_digest"],
                    "winner_idx": 0,
                    "scores": [0.9, 0.1],
                    "verifier": {"name": "test-verifier", "version": "v1"},
                }
            )
            + "\n"
            for g in groups
        ),
        encoding="utf-8",
    )

    bad_parent = tmp_path / "bad_emit_parent_file"
    bad_parent.write_text("i am a file", encoding="utf-8")
    blocked_emit_pairs = str(bad_parent / "pairs.jsonl")

    res = CliRunner().invoke(
        app,
        [
            "best-of-n",
            "--candidate-artifact",
            str(artifact),
            "--judgments",
            str(judgments),
            "--output",
            str(tmp_path / "sft.jsonl"),
            "--emit-pairs",
            blocked_emit_pairs,
        ],
    )

    assert res.exit_code == 2
    # The real filesystem error must be reported, never WinError 32
    assert "[WinError 32]" not in res.output
    if os.name == "nt":
        assert "183" in res.output or "already exists" in res.output.lower()

    leftovers = [str(p) for p in tmp_path.glob(".soup.group.*.tmp")]
    assert not leftovers, f"Leftover staging files in directory: {leftovers}"


def test_stage_offline_datasets_second_fdopen_failure_closes_dpo_fd_and_cleans_up(
    tmp_path, monkeypatch
):
    """When the second os.fdopen fails, dpo_fd is closed before unlinking (#1464)."""
    monkeypatch.chdir(tmp_path)

    sft_out = str(tmp_path / "sft.jsonl")
    dpo_out = str(tmp_path / "dpo.jsonl")

    index = Mock()
    index.iter_rows = Mock(return_value=[])

    orig_fdopen = os.fdopen
    calls = 0

    def _mock_fdopen(fd, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated second fdopen failure")
        return orig_fdopen(fd, *args, **kwargs)

    monkeypatch.setattr(os, "fdopen", _mock_fdopen)

    with pytest.raises(OSError) as excinfo:
        stage_offline_datasets(index, sft_out, dpo_out)

    assert "simulated second fdopen failure" in str(excinfo.value)
    assert not isinstance(excinfo.value, PermissionError), (
        f"Expected real OSError, got PermissionError: {excinfo.value}"
    )

    leftovers = [str(p) for p in tmp_path.glob(".soup.group.*.tmp")]
    assert not leftovers, f"Leftover staging files were not cleaned up: {leftovers}"


def test_stage_offline_datasets_unlinks_after_descriptors_closed_cross_platform(
    tmp_path, monkeypatch
):
    """Staging files must have their descriptors closed before unlinking (#1464)."""
    import tempfile

    monkeypatch.chdir(tmp_path)

    sft_out = str(tmp_path / "sft.jsonl")
    dpo_out = str(tmp_path / "dpo.jsonl")

    tracked_fds: dict[int, str] = {}
    orig_mkstemp = tempfile.mkstemp

    def _tracking_mkstemp(*args, **kwargs):
        fd, path = orig_mkstemp(*args, **kwargs)
        tracked_fds[fd] = os.path.abspath(path)
        return fd, path

    monkeypatch.setattr(tempfile, "mkstemp", _tracking_mkstemp)

    orig_unlink = os.unlink
    unlinked_paths: list[str] = []

    def _spying_unlink(path):
        abs_p = os.path.abspath(path)
        unlinked_paths.append(abs_p)
        for fd, fpath in tracked_fds.items():
            if fpath == abs_p:
                try:
                    os.fstat(fd)
                    pytest.fail(f"File {path} was unlinked while descriptor {fd} was still open!")
                except OSError:
                    # Descriptor is closed as required
                    pass
        return orig_unlink(path)

    monkeypatch.setattr(os, "unlink", _spying_unlink)

    index = Mock()

    def _fail_iter():
        raise RuntimeError("simulated stream failure")
        yield

    index.iter_rows = _fail_iter

    with pytest.raises(RuntimeError, match="simulated stream failure"):
        stage_offline_datasets(index, sft_out, dpo_out)

    assert len(unlinked_paths) == 2, f"Expected 2 unlinked files, got {unlinked_paths}"
    leftovers = [str(p) for p in tmp_path.glob(".soup.group.*.tmp")]
    assert not leftovers, f"Leftover staging files were not cleaned up: {leftovers}"
