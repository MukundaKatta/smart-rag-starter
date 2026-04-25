"""Tests for the `srag` command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smart_rag.cli import main


def _write_corpus(root: Path) -> Path:
    docs = root / "docs"
    docs.mkdir()
    (docs / "refunds.md").write_text(
        "Our refund policy allows refunds within 30 days of purchase.\n"
        "Contact support to start a refund request.\n",
        encoding="utf-8",
    )
    (docs / "cats.md").write_text(
        "Cats are small, carnivorous mammals often kept as pets.\n",
        encoding="utf-8",
    )
    (docs / "python.md").write_text(
        "Python is a high-level, general-purpose programming language.\n",
        encoding="utf-8",
    )
    return docs


def test_retrieve_prints_prompt(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    docs = _write_corpus(tmp_path)
    rc = main(
        [
            "retrieve",
            "refund policy",
            str(docs / "refunds.md"),
            str(docs / "cats.md"),
            str(docs / "python.md"),
            "--k",
            "2",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert out.strip(), "expected non-empty stdout"
    assert "refund policy" in out
    assert "Sources:" in out
    assert "[1]" in out


def test_retrieve_json_emits_valid_payload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    docs = _write_corpus(tmp_path)
    rc = main(
        [
            "retrieve",
            "refund policy",
            str(docs),  # directory path: should be expanded
            "--k",
            "2",
            "--json",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["query"] == "refund policy"
    assert payload["k"] == 2
    assert isinstance(payload["results"], list) and payload["results"]
    assert "prompt" in payload and isinstance(payload["prompt"], str)
    # The refunds doc should be in the results.
    assert any("refunds.md" in r["source"] for r in payload["results"])


def test_retrieve_with_no_documents_returns_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    rc = main(["retrieve", "anything", str(tmp_path / "nope-*.md")])
    err = capsys.readouterr().err
    assert rc == 2
    assert "no documents" in err


def test_version_flag(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    # argparse exits 0 for --version.
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "srag" in out
