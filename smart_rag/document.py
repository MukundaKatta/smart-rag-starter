"""Document loading utilities.

Supports plain text and markdown files. Markdown files with a simple YAML
frontmatter block (delimited by lines containing only `---`) have the
frontmatter stripped from the body and exposed as `Document.metadata`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List


SUPPORTED_EXTENSIONS = (".txt", ".md", ".markdown")


@dataclass
class Document:
    """A loaded source document."""

    source: str
    text: str
    metadata: Dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)


def _parse_frontmatter(raw: str) -> "tuple[Dict[str, str], str]":
    """Strip a YAML-ish frontmatter block from `raw`.

    This intentionally implements a minimal subset (key: value lines) so the
    project has zero YAML dependency. Anything more exotic is left in the body.
    """
    stripped = raw.lstrip("﻿")
    if not stripped.startswith("---"):
        return {}, raw
    lines = stripped.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw
    end_index = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_index = i
            break
    if end_index == -1:
        return {}, raw
    metadata: Dict[str, str] = {}
    for line in lines[1:end_index]:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        metadata[key.strip()] = value.strip().strip('"').strip("'")
    body = "\n".join(lines[end_index + 1 :]).lstrip("\n")
    return metadata, body


def load_file(path: str | os.PathLike) -> Document:
    """Load a single supported file into a `Document`."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"No such file: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported extension {file_path.suffix!r}. "
            f"Supported: {SUPPORTED_EXTENSIONS}"
        )
    raw = file_path.read_text(encoding="utf-8")
    metadata, body = _parse_frontmatter(raw)
    metadata.setdefault("filename", file_path.name)
    return Document(source=str(file_path), text=body, metadata=metadata)


def load_directory(
    directory: str | os.PathLike,
    extensions: Iterable[str] = SUPPORTED_EXTENSIONS,
    recursive: bool = True,
) -> List[Document]:
    """Load all supported files in a directory into `Document` objects."""
    base = Path(directory)
    if not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {base}")
    pattern = "**/*" if recursive else "*"
    docs: List[Document] = []
    for path in sorted(base.glob(pattern)):
        if path.is_file() and path.suffix.lower() in extensions:
            docs.append(load_file(path))
    return docs
