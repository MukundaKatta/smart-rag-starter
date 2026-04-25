"""Document loading: read text-ish files (md/txt/py/rst) into Document objects.

Globs are expanded against the filesystem; non-matching patterns are skipped
silently. Each file becomes one Document; binary or otherwise unreadable files
raise during `read_text` with the underlying error.
"""

from __future__ import annotations

import glob
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# Extensions we treat as plain text. Keep the set conservative; users can
# rename/alias other extensions before passing them in.
TEXT_EXTENSIONS = {".md", ".txt", ".py", ".rst"}


@dataclass
class Document:
    """A single loaded text document."""

    id: str
    source: str
    text: str
    metadata: dict = field(default_factory=dict)


def _doc_id(source: str) -> str:
    """Stable short id derived from the source path."""
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]


def _expand(pattern: str | Path) -> list[Path]:
    """Expand a single path-or-glob into concrete file paths.

    A non-glob pattern that points at an existing file returns just that file.
    A glob is expanded with `glob.glob(..., recursive=True)`. A directory is
    walked for any supported text extension.
    """
    s = str(pattern)
    p = Path(s)
    if p.is_file():
        return [p]
    if p.is_dir():
        return [
            child
            for child in sorted(p.rglob("*"))
            if child.is_file() and child.suffix.lower() in TEXT_EXTENSIONS
        ]
    # Treat as glob pattern (supports ** when recursive=True).
    matches = sorted(glob.glob(s, recursive=True))
    return [Path(m) for m in matches if Path(m).is_file()]


def load_documents(paths: Iterable[str | Path]) -> list[Document]:
    """Load text files from one or more paths/globs into `Document` objects.

    Files with extensions outside `TEXT_EXTENSIONS` are skipped so that broad
    globs (for example `**/*`) don't try to read binary files.
    """
    seen: set[str] = set()
    docs: list[Document] = []
    for pattern in paths:
        for file_path in _expand(pattern):
            if file_path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            source = str(file_path)
            if source in seen:
                continue
            seen.add(source)
            text = file_path.read_text(encoding="utf-8", errors="replace")
            docs.append(
                Document(
                    id=_doc_id(source),
                    source=source,
                    text=text,
                    metadata={"filename": file_path.name},
                )
            )
    return docs
