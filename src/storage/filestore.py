"""File persistence layer — local disk storage."""

from __future__ import annotations

import shutil
from pathlib import Path


class FileStore:
    """Manages raw file upload storage."""

    def __init__(self, base_dir: str = "./data/uploads") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, content: bytes, filename: str) -> Path:
        """Save raw bytes to disk. Returns file path."""
        file_path = self.base_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)
        return file_path

    def delete(self, filename: str) -> bool:
        file_path = self.base_dir / filename
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def exists(self, filename: str) -> bool:
        return (self.base_dir / filename).exists()
