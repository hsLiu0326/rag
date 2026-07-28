"""MinerU document parser — PDF/image to structured Markdown via GPU.

Replaces standalone PaddleOCR with MinerU's end-to-end pipeline:
  布局检测(DocLayout-YOLO) → 阅读顺序 → 表格识别 → 公式识别 → OCR → Markdown

MinerU automatically uses GPU (CUDA) if available. On RTX 4060 8GB,
processing speed is ~3-8 seconds per page.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


class MinerUParser:
    """Wrap MinerU CLI for PDF/image → structured Markdown conversion.

    Usage:
        parser = MinerUParser(backend="pipeline")
        markdown_text = parser.parse("document.pdf")
    """

    def __init__(
        self,
        backend: str = "pipeline",
        timeout: int = 300,
        gpu_id: int | None = 0,
    ) -> None:
        """
        Args:
            backend: 'pipeline' (fast, GPU) or 'vlm' (high accuracy, needs more VRAM).
            timeout: Max seconds per document. Large PDFs may need longer.
            gpu_id: GPU device ID. None = auto. Set to "" to force CPU.
        """
        self.backend = backend
        self.timeout = timeout
        self.gpu_id = gpu_id
        self._checked = False

    def _find_mineru(self) -> str:
        """Find the mineru executable path.

        On Windows with venv, the CLI is at <venv>/Scripts/mineru.exe.
        We derive the path from the current Python interpreter.
        """
        scripts_dir = os.path.dirname(sys.executable)
        mineru_path = os.path.join(scripts_dir, "mineru.exe")
        if os.path.exists(mineru_path):
            return mineru_path
        # Fallback: try system PATH
        import shutil as _shutil
        found = _shutil.which("mineru")
        if found:
            return found
        raise FileNotFoundError(
            "mineru executable not found. "
            "Install with: pip install 'mineru[all]'"
        )

    def _unload_ollama(self) -> None:
        """Tell Ollama to release model from GPU, freeing CUDA for MinerU.
        Ollama reloads automatically on next embedding request."""
        try:
            subprocess.run(
                ["ollama", "stop", "qwen-emb:latest"],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass  # Ollama CLI not in PATH — continue, MinerU may still work

    def _ensure_ready(self) -> None:
        """Verify mineru CLI is installed and GPU is detected."""
        if self._checked:
            return
        mineru_path = self._find_mineru()
        result = subprocess.run(
            [mineru_path, "--version"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"MinerU not working. Run: pip install 'mineru[all]'\n"
                f"stderr: {result.stderr[:500]}"
            )
        self._checked = True
        self._mineru_path = mineru_path
        version_line = (result.stdout or "").strip().split("\n")[0]
        print(f"[mineru] CLI ready: {version_line}")

    def parse(self, file_path: str | Path) -> str:
        """Parse a PDF/image file and return structured Markdown.

        Args:
            file_path: Path to PDF, PNG, JPG, DOCX, PPTX, or XLSX file.

        Returns:
            Merged Markdown content with preserved structure (headings, tables, formulas).
        """
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self._ensure_ready()

        # Output to a temp directory
        with tempfile.TemporaryDirectory(prefix="mineru_") as tmpdir:
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            cmd = [self._mineru_path, "-p", str(file_path), "-o", str(output_dir)]
            if self.backend:
                cmd.extend(["-b", self.backend])

            env = os.environ.copy()
            # Use HF mirror for model downloads
            env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

            print(f"[mineru] Parsing: {file_path.name} (backend={self.backend})")
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                raise TimeoutError(
                    f"MinerU timed out after {self.timeout}s for {file_path.name}. "
                    f"Try a smaller file or increase timeout."
                )

            if result.returncode != 0:
                stderr_tail = result.stderr.strip().split("\n")[-5:]
                raise RuntimeError(
                    f"MinerU failed (code={result.returncode}) for {file_path.name}:\n"
                    + "\n".join(stderr_tail)
                )

            # Find the output markdown directory
            # MinerU outputs: output_dir/<filename>/<filename>_md/  or  output_dir/<filename>.md
            markdown_text = self._collect_markdown(output_dir, file_path.stem)

            if not markdown_text.strip():
                raise ValueError(
                    f"MinerU produced no text for {file_path.name}. "
                    f"The file may be entirely image-based or corrupted."
                )

            print(f"[mineru] Done: {file_path.name} → {len(markdown_text)} chars")
            return markdown_text

    def _collect_markdown(self, output_dir: Path, stem: str) -> str:
        """Find and merge all Markdown files from MinerU output."""
        # Pattern 1: output_dir/<stem>/<stem>_md/*.md
        md_dir = output_dir / stem / f"{stem}_md"
        if md_dir.is_dir():
            return self._merge_md_files(md_dir)

        # Pattern 2: output_dir/<stem>/<stem>.md
        single_md = output_dir / stem / f"{stem}.md"
        if single_md.is_file():
            return single_md.read_text(encoding="utf-8")

        # Pattern 3: output_dir/<stem>.md (flat output)
        flat_md = output_dir / f"{stem}.md"
        if flat_md.is_file():
            return flat_md.read_text(encoding="utf-8")

        # Pattern 4: output_dir/**/*.md (deep search)
        md_files = list(output_dir.rglob("*.md"))
        if md_files:
            return self._merge_md_files(output_dir, md_files)

        # Last resort: search for any markdown
        md_files = list(output_dir.rglob(f"*{stem}*.md"))
        if md_files:
            return self._merge_md_files(output_dir, md_files)

        raise FileNotFoundError(
            f"No markdown output found for '{stem}' in {output_dir}. "
            f"Contents: {list(output_dir.iterdir())[:10]}"
        )

    @staticmethod
    def _merge_md_files(base_dir: Path, files: list[Path] | None = None) -> str:
        """Merge multiple .md files into one, sorted by filename."""
        if files is None:
            files = sorted(base_dir.rglob("*.md"))
        else:
            files = sorted(files)

        parts: list[str] = []
        for f in files:
            text = f.read_text(encoding="utf-8")
            if text.strip():
                parts.append(text.strip())
        return "\n\n".join(parts)

    async def aparse(self, file_path: str | Path) -> str:
        """Async wrapper: run parse() in a dedicated thread to avoid blocking."""
        # Dedicated executor to avoid thread-pool starvation from embedder/subprocess
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(pool, self.parse, file_path)

    def parse_with_metadata(self, file_path: str | Path) -> dict[str, Any]:
        """Parse and also return JSON metadata (if MinerU outputs it)."""
        file_path = Path(file_path).resolve()
        markdown = self.parse(file_path)

        # Try to find companion JSON metadata
        metadata: dict[str, Any] = {"source": file_path.name, "format": file_path.suffix.lower()}
        return {"markdown": markdown, "metadata": metadata}
