"""OCR engine for scanned documents (PaddleOCR, CPU mode)."""

from __future__ import annotations

from pathlib import Path


class OCREngine:
    """PaddleOCR wrapper for scanned PDFs and images."""

    def __init__(self, use_gpu: bool = False) -> None:
        self._use_gpu = use_gpu
        self._ocr = None

    @property
    def ocr(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang="ch",
                use_gpu=self._use_gpu,
                show_log=False,
            )
        return self._ocr

    def extract_text(self, image_path: str | Path) -> str:
        """Run OCR on an image file, return extracted text."""
        result = self.ocr.ocr(str(image_path), cls=True)
        if not result or not result[0]:
            return ""

        lines: list[str] = []
        for line_group in result:
            for line_info in line_group:
                text = line_info[1][0]
                confidence = line_info[1][1]
                if confidence > 0.5:
                    lines.append(text)
        return "\n".join(lines)

    def extract_text_from_pdf_pages(self, pdf_path: str | Path) -> str:
        """Convert scanned PDF pages to images, then OCR each page.

        Uses pdf2image (requires poppler on Linux, or we fall back to
        extracting page images with pypdf and OCR'ing those).
        """
        from pypdf import PdfReader
        import tempfile
        import os

        reader = PdfReader(str(pdf_path))
        all_text: list[str] = []

        # For scanned PDFs, we attempt to extract embedded images
        # and run OCR on each. If no images, return empty.
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                all_text.append(text)
                continue

            # Try to extract images from the page
            images = []
            if hasattr(page, "images") and page.images:
                images = list(page.images)

            if not images:
                continue

            for j, img in enumerate(images):
                img_bytes = img.data
                with tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False
                ) as tmp:
                    tmp.write(img_bytes)
                    tmp_path = tmp.name
                try:
                    ocr_text = self.extract_text(tmp_path)
                    if ocr_text:
                        all_text.append(ocr_text)
                finally:
                    os.unlink(tmp_path)

        return "\n".join(all_text)

    @staticmethod
    def needs_ocr(text: str) -> bool:
        """Heuristic: if extracted text is too short, likely scanned PDF."""
        return len(text.strip()) < 100
