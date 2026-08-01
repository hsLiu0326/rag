"""Hierarchical Markdown chunker — H1/H2/H3 aware splitting.

Produces parent (~1024 token) and child (~256 token) chunk pairs
for the parent-document retrieval architecture.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

import tiktoken

from src.models.document import Section, Chunk, ChunkPair


class HierarchicalChunker:
    """Splits Markdown into parent-child chunk pairs based on heading hierarchy."""

    HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

    def __init__(
        self,
        parent_size: int = 1024,
        child_size: int = 256,
        overlap: int = 32,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = overlap
        try:
            self._enc = tiktoken.get_encoding(encoding_name)
        except Exception:
            self._enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def split(self, markdown_text: str, base_metadata: dict[str, Any]) -> list[ChunkPair]:
        """Main entry: parse heading tree, then split into ChunkPairs."""
        sections = self._parse_sections(markdown_text)
        if not sections:
            # Fallback: no headings found -> uniform split
            return self._uniform_split(markdown_text, base_metadata)

        pairs: list[ChunkPair] = []
        for section in sections:
            pairs.extend(self._split_section(section, base_metadata))
        return pairs

    # ── Section tree parsing ────────────────────────────────────────

    def _parse_sections(self, text: str) -> list[Section]:
        """Build section tree from Markdown headings."""
        matches = list(self.HEADING_PATTERN.finditer(text))
        if not matches:
            return []

        sections: list[Section] = []
        for i, m in enumerate(matches):
            level = len(m.group(1))
            title = m.group(2).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()

            section = Section(
                level=level,
                title=title,
                content=content,
                start_offset=start,
                end_offset=end,
            )

            # Build hierarchy: find the nearest parent section
            parent = self._find_parent(sections, level)
            if parent is not None:
                parent.children.append(section)
            else:
                sections.append(section)

        return sections

    @staticmethod
    def _find_parent(sections: list[Section], level: int) -> Section | None:
        """Find the nearest ancestor whose heading level is strictly lower.

        Walks the section tree built so far and descends into children so that
        an H3 is attached under its own H2 rather than jumping straight to the
        H1 (or top-level) section.
        """
        best: Section | None = None
        for s in sections:
            if s.level < level:
                # A candidate ancestor; prefer a deeper one inside its subtree
                best = HierarchicalChunker._find_parent(s.children, level) or s
        return best

    # ── Splitting logic ─────────────────────────────────────────────

    def _split_section(self, section: Section, meta: dict[str, Any]) -> list[ChunkPair]:
        """Recursively split a section into ChunkPairs."""
        pairs: list[ChunkPair] = []

        # Update metadata with title path
        local_meta = dict(meta)
        local_meta[f"h{section.level}"] = section.title

        if section.children:
            for child in section.children:
                pairs.extend(self._split_section(child, local_meta))
            return pairs

        # Leaf section: split content
        content = section.content
        if not content.strip():
            return pairs

        tokens = self.count_tokens(content)
        if tokens <= self.parent_size:
            # Single parent → multiple children
            pair = self._make_pair(content, local_meta)
            if pair:
                pairs.append(pair)
        else:
            # Split large section into multiple parent chunks
            paragraphs = self._split_by_paragraphs(content)
            buffer = ""
            for para in paragraphs:
                if self.count_tokens(buffer + para) > self.parent_size and buffer:
                    pair = self._make_pair(buffer.strip(), local_meta)
                    if pair:
                        pairs.append(pair)
                    buffer = para
                else:
                    buffer += ("\n\n" if buffer else "") + para
            if buffer.strip():
                pair = self._make_pair(buffer.strip(), local_meta)
                if pair:
                    pairs.append(pair)

        return pairs

    def _make_pair(self, parent_text: str, meta: dict[str, Any]) -> ChunkPair | None:
        """Create a ChunkPair from parent text: one parent → N children."""
        if not parent_text.strip():
            return None

        parent_id = uuid.uuid4().hex
        token_count = self.count_tokens(parent_text)

        parent_meta = {
            **meta,
            "parent_id": parent_id,
            "title_path": self._build_title_path(meta),
            "token_count": token_count,
        }

        # Generate child chunks
        children = self._split_children(parent_text, parent_id, parent_meta)

        return ChunkPair(
            parent_id=parent_id,
            parent_text=parent_text,
            parent_metadata=parent_meta,
            children=children,
        )

    def _split_children(
        self, text: str, parent_id: str, meta: dict[str, Any]
    ) -> list[Chunk]:
        """Split parent text into child chunks with overlap."""
        tokens = self.count_tokens(text)
        if tokens <= self.child_size:
            chunk_id = uuid.uuid4().hex
            return [Chunk(
                chunk_id=chunk_id,
                text=text,
                metadata={
                    **meta,
                    "parent_id": parent_id,
                    "chunk_type": "child",
                    "chunk_index": 0,
                },
            )]

        children: list[Chunk] = []
        paragraphs = self._split_by_paragraphs(text)

        buffer = ""
        index = 0
        for para in paragraphs:
            tentative = buffer + ("\n\n" if buffer else "") + para
            if self.count_tokens(tentative) > self.child_size and buffer:
                chunk_id = uuid.uuid4().hex
                children.append(Chunk(
                    chunk_id=chunk_id,
                    text=buffer,
                    metadata={
                        **meta,
                        "parent_id": parent_id,
                        "chunk_type": "child",
                        "chunk_index": index,
                    },
                ))
                # Overlap: carry last ~overlap tokens
                index += 1
                buffer = para
            else:
                buffer = tentative

        if buffer.strip():
            chunk_id = uuid.uuid4().hex
            children.append(Chunk(
                chunk_id=chunk_id,
                text=buffer,
                metadata={
                    **meta,
                    "parent_id": parent_id,
                    "chunk_type": "child",
                    "chunk_index": index,
                },
            ))

        return children

    # ── Fallback: uniform splitting without headings ─────────────────

    def _uniform_split(self, text: str, meta: dict[str, Any]) -> list[ChunkPair]:
        """Fallback: split text uniformly when no headings exist."""
        paragraphs = self._split_by_paragraphs(text)
        if not paragraphs:
            return []

        pairs: list[ChunkPair] = []
        parent_buffer = ""
        group_count = 0

        for para in paragraphs:
            tentative = parent_buffer + ("\n\n" if parent_buffer else "") + para
            if self.count_tokens(tentative) > self.parent_size and parent_buffer:
                pair = self._make_pair(parent_buffer.strip(), meta)
                if pair:
                    pairs.append(pair)
                parent_buffer = para
                group_count += 1
            else:
                parent_buffer = tentative

        if parent_buffer.strip():
            pair = self._make_pair(parent_buffer.strip(), meta)
            if pair:
                pairs.append(pair)

        return pairs

    # ── Utilities ───────────────────────────────────────────────────

    @staticmethod
    def _split_by_paragraphs(text: str) -> list[str]:
        """Split text on double newlines (paragraph boundaries)."""
        parts = re.split(r"\n\s*\n", text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _build_title_path(meta: dict[str, Any]) -> str:
        """Build breadcrumb title path like 'H1 > H2 > H3'."""
        parts = []
        for key in ("h1", "h2", "h3"):
            if key in meta and meta[key]:
                parts.append(meta[key])
        return " > ".join(parts)
