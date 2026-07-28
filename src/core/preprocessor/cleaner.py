"""Text cleaner — pluggable rules for post-MinerU / post-load text cleanup."""

from __future__ import annotations

import re
from typing import Callable

# ── Rule type: Callable[[str] -> str] ────────────────────────────────
CleanRule = Callable[[str], str]

# ── Built-in Rules ────────────────────────────────────────────────────


def collapse_blank_lines(text: str, max_consecutive: int = 2) -> str:
    """Collapse excessive blank lines to at most `max_consecutive`."""
    return re.sub(r"\n{" + str(max_consecutive + 1) + r",}", "\n" * max_consecutive, text)


def normalize_unicode(text: str) -> str:
    """Replace common encoding artifacts with proper characters."""
    replacements = {
        " ": " ",      # non-breaking space → space
        "–": "--",     # en dash
        "—": "——",     # em dash
        "‘": "'",      # left single quote
        "’": "'",      # right single quote
        "“": '"',      # left double quote
        "”": '"',      # right double quote
        "…": "...",    # ellipsis
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def trim_trailing_spaces(text: str) -> str:
    """Remove trailing whitespace from each line."""
    return "\n".join(line.rstrip() for line in text.split("\n"))


def strip_html_tags(text: str) -> str:
    """Remove HTML tags that may leak through from DOCX/PDF parsing."""
    return re.sub(r"<[^>]+>", "", text)


def normalize_table_spacing(text: str) -> str:
    """Ensure exactly one blank line before/after markdown tables."""
    # Tables start with | ... | followed by |---|
    lines = text.split("\n")
    out: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        is_table_line = stripped.startswith("|") and stripped.endswith("|")
        prev_is_table = i > 0 and out[-1].strip().startswith("|") if out else False

        if is_table_line and not prev_is_table and out and out[-1] != "":
            out.append("")  # blank line before table
        elif not is_table_line and prev_is_table and stripped:
            out.append("")  # blank line after table

        out.append(line)

    # Deduplicate blank lines
    result = "\n".join(out)
    return collapse_blank_lines(result, max_consecutive=2)


def fix_merged_words(text: str) -> str:
    """Fix CJK+Latin word merging: add space between Chinese char and Latin word."""
    # Chinese char followed by Latin
    text = re.sub(r"([一-鿿])([A-Za-z0-9])", r"\1 \2", text)
    # Latin followed by Chinese char
    text = re.sub(r"([A-Za-z0-9])([一-鿿])", r"\1 \2", text)
    return text


# ── Default Rule Set ──────────────────────────────────────────────────

DEFAULT_RULES: list[CleanRule] = [
    normalize_unicode,
    strip_html_tags,
    trim_trailing_spaces,
    normalize_table_spacing,
    fix_merged_words,
    collapse_blank_lines,
]

# ── Cleaner ───────────────────────────────────────────────────────────


class TextCleaner:
    """Apply ordered cleaning rules to raw text.

    Usage:
        cleaner = TextCleaner(rules=DEFAULT_RULES)
        clean_text = cleaner.clean(raw_text)

        # Add custom rules:
        cleaner.add_rule(my_custom_rule, position=0)  # insert at front
        cleaner.remove_rule("collapse_blank_lines")    # remove by name
    """

    def __init__(self, rules: list[CleanRule] | None = None) -> None:
        self._rules: list[CleanRule] = list(rules or DEFAULT_RULES)

    @property
    def rules(self) -> list[str]:
        """Return current rule names for inspection."""
        return [r.__name__ for r in self._rules]

    def clean(self, text: str) -> str:
        """Apply all rules in order, return cleaned text."""
        for rule in self._rules:
            try:
                text = rule(text)
            except Exception:
                pass  # A single broken rule shouldn't fail the whole pipeline
        return text

    def add_rule(self, rule: CleanRule, position: int | None = None) -> None:
        """Add a rule. position=None appends to end."""
        if position is None:
            self._rules.append(rule)
        else:
            self._rules.insert(position, rule)

    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by function name. Returns True if removed."""
        for i, r in enumerate(self._rules):
            if r.__name__ == rule_name:
                self._rules.pop(i)
                return True
        return False
