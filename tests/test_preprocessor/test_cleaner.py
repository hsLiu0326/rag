"""Tests for the pluggable text cleaning pipeline."""

from src.core.preprocessor.cleaner import (
    TextCleaner,
    DEFAULT_RULES,
    collapse_blank_lines,
    fix_merged_words,
    normalize_table_spacing,
    normalize_unicode,
    strip_html_tags,
    trim_trailing_spaces,
)


def test_default_rule_set_has_six_rules():
    assert len(DEFAULT_RULES) == 6


def test_cleaner_reports_rule_names():
    cleaner = TextCleaner()
    names = cleaner.rules
    assert len(names) == 6
    assert "normalize_unicode" in names
    assert "collapse_blank_lines" in names


def test_normalize_unicode_replaces_artifacts():
    text = "a\u00a0b \u2013 \u2014 \u2018x\u2019 \u201cy\u201d \u2026"
    cleaned = normalize_unicode(text)
    assert cleaned == "a b -- —— 'x' \"y\" ..."


def test_strip_html_tags():
    assert strip_html_tags("<p>正文<b>加粗</b></p>") == "正文加粗"


def test_trim_trailing_spaces():
    assert trim_trailing_spaces("行一  \n行二\t\n行三") == "行一\n行二\n行三"


def test_collapse_blank_lines():
    assert collapse_blank_lines("a\n\n\n\n\nb") == "a\n\nb"


def test_fix_merged_words_adds_spaces_around_cjk_and_latin():
    assert fix_merged_words("中文ABC测试") == "中文 ABC 测试"
    assert fix_merged_words("Python和Java") == "Python 和 Java"


def test_normalize_table_spacing():
    text = "说明\n| A | B |\n| --- | --- |\n| 1 | 2 |\n结尾"
    cleaned = normalize_table_spacing(text)
    # Exactly one blank line before and after the table
    assert "说明\n\n| A | B |" in cleaned
    assert "| 1 | 2 |\n\n结尾" in cleaned


def test_clean_applies_all_rules_in_order():
    raw = "<p>中文Python测试\u00a0\u2026</p>\n\n\n\n  尾部空格  \n"
    cleaned = TextCleaner().clean(raw)
    assert "中文 Python 测试 ..." in cleaned
    assert "<p>" not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned.rstrip().endswith("尾部空格")


def test_clean_tolerates_broken_rule():
    cleaner = TextCleaner()

    def broken(text):
        raise RuntimeError("boom")

    cleaner.add_rule(broken)
    assert cleaner.clean("正常文本") == "正常文本"


def test_add_and_remove_rule():
    cleaner = TextCleaner()
    name_before = len(cleaner.rules)
    cleaner.add_rule(str.upper)
    assert len(cleaner.rules) == name_before + 1
    assert cleaner.remove_rule("upper") is True
    assert len(cleaner.rules) == name_before
    assert cleaner.remove_rule("does_not_exist") is False
