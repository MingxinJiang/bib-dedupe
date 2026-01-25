#! /usr/bin/env python
"""Preparation of title field"""
import html
import re
from typing import Optional, Tuple

import numpy as np
from number_parser import parse
from rapidfuzz import fuzz


TITLE_STOPWORDS = [
    "a",
    "an",
    "the",
    "in",
    "of",
    "on",
    "for",
    "from",
    "does",
    "do",
    "and",
    "are",
    "on",
    "with",
    "to",
    "or",
    "as",
    "by",
    "their",
    "the",
]

def normalize_for_journal_match(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    # Keep only letters; remove spaces, punctuation, digits
    s = re.sub(r"[^a-z]", "", s)
    return s


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (ca != cb)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def _normalized_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    dist = _levenshtein_distance(a, b)
    return 1.0 - dist / max(len(a), len(b))


def _has_doi(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and np.isnan(value):
        return False
    text = str(value).strip()
    return text != "" and text.lower() not in {"nan", "none"}


def mark_title_equals_journal(
    title_array: np.array,
    container_array: np.array,
    doi_array: np.array,
    *,
    pdf_only_dataset: bool,
    min_title_sim: float = 0.9,
) -> tuple[np.array, np.array]:
    """Detect titles that are actually journal names (title vs container_title)."""
    matches = np.zeros(len(title_array), dtype=bool)
    canonical_names = np.array([""] * len(title_array), dtype=object)
    if not pdf_only_dataset:
        return matches, canonical_names
    for idx, (title, container, doi) in enumerate(
        zip(title_array, container_array, doi_array)
    ):
        if _has_doi(doi):
            continue

        title_norm = normalize_for_journal_match(str(title or ""))
        container_norm = normalize_for_journal_match(str(container or ""))
        if not title_norm or not container_norm:
            continue

        sim = fuzz.partial_ratio(title_norm, container_norm) / 100
        if sim >= min_title_sim:
            matches[idx] = True
            canonical_names[idx] = str(container or "")

    return matches, canonical_names


TEMPLATE_PHRASES = (
    r"(?:guest\s+)?editorial",
    r"editor'?s?\s+comments?",
    r"book\s+review",
)

TEMPLATE_PREFIX_RE = re.compile(
    rf"^\s*(?P<lead>.*?)(?P<template>{'|'.join(TEMPLATE_PHRASES)})\b(?P<suffix>.*)$",
    flags=re.IGNORECASE,
)


def _normalize_template_prefix(prefix: str) -> str:
    cleaned = (
        prefix.replace("\u2019", "'").replace("\u2018", "'").lower()
        if prefix
        else ""
    )
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    return " ".join(cleaned.split())


def _is_template_prefix(prefix: str) -> bool:
    normalized = _normalize_template_prefix(prefix)
    if not normalized:
        return False
    if "editorial" in normalized:
        return True
    if "editor" in normalized and "comment" in normalized:
        return True
    if "book" in normalized and "review" in normalized:
        return True
    return False


def _is_template_leadin(lead: str) -> bool:
    if not lead or not lead.strip():
        return True
    stripped = lead.strip()
    if stripped.endswith(("/", "\\")):
        return True
    return bool(re.fullmatch(r"[\s\-/\u2013\u2014:;]*", lead))


def _extract_template_prefix_suffix(title: str) -> Optional[Tuple[str, str]]:
    match = TEMPLATE_PREFIX_RE.match(title)
    if not match:
        return None
    lead = match.group("lead")
    if not _is_template_leadin(lead):
        return None
    prefix = f"{lead}{match.group('template')}".strip()
    suffix = match.group("suffix").strip()
    return prefix, suffix


def strip_template_title(title: str, *, min_suffix_len: int = 10) -> str:
    if not isinstance(title, str) or not title:
        return title
    extracted = _extract_template_prefix_suffix(title)
    if extracted is None:
        return title
    prefix, suffix = extracted
    if not _is_template_prefix(prefix):
        return title
    suffix = re.sub(r"^[\s:;/\-\u2013\u2014]+", "", suffix)
    if len(re.sub(r"\s+", "", suffix)) <= min_suffix_len:
        return title
    return suffix


def remove_erratum_suffix(title: str) -> str:
    erratum_phrases = ["erratum appears in ", "erratum in "]
    for phrase in erratum_phrases:
        if phrase in title.lower():
            title = title[: title.lower().rfind(phrase) - 2]

    title = re.sub(r" review \d+ refs$", "", title)

    return title


# flake8: noqa: E501
# pylint: disable=line-too-long
def prep_title(title_array: np.array) -> np.array:
    # Remove language and contents after if "Russian" or "Chinese" followed by a newline
    title_array = np.array(
        [
            re.sub(
                r"\. (Russian|Chinese|Spanish|Czech|Italian|Polish|Dutch|Ukrainian|German|French|Japanese|Slovak|Hungarian|Portuguese English|Turkish|Norwegian|Portuguese)(\r?\n)?.*$",
                "",
                title,
                flags=re.IGNORECASE,
            )
            if ". " in title
            else title
            for title in title_array
        ]
    )

    title_array = np.array(
        [
            title.replace("-like", "like")
            .replace("co-", "co")
            .replace("post-", "post")
            .replace("three-dimensional", "threedimensional")
            .replace("+", " plus ")
            for title in title_array
        ]
    )

    # Replace 'withdrawn' at the beginning and '(review') at the end (case insensitive)
    title_array = np.array(
        [
            re.sub(
                r"^(withdrawn[.:] )|^(proceedings: )|^(reprint)|( \(review\))$|( \(vol \d+.*\))",
                "",
                title,
                flags=re.IGNORECASE,
            )
            for title in title_array
        ]
    )

    # Replace roman numbers (title similarity is sensitive to numbers)
    title_array = np.array(
        [
            re.sub(
                r"\biv\b",
                " 4 ",
                re.sub(
                    r"\biii\b",
                    " 3 ",
                    re.sub(
                        r"\bii\b",
                        " 2 ",
                        re.sub(r"\bi\b", " 1 ", title, flags=re.IGNORECASE),
                        flags=re.IGNORECASE,
                    ),
                    flags=re.IGNORECASE,
                ),
                flags=re.IGNORECASE,
            )
            for title in title_array
        ]
    )

    # Remove html tags
    title_array = np.array([re.sub(r"<.*?>", " ", title) for title in title_array])
    # Replace html special entities
    title_array = np.array([html.unescape(title) for title in title_array])

    # Remove language tags added by some databases (at the end)
    title_array = np.array(
        [re.sub(r"\. \[[A-Z][a-z]*\]$", "", title) for title in title_array]
    )

    # Remove trailing "1" if it is not preceded by "part"
    title_array = np.array(
        [
            re.sub(r"1$", "", title) if "part" not in title[-10:].lower() else title
            for title in title_array
        ]
    )
    # Remove erratum suffix
    # https://www.nlm.nih.gov/bsd/policy/errata.html
    title_array = np.array([remove_erratum_suffix(title) for title in title_array])

    # Replace words in parentheses at the end
    title_array = np.array(
        [re.sub(r"\s*\([^)]*\)\s*$", "", value) for value in title_array]
    )

    # Remove '[Review] [33 refs]' and ' [abstract no: 134]' ignoring case
    title_array = np.array(
        [
            re.sub(
                r"\[Review\] \[\d+ refs\]| \[abstract no: \d+\]",
                "",
                title,
                flags=re.IGNORECASE,
            )
            for title in title_array
        ]
    )

    # Replace brackets (often used in formulae)
    title_array = np.array(
        [re.sub(r"([A-Za-z])\(([0-9]*)\)", r"\1\2", title) for title in title_array]
    )

    # Replace special characters
    title_array = np.array(
        [re.sub(r"[^A-Za-z0-9,\[\]]+", " ", title.lower()) for title in title_array]
    )

    # Remove common stopwords
    title_array = np.array(
        [
            " ".join(word for word in title.split() if word not in TITLE_STOPWORDS)
            for title in title_array
        ]
    )

    # Apply parse function to replace numbers
    title_array = np.array([parse(title) for title in title_array])

    # Replace spaces between digits
    title_array = np.array(
        [
            re.sub(r"(\d) (\d)", r"\1\2", title).rstrip(" ].").lstrip("[ ")
            for title in title_array
        ]
    )

    # Replace multiple spaces with a single space
    title_array = np.array(
        [re.sub(r"\s+", " ", title).rstrip().lstrip() for title in title_array]
    )
    return title_array
