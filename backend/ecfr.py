"""
ecfr.py
This module provides functions to download and process regulatory data from the
Electronic Code of Federal Regulations (eCFR) API.  It exposes a single
function, ``build_dataset``, that walks through every title, chapter and part
published on the eCFR website, computes a few simple metrics and returns the
results as a list of dictionaries.  The metrics include a total word count,
an MD5 checksum of all text for the chapter, a count of available versions
for the title (as a proxy for historical churn) and a custom ratio of the
frequency of the word "shall" relative to the total word count.  The
structure of the eCFR API is described in the official SDK documentation
which notes that ``/api/versioner/v1/titles.json`` returns summary
information about every title and that endpoints such as
``/api/versioner/v1/structure/{date}/title-{title}.json`` and
``/api/versioner/v1/full/{date}/title-{title}.xml`` expose the full
hierarchy and text of a title【360949012315245†L126-L133】.

The code in this module is deliberately short and easy to read.  It makes a
few simplifying assumptions: chapters are treated as agencies, and all parts
within a chapter are aggregated to compute the chapter‑level metrics.  The
latest available date for a title is taken from the ``up_to_date_as_of``
field returned by the titles endpoint【590897020559311†L0-L20】.  If API calls
fail the corresponding chapter is skipped.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import requests

# Base URL for all versioning endpoints.  See the SDK reference for details【360949012315245†L126-L133】.
BASE = "https://www.ecfr.gov/api/versioner/v1"


def _fetch_json(path: str) -> Optional[Dict[str, Any]]:
    """Fetch JSON from the eCFR API, returning None on errors."""
    try:
        resp = requests.get(f"{BASE}{path}", timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _fetch_text(path: str) -> Optional[str]:
    """Fetch raw text (XML) from the eCFR API, returning None on errors."""
    try:
        resp = requests.get(f"{BASE}{path}", timeout=60)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


def _collect_parts(node: Dict[str, Any], parts: List[str]) -> None:
    """Recursively collect part identifiers from a structure node.

    The eCFR structure tree uses a ``type`` field to identify the kind of
    element (e.g. 'part', 'subpart', 'section').  When we encounter a part
    node we append its identifier to the provided list.  Children are
    traversed depth‑first.
    """
    if not node:
        return
    if node.get("type") == "part":
        ident = node.get("identifier") or node.get("label") or ""
        # identifiers may be like '1' or 'Part 1'; strip out non‑digit
        num = re.sub(r"\D", "", ident)
        if num:
            parts.append(num)
    for child in node.get("children", []):
        _collect_parts(child, parts)


def _extract_text(xml: str) -> str:
    """Extract plain text from a piece of regulation XML.

    The ``full`` endpoint returns an XML document that contains tags for
    sections, paragraphs and other structural elements.  A quick way to turn
    this into plain text is to strip out all markup with a regular
    expression.  While this approach is naive it works reasonably well for
    word counting and does not require any external libraries.
    """
    # remove tags
    text = re.sub(r"<[^>]+>", " ", xml)
    # collapse whitespace
    return re.sub(r"\s+", " ", text).strip()


def build_dataset(
    selected_titles: Optional[List[int]] = None,
    skip_titles: Optional[List[int]] = None,
    log_progress: Optional[callable] = None,
    log_error: Optional[callable] = None,
) -> List[Dict[str, Any]]:
    """Build a list of metrics for every chapter of every title.

    Parameters
    ----------
    selected_titles: list[int] | None
        Optional list of title numbers to process.  If omitted all
        available titles are processed.  Passing a single number makes it
        possible to refresh or summarise a specific title without
        downloading the entire CFR.
    log_progress: callable | None
        Optional callback that will be invoked with a short message at
        various points in the computation.  This can be used to write a
        process log for observability.
    log_error: callable | None
        Optional callback invoked whenever an error occurs.  The callback
        receives a string describing the context and the exception.
    """
    titles_data = _fetch_json("/titles.json")
    if not titles_data or "titles" not in titles_data:
        return []

    dataset: List[Dict[str, Any]] = []
    for t in titles_data["titles"]:
        # Convert the title number to an integer if possible so that
        # comparisons in callers work reliably.  Some entries may store
        # the number as a string.
        try:
            num = int(t.get("number"))
        except Exception:
            # if the title number cannot be converted to an int, skip it
            continue
        # allow callers to restrict the set of titles processed
        if selected_titles and num not in selected_titles:
            continue
        # skip titles that have already been processed
        if skip_titles and num in skip_titles:
            continue
        if t.get("reserved"):
            continue
        date = t.get("up_to_date_as_of")
        if not date:
            continue
        if log_progress:
            log_progress(f"Processing title {num}: {t.get('name')}")
        try:
            versions_info = _fetch_json(f"/versions/title-{num}.json")
            version_count = len(versions_info.get("versions", [])) if versions_info else 0
        except Exception as exc:
            version_count = 0
            if log_error:
                log_error(f"versions title-{num}", exc)
        try:
            struct = _fetch_json(f"/structure/{date}/title-{num}.json")
        except Exception as exc:
            struct = None
            if log_error:
                log_error(f"structure title-{num}", exc)
        if not struct:
            continue
        for ch in struct.get("children", []):
            if ch.get("type") != "chapter":
                continue
            agency_name = ch.get("heading") or ch.get("label") or ch.get("identifier")
            parts: List[str] = []
            _collect_parts(ch, parts)
            total_words = 0
            combined_text_parts: List[str] = []
            if log_progress:
                log_progress(f"  Chapter {ch.get('identifier')} ({agency_name}): {len(parts)} parts")
            for p in parts:
                if log_progress:
                    log_progress(f"    Part {p}")
                try:
                    xml = _fetch_text(f"/full/{date}/title-{num}.xml?part={p}")
                except Exception as exc:
                    xml = None
                    if log_error:
                        log_error(f"full title-{num} part {p}", exc)
                if not xml:
                    continue
                text = _extract_text(xml)
                words = re.findall(r"\b\w+\b", text)
                total_words += len(words)
                combined_text_parts.append(text)
            if not combined_text_parts:
                continue
            combined_text = " ".join(combined_text_parts)
            checksum = hashlib.md5(combined_text.encode("utf-8")).hexdigest()
            shall_count = combined_text.lower().count("shall")
            shall_ratio = float(shall_count) / max(total_words, 1)
            dataset.append(
                {
                    "title_number": num,
                    "title_name": t.get("name"),
                    "chapter_identifier": ch.get("identifier"),
                    "agency": agency_name,
                    "word_count": total_words,
                    "checksum": checksum,
                    "version_count": version_count,
                    "shall_ratio": round(shall_ratio, 6),
                }
            )
    return dataset


def save_dataset(data: List[Dict[str, Any]], path: str) -> None:
    """Save the computed dataset as JSON to the given path."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load a previously saved dataset from disk."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []