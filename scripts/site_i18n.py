#!/usr/bin/env python
"""Chinese version of the website. The English page is built first (build_site.py); this module translates it
segment by segment with the dictionary docs/site_zh.json {normalised English inner HTML: Chinese inner HTML}.

A segment is a block element (heading, paragraph, list item, table cell, caption, ...) that contains no other block
element; its inner HTML (inline tags kept) is the dictionary key. Segments without letters (numbers, symbols) are left
as they are. Missing translations fall back to English and are listed by `python scripts/site_i18n.py missing`, so a
rebuild with new numbers shows exactly which sentences need a new translation."""
import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
DICT = ROOT / "docs/site_zh.json"
BLOCK = {"h1", "h2", "h3", "h4", "p", "li", "th", "td", "figcaption", "caption", "summary", "pre", "a", "div", "span"}
SEG_CLASSES = {"kl", "kv", "eyebrow", "lede"}  # div/span segments only with these classes; <a> only inside the toc


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _is_seg_tag(t) -> bool:
    if t.name in ("div", "span"):
        return bool(SEG_CLASSES & set(t.get("class") or []))
    if t.name == "a":
        return t.find_parent("nav") is not None
    return t.name in BLOCK


def segments(soup):
    out = []
    for t in soup.find_all(_is_seg_tag):
        if any(_is_seg_tag(c) for c in t.find_all(True)):
            continue  # container: its children are the segments
        key = norm(t.decode_contents()) if t.name != "pre" else t.decode_contents().strip()
        if re.search(r"[A-Za-z]{2,}", BeautifulSoup(key, "html.parser").get_text()):
            out.append((t, key))
    return out


def load_dict():
    return json.loads(DICT.read_text()) if DICT.exists() else {}


def translate(body_html: str):
    """English body -> (Chinese body with ids prefixed 'zh-', list of missing keys)."""
    soup = BeautifulSoup(body_html, "html.parser")
    zh = load_dict()
    missing = []
    for t, key in segments(soup):
        if key in zh:
            t.clear()
            t.append(BeautifulSoup(zh[key], "html.parser"))
        else:
            missing.append(key)
    for t in soup.find_all(id=True):
        t["id"] = "zh-" + t["id"]
    for t in soup.find_all("a", href=True):
        if t["href"].startswith("#"):
            t["href"] = "#zh-" + t["href"][1:]
    return str(soup), list(dict.fromkeys(missing))


if __name__ == "__main__":  # python scripts/site_i18n.py missing [out.json]  -> untranslated segments of the built page
    html = (ROOT / "outputs/rasr/site/index.html").read_text()
    body = html[html.index('<div class="lang-en"'):html.index('<div class="lang-zh"')] if '<div class="lang-en"' in html else html
    _, miss = translate(body)
    print(f"{len(miss)} untranslated segments")
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps({k: "" for k in miss}, indent=1, ensure_ascii=False))
