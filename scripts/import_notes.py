#!/usr/bin/env python3
"""
Import Markdown, TXT, PPTX, or PDF study material into a Hugo Academic post.
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import sys
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
CONTENT_POST = ROOT / "content" / "post"
CONTENT_AUTHORS = ROOT / "content" / "authors"
STATIC_IMPORTED = ROOT / "static" / "files" / "imported"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import notes into a Hugo post with a Mermaid mindmap."
    )
    parser.add_argument("source", help="Path to a .md, .txt, .pptx, or .pdf file")
    parser.add_argument("--title", help="Override generated title")
    parser.add_argument("--slug", help="Override generated post slug")
    parser.add_argument("--author", help="Author slug in content/authors/")
    parser.add_argument("--category", default="Notes", help="Category for front matter")
    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        default=[],
        help="Extra tag to append. Can be used multiple times.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing target post directory if present.",
    )
    return parser.parse_args()


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "cp950", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode text file: {path}")


def extract_pdf_text(path: Path) -> str:
    errors: list[str] = []
    for module_name in ("pypdf", "PyPDF2"):
        try:
            module = __import__(module_name)
            reader = module.PdfReader(str(path))
            pages = [(page.extract_text() or "") for page in reader.pages]
            text = "\n\n".join(pages).strip()
            if text:
                return text
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{module_name}: {exc}")

    details = " | ".join(errors) if errors else "no PDF backend available"
    raise RuntimeError(
        "PDF text extraction failed. Install `pypdf` or convert the PDF to "
        f"Markdown/TXT first. Details: {details}"
    )


def extract_pptx_text(path: Path) -> str:
    ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
    sections: list[str] = []
    with zipfile.ZipFile(path) as zf:
        slides = sorted(
            name
            for name in zf.namelist()
            if name.startswith("ppt/slides/slide") and name.endswith(".xml")
        )
        for idx, slide_name in enumerate(slides, start=1):
            root = ET.fromstring(zf.read(slide_name))
            chunks = [
                node.text.strip()
                for node in root.findall(".//a:t", ns)
                if node.text and node.text.strip()
            ]
            if chunks:
                sections.append(f"## Slide {idx}\n\n" + "\n\n".join(chunks))
    text = "\n\n".join(sections).strip()
    if not text:
        raise RuntimeError(f"No extractable text found in PPTX: {path}")
    return text


def load_source(path: Path) -> tuple[str, bool]:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return read_text_file(path), True
    if suffix == ".txt":
        return read_text_file(path), False
    if suffix == ".pptx":
        return extract_pptx_text(path), False
    if suffix == ".pdf":
        return extract_pdf_text(path), False
    raise RuntimeError(f"Unsupported file type: {suffix}")


def strip_front_matter(text: str) -> str:
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return parts[2].lstrip()
    if text.startswith("+++"):
        parts = text.split("+++", 2)
        if len(parts) == 3:
            return parts[2].lstrip()
    return text


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    base = ascii_value if ascii_value.strip() else value
    base = re.sub(r"[^\w\s-]", "", base, flags=re.UNICODE).strip().lower()
    slug = re.sub(r"[-\s]+", "-", base).strip("-")
    return slug or "imported-notes"


def title_from_path(path: Path) -> str:
    raw = path.stem.replace("_", " ").replace("-", " ").strip()
    return raw or "Imported Notes"


def clean_lines(text: str) -> list[str]:
    return [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]


def infer_summary(markdown: str) -> str:
    cleaned = re.sub(r"```.*?```", "", markdown, flags=re.S)
    cleaned = re.sub(r"!\[.*?\]\(.*?\)", "", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"^#+\s*", "", cleaned, flags=re.M)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:140] + ("..." if len(cleaned) > 140 else "")


def build_markdown_body(text: str, source_is_markdown: bool) -> str:
    if source_is_markdown:
        return strip_front_matter(text).strip()

    lines = clean_lines(text)
    blocks: list[str] = []
    paragraph: list[str] = []
    list_mode = False

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            blocks.append(" ".join(paragraph).strip())
            paragraph = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            list_mode = False
            continue

        if re.match(r"^(#{1,6})\s+", line):
            flush_paragraph()
            blocks.append(line)
            list_mode = False
            continue

        if len(line) <= 80 and (line.endswith(":") or re.match(r"^(slide|chapter|section)\s+\d+", line, flags=re.I)):
            flush_paragraph()
            heading = line[:-1] if line.endswith(":") else line
            blocks.append(f"## {heading}")
            list_mode = False
            continue

        if re.match(r"^[-*]\s+", line) or re.match(r"^\d+[.)]\s+", line):
            flush_paragraph()
            blocks.append(re.sub(r"^\d+[.)]\s+", "- ", line))
            list_mode = True
            continue

        if list_mode and len(line) <= 120:
            blocks.append(f"- {line}")
            continue

        paragraph.append(line)

    flush_paragraph()
    return "\n\n".join(blocks).strip()


def extract_outline(markdown: str) -> list[tuple[int, str]]:
    outline: list[tuple[int, str]] = []
    for line in clean_lines(markdown):
        match = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if match:
            level = len(match.group(1))
            title = re.sub(r"`+", "", match.group(2)).strip()
            if title:
                outline.append((level, title))

    if outline:
        return outline[:20]

    fallback: list[tuple[int, str]] = []
    for line in clean_lines(markdown):
        stripped = line.strip("-* ").strip()
        if 2 <= len(stripped) <= 50:
            fallback.append((2, stripped))
        if len(fallback) >= 10:
            break
    return fallback


def mermaid_safe(text: str) -> str:
    text = html.unescape(text)
    text = text.replace('"', "'")
    text = re.sub(r"[{}[\]<>]", "", text)
    return text.strip()


def build_mermaid(title: str, markdown: str) -> str:
    outline = extract_outline(markdown)
    root = mermaid_safe(title)
    lines = ["mindmap", f"  root(({root}))"]
    for level, item in outline:
        indent = "  " * max(2, min(level + 1, 5))
        safe_item = mermaid_safe(item)
        if safe_item and safe_item != root:
            lines.append(f"{indent}{safe_item}")
    return "\n".join(lines)


def guess_tags(path: Path, markdown: str) -> list[str]:
    tags: list[str] = []
    stem = path.stem.lower()
    lowered = markdown.lower()
    if "nlp" in stem or "natural language" in lowered or "language model" in lowered:
        tags.append("NLP")
    if path.suffix.lower() == ".pptx":
        tags.append("Slides")
    if path.suffix.lower() == ".pdf":
        tags.append("PDF")
    if path.suffix.lower() in {".md", ".markdown", ".txt"}:
        tags.append("Notes")
    seen: set[str] = set()
    deduped: list[str] = []
    for tag in tags:
        key = tag.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(tag)
    return deduped


def detect_author_slug(explicit: str | None) -> str:
    if explicit:
        return explicit
    if not CONTENT_AUTHORS.exists():
        return "admin"
    candidates = sorted(
        path.name
        for path in CONTENT_AUTHORS.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    return candidates[0] if candidates else "admin"


def escape_yaml(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def render_post(
    *,
    title: str,
    summary: str,
    author: str,
    category: str,
    tags: Iterable[str],
    source_link: str,
    mermaid: str,
    body: str,
) -> str:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    tag_lines = "\n".join(f"- {tag}" for tag in tags) or "- Notes"
    return f"""---
title: "{escape_yaml(title)}"
date: {timestamp}
draft: false
diagram: true

authors:
- {author}

tags:
{tag_lines}

categories:
- {category}

summary: "{escape_yaml(summary)}"
---

## Source File

[Download the original file]({source_link})

## Mindmap

```mermaid
{mermaid}
```

## Notes

{body}
"""


def main() -> int:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()

    if not source.exists():
        print(f"Source file not found: {source}", file=sys.stderr)
        return 1

    try:
        raw_text, source_is_markdown = load_source(source)
        title = args.title or title_from_path(source)
        slug = args.slug or slugify(title)
        author = detect_author_slug(args.author)
        body = build_markdown_body(raw_text, source_is_markdown)
        summary = infer_summary(body) or title
        mermaid = build_mermaid(title, body)
        tags = guess_tags(source, body)
        for extra_tag in args.tags:
            if extra_tag not in tags:
                tags.append(extra_tag)

        post_dir = CONTENT_POST / slug
        post_file = post_dir / "index.md"
        copied_name = f"{slug}{source.suffix.lower()}"
        copied_source = STATIC_IMPORTED / copied_name

        if post_dir.exists():
            if not args.force:
                raise RuntimeError(
                    f"Target post already exists: {post_dir}. Use --force to overwrite."
                )
            shutil.rmtree(post_dir)

        post_dir.mkdir(parents=True, exist_ok=True)
        copied_source.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, copied_source)

        post_text = render_post(
            title=title,
            summary=summary,
            author=author,
            category=args.category,
            tags=tags,
            source_link=f"/files/imported/{copied_name}",
            mermaid=mermaid,
            body=body,
        )
        post_file.write_text(post_text, encoding="utf-8")

        print(f"Created post: {post_file}")
        print(f"Copied source: {copied_source}")
        print(f"Preview URL after `hugo server`: /post/{slug}/")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
