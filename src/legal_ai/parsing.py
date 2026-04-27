import re
import unicodedata
from pathlib import Path

import fitz
from bs4 import BeautifulSoup
from docx import Document


def clean_extracted_text(text: str) -> str:
    text = normalize_pdf_artifacts(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_pdf_artifacts(text: str) -> str:
    replacements = {
        "\uf0a7": "-",
        "\uf0b7": "-",
        "\u2022": "-",
        "\u00a0": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
    }
    for bad, replacement in replacements.items():
        text = text.replace(bad, replacement)

    cleaned_chars = []
    for char in text:
        category = unicodedata.category(char)
        if category == "So" and char not in {"§"}:
            cleaned_chars.append(" [illegible mark] ")
        else:
            cleaned_chars.append(char)

    text = "".join(cleaned_chars)
    text = re.sub(r"(\s*\[illegible mark\]\s*){2,}", " [illegible marks] ", text)
    return text


def remove_repeated_lines(text: str, min_repeats: int = 3) -> str:
    lines = [line.strip() for line in text.splitlines()]
    counts = {}
    for line in lines:
        if line:
            counts[line] = counts.get(line, 0) + 1

    repeated = {line for line, count in counts.items() if count >= min_repeats}
    cleaned = [line for line in lines if not (line in repeated and len(line) < 120)]
    return "\n".join(cleaned)


def parse_pdf(file_path: str | Path) -> str:
    doc = fitz.open(str(file_path))
    pages = [page.get_text("text") for page in doc]
    return remove_repeated_lines(clean_extracted_text("\n\n".join(pages)))


def parse_docx(file_path: str | Path) -> str:
    doc = Document(str(file_path))
    paragraphs = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
    return clean_extracted_text("\n\n".join(paragraphs))


def parse_html(file_path: str | Path) -> str:
    html = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return clean_extracted_text(soup.get_text(separator="\n"))


def parse_document(file_path: str | Path) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(file_path)
    if suffix == ".docx":
        return parse_docx(file_path)
    if suffix in {".html", ".htm"}:
        return parse_html(file_path)
    if suffix == ".doc":
        raise ValueError("Legacy .doc files must be converted to PDF or DOCX first.")
    raise ValueError(f"Unsupported file type: {suffix}")
