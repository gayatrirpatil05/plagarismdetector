"""Text utility functions for SemanticEye Plagiarism & AI Detector.
Handles sentence tokenization, document extraction (.txt, .pdf, .docx),
and word-level diff matching.
"""
import html
import io
import re
import difflib
import zipfile
from typing import List, Dict, Any, Tuple

from app import config


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using safe, None-guard regex split."""
    if not text:
        return []
    # Normalize all line endings and collapse whitespace
    cleaned = re.sub(r"\r\n|\r", "\n", text).strip()
    cleaned = re.sub(r"\n+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return []

    # Simple reliable split on .!? followed by space + capital letter
    raw_parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", cleaned)
    sentences = []
    for part in raw_parts:
        if part is None:
            continue
        s = str(part).strip()
        if len(s) > 2:
            sentences.append(s)

    # Fallback if nothing was split
    if not sentences and cleaned:
        sentences = [cleaned]
    return sentences


def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    """Extract text from an uploaded .txt / .pdf / .docx file.

    Raises ValueError (with a user-safe message) if the file is unsupported,
    malformed, or too large - errors are never returned as if they were text.
    """
    name = (filename or "").lower()

    if name.endswith(".txt"):
        if b"\x00" in file_bytes[:4096]:
            raise ValueError("This does not look like a plain-text file.")
        for encoding in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                return file_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        return file_bytes.decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        if not file_bytes.startswith(b"%PDF"):
            raise ValueError("This file is not a valid PDF.")
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            if reader.is_encrypted:
                raise ValueError("Password-protected PDFs are not supported.")
            if len(reader.pages) > config.MAX_PDF_PAGES:
                raise ValueError(f"PDF has too many pages (limit {config.MAX_PDF_PAGES}).")
            pages_text = [t for t in (p.extract_text() for p in reader.pages) if t]
            return "\n\n".join(pages_text)
        except ValueError:
            raise
        except Exception:
            raise ValueError("Could not read this PDF. It may be corrupted or scanned images only.")

    if name.endswith(".docx"):
        if not file_bytes.startswith(b"PK"):
            raise ValueError("This file is not a valid DOCX document.")
        try:
            with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:  # guard against zip bombs
                if sum(i.file_size for i in z.infolist()) > 100 * 1024 * 1024:
                    raise ValueError("Document is too large when decompressed.")
            from docx import Document
            doc = Document(io.BytesIO(file_bytes))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ValueError:
            raise
        except Exception:
            raise ValueError("Could not read this DOCX file. It may be corrupted.")

    raise ValueError("Unsupported file type. Please upload .txt, .pdf or .docx.")


def calculate_text_stats(text: str, sentences: List[str]) -> Dict[str, Any]:
    """Compute high-level textual metrics (words, chars, reading time, etc.)."""
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)
    char_count = len(text)
    avg_sentence_len = round(word_count / max(1, len(sentences)), 1)
    reading_time_min = round(max(0.1, word_count / 200.0), 1)

    unique_words = set(w.lower() for w in words)
    lexical_richness = round((len(unique_words) / max(1, word_count)) * 100, 1)

    return {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": len(sentences),
        "avg_sentence_length": avg_sentence_len,
        "reading_time_minutes": reading_time_min,
        "unique_words": len(unique_words),
        "lexical_richness_pct": lexical_richness,
    }


def highlight_word_diff(s1: str, s2: str) -> Tuple[str, str]:
    """Generate inline HTML diff tags showing matching and unique words."""
    words1 = s1.split()
    words2 = s2.split()
    esc = lambda w: html.escape(w, quote=True)

    matcher = difflib.SequenceMatcher(
        None,
        [w.lower().strip(".,!?;:\"'") for w in words1],
        [w.lower().strip(".,!?;:\"'") for w in words2]
    )

    h1_parts: List[str] = []
    h2_parts: List[str] = []

    matching_blocks = matcher.get_matching_blocks()

    idx1 = 0
    for block in matching_blocks:
        while idx1 < block.a:
            h1_parts.append(f"<span class='diff-unique-1'>{esc(words1[idx1])}</span>")
            idx1 += 1
        for _ in range(block.size):
            if idx1 < len(words1):
                h1_parts.append(f"<span class='diff-match'>{esc(words1[idx1])}</span>")
                idx1 += 1
    while idx1 < len(words1):
        h1_parts.append(f"<span class='diff-unique-1'>{esc(words1[idx1])}</span>")
        idx1 += 1

    idx2 = 0
    for block in matching_blocks:
        while idx2 < block.b:
            h2_parts.append(f"<span class='diff-unique-2'>{esc(words2[idx2])}</span>")
            idx2 += 1
        for _ in range(block.size):
            if idx2 < len(words2):
                h2_parts.append(f"<span class='diff-match'>{esc(words2[idx2])}</span>")
                idx2 += 1
    while idx2 < len(words2):
        h2_parts.append(f"<span class='diff-unique-2'>{esc(words2[idx2])}</span>")
        idx2 += 1

    return " ".join(h1_parts), " ".join(h2_parts)
