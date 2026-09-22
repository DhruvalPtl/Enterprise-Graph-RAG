"""
Document loaders for PDF, Markdown, and Text files.

Design Philosophy:
- Keep loaders isolated and simple.
- Preserve document structural boundaries (e.g., pages in PDFs) so downstream
  chunkers can attach accurate citations and page references.
- Generate deterministic document IDs using content hashes for idempotency.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Dict, Any, List
import re

from app.models import Document, DocumentPage, generate_content_hash


class BaseLoader(ABC):
    """Abstract base class for all document loaders."""

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path).resolve()
        if not self.file_path.exists():
            raise FileNotFoundError(f"Source file not found: {self.file_path}")

    @abstractmethod
    def load(self) -> Document:
        """Extract text and metadata from the document."""
        pass


class TextLoader(BaseLoader):
    """Loads plain text (.txt) files."""

    def load(self) -> Document:
        with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        doc_id = f"doc_{generate_content_hash(content)}"
        metadata: Dict[str, Any] = {
            "document_name": self.file_path.name,
            "source_path": str(self.file_path),
            "file_type": "text",
            "file_size_bytes": self.file_path.stat().st_size,
            "char_count": len(content),
        }

        # For text files, the entire document is treated as a single page (page 1)
        pages = [DocumentPage(page_number=1, text=content)]

        return Document(
            id=doc_id,
            content=content,
            metadata=metadata,
            pages=pages,
        )


class MarkdownLoader(BaseLoader):
    """Loads Markdown (.md) documents and captures high-level headings."""

    def load(self) -> Document:
        with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Extract top-level heading if available (# Heading)
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        document_title = title_match.group(1).strip() if title_match else self.file_path.stem

        doc_id = f"doc_{generate_content_hash(content)}"
        metadata: Dict[str, Any] = {
            "document_name": self.file_path.name,
            "source_path": str(self.file_path),
            "file_type": "markdown",
            "title": document_title,
            "file_size_bytes": self.file_path.stat().st_size,
            "char_count": len(content),
        }

        pages = [DocumentPage(page_number=1, text=content)]

        return Document(
            id=doc_id,
            content=content,
            metadata=metadata,
            pages=pages,
        )


MONTH_EXPANSIONS = {
    "jan": "January", "feb": "February", "mar": "March", "apr": "April",
    "may": "May", "jun": "June", "jul": "July", "aug": "August",
    "sep": "September", "sept": "September", "oct": "October",
    "nov": "November", "dec": "December"
}


def normalize_extracted_text(text: str) -> str:
    """
    Normalizes extracted text while preserving faithful searchable content:
    - Eliminates NUL bytes (PostgreSQL safety)
    - Strips non-standard control characters (preserves \\t, \\n, \\r)
    - Removes obvious hyphenation across line breaks without aggressive modification
    - Unifies multi-column timeline date badges where 4-digit year was separated from date badge
    - Expands month abbreviations in dates so events are searchable by both full and abbreviated month
    - Collapses excessive runs of empty lines (3+ into 2)
    """
    if not text:
        return ""
    # NUL bytes
    text = text.replace("\x00", "")
    # Strip non-standard control characters (ord < 32 excluding \t, \n, \r, and delete \x7f)
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Obvious word hyphenation across line breaks: e.g. "dis- \ncuss" -> "discuss"
    text = re.sub(r"(\b[A-Za-z]{2,})-\s*\n\s*([a-z]{2,}\b)", r"\1\2", text)
    # Unify timeline date badges (e.g. "Jul. 25,\n\nTitle\n\n2023" -> "Jul. 25, 2023\n\nTitle")
    text = re.sub(
        r"(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{1,2},?)\s*\n\n([^\n]+(?:\n[^\n]+)?)\n\n(20\d\d)\b",
        lambda m: f"{m.group(1).rstrip(',')}, {m.group(3)}\n\n{m.group(2)}",
        text,
    )
    # Expand month abbreviations in dates so events are searchable by both full month and abbreviation
    # e.g. "Jul. 25, 2023" -> "July 25, 2023 (Jul. 25, 2023)"
    def _expand_date_month(m: re.Match) -> str:
        raw_month = m.group(1)
        abbr = raw_month.lower().rstrip(".")
        full = MONTH_EXPANSIONS.get(abbr)
        if full and full.lower() != raw_month.lower():
            clean_raw = raw_month.rstrip(".")
            return f"{full} {m.group(2)}, {m.group(3)} ({clean_raw}. {m.group(2)}, {m.group(3)})"
        return m.group(0)

    text = re.sub(
        r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s*(\d{1,2}),?\s*(19\d\d|20\d\d)\b",
        _expand_date_month,
        text,
        flags=re.IGNORECASE,
    )
    # Collapse runs of 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()



class PDFLoader(BaseLoader):
    """
    Loads PDF (.pdf) documents page-by-page using PyMuPDF (fitz).
    Extracts text in visual reading order using sorted blocks,
    detects and serializes tables to Markdown, and preserves page
    boundaries for accurate citation.
    """

    def load(self) -> Document:
        try:
            import pymupdf as fitz
        except ImportError:
            try:
                import fitz
            except ImportError:
                raise ImportError(
                    "pymupdf is required to load PDF documents. "
                    "Please run: pip install pymupdf"
                )

        doc = fitz.open(str(self.file_path))
        pages: List[DocumentPage] = []
        full_text_parts: List[str] = []

        try:
            for page_index, page in enumerate(doc):
                page_number = page_index + 1

                # Detect tables if available
                tables = []
                try:
                    tabs = page.find_tables()
                    if hasattr(tabs, "tables"):
                        tables = list(tabs.tables)
                    else:
                        tables = list(tabs)
                except Exception:
                    tables = []

                # Block extraction with reading order sorting
                blocks = page.get_text("blocks", sort=True)

                table_emitted = {i: False for i in range(len(tables))}
                page_elements: List[str] = []

                for b in blocks:
                    # b: (x0, y0, x1, y1, text, block_no, block_type)
                    if b[6] != 0:  # Skip image blocks
                        continue
                    b_rect = fitz.Rect(b[:4])
                    matched_tab_idx = None
                    for idx, t in enumerate(tables):
                        t_rect = fitz.Rect(t.bbox)
                        inter = b_rect & t_rect
                        if b_rect.get_area() > 0 and (inter.get_area() / b_rect.get_area()) > 0.5:
                            matched_tab_idx = idx
                            break

                    if matched_tab_idx is not None:
                        if not table_emitted[matched_tab_idx]:
                            try:
                                md = tables[matched_tab_idx].to_markdown().strip()
                                if md:
                                    page_elements.append(md)
                            except Exception:
                                pass
                            table_emitted[matched_tab_idx] = True
                    else:
                        text = b[4].strip()
                        if text:
                            page_elements.append(text)

                # Any table not overlapping text blocks
                for idx, emitted in table_emitted.items():
                    if not emitted:
                        try:
                            md = tables[idx].to_markdown().strip()
                            if md:
                                page_elements.append(md)
                        except Exception:
                            pass

                raw_page_text = "\n\n".join(page_elements)
                page_text = normalize_extracted_text(raw_page_text)

                pages.append(
                    DocumentPage(
                        page_number=page_number,
                        text=page_text,
                        metadata={"page_char_count": len(page_text)},
                    )
                )
                if page_text:
                    full_text_parts.append(page_text)
        finally:
            total_pages = len(doc)
            doc.close()

        full_content = "\n\n".join(full_text_parts)
        doc_id = f"doc_{generate_content_hash(full_content if full_content else self.file_path.name)}"

        metadata: Dict[str, Any] = {
            "document_name": self.file_path.name,
            "source_path": str(self.file_path),
            "file_type": "pdf",
            "total_pages": total_pages,
            "file_size_bytes": self.file_path.stat().st_size,
            "char_count": len(full_content),
        }

        return Document(
            id=doc_id,
            content=full_content,
            metadata=metadata,
            pages=pages,
        )


def load_document(file_path: Union[str, Path]) -> Document:
    """
    Factory function: detects the file type and dispatches to the appropriate loader.
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension == ".pdf":
        return PDFLoader(path).load()
    elif extension == ".md":
        return MarkdownLoader(path).load()
    elif extension == ".txt":
        return TextLoader(path).load()
    else:
        raise ValueError(
            f"Unsupported file extension '{extension}'. "
            f"Supported extensions are: .pdf, .md, .txt"
        )
