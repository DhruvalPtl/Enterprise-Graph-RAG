"""
Structural Document Chunker for Enterprise RAG.

Key Design Decisions:
1. Structural Boundaries: Respects natural document breaks (headings, paragraphs,
   lists, and sentences) rather than blindly slicing at fixed character counts.
2. Section Awareness: Tracks active headings and attaches the current section title
   to each chunk's metadata, providing vital context for downstream retrieval.
3. Page Tracking: In multi-page documents (e.g., PDFs), chunks preserve their source
   page number for accurate UI citation and verification.
4. Deterministic Overlap: Smoothly bridges adjacent chunks to prevent context loss
   at boundary cuts.
"""
from typing import List, Optional, Tuple, Dict, Any
import re

from app.models import Document, DocumentPage, Chunk
from app.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


class RecursiveStructuralChunker:
    """
    Splits documents into coherent chunks respecting structural boundaries:
    Headings -> Paragraphs -> List items / Lines -> Sentences -> Words.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than "
                f"chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Regex to detect markdown or uppercase section headers
        # Matches: "# Heading", "## Subheading", "SECTION 1:", "CHAPTER 2", etc.
        # Requires at least one alphabetic letter so pure numbers (e.g. "2023", "372") are not treated as headings.
        self._heading_pattern = re.compile(
            r"^(?:#{1,6}\s+(.+)|(?=.*[A-Za-z])([A-Z0-9\s\-_:]{3,50}:?|[0-9]+\.[0-9]*\s+[A-Z].*))$"
        )

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Chunks an entire Document, processing page-by-page or the full document content.
        Preserves section titles, page numbers, and sequential indices.
        """
        chunks: List[Chunk] = []
        chunk_index = 0
        current_section = document.metadata.get("title", "General")

        # Determine segments to process: either page-by-page (e.g. PDFs) or whole document
        pages = document.pages if document.pages else [
            DocumentPage(page_number=1, text=document.content)
        ]

        for page in pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            # Chunk the page text while tracking section changes
            page_chunks, current_section, chunk_index = self._chunk_text_block(
                text=page_text,
                document=document,
                page_number=page.page_number,
                current_section=current_section,
                start_index=chunk_index,
            )
            chunks.extend(page_chunks)

        return chunks

    def _chunk_text_block(
        self,
        text: str,
        document: Document,
        page_number: int,
        current_section: str,
        start_index: int,
    ) -> Tuple[List[Chunk], str, int]:
        """
        Chunks a block of text (such as a page or document section).
        Returns: (chunks_list, updated_section, next_chunk_index)
        """
        chunks: List[Chunk] = []
        chunk_index = start_index

        # Split block into paragraphs (double newline)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

        current_buffer: List[str] = []
        current_buffer_len = 0

        for paragraph in paragraphs:
            # Check if this paragraph is a section heading
            heading_title = self._extract_heading(paragraph)
            if heading_title:
                # If we have accumulated text under the previous section,
                # flush it now so sections stay cleanly separated.
                if current_buffer:
                    chunk_text = "\n\n".join(current_buffer).strip()
                    if chunk_text:
                        chunks.append(
                            self._create_chunk(
                                text=chunk_text,
                                document=document,
                                page_number=page_number,
                                section=current_section,
                                chunk_index=chunk_index,
                            )
                        )
                        chunk_index += 1
                    current_buffer = []
                    current_buffer_len = 0

                current_section = heading_title

            # If a single paragraph is longer than chunk_size, break it into smaller units (sentences)
            units = self._split_into_units(paragraph)

            for unit in units:
                unit_len = len(unit)

                # Check if adding this unit exceeds target chunk_size
                projected_len = current_buffer_len + unit_len + (2 if current_buffer else 0)

                if projected_len > self.chunk_size and current_buffer:
                    # Flush current buffer as a chunk
                    chunk_text = "\n\n".join(current_buffer).strip()
                    if chunk_text:
                        chunks.append(
                            self._create_chunk(
                                text=chunk_text,
                                document=document,
                                page_number=page_number,
                                section=current_section,
                                chunk_index=chunk_index,
                            )
                        )
                        chunk_index += 1

                    # Compute overlap from the end of current buffer
                    overlap_buffer = self._compute_overlap(current_buffer)
                    current_buffer = overlap_buffer
                    current_buffer_len = sum(len(u) for u in current_buffer) + max(0, (len(current_buffer) - 1) * 2)

                # If the unit itself is still longer than chunk_size (e.g. huge single sentence)
                if unit_len > self.chunk_size:
                    sub_chunks = self._hard_slice_text(unit)
                    for sc in sub_chunks:
                        chunks.append(
                            self._create_chunk(
                                text=sc,
                                document=document,
                                page_number=page_number,
                                section=current_section,
                                chunk_index=chunk_index,
                            )
                        )
                        chunk_index += 1
                    current_buffer = []
                    current_buffer_len = 0
                else:
                    current_buffer.append(unit)
                    current_buffer_len += unit_len + (2 if len(current_buffer) > 1 else 0)

        # Flush any remaining text in buffer
        if current_buffer:
            chunk_text = "\n\n".join(current_buffer).strip()
            if chunk_text:
                chunks.append(
                    self._create_chunk(
                        text=chunk_text,
                        document=document,
                        page_number=page_number,
                        section=current_section,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

        return chunks, current_section, chunk_index

    def _split_into_units(self, text: str) -> List[str]:
        """
        Splits text if it exceeds chunk_size into smaller structural units:
        Lines/Lists -> Sentences.
        """
        if len(text) <= self.chunk_size:
            return [text]

        # Try splitting by single line (e.g. bullet points or list items)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if len(lines) > 1 and all(len(line) <= self.chunk_size for line in lines):
            return lines

        # If lines still too long or text is one big block, split by sentence punctuation
        # Regex uses lookbehind to keep the punctuation (.!?) with the preceding sentence
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences if sentences else [text]

    def _compute_overlap(self, buffer: List[str]) -> List[str]:
        """
        Picks trailing units from buffer whose total character length fits within chunk_overlap.
        """
        overlap_units: List[str] = []
        running_length = 0

        for unit in reversed(buffer):
            unit_len = len(unit) + (2 if overlap_units else 0)
            if running_length + unit_len <= self.chunk_overlap:
                overlap_units.insert(0, unit)
                running_length += unit_len
            else:
                break

        return overlap_units

    def _hard_slice_text(self, text: str) -> List[str]:
        """
        Fallback sliding window for exceptionally long single sentences or unbroken tokens.
        """
        slices = []
        start = 0
        step = max(1, self.chunk_size - self.chunk_overlap)

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_str = text[start:end].strip()
            if chunk_str:
                slices.append(chunk_str)
            if end == len(text):
                break
            start += step

        return slices

    def _extract_heading(self, paragraph: str) -> Optional[str]:
        """
        Detects if a single-line paragraph is a heading and returns clean title.
        """
        lines = paragraph.split("\n")
        if len(lines) == 1:
            line = lines[0].strip()
            match = self._heading_pattern.match(line)
            if match:
                # Group 1 is markdown (# Heading), Group 2 is text-based (SECTION 1)
                heading = match.group(1) or match.group(2)
                return heading.strip()
        return None

    def _create_chunk(
        self,
        text: str,
        document: Document,
        page_number: int,
        section: str,
        chunk_index: int,
    ) -> Chunk:
        """
        Constructs a Chunk object with deterministic ID and rich metadata.
        """
        chunk_id = f"{document.id}_c{chunk_index:04d}"
        doc_name = document.metadata.get("document_name", "unknown")

        metadata: Dict[str, Any] = {
            "document_name": doc_name,
            "file_type": document.metadata.get("file_type", "unknown"),
            "page_number": page_number,
            "section": section,
            "chunk_index": chunk_index,
            "char_count": len(text),
            "word_count": len(text.split()),
        }

        return Chunk(
            chunk_id=chunk_id,
            document_id=document.id,
            text=text,
            metadata=metadata,
        )
