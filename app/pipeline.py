"""
Document Ingestion Pipeline.

Orchestrates the entire ingestion workflow:
1. Discover supported files in the raw data directory.
2. Extract text and metadata using format-specific loaders.
3. Apply structural chunking with heading/page awareness and overlap.
4. Save the processed chunks into local JSON files for inspection and downstream stages.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from app.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SUPPORTED_EXTENSIONS
from app.models import Chunk, Document
from app.loaders import load_document
from app.chunker import RecursiveStructuralChunker


class IngestionPipeline:
    """
    Coordinates document loading, chunking, and local storage.
    """

    def __init__(
        self,
        raw_dir: Path = RAW_DATA_DIR,
        processed_dir: Path = PROCESSED_DATA_DIR,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        chunker_kwargs = {}
        if chunk_size is not None:
            chunker_kwargs["chunk_size"] = chunk_size
        if chunk_overlap is not None:
            chunker_kwargs["chunk_overlap"] = chunk_overlap

        self.chunker = RecursiveStructuralChunker(**chunker_kwargs)

    def process_file(self, file_path: Path) -> List[Chunk]:
        """Loads and chunks a single file, saving its individual chunk output."""
        file_path = Path(file_path)
        document: Document = load_document(file_path)
        chunks: List[Chunk] = self.chunker.chunk_document(document)

        # Save per-document chunk file
        output_file = self.processed_dir / f"{file_path.stem}_chunks.json"
        self._save_chunks(chunks, output_file)

        return chunks

    def process_directory(self) -> Dict[str, Any]:
        """
        Discovers and processes all supported documents in the raw data directory.
        Saves individual chunk files and an aggregated 'all_chunks.json'.
        """
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw data directory does not exist: {self.raw_dir}")

        # Scan for supported files
        files_to_process: List[Path] = [
            p for p in self.raw_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

        if not files_to_process:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "files_summary": [],
                "message": f"No supported files found in {self.raw_dir}",
            }

        all_chunks: List[Chunk] = []
        files_summary: List[Dict[str, Any]] = []

        for file_path in sorted(files_to_process):
            chunks = self.process_file(file_path)
            all_chunks.extend(chunks)

            files_summary.append({
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower(),
                "chunks_count": len(chunks),
                "total_chars": sum(c.metadata.get("char_count", 0) for c in chunks),
            })

        # Save aggregated chunks
        aggregated_file = self.processed_dir / "all_chunks.json"
        self._save_chunks(all_chunks, aggregated_file)

        avg_chars = (
            sum(c.metadata.get("char_count", 0) for c in all_chunks) // len(all_chunks)
            if all_chunks else 0
        )

        return {
            "total_documents": len(files_to_process),
            "total_chunks": len(all_chunks),
            "average_chunk_size_chars": avg_chars,
            "processed_dir": str(self.processed_dir),
            "files_summary": files_summary,
        }

    def _save_chunks(self, chunks: List[Chunk], output_file: Path) -> None:
        """Serializes chunks to formatted JSON."""
        data = [chunk.to_dict() for chunk in chunks]
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
