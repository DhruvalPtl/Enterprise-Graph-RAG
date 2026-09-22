"""
Isolated Benchmark Baseline Extractor (scripts/benchmark/run_baselines.py).

Extracts raw baseline text from the selected 6 representative PDF benchmark pages
using both:
1. Current Production Baseline: pypdf (stream drawing order)
2. Alternative Layout-Aware Baseline: PyMuPDF / fitz (block reading-order + table detection)

Also renders high-resolution 300 DPI page images for vision/OCR analysis
matching the official baidu/Unlimited-OCR input specification: fitz.Matrix(300 / 72, 300 / 72).

Saves all outputs under reports/ocr_benchmark/.
DOES NOT alter any production files, database tables, chunks, or retrieval behavior.
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add repository root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pypdf import PdfReader
import fitz  # PyMuPDF


BENCHMARK_SPECS = [
    {
        "test_id": "TEST_A1",
        "category": "Known Layout Failure (Timeline)",
        "file": "Artificial-Intelligence-Index-Report-2024-Stanford-University.pdf",
        "page_number": 372,
        "description": "3-column visual timeline with dates, event titles, descriptions, and 3 figure thumbnails/captions."
    },
    {
        "test_id": "TEST_A2",
        "category": "Chart/Figure & Clustered Captions",
        "file": "Artificial-Intelligence-Index-Report-2024-Stanford-University.pdf",
        "page_number": 51,
        "description": "Infographic page with 8 embedded image objects and 3 clustered figure captions."
    },
    {
        "test_id": "TEST_B1",
        "category": "Complex Textbook Layout & Raw NUL Bytes",
        "file": "speech-and-language-processing-daniel-jurafsky-and-james-h-martin-837.pdf",
        "page_number": 97,
        "description": "Phonetics/syntax textbook page containing 22 raw NUL bytes (0x00) and multi-column IPA notation."
    },
    {
        "test_id": "TEST_B2",
        "category": "Complex Textbook Layout & Syntax Trees",
        "file": "speech-and-language-processing-daniel-jurafsky-and-james-h-martin-837.pdf",
        "page_number": 25,
        "description": "Regular expressions and finite-state automata diagrams with 4 image objects."
    },
    {
        "test_id": "TEST_C1",
        "category": "Structured Comparison Table & Architecture",
        "file": "foundation-models-for-natural-language-processing-gerhard-paa-and-sven-giesselbach-839.pdf",
        "page_number": 47,
        "description": "Comparative model benchmark table and transformer architecture diagram."
    },
    {
        "test_id": "TEST_C2",
        "category": "Mathematical Derivations & LaTeX Equations",
        "file": "natural-language-processing-jacob-eisenstein-838.pdf",
        "page_number": 32,
        "description": "Multi-line mathematical probability formulas, vector algebra, and dense LaTeX notation."
    },
]


def extract_pypdf(pdf_path: Path, page_number: int) -> Dict[str, Any]:
    """Extracts text using current production pypdf implementation."""
    t0 = time.perf_counter()
    reader = PdfReader(str(pdf_path))
    page = reader.pages[page_number - 1]
    raw_text = page.extract_text() or ""
    sanitized_text = raw_text.replace("\x00", "").strip()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "text": sanitized_text,
        "raw_text": raw_text,
        "char_count": len(sanitized_text),
        "word_count": len(sanitized_text.split()),
        "line_count": len(sanitized_text.splitlines()),
        "nul_bytes_count": raw_text.count("\x00"),
        "latency_ms": round(elapsed_ms, 2),
    }


def extract_pymupdf(pdf_path: Path, page_number: int) -> Dict[str, Any]:
    """Extracts text using layout-aware PyMuPDF with block sorting and table detection."""
    t0 = time.perf_counter()
    doc = fitz.open(str(pdf_path))
    page = doc[page_number - 1]

    # 1. Block-level extraction with visual reading order sorting
    # blocks returns list of tuples: (x0, y0, x1, y1, "text", block_no, block_type)
    # block_type 0 is text, 1 is image
    blocks = page.get_text("blocks", sort=True)
    text_blocks = []
    for b in blocks:
        if b[6] == 0:  # text block
            block_text = b[4].strip()
            if block_text:
                text_blocks.append(block_text)

    # 2. Extract tables if present using PyMuPDF's built-in table finder
    tables = []
    try:
        tabs = page.find_tables()
        for tab in tabs:
            df_markdown = tab.to_markdown()
            if df_markdown:
                tables.append(df_markdown)
    except Exception:
        pass

    full_text = "\n\n".join(text_blocks)
    if tables:
        full_text += "\n\n[EXTRACTED TABLES]:\n" + "\n\n".join(tables)

    doc.close()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "text": full_text,
        "char_count": len(full_text),
        "word_count": len(full_text.split()),
        "line_count": len(full_text.splitlines()),
        "block_count": len(text_blocks),
        "table_count": len(tables),
        "tables_markdown": tables,
        "latency_ms": round(elapsed_ms, 2),
    }


def render_page_image(pdf_path: Path, page_number: int, output_path: Path, dpi: int = 300) -> Dict[str, Any]:
    """Renders PDF page to 300 DPI image as required by vision models & Unlimited-OCR."""
    t0 = time.perf_counter()
    doc = fitz.open(str(pdf_path))
    page = doc[page_number - 1]
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    pix.save(str(output_path))
    doc.close()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "width": pix.width,
        "height": pix.height,
        "file_size_bytes": output_path.stat().st_size,
        "render_latency_ms": round(elapsed_ms, 2),
    }


def main():
    raw_dir = Path("data/raw")
    output_dir = Path("reports/ocr_benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  RUNNING ISOLATED BENCHMARK BASELINES (pypdf vs PyMuPDF)")
    print("=" * 80)
    print(f"  Source directory: {raw_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Total Benchmark Pages: {len(BENCHMARK_SPECS)}\n")

    summary_results = []

    for spec in BENCHMARK_SPECS:
        test_id = spec["test_id"]
        doc_name = spec["file"]
        page_num = spec["page_number"]
        pdf_path = raw_dir / doc_name

        print(f"[{test_id}] {doc_name} (Page {page_num}) - {spec['category']}")
        if not pdf_path.exists():
            print(f"  [ERROR] File not found: {pdf_path}")
            continue

        # 1. pypdf extraction
        pypdf_res = extract_pypdf(pdf_path, page_num)
        pypdf_file = output_dir / f"page_{page_num}_pypdf.txt"
        with open(pypdf_file, "w", encoding="utf-8") as f:
            f.write(pypdf_res["text"])
        print(f"  -> pypdf: {pypdf_res['char_count']} chars, {pypdf_res['word_count']} words, NULs={pypdf_res['nul_bytes_count']} in {pypdf_res['latency_ms']} ms")

        # 2. PyMuPDF extraction
        pymupdf_res = extract_pymupdf(pdf_path, page_num)
        pymupdf_file = output_dir / f"page_{page_num}_pymupdf.txt"
        with open(pymupdf_file, "w", encoding="utf-8") as f:
            f.write(pymupdf_res["text"])
        print(f"  -> PyMuPDF: {pymupdf_res['char_count']} chars, {pymupdf_res['block_count']} blocks, {pymupdf_res['table_count']} tables in {pymupdf_res['latency_ms']} ms")

        # 3. Render 300 DPI image
        img_file = output_dir / f"page_{page_num}.png"
        img_res = render_page_image(pdf_path, page_num, img_file, dpi=300)
        print(f"  -> Rendered 300 DPI image: {img_res['width']}x{img_res['height']} ({img_res['file_size_bytes']/1024:.1f} KB) in {img_res['render_latency_ms']} ms\n")

        summary_results.append({
            "test_id": test_id,
            "category": spec["category"],
            "document": doc_name,
            "page_number": page_num,
            "description": spec["description"],
            "pypdf": pypdf_res,
            "pymupdf": pymupdf_res,
            "image": img_res,
            "files": {
                "pypdf_txt": str(pypdf_file),
                "pymupdf_txt": str(pymupdf_file),
                "image_png": str(img_file),
            }
        })

    # Save summary metadata
    meta_path = output_dir / "baseline_summary.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    print("=" * 80)
    print(f"Baseline extraction completed successfully. Metadata saved to: {meta_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
