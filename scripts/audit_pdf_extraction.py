"""
Read-Only PDF Extraction Audit Tool (Enterprise RAG).

Diagnoses extraction fidelity, reading-order consistency, image object density,
NUL/control byte occurrences, and layout-induced text fragmentation across
raw PDF corpora without modifying any database records, chunks, or pipeline code.

Usage:
    python scripts/audit_pdf_extraction.py
    python scripts/audit_pdf_extraction.py --file Artificial-Intelligence-Index-Report-2024-Stanford-University.pdf --page 372
    python scripts/audit_pdf_extraction.py --inspect-372
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add repository root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pypdf import PdfReader
try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None


@dataclass
class PageAuditRecord:
    """Diagnostic metrics collected for a single PDF page."""
    document: str
    page_number: int
    char_count: int
    word_count: int
    image_count: int
    nul_count: int
    control_char_count: int
    replacement_char_count: int
    is_sparse: bool
    caption_count: int
    has_clustered_captions: bool
    has_clustered_dates: bool
    reading_order_inversion: bool
    is_suspicious: bool
    reasons: List[str] = field(default_factory=list)
    table_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["reasons_str"] = "; ".join(self.reasons) if self.reasons else "NONE"
        return d


def count_page_images(page) -> int:
    """
    Safely counts PDF image XObjects from page resources dictionary.
    Avoids uncompressing / decoding image payloads for maximum audit speed.
    """
    try:
        res = page.get("/Resources")
        if not res:
            return 0
        if hasattr(res, "get_object"):
            res = res.get_object()
        if not isinstance(res, dict):
            return 0

        xobj = res.get("/XObject")
        if not xobj:
            return 0
        if hasattr(xobj, "get_object"):
            xobj = xobj.get_object()
        if not isinstance(xobj, dict):
            return 0

        image_count = 0
        for key in xobj:
            obj = xobj[key]
            if hasattr(obj, "get_object"):
                obj = obj.get_object()
            if isinstance(obj, dict) and obj.get("/Subtype") == "/Image":
                image_count += 1
        return image_count
    except Exception:
        # Fallback to len(page.images) if structural dictionary check fails
        try:
            return len(page.images)
        except Exception:
            return 0


def detect_spatial_stream_inversion(page) -> bool:
    """
    Detects if the text stream extraction order conflicts significantly with
    the visual vertical (top-to-bottom) reading order.
    """
    try:
        positions: List[Tuple[float, float]] = []

        def visitor(text, cm, tm, font_dict, font_size):
            if text and text.strip():
                # tm[4] = X, tm[5] = Y
                positions.append((float(tm[4]), float(tm[5])))

        page.extract_text(visitor_text=visitor)

        if len(positions) < 4:
            return False

        # In standard top-to-bottom reading order, Y generally decreases.
        # Significant positive jumps in Y (e.g. text jumping from bottom back to top)
        # across sequential text stream tokens indicate non-linear stream drawing.
        significant_y_jumps = 0
        for i in range(len(positions) - 1):
            curr_y = positions[i][1]
            next_y = positions[i + 1][1]
            # If Y jumps upwards by more than 120 points between consecutive stream tokens
            if (next_y - curr_y) > 120.0:
                significant_y_jumps += 1

        return significant_y_jumps >= 2
    except Exception:
        return False


def audit_single_page(page, page_number: int, doc_name: str) -> PageAuditRecord:
    """
    Performs comprehensive diagnostic inspection of a single page without modifying state.
    """
    # 1. Extract raw text BEFORE any sanitization
    raw_text = page.extract_text() or ""

    char_count = len(raw_text)
    words = raw_text.split()
    word_count = len(words)

    # 2. Inspect raw byte anomalies
    nul_count = raw_text.count("\x00")
    # Control chars excluding standard whitespace (\t, \n, \r)
    control_char_count = sum(1 for ch in raw_text if ord(ch) < 32 and ch not in "\t\n\r")
    replacement_char_count = raw_text.count("\ufffd")

    # 3. Detect image objects
    image_count = count_page_images(page)

    # 4. Sparse / Empty text heuristic
    is_sparse = (char_count < 50 or word_count < 10)

    # 5. Figure / Table captions count
    caption_matches = re.findall(r"\b(Figure|Fig\.|Table)\s+[0-9]+[A-Za-z0-9\.\-]*", raw_text, re.IGNORECASE)
    source_matches = re.findall(r"\bSource:\s+.*", raw_text, re.IGNORECASE)
    caption_count = len(caption_matches) + len(source_matches)

    # 6. Clustered captions / orphaned captions heuristic
    # If 2 or more Figure / Source captions occur sequentially in the last 6 lines of text
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    trailing_lines = lines[-6:] if len(lines) >= 6 else lines
    trailing_captions = sum(
        1 for ln in trailing_lines
        if re.search(r"^(Figure|Fig\.|Table|Source:)", ln, re.IGNORECASE)
    )
    has_clustered_captions = trailing_captions >= 2

    # 7. Clustered dates / orphaned timeline badges
    date_matches = [
        ln for ln in trailing_lines
        if re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+[0-9]{1,2},?\s+[0-9]{4}\b", ln, re.IGNORECASE)
        or re.search(r"\b[0-9]{4}\b", ln) and len(ln) <= 10
    ]
    has_clustered_dates = len(date_matches) >= 2

    # 8. Spatial stream inversion
    reading_order_inversion = False
    if char_count > 100:
        reading_order_inversion = detect_spatial_stream_inversion(page)

    # 9. Evaluate suspicion reasons
    reasons = []
    if nul_count > 0:
        reasons.append(f"Contains {nul_count} NUL (0x00) bytes")
    if replacement_char_count > 0:
        reasons.append(f"Contains {replacement_char_count} unmapped glyphs (U+FFFD)")
    if is_sparse and image_count > 0:
        reasons.append(f"Sparse text ({char_count} chars) with {image_count} images (visual diagram/scan)")
    elif is_sparse and image_count == 0:
        reasons.append(f"Sparse text ({char_count} chars, cover/blank page)")
    if reading_order_inversion:
        reasons.append("Stream drawing order conflicts with vertical reading order")
    if has_clustered_captions:
        reasons.append(f"Orphaned/clustered figure captions ({trailing_captions} in trailing lines)")
    if has_clustered_dates:
        reasons.append(f"Orphaned dates/timeline badges ({len(date_matches)} in trailing lines)")

    is_suspicious = len(reasons) > 0

    return PageAuditRecord(
        document=doc_name,
        page_number=page_number,
        char_count=char_count,
        word_count=word_count,
        image_count=image_count,
        nul_count=nul_count,
        control_char_count=control_char_count,
        replacement_char_count=replacement_char_count,
        is_sparse=is_sparse,
        caption_count=caption_count,
        has_clustered_captions=has_clustered_captions,
        has_clustered_dates=has_clustered_dates,
        reading_order_inversion=reading_order_inversion,
        is_suspicious=is_suspicious,
        reasons=reasons,
    )


def audit_single_page_pymupdf(page, page_number: int, doc_name: str) -> PageAuditRecord:
    """
    Diagnostic inspection of a single page using PyMuPDF block extraction and table detection.
    """
    tables = []
    try:
        tabs = page.find_tables()
        if hasattr(tabs, "tables"):
            tables = list(tabs.tables)
        else:
            tables = list(tabs)
    except Exception:
        tables = []
    table_count = len(tables)

    blocks = page.get_text("blocks", sort=True)
    table_emitted = {i: False for i in range(len(tables))}
    page_elements: List[str] = []

    for b in blocks:
        if b[6] != 0:
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

    for idx, emitted in table_emitted.items():
        if not emitted:
            try:
                md = tables[idx].to_markdown().strip()
                if md:
                    page_elements.append(md)
            except Exception:
                pass

    raw_text = "\n\n".join(page_elements)

    char_count = len(raw_text)
    words = raw_text.split()
    word_count = len(words)

    nul_count = raw_text.count("\x00")
    control_char_count = sum(1 for ch in raw_text if ord(ch) < 32 and ch not in "\t\n\r")
    replacement_char_count = raw_text.count("\ufffd")

    try:
        image_count = len(page.get_images())
    except Exception:
        image_count = 0

    is_sparse = (char_count < 50 or word_count < 10)

    caption_matches = re.findall(r"\b(Figure|Fig\.|Table)\s+[0-9]+[A-Za-z0-9\.\-]*", raw_text, re.IGNORECASE)
    source_matches = re.findall(r"\bSource:\s+.*", raw_text, re.IGNORECASE)
    caption_count = len(caption_matches) + len(source_matches)

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    trailing_lines = lines[-6:] if len(lines) >= 6 else lines
    trailing_captions = sum(
        1 for ln in trailing_lines
        if re.search(r"^(Figure|Fig\.|Table|Source:)", ln, re.IGNORECASE)
    )
    has_clustered_captions = trailing_captions >= 2

    date_matches = [
        ln for ln in trailing_lines
        if re.search(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+[0-9]{1,2},?\s+[0-9]{4}\b", ln, re.IGNORECASE)
        or (re.search(r"\b[0-9]{4}\b", ln) and len(ln) <= 10)
    ]
    has_clustered_dates = len(date_matches) >= 2

    reading_order_inversion = False

    reasons = []
    if nul_count > 0:
        reasons.append(f"Contains {nul_count} NUL (0x00) bytes")
    if replacement_char_count > 0:
        reasons.append(f"Contains {replacement_char_count} unmapped glyphs (U+FFFD)")
    if is_sparse and image_count > 0:
        reasons.append(f"Sparse text ({char_count} chars) with {image_count} images (visual diagram/scan)")
    elif is_sparse and image_count == 0:
        reasons.append(f"Sparse text ({char_count} chars, cover/blank page)")
    if has_clustered_captions:
        reasons.append(f"Orphaned/clustered figure captions ({trailing_captions} in trailing lines)")
    if has_clustered_dates:
        reasons.append(f"Orphaned dates/timeline badges ({len(date_matches)} in trailing lines)")

    is_suspicious = len(reasons) > 0

    return PageAuditRecord(
        document=doc_name,
        page_number=page_number,
        char_count=char_count,
        word_count=word_count,
        image_count=image_count,
        nul_count=nul_count,
        control_char_count=control_char_count,
        replacement_char_count=replacement_char_count,
        is_sparse=is_sparse,
        caption_count=caption_count,
        has_clustered_captions=has_clustered_captions,
        has_clustered_dates=has_clustered_dates,
        reading_order_inversion=reading_order_inversion,
        is_suspicious=is_suspicious,
        reasons=reasons,
        table_count=table_count,
    )


def audit_pdf_file(pdf_path: Path, max_pages: Optional[int] = None, engine: str = "pymupdf") -> List[PageAuditRecord]:
    """Audits all pages in a given PDF document using the selected engine."""
    pdf_path = Path(pdf_path)
    records: List[PageAuditRecord] = []

    if engine == "pymupdf" and fitz is not None:
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        pages_to_audit = min(total_pages, max_pages) if max_pages else total_pages
        for idx in range(pages_to_audit):
            page_number = idx + 1
            page = doc[idx]
            rec = audit_single_page_pymupdf(page, page_number=page_number, doc_name=pdf_path.name)
            records.append(rec)
        doc.close()
    else:
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        pages_to_audit = min(total_pages, max_pages) if max_pages else total_pages
        for idx in range(pages_to_audit):
            page_number = idx + 1
            page = reader.pages[idx]
            rec = audit_single_page(page, page_number=page_number, doc_name=pdf_path.name)
            records.append(rec)

    return records


def audit_all_raw_pdfs(raw_dir: Path, max_pages_per_doc: Optional[int] = None, engine: str = "pymupdf") -> List[PageAuditRecord]:
    """Audits all PDF files in the raw data directory."""
    raw_dir = Path(raw_dir)
    pdf_files = sorted(list(raw_dir.glob("*.pdf")))

    all_records: List[PageAuditRecord] = []
    for pdf in pdf_files:
        records = audit_pdf_file(pdf, max_pages=max_pages_per_doc, engine=engine)
        all_records.extend(records)
    return all_records


def inspect_ai_index_page_372(pdf_path: Path) -> Dict[str, Any]:
    """
    Performs an explicit deep inspection of AI Index 2024 page 372.
    Exposes spatial coordinates, image objects, and reading-order layout without hardcoding text.
    """
    reader = PdfReader(str(pdf_path))
    page = reader.pages[371]  # 0-indexed page 372

    raw_text = page.extract_text() or ""
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]

    # Collect spatial coordinates
    spatial_elements: List[Dict[str, Any]] = []

    def visitor(text, cm, tm, font_dict, font_size):
        if text and text.strip():
            spatial_elements.append({
                "text": text.strip(),
                "x": round(float(tm[4]), 1),
                "y": round(float(tm[5]), 1),
                "font_size": round(float(font_size), 1) if font_size else None,
            })

    page.extract_text(visitor_text=visitor)

    # Inspect images and XObjects
    res = page.get("/Resources", {})
    if hasattr(res, "get_object"):
        res = res.get_object()
    xobj = res.get("/XObject", {}) if isinstance(res, dict) else {}
    if hasattr(xobj, "get_object"):
        xobj = xobj.get_object()

    image_details = []
    if isinstance(xobj, dict):
        for key, obj in xobj.items():
            if hasattr(obj, "get_object"):
                obj = obj.get_object()
            if isinstance(obj, dict) and obj.get("/Subtype") == "/Image":
                image_details.append({
                    "id": str(key),
                    "width": obj.get("/Width"),
                    "height": obj.get("/Height"),
                    "color_space": str(obj.get("/ColorSpace")),
                })

    # Sort elements spatially by: Y descending (top to bottom), then X ascending (left to right)
    spatially_sorted = sorted(spatial_elements, key=lambda e: (-e["y"], e["x"]))

    return {
        "document": pdf_path.name,
        "page_number": 372,
        "raw_text_length": len(raw_text),
        "raw_line_count": len(lines),
        "image_count": len(image_details),
        "image_details": image_details,
        "raw_stream_order_snippet": lines[-12:],  # Trailing lines showing orphaned captions/dates
        "spatial_layout_sample": spatial_elements[:15],
        "spatially_sorted_sample": spatially_sorted[:15],
        "dates_stream_positions": [
            e for e in spatial_elements
            if re.search(r"Jul\.\s*[0-9]+", e["text"], re.IGNORECASE)
        ],
    }


def save_reports(records: List[PageAuditRecord], output_dir: Path, engine: str = "pymupdf") -> Tuple[Path, Path]:
    """Saves both JSON and CSV audit reports under output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "pdf_extraction_audit.json"
    csv_path = output_dir / "pdf_extraction_audit.csv"

    # Save JSON
    json_data = {
        "audit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "engine": engine,
        "total_pages_audited": len(records),
        "suspicious_pages_count": sum(1 for r in records if r.is_suspicious),
        "pages_with_images_count": sum(1 for r in records if r.image_count > 0),
        "pages_with_nul_bytes_count": sum(1 for r in records if r.nul_count > 0),
        "total_tables_detected": sum(r.table_count for r in records),
        "records": [r.to_dict() for r in records],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    # Save CSV
    fieldnames = [
        "document",
        "page_number",
        "char_count",
        "word_count",
        "image_count",
        "nul_count",
        "control_char_count",
        "replacement_char_count",
        "is_sparse",
        "caption_count",
        "has_clustered_captions",
        "has_clustered_dates",
        "reading_order_inversion",
        "is_suspicious",
        "table_count",
        "reasons_str",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.to_dict()[k] for k in fieldnames})

    return json_path, csv_path


def print_console_summary(records: List[PageAuditRecord], detailed_suspicious: bool = True, engine: str = "pymupdf"):
    """Prints a clean, informative terminal audit report."""
    total_pages = len(records)
    if total_pages == 0:
        print("[AUDIT] No pages audited.")
        return

    docs = sorted(list(set(r.document for r in records)))
    suspicious_records = [r for r in records if r.is_suspicious]
    image_pages = [r for r in records if r.image_count > 0]
    total_images = sum(r.image_count for r in records)
    nul_pages = [r for r in records if r.nul_count > 0]
    total_nul_bytes = sum(r.nul_count for r in records)
    sparse_pages = [r for r in records if r.is_sparse]
    reading_order_pages = [r for r in records if r.reading_order_inversion]
    total_tables = sum(r.table_count for r in records)

    print("=" * 80)
    print("  ENTERPRISE RAG - PDF EXTRACTION QUALITY AUDIT REPORT")
    print("=" * 80)
    print(f"  Extraction Engine       : {engine.upper()}")
    print(f"  Total Documents Audited : {len(docs)}")
    print(f"  Total Pages Audited     : {total_pages}")
    print(f"  Pages with Images       : {len(image_pages)} ({len(image_pages)/total_pages*100:.1f}%) [Total images: {total_images}]")
    print(f"  Pages with NUL Bytes    : {len(nul_pages)} [Total raw NUL bytes: {total_nul_bytes}]")
    print(f"  Sparse/Empty Text Pages : {len(sparse_pages)} ({len(sparse_pages)/total_pages*100:.1f}%)")
    print(f"  Reading Order Inversions: {len(reading_order_pages)}")
    print(f"  Tables Detected         : {total_tables}")
    print(f"  Suspicious Pages Flagged: {len(suspicious_records)} ({len(suspicious_records)/total_pages*100:.1f}%)")
    print("-" * 80)

    # Breakdown by document
    print("\n[PER-DOCUMENT AUDIT BREAKDOWN]")
    hdr = f"  {'Document':<40} | {'Pages':<6} | {'Suspicious':<10} | {'Images':<8} | {'NUL Bytes':<9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for doc in docs:
        doc_recs = [r for r in records if r.document == doc]
        doc_susp = sum(1 for r in doc_recs if r.is_suspicious)
        doc_imgs = sum(r.image_count for r in doc_recs)
        doc_nuls = sum(r.nul_count for r in doc_recs)
        short_doc = doc if len(doc) <= 40 else doc[:37] + "..."
        print(f"  {short_doc:<40} | {len(doc_recs):<6} | {doc_susp:<10} | {doc_imgs:<8} | {doc_nuls:<9}")

    # Top sample of suspicious pages
    if detailed_suspicious and suspicious_records:
        print("\n[SAMPLE SUSPICIOUS PAGES (Top 25)]")
        shdr = f"  {'Document':<35} | {'Page':<5} | {'Chars':<6} | {'Img':<4} | {'Reason':<45}"
        print(shdr)
        print("  " + "-" * (len(shdr) - 2))
        for r in suspicious_records[:25]:
            short_doc = r.document if len(r.document) <= 35 else r.document[:32] + "..."
            reason_str = "; ".join(r.reasons)
            if len(reason_str) > 45:
                reason_str = reason_str[:42] + "..."
            print(f"  {short_doc:<35} | {r.page_number:<5} | {r.char_count:<6} | {r.image_count:<4} | {reason_str:<45}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Audit PDF extraction quality for Enterprise RAG.")
    parser.add_argument(
        "--engine",
        type=str,
        choices=["pymupdf", "pypdf"],
        default="pymupdf",
        help="Extraction engine to audit (default: pymupdf, baseline: pypdf)",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Path to raw documents directory (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Path to output report directory (default: reports)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Specific PDF file to audit (relative or absolute)",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=None,
        help="Specific 1-indexed page number to audit",
    )
    parser.add_argument(
        "--inspect-372",
        action="store_true",
        help="Explicitly perform deep coordinate and layout inspection on AI Index page 372",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional limit on pages audited per document for fast checks",
    )

    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    output_path = Path(args.output_dir)

    print(f"Initializing PDF Extraction Audit from: {raw_path}")
    start_time = time.time()

    if args.inspect_372 or (args.page == 372 and args.file and "Artificial-Intelligence" in args.file):
        ai_index_file = raw_path / "Artificial-Intelligence-Index-Report-2024-Stanford-University.pdf"
        if not ai_index_file.exists():
            print(f"[ERROR] AI Index file not found at: {ai_index_file}")
            sys.exit(1)

        print("\n" + "=" * 80)
        print("  DEEP INSPECTION: AI INDEX 2024 (PAGE 372)")
        print("=" * 80)
        details = inspect_ai_index_page_372(ai_index_file)
        print(f"  Extracted Text Length : {details['raw_text_length']} characters across {details['raw_line_count']} lines")
        print(f"  Detected Image Objects: {details['image_count']} image objects:")
        for img in details["image_details"]:
            print(f"    - ID: {img['id']:<8} Dimensions: {img['width']}x{img['height']} ({img['color_space']})")

        print("\n  [Stream Order Anomaly - Trailing Lines Dumped at Page Bottom]:")
        for ln in details["raw_stream_order_snippet"]:
            print(f"    | {ln}")

        print("\n  [Date Badges on Timeline with 2D Coordinates (Left Margin X ~ 60)]:")
        for dt in details["dates_stream_positions"]:
            print(f"    - Date: {dt['text']:<12} Coordinates: X={dt['x']:<6} Y={dt['y']:<6} (Stream position: separated from event titles)")

        print("=" * 80)
        if not args.file and not args.page and not args.max_pages:
            # If only --inspect-372 was passed, exit after showing inspection
            return

    # Run audit on requested scope
    if args.file:
        target_file = Path(args.file)
        if not target_file.is_absolute():
            target_file = raw_path / target_file
        if not target_file.exists():
            print(f"[ERROR] Specified file not found: {target_file}")
            sys.exit(1)

        if args.page:
            if args.engine == "pymupdf" and fitz is not None:
                doc = fitz.open(str(target_file))
                if args.page < 1 or args.page > len(doc):
                    print(f"[ERROR] Page {args.page} out of bounds (1..{len(doc)})")
                    sys.exit(1)
                records = [audit_single_page_pymupdf(doc[args.page - 1], args.page, target_file.name)]
                doc.close()
            else:
                reader = PdfReader(str(target_file))
                if args.page < 1 or args.page > len(reader.pages):
                    print(f"[ERROR] Page {args.page} out of bounds (1..{len(reader.pages)})")
                    sys.exit(1)
                records = [audit_single_page(reader.pages[args.page - 1], args.page, target_file.name)]
        else:
            records = audit_pdf_file(target_file, max_pages=args.max_pages, engine=args.engine)
    else:
        records = audit_all_raw_pdfs(raw_path, max_pages_per_doc=args.max_pages, engine=args.engine)

    elapsed = time.time() - start_time
    print(f"\nAudit completed in {elapsed:.2f} seconds.")

    # Save reports
    json_file, csv_file = save_reports(records, output_path, engine=args.engine)
    print(f"Saved audit reports to:")
    print(f"  - JSON: {json_file}")
    print(f"  - CSV : {csv_file}\n")

    # Display console summary
    print_console_summary(records, engine=args.engine)


if __name__ == "__main__":
    main()
