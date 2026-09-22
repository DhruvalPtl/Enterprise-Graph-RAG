"""
Generates a realistic multi-page PDF document for testing the RAG ingestion pipeline.

This document simulates an Enterprise AI Platform Architecture guide with
multiple sections, headings, lists, and multi-page content.
"""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def generate_enterprise_pdf(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#1a365d"),
        spaceAfter=14,
    )
    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Heading2"],
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#2b6cb0"),
        spaceBefore=12,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2d3748"),
        spaceAfter=8,
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=18,
        firstLineIndent=-12,
        spaceAfter=4,
    )

    story = []

    # --- PAGE 1 ---
    story.append(Paragraph("Enterprise AI Platform Architecture", title_style))
    story.append(Paragraph("Document Version: 1.0 | Classification: Internal Engineering Guide", body_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Executive Summary & Mission", h1_style))
    story.append(Paragraph(
        "The Enterprise AI Platform is engineered to deliver reliable, deterministic, and auditable "
        "autonomous workflows for enterprise environments. The core challenge in enterprise GenAI deployment "
        "is not model capacity, but context precision, hallucination mitigation, and strict enterprise permissioning.",
        body_style
    ))
    story.append(Paragraph(
        "Our unified knowledge retrieval layer powers Forward Deployed AI Engineers and internal systems by combining "
        "vector similarity with inverted index keyword search, reranking mechanisms, and citation tracking. This document "
        "serves as the reference specification for data ingestion and text chunking requirements.",
        body_style
    ))

    story.append(Paragraph("2. Ingestion Guarantees", h1_style))
    story.append(Paragraph(
        "To maintain audit compliance across banking, healthcare, and enterprise software customers, the ingestion pipeline "
        "must enforce the following guarantees on every processed document:",
        body_style
    ))
    story.append(Paragraph("• <b>Deterministic Identifiers:</b> Document IDs must be calculated from cryptographic hashes of source content.", bullet_style))
    story.append(Paragraph("• <b>Traceable Metadata:</b> Source document path, page numbers, and detected section headings must accompany every chunk.", bullet_style))
    story.append(Paragraph("• <b>Lossless Boundaries:</b> Chunk boundaries must preserve sentence and paragraph coherence rather than performing naive character cuts.", bullet_style))
    story.append(Paragraph("• <b>Idempotent Execution:</b> Re-running the pipeline on unchanged source files must produce identical chunk keys.", bullet_style))

    story.append(PageBreak())

    # --- PAGE 2 ---
    story.append(Paragraph("3. Chunking Strategy & Retrieval Relevance", h1_style))
    story.append(Paragraph(
        "Traditional naive chunking slices text into fixed token windows (e.g., 500 characters) regardless of document structure. "
        "In production enterprise systems, this causes severe degradation in retrieval precision because critical context—such as table headers, "
        "clauses in legal contracts, or code blocks—gets severed across chunk boundaries.",
        body_style
    ))
    story.append(Paragraph(
        "We recommend a recursive structural chunking approach. The algorithm attempts to split at structural headers, then paragraphs, "
        "then list items, and finally sentence boundaries. Controlled overlap of 15-20% ensures that boundary-crossing thoughts maintain "
        "sufficient co-occurrence signal for dense embedding models.",
        body_style
    ))

    story.append(Paragraph("4. Security and Isolation Protocols", h1_style))
    story.append(Paragraph(
        "Customer documents often contain sensitive Intellectual Property or Personally Identifiable Information (PII). "
        "The data plane must adhere to the principle of least privilege:",
        body_style
    ))
    story.append(Paragraph("• Role-Based Access Control (RBAC) must be propagated from the source document down to individual chunk metadata.", bullet_style))
    story.append(Paragraph("• Encryption at rest must utilize AES-256 for all serialized chunk artifacts and vector databases.", bullet_style))
    story.append(Paragraph("• No customer data is transmitted to external endpoints without customer-managed encryption key (CMEK) approval.", bullet_style))

    doc.build(story)
    print(f"Generated sample PDF at: {output_path}")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    target_file = out_dir / "enterprise_platform_architecture.pdf"
    generate_enterprise_pdf(target_file)
