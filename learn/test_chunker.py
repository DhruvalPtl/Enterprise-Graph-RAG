import sys
from pathlib import Path

# Add project root directory to Python path so we can import 'app'
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models import Document, DocumentPage
from app.chunker import RecursiveStructuralChunker

# Sample text with headings and paragraphs
sample_text = """
# Company Security Guidelines

All employees must enable Multi-Factor Authentication (MFA) on their accounts.
Passwords must be at least 16 characters in length and updated every 90 days.

## Data Retention Policy

Customer personal data must not be stored on local machines.
All data must be uploaded to encrypted cloud storage within 24 hours of collection.
Records will be purged automatically after 7 years.
"""

doc = Document(
    id="doc_test",
    content=sample_text,
    metadata={"document_name": "security.txt"},
    pages=[DocumentPage(page_number=1, text=sample_text)]
)

chunker = RecursiveStructuralChunker(chunk_size=200, chunk_overlap=50)
chunks = chunker.chunk_document(doc)

print(f"Total chunks created: {len(chunks)}\n")
for c in chunks:
    print(f"[{c.chunk_id}] Section: '{c.metadata['section']}' | Chars: {c.metadata['char_count']}")
    print(f"Text:\n{c.text}")
    print("-" * 50)
