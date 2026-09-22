"""
Selects candidate benchmark pages from reports/pdf_extraction_audit.json.
"""
import json
from pathlib import Path

with open("reports/pdf_extraction_audit.json", encoding="utf-8") as f:
    data = json.load(f)

records = data["records"]

print("=== CANDIDATES FOR BENCHMARK ===")

# Test A: AI Index 2024
ai_pages = [r for r in records if "Artificial-Intelligence" in r["document"]]
p372 = next(r for r in ai_pages if r["page_number"] == 372)
print(f"TEST A1 (Mandatory): AI Index Page 372 -> chars={p372['char_count']}, images={p372['image_count']}, captions={p372['caption_count']}, reasons={p372['reasons']}")

# Another interesting AI Index page: e.g. a chart/table benchmark page
ai_chart = [r for r in ai_pages if r["image_count"] >= 2 and r["char_count"] > 1000 and r["page_number"] != 372][0]
print(f"TEST A2 (Chart/Figure): AI Index Page {ai_chart['page_number']} -> chars={ai_chart['char_count']}, images={ai_chart['image_count']}, captions={ai_chart['caption_count']}, reasons={ai_chart['reasons']}")

# Test B: Speech and Language Processing
slp_pages = [r for r in records if "speech-and-language" in r["document"]]
slp_nul = [r for r in slp_pages if r["nul_count"] > 0][0] # e.g. page 97 (22 NUL bytes)
print(f"TEST B1 (NUL Bytes & Tables): SLP Page {slp_nul['page_number']} -> chars={slp_nul['char_count']}, nul={slp_nul['nul_count']}, images={slp_nul['image_count']}")

slp_complex = [r for r in slp_pages if r["image_count"] >= 3 and r["char_count"] > 1500][0]
print(f"TEST B2 (Complex Layout/Trees): SLP Page {slp_complex['page_number']} -> chars={slp_complex['char_count']}, images={slp_complex['image_count']}, reasons={slp_complex['reasons']}")

# Test C: Structured / Visual pages
fm_pages = [r for r in records if "foundation-models" in r["document"]]
fm_table = [r for r in fm_pages if r["caption_count"] >= 2 and r["image_count"] >= 1 and r["char_count"] > 1200][0]
print(f"TEST C1 (Table/Architecture): Foundation Models Page {fm_table['page_number']} -> chars={fm_table['char_count']}, images={fm_table['image_count']}, captions={fm_table['caption_count']}")

eis_pages = [r for r in records if "eisenstein" in r["document"]]
eis_math = [r for r in eis_pages if r["reading_order_inversion"] and r["char_count"] > 1500][0]
print(f"TEST C2 (Equations/LaTeX): Eisenstein NLP Page {eis_math['page_number']} -> chars={eis_math['char_count']}, reasons={eis_math['reasons']}")
