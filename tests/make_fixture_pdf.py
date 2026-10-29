"""Write a small multi-page text PDF without third-party libraries (used for test fixtures).

Run: python tests/make_fixture_pdf.py  (regenerates the committed fixture PDF)
"""

from __future__ import annotations

from pathlib import Path


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(pages: list[list[str]]) -> bytes:
    """Each page is a list of text lines. Helvetica 11pt, simple top-down layout."""
    objects: list[bytes] = []  # 1-indexed object bodies, filled below

    n_pages = len(pages)
    font_id = 3 + 2 * n_pages  # catalog=1, pages=2, then (page, content) pairs, then font
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n_pages))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    for i, lines in enumerate(pages):
        page_id, content_id = 3 + 2 * i, 4 + 2 * i
        ops = ["BT", "/F1 11 Tf", "14 TL", "72 740 Td"]
        for line in lines:
            ops.append(f"({_escape(line)}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()
        )
        objects.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


MEMO_PAGES = [
    [
        "FICTIONAL DEMO DOCUMENT - not real data",
        "Kestrel Field Reserves Memo (Marrow Basin, fictional)",
        "Prepared by Tallgrass Fictional Energy for the 2025 year-end review.",
        "",
        "1. Reserves summary as of 2025-12-31",
        "Proved (1P) reserves for Kestrel Field are 14.2 MMbbl of oil.",
        "Proved plus probable (2P) reserves are 21.7 MMbbl of oil.",
        "Proved, probable and possible (3P) reserves are 30.1 MMbbl of oil.",
        "The booked recovery factor is 28 percent of original oil in place.",
        "Original oil in place is estimated at 77 MMbbl.",
    ],
    [
        "FICTIONAL DEMO DOCUMENT - not real data",
        "2. Development plan and risks",
        "The 2026 program calls for three infill wells: Kestrel-7, Kestrel-8 and Kestrel-9.",
        "Base decline for the existing wells is 11 percent per year.",
        "The main subsurface risk is rising water cut on the eastern flank,",
        "where the produced water fraction reached 38 percent in December 2025.",
        "Capital cost for the three-well program is 18.5 million USD.",
        "Management recommends a waterflood pilot only if the water cut exceeds 50 percent.",
    ],
]

FACILITIES_PAGES = [
    [
        "FICTIONAL DEMO DOCUMENT - not real data",
        "Kestrel Field Facilities Study (Marrow Basin, fictional)",
        "Prepared by Tallgrass Fictional Energy, Facilities Engineering.",
        "",
        "1. Central tank battery",
        "The central tank battery has three 500 barrel oil tanks and two 400 barrel water tanks.",
        "The inlet separator is rated for 15,000 barrels per day of total liquid.",
        "Oil is sold by truck. Trucks collect on Tuesdays and Fridays.",
    ],
    [
        "FICTIONAL DEMO DOCUMENT - not real data",
        "2. Gas compression",
        "A single-stage reciprocating compressor was installed in 2019 and is rated for 2.4 MMscf per day.",
        "The availability target for the compressor is 95 percent.",
        "The compressor runs on grid power. A 250 kW diesel generator provides backup.",
    ],
    [
        "FICTIONAL DEMO DOCUMENT - not real data",
        "3. Produced water handling",
        "Produced water is injected into the disposal well Kestrel-D1 at an average of 9,800 barrels per day.",
        "The permitted maximum injection rate for Kestrel-D1 is 12,000 barrels per day.",
        "The study recommends adding a second water filter skid in the fourth quarter of 2026,",
        "at an estimated cost of 1.2 million USD.",
    ],
]

if __name__ == "__main__":
    corpus = Path(__file__).parent / "fixtures" / "corpus"
    for name, pages in (("kestrel_reserves_memo.pdf", MEMO_PAGES), ("kestrel_facilities_study.pdf", FACILITIES_PAGES)):
        (corpus / name).write_bytes(build_pdf(pages))
        print(f"wrote {corpus / name}")
