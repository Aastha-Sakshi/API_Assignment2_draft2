"""
Generate data/sample_resume.pdf — a fabricated resume for demos and screenshots.

    python scripts/make_sample_resume.py

Why this exists: the screenshots in the report render the resume's contact block
in full. Driving the UI with a real CV means every captured image carries that
person's name, phone and email, which cannot be published. This file is entirely
invented, so the same screenshots are safe to commit.

It is a PDF with a real text layer, not an image, so it exercises the same path
a real resume does: DiT classifies the page, PyMuPDF extracts losslessly, and
OCR is correctly skipped.
"""

from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample_resume.pdf"

NAME = "Jane Doe"
CONTACT = "+1-555-0142  |  Bengaluru, India  |  jane.doe@example.com  |  linkedin.com/in/janedoe"

SECTIONS = [
    ("SUMMARY", [
        "Backend engineer with 6 years building and operating cloud services. Owns systems",
        "end to end: design, delivery, on-call. Strongest in Python, distributed systems and",
        "the operational side of machine learning.",
    ]),
    ("EXPERIENCE", [
        "Senior Backend Engineer — Northwind Cloud, Bengaluru            2022 - Present",
        "  - Led the migration of a monolith to 14 services on Kubernetes, cutting deploy",
        "    time from 40 minutes to under 5.",
        "  - Built an event pipeline on Kafka handling 120k messages/second at p99 < 80 ms.",
        "  - Introduced structured logging and SLO dashboards; halved mean time to detect.",
        "  - Mentored four engineers; ran the design review process for the platform team.",
        "",
        "Backend Engineer — Crestline Systems, Pune                      2019 - 2022",
        "  - Designed a multi-tenant billing service in Python and PostgreSQL serving",
        "    500000 accounts with zero data-loss incidents over three years.",
        "  - Cut AWS spend 31% by right-sizing workloads and adding autoscaling policies.",
        "  - Wrote the company's first CI pipeline; raised test coverage from 22% to 78%.",
    ]),
    ("SKILLS", [
        "Languages     Python, Go, SQL, TypeScript",
        "Cloud         AWS (EC2, S3, Lambda, RDS), Docker, Kubernetes, Terraform",
        "Data          PostgreSQL, Redis, Kafka, Elasticsearch",
        "Practices     CI/CD, observability, infrastructure as code, incident response",
    ]),
    ("EDUCATION", [
        "B.E. Computer Science — Pune Institute of Technology             2015 - 2019",
        "First class with distinction.",
    ]),
    ("CERTIFICATIONS", [
        "AWS Certified Solutions Architect - Associate                    2021",
        "Certified Kubernetes Administrator (CKA)                         2023",
    ]),
]


def build() -> None:
    doc = fitz.open()
    page = doc.new_page()  # A4 by default
    left, y = 56, 60

    page.insert_text((left, y), NAME, fontname="hebo", fontsize=20)
    y += 20
    page.insert_text((left, y), CONTACT, fontname="helv", fontsize=8.5)
    y += 12
    page.draw_line(fitz.Point(left, y), fitz.Point(539, y), color=(0.4, 0.4, 0.4), width=0.8)
    y += 20

    for heading, body in SECTIONS:
        page.insert_text((left, y), heading, fontname="hebo", fontsize=10.5)
        y += 6
        page.draw_line(fitz.Point(left, y), fitz.Point(539, y), color=(0.75, 0.75, 0.75), width=0.5)
        y += 13
        for line in body:
            page.insert_text((left, y), line, fontname="helv", fontsize=8.8)
            y += 11.5
        y += 9

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT))
    doc.close()

    text = "\n".join(p.get_text("text") for p in fitz.open(str(OUT)))
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size // 1024} KB)")
    print(f"  text layer: {len(text)} chars — extraction will not need OCR")


if __name__ == "__main__":
    build()
