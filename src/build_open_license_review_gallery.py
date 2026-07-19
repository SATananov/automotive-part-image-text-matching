from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any

from src.open_license_dataset_config import (
    OPEN_LICENSE_REVIEW_COLUMNS,
    OPEN_LICENSE_REVIEW_GALLERY_PATH,
    OPEN_LICENSE_REVIEW_PATH,
)


def read_review_rows() -> list[dict[str, str]]:
    if not OPEN_LICENSE_REVIEW_PATH.is_file():
        return []

    with OPEN_LICENSE_REVIEW_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != OPEN_LICENSE_REVIEW_COLUMNS:
            raise RuntimeError(
                "The open-license review workbook has an "
                "unexpected schema."
            )
        return [
            {
                column: str(row.get(column, "")).strip()
                for column in OPEN_LICENSE_REVIEW_COLUMNS
            }
            for row in reader
        ]


def escape(value: object) -> str:
    return html.escape(str(value or ""), quote=True)


def render_gallery(rows: list[dict[str, str]]) -> str:
    cards: list[str] = []
    report_directory = OPEN_LICENSE_REVIEW_GALLERY_PATH.parent

    for row in sorted(
        rows,
        key=lambda item: (
            item["part_category"],
            item["asset_id"],
        ),
    ):
        image_path = Path(row["local_path"])
        relative_image = Path(
            "..",
            "..",
            *image_path.parts,
        ).as_posix()

        cards.append(
            f"""
<article class="card">
  <img src="{escape(relative_image)}"
       alt="{escape(row['commons_title'])}">
  <div class="body">
    <div class="category">{escape(row['part_category'])}</div>
    <h2>{escape(row['asset_id'])}</h2>
    <p><strong>Decision:</strong>
       {escape(row['operator_decision'])}</p>
    <p><strong>Title:</strong>
       {escape(row['commons_title'])}</p>
    <p><strong>Author:</strong>
       {escape(row['author'] or 'See Commons source')}</p>
    <p><strong>License:</strong>
       <a href="{escape(row['license_url'])}">
       {escape(row['license_short_name'])}</a></p>
    <p><a href="{escape(row['description_url'])}">
       Open Wikimedia Commons source</a></p>
  </div>
</article>
""".strip()
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Open-License Image Review</title>
<style>
body {{
  font-family: Arial, sans-serif;
  margin: 0;
  background: #f4f5f7;
  color: #1f2937;
}}
header {{
  padding: 24px;
  background: white;
  border-bottom: 1px solid #d1d5db;
}}
main {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 18px;
  padding: 24px;
}}
.card {{
  background: white;
  border: 1px solid #d1d5db;
  border-radius: 12px;
  overflow: hidden;
}}
.card img {{
  width: 100%;
  height: 220px;
  object-fit: contain;
  background: #eef0f3;
}}
.body {{
  padding: 16px;
}}
.category {{
  font-size: 12px;
  font-weight: bold;
  text-transform: uppercase;
}}
h2 {{
  font-size: 16px;
  overflow-wrap: anywhere;
}}
p {{
  font-size: 14px;
  line-height: 1.4;
}}
a {{
  color: #075985;
}}
</style>
</head>
<body>
<header>
  <h1>Step 010.1 Open-License Review</h1>
  <p>{len(rows)} candidate images. Edit only the operator columns
  in <code>data/external/open_license/open_license_review.csv</code>.</p>
</header>
<main>
{chr(10).join(cards)}
</main>
</body>
</html>
"""


def build_gallery() -> dict[str, Any]:
    rows = read_review_rows()
    OPEN_LICENSE_REVIEW_GALLERY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    OPEN_LICENSE_REVIEW_GALLERY_PATH.write_text(
        render_gallery(rows),
        encoding="utf-8",
        newline="\n",
    )
    return {
        "status": "PASS",
        "images": len(rows),
        "gallery": str(OPEN_LICENSE_REVIEW_GALLERY_PATH),
    }


def main() -> None:
    report = build_gallery()
    print("Step 010.1 open-license review gallery")
    print(f"- Status: {report['status']}")
    print(f"- Images: {report['images']}")
    print(f"- Gallery: {report['gallery']}")


if __name__ == "__main__":
    main()
