from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from src.dataset_config import (
    METADATA_COLUMNS,
    PART_CATEGORIES,
    PART_FAMILIES,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "development"
IMAGE_DIRECTORY = OUTPUT_DIRECTORY / "images"
METADATA_PATH = OUTPUT_DIRECTORY / "metadata.csv"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "development_dataset_summary.md"

IMAGE_SIZE = 224
PARTS_PER_CATEGORY = 2

DISPLAY_NAMES = {
    "starter": "starter motor",
    "alternator": "alternator",
    "brake_disc": "brake disc",
    "brake_pad": "brake pad set",
    "shock_absorber": "shock absorber",
    "coil_spring": "coil spring",
    "headlight": "headlight assembly",
    "taillight": "taillight assembly",
    "oil_filter": "oil filter",
    "air_filter": "air filter",
}

CATEGORY_TO_FAMILY = {
    category: family
    for family, categories in PART_FAMILIES.items()
    for category in categories
}

FAMILY_ORDER = tuple(PART_FAMILIES)


def get_partial_match_category(category: str) -> str:
    family = CATEGORY_TO_FAMILY[category]
    family_categories = PART_FAMILIES[family]

    return next(
        other_category
        for other_category in family_categories
        if other_category != category
    )


def get_mismatch_category(category: str) -> str:
    current_family = CATEGORY_TO_FAMILY[category]
    current_index = FAMILY_ORDER.index(current_family)
    next_family = FAMILY_ORDER[(current_index + 1) % len(FAMILY_ORDER)]

    return PART_FAMILIES[next_family][0]


def create_part_image(
    category: str,
    variant: int,
    destination: Path,
) -> None:
    background = (245, 245, 245)
    dark = (45, 45, 45)
    medium = (105, 105, 105)
    light = (180, 180, 180)

    image = Image.new(
        mode="RGB",
        size=(IMAGE_SIZE, IMAGE_SIZE),
        color=background,
    )

    draw = ImageDraw.Draw(image)
    shift = (variant - 1) * 6

    if category == "starter":
        draw.rounded_rectangle(
            (48 + shift, 82, 158 + shift, 142),
            radius=16,
            fill=medium,
            outline=dark,
            width=5,
        )
        draw.ellipse(
            (132 + shift, 68, 190 + shift, 128),
            fill=light,
            outline=dark,
            width=5,
        )
        draw.rectangle(
            (36 + shift, 96, 58 + shift, 128),
            fill=dark,
        )

    elif category == "alternator":
        draw.ellipse(
            (50 + shift, 48, 174 + shift, 176),
            fill=medium,
            outline=dark,
            width=6,
        )
        draw.ellipse(
            (88 + shift, 86, 136 + shift, 134),
            fill=background,
            outline=dark,
            width=5,
        )

        for angle in range(0, 360, 45):
            radians = math.radians(angle)
            x1 = 112 + shift + int(31 * math.cos(radians))
            y1 = 111 + int(31 * math.sin(radians))
            x2 = 112 + shift + int(52 * math.cos(radians))
            y2 = 111 + int(52 * math.sin(radians))
            draw.line((x1, y1, x2, y2), fill=dark, width=5)

    elif category == "brake_disc":
        draw.ellipse(
            (42 + shift, 42, 182 + shift, 182),
            fill=light,
            outline=dark,
            width=6,
        )
        draw.ellipse(
            (73 + shift, 73, 151 + shift, 151),
            fill=background,
            outline=medium,
            width=5,
        )
        draw.ellipse(
            (99 + shift, 99, 125 + shift, 125),
            fill=dark,
        )

    elif category == "brake_pad":
        draw.rounded_rectangle(
            (45 + shift, 62, 105 + shift, 160),
            radius=12,
            fill=medium,
            outline=dark,
            width=5,
        )
        draw.rounded_rectangle(
            (119 + shift, 62, 179 + shift, 160),
            radius=12,
            fill=medium,
            outline=dark,
            width=5,
        )

    elif category == "shock_absorber":
        draw.rectangle(
            (100 + shift, 38, 124 + shift, 92),
            fill=dark,
        )
        draw.rounded_rectangle(
            (82 + shift, 82, 142 + shift, 170),
            radius=18,
            fill=medium,
            outline=dark,
            width=5,
        )
        draw.line(
            (112 + shift, 170, 112 + shift, 196),
            fill=dark,
            width=9,
        )

    elif category == "coil_spring":
        points = []

        for y_position in range(38, 190, 4):
            x_position = (
                112
                + shift
                + int(31 * math.sin((y_position - 38) / 10))
            )
            points.append((x_position, y_position))

        draw.line(points, fill=dark, width=8)

    elif category == "headlight":
        draw.polygon(
            (
                (38 + shift, 95),
                (74 + shift, 56),
                (178 + shift, 72),
                (190 + shift, 140),
                (67 + shift, 158),
            ),
            fill=light,
            outline=dark,
        )
        draw.ellipse(
            (87 + shift, 82, 143 + shift, 138),
            fill=background,
            outline=dark,
            width=5,
        )

    elif category == "taillight":
        draw.rounded_rectangle(
            (48 + shift, 57, 176 + shift, 167),
            radius=32,
            fill=medium,
            outline=dark,
            width=6,
        )
        draw.line(
            (91 + shift, 60, 91 + shift, 164),
            fill=background,
            width=6,
        )
        draw.line(
            (133 + shift, 60, 133 + shift, 164),
            fill=background,
            width=6,
        )

    elif category == "oil_filter":
        draw.ellipse(
            (69 + shift, 45, 155 + shift, 79),
            fill=light,
            outline=dark,
            width=5,
        )
        draw.rectangle(
            (69 + shift, 62, 155 + shift, 165),
            fill=medium,
            outline=dark,
            width=5,
        )
        draw.ellipse(
            (69 + shift, 148, 155 + shift, 182),
            fill=medium,
            outline=dark,
            width=5,
        )

    elif category == "air_filter":
        draw.rounded_rectangle(
            (42 + shift, 55, 182 + shift, 169),
            radius=12,
            fill=light,
            outline=dark,
            width=6,
        )

        for x_position in range(60 + shift, 175 + shift, 14):
            draw.line(
                (x_position, 68, x_position, 156),
                fill=medium,
                width=4,
            )

    else:
        raise ValueError(f"Unsupported category: {category}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG")


def create_metadata_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for category in PART_CATEGORIES:
        family = CATEGORY_TO_FAMILY[category]
        partial_category = get_partial_match_category(category)
        mismatch_category = get_mismatch_category(category)

        for group_number in range(1, PARTS_PER_CATEGORY + 1):
            part_group_id = f"{category}_{group_number:03d}"
            image_id = f"{part_group_id}_01"
            image_filename = f"{image_id}.png"
            relative_image_path = f"data/development/images/{image_filename}"
            absolute_image_path = PROJECT_ROOT / relative_image_path

            create_part_image(
                category=category,
                variant=group_number,
                destination=absolute_image_path,
            )

            descriptions = (
                (
                    "MATCH",
                    f"Automotive {DISPLAY_NAMES[category]}.",
                ),
                (
                    "PARTIAL_MATCH",
                    f"Automotive {DISPLAY_NAMES[partial_category]}.",
                ),
                (
                    "MISMATCH",
                    f"Automotive {DISPLAY_NAMES[mismatch_category]}.",
                ),
            )

            for label, description in descriptions:
                rows.append(
                    {
                        "sample_id": f"{image_id}_{label.lower()}",
                        "image_id": image_id,
                        "part_group_id": part_group_id,
                        "image_path": relative_image_path,
                        "part_family": family,
                        "part_category": category,
                        "description": description,
                        "label": label,
                        "source": "generated_development",
                    }
                )

    return rows


def write_metadata(rows: list[dict[str, str]]) -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    with METADATA_PATH.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as metadata_file:
        writer = csv.DictWriter(
            metadata_file,
            fieldnames=METADATA_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, str]]) -> None:
    label_counts = Counter(row["label"] for row in rows)
    category_counts = Counter(row["part_category"] for row in rows)

    summary_lines = [
        "# Development Dataset Summary",
        "",
        f"- Images: {len({row['image_id'] for row in rows})}",
        f"- Physical part groups: {len({row['part_group_id'] for row in rows})}",
        f"- Image-text samples: {len(rows)}",
        "",
        "## Label distribution",
        "",
    ]

    for label, count in sorted(label_counts.items()):
        summary_lines.append(f"- {label}: {count}")

    summary_lines.extend(
        [
            "",
            "## Category distribution",
            "",
        ]
    )

    for category, count in sorted(category_counts.items()):
        summary_lines.append(f"- {category}: {count}")

    summary_lines.extend(
        [
            "",
            "This dataset is intended only for pipeline development and testing.",
            "It is not used for the final model evaluation.",
        ]
    )

    SUMMARY_PATH.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    rows = create_metadata_rows()
    write_metadata(rows)
    write_summary(rows)

    print("Development dataset created successfully.")
    print(f"Images: {len({row['image_id'] for row in rows})}")
    print(f"Samples: {len(rows)}")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()