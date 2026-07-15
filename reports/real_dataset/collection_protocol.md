# Real Automotive Part Dataset Collection Protocol

## Purpose

The real dataset contains photographs of automotive parts and short text descriptions.

Each physical part receives one `part_group_id`. Different photographs of the same physical part must use the same group identifier.

## Initial target

The minimum target is:

- 10 automotive part categories
- 10 physical parts per category
- 2 photographs per physical part
- 200 real images
- 600 image-text samples

Each image will later be paired with three descriptions:

- `MATCH`
- `PARTIAL_MATCH`
- `MISMATCH`

## Categories

- starter
- alternator
- brake_disc
- brake_pad
- shock_absorber
- coil_spring
- headlight
- taillight
- oil_filter
- air_filter

## File naming

The filename contains the category, physical part number, and view.

Examples:

```text
starter_001_front.jpg
starter_001_detail.jpg
brake_disc_004_front.jpg
brake_disc_004_rear.jpg