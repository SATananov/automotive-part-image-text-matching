# First Batch Capture Checklist

This checklist is the operational guide for capturing and importing the first real automotive-parts batch.

## File naming rule

Use the exact descriptive filename from the table below. The pattern is:

```text
real_<part_category>_001_<view>.jpg
```

Examples:

```text
real_starter_001_front.jpg
real_brake_disc_001_detail.jpg
```

Do not rename photographs to technical intake IDs. The mapping file keeps the internal `intake_id` separately.

## Capture rules

- Photograph one physical part per `part_group_id`.
- Use the same physical part for its `front` and `detail` images.
- Keep the full part visible in the `front` image.
- Use the `detail` image for a meaningful connector, label, mounting point, surface, or distinguishing feature.
- Use even light, a simple background, sharp focus, and no digital watermark.
- Avoid hands, faces, licence plates, invoices, serial numbers, or unrelated warehouse information.
- Do not reuse one image for two filenames.
- JPEG is recommended. PNG and JPEG are accepted by the local importer.

## Exact first-batch filenames

| No. | Category | View | Filename | Internal intake ID |
|---:|---|---|---|---|
| 1 | `starter` | `front` | `real_starter_001_front.jpg` | `intake_000001` |
| 2 | `starter` | `detail` | `real_starter_001_detail.jpg` | `intake_000002` |
| 3 | `alternator` | `front` | `real_alternator_001_front.jpg` | `intake_000003` |
| 4 | `alternator` | `detail` | `real_alternator_001_detail.jpg` | `intake_000004` |
| 5 | `brake_disc` | `front` | `real_brake_disc_001_front.jpg` | `intake_000005` |
| 6 | `brake_disc` | `detail` | `real_brake_disc_001_detail.jpg` | `intake_000006` |
| 7 | `brake_pad` | `front` | `real_brake_pad_001_front.jpg` | `intake_000007` |
| 8 | `brake_pad` | `detail` | `real_brake_pad_001_detail.jpg` | `intake_000008` |
| 9 | `shock_absorber` | `front` | `real_shock_absorber_001_front.jpg` | `intake_000009` |
| 10 | `shock_absorber` | `detail` | `real_shock_absorber_001_detail.jpg` | `intake_000010` |
| 11 | `coil_spring` | `front` | `real_coil_spring_001_front.jpg` | `intake_000011` |
| 12 | `coil_spring` | `detail` | `real_coil_spring_001_detail.jpg` | `intake_000012` |
| 13 | `headlight` | `front` | `real_headlight_001_front.jpg` | `intake_000013` |
| 14 | `headlight` | `detail` | `real_headlight_001_detail.jpg` | `intake_000014` |
| 15 | `taillight` | `front` | `real_taillight_001_front.jpg` | `intake_000015` |
| 16 | `taillight` | `detail` | `real_taillight_001_detail.jpg` | `intake_000016` |
| 17 | `oil_filter` | `front` | `real_oil_filter_001_front.jpg` | `intake_000017` |
| 18 | `oil_filter` | `detail` | `real_oil_filter_001_detail.jpg` | `intake_000018` |
| 19 | `air_filter` | `front` | `real_air_filter_001_front.jpg` | `intake_000019` |
| 20 | `air_filter` | `detail` | `real_air_filter_001_detail.jpg` | `intake_000020` |

## Local import procedure

1. Place the renamed photographs in `data/real/capture_inbox/batch_001/`.
2. Run `python -m src.project_cli import-first-real-batch`.
3. Review `data/real/processed/first_batch_local_import_inventory.csv` and `reports/real_dataset/first_batch_local_import_readiness.md`.
4. When readiness is `READY_FOR_STAGING`, run `python -m src.project_cli stage-first-real-batch-capture`.
5. Review the staging inventory and pending queue draft before any live queue change.

The importer copies original bytes without pixel conversion. Pixel normalization happens only in the staging workflow.
