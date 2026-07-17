# First Batch Operator Guide

This guide is for the person photographing the first controlled real-data batch. It keeps the capture session simple: one physical part at a time, one `front` photograph, one meaningful `detail` photograph, and the exact descriptive filenames already reserved by the project.

## Before the capture session

- Charge the phone or camera and clean the lens.
- Prepare a plain, non-reflective background with even light.
- Keep one physical part in the capture area at a time.
- Remove invoices, customer labels, registration plates, faces, hands, and unrelated warehouse information from the frame.
- Keep the full part visible in the `front` view.
- Use the `detail` view for a connector, mounting point, shape feature, friction surface, pleat pattern, or another useful identifying feature.
- Do not apply filters, watermarks, stickers, or screenshots.
- Do not reuse the same photograph under two filenames.

## Session preparation command

Run this command before, during, and after the capture session:

```powershell
python -m src.project_cli prepare-first-real-batch-session
```

It creates the operator worksheet:

```text
data/real/processed/first_batch_capture_session.csv
```

and the readable status report:

```text
reports/real_dataset/first_batch_capture_session_readiness.md
```

The command is read-only for the inbox, originals, staging, annotations, live queue, approval log, and manifest.

## Capture order

| No. | Physical part | Front filename | Detail filename | Useful detail subject |
|---:|---|---|---|---|
| 1 | Starter | `real_starter_001_front.jpg` | `real_starter_001_detail.jpg` | Pinion, solenoid connector, or mounting flange |
| 2 | Alternator | `real_alternator_001_front.jpg` | `real_alternator_001_detail.jpg` | Pulley, regulator connector, vents, or mounting ear |
| 3 | Brake disc | `real_brake_disc_001_front.jpg` | `real_brake_disc_001_detail.jpg` | Friction surface, ventilation channel, or hub holes |
| 4 | Brake pad | `real_brake_pad_001_front.jpg` | `real_brake_pad_001_detail.jpg` | Friction material, backing plate, clip, or profile |
| 5 | Shock absorber | `real_shock_absorber_001_front.jpg` | `real_shock_absorber_001_detail.jpg` | Rod, top mount, lower mount, or body marking |
| 6 | Coil spring | `real_coil_spring_001_front.jpg` | `real_coil_spring_001_detail.jpg` | End geometry, coil spacing, or paint mark |
| 7 | Headlight | `real_headlight_001_front.jpg` | `real_headlight_001_detail.jpg` | Rear connector, mounting tab, housing, or lens feature |
| 8 | Taillight | `real_taillight_001_front.jpg` | `real_taillight_001_detail.jpg` | Rear connector, mounting point, housing, or lens feature |
| 9 | Oil filter | `real_oil_filter_001_front.jpg` | `real_oil_filter_001_detail.jpg` | Thread, gasket, can shape, or permitted product marking |
| 10 | Air filter | `real_air_filter_001_front.jpg` | `real_air_filter_001_detail.jpg` | Pleats, frame edge, seal, or overall profile |

## File placement

Place each renamed photograph in:

```text
data/real/capture_inbox/batch_001/
```

JPEG is recommended. JPEG and PNG are accepted by the local importer. Keep only one extension for each planned filename stem.

## Status meanings

- `AWAITING_CAPTURE`: no planned photographs are available yet.
- `CAPTURE_SESSION_IN_PROGRESS`: at least one photograph exists, but the 20-file set is incomplete.
- `READY_FOR_LOCAL_IMPORT`: all 20 photographs are available in the local inbox or originals, but not all have been imported to originals.
- `READY_FOR_STAGING`: all 20 photographs are present in immutable originals storage.
- `CAPTURE_SESSION_BLOCKED`: a filename, duplicate, readability, or mapping problem must be corrected.

## End-of-session sequence

1. Run `prepare-first-real-batch-session` and review the missing-file count.
2. Correct any blocked or missing `front/detail` pair.
3. Run `python -m src.project_cli import-first-real-batch`.
4. Run `prepare-first-real-batch-session` again and confirm `READY_FOR_STAGING`.
5. Run `python -m src.project_cli stage-first-real-batch-capture`.
6. Review the staging inventory and pending queue draft before any manual live-queue import.

No command in this guide approves samples automatically.
