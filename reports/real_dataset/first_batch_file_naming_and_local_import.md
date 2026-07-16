# First Batch File Naming and Local Import

The first real automotive-parts batch uses descriptive capture filenames instead of technical intake identifiers.

## Naming standard

Each source photograph uses:

```text
real_<part_category>_001_<view>.jpg
```

Examples:

```text
real_starter_001_front.jpg
real_starter_001_detail.jpg
real_brake_disc_001_front.jpg
real_air_filter_001_detail.jpg
```

The committed `first_batch_capture_file_map.csv` connects each descriptive filename to its internal `intake_id`, physical group, category, view, and staging destination.

## Local import boundary

Photographs are placed in the ignored local directory:

```text
data/real/capture_inbox/batch_001/
```

The local import command validates filenames, image readability, duplicates, destination conflicts, and transaction safety. It copies original bytes into `data/real/originals/batch_001/` without pixel conversion and leaves the inbox files unchanged.

The command does not modify staging, annotations, the live intake queue, approval history, or the approved-image manifest. Pixel normalization remains the responsibility of the Step 009.3 staging command.

## Generated evidence

The workflow generates:

```text
data/real/processed/first_batch_local_import_inventory.csv
reports/real_dataset/first_batch_local_import_readiness.json
reports/real_dataset/first_batch_local_import_readiness.md
reports/real_dataset/first_batch_capture_checklist.md
```

The initial committed state is `PASS / AWAITING_LOCAL_FILES`. A partially imported batch is `LOCAL_IMPORT_IN_PROGRESS`; all 20 originals produce `READY_FOR_STAGING`.
