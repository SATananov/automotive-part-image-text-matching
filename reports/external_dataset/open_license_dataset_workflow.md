# Step 010.1 — Open-License Internet Image Collection & Source Manifest

## Goal

This step creates a separate external image collection for model development.
It does not relabel internet images as warehouse photographs and does not
modify the Step 010 real-capture queue.

The source is Wikimedia Commons. Every accepted candidate must have:

- a supported open or public-domain license;
- a source file page;
- a license URL;
- creator or credit metadata when attribution is required;
- a local SHA-256 fingerprint;
- a manual review decision.

## Official source and reuse references

- Wikimedia Commons reuse guide:
  `https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia`
- MediaWiki Imageinfo API:
  `https://www.mediawiki.org/wiki/API:Imageinfo`
- Commons metadata extension:
  `https://www.mediawiki.org/wiki/Extension:CommonsMetadata`

The individual Commons file page remains the authority for each image's
license and attribution requirements.

## Collection

```powershell
python -m src.project_cli collect-open-license-images
```

The collector searches all ten project categories and targets five candidate
images per category. It uses a maximum-width 1024-pixel Wikimedia thumbnail to
keep the repository manageable.

Generated files:

```text
data/external/open_license/images/
data/external/open_license/open_license_manifest.csv
data/external/open_license/open_license_review.csv
data/external/open_license/ATTRIBUTION.md
reports/external_dataset/open_license_collection_summary.md
```

The initial review decision is always `pending`. The collector does not approve
semantic category membership.

## Review gallery

```powershell
python -m src.project_cli build-open-license-review-gallery
```

Open:

```text
reports/external_dataset/open_license_review_gallery.html
```

Edit only these columns in `open_license_review.csv`:

- `operator_decision`
- `rejection_reason`
- `operator_notes`

Allowed decisions:

- `pending`
- `approved`
- `rejected`

Rejected rows require a reason.

## Validation

```powershell
python -m src.project_cli validate-open-license-images
```

Safe readiness values:

- `AWAITING_COLLECTION`
- `COLLECTION_INCOMPLETE`
- `MANUAL_REVIEW_REQUIRED`
- `READY_FOR_EXTERNAL_DATASET`
- `REPLACEMENT_IMAGES_REQUIRED`
- `VALIDATION_BLOCKED`

The external collection is ready only when every category has at least five
approved images.

## Verification

```powershell
python -m src.project_cli verify-open-license-dataset
```
