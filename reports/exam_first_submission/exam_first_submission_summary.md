# Step 011.5 — Exam-First Submission Architecture

Status: **PASS**

Readiness: `EXAM_FIRST_SUBMISSION_FOCUSED_TEST_LOCKED`

## Purpose

The repository now opens with one precise Deep Learning question and one
focused executed notebook. Course exercises, historical notebooks, manifests,
and engineering audits remain available as supporting evidence without
competing with the main analysis.

## Primary result

- retained model: `keras_multimodal`;
- grouped validation samples: 60;
- independent physical-part groups: 20;
- validation accuracy: 0.5333;
- validation macro F1: 0.5208;
- validation errors analyzed: 35;
- `PARTIAL_MATCH` errors: 20.

## Safety boundary

- model training performed: false;
- locked test CSV files opened: false;
- test split used: false;
- final test evaluation authorized: false;
- production final model changed: false;
- model selection changed: false.

## Repository design decision

Standard code, data, report, and test directories were not physically moved.
Moving them would break stable paths and historical manifests. Instead, the
new `exam/` directory is the teacher-facing entry point and
`supplementary/README.md` is the index to deeper evidence.
