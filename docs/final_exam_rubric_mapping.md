# Final Exam Rubric Mapping

| Criterion | Evidence in the repository |
|---|---|
| Problem statement | `README.md` and notebook Sections 1 and 9 define the real catalogue/warehouse consistency problem, labels, audience, and cautious conclusion. |
| Layout | Executed `project.ipynb` follows problem → data → validity → models → results → errors → research → limitations → conclusion. |
| Code quality | Functional modules under `src/`, type hints, deterministic builders, explicit constants, error handling, and automated tests. |
| Previous research | Notebook cites and distinguishes VSE++, VisualBERT, CLIP, ResNet, and the external dataset; no false benchmark comparability claim. |
| Data acquisition/cleaning | `src/import_dataset_v2.py`, the original Wikimedia license manifest and unified Dataset V2 provenance manifest, hashes, standardisation notes, deterministic selection, duplicate screening, and `docs/dataset_v2_protocol.md`. |
| Testing | Tests cover schema, counts, locking, identity leakage, image validity, licensing, statistical grouping, model artifacts, notebook execution, and metric recomputation. |
| Visualization | Notebook includes source/split counts, model comparison with grouped intervals, confusion matrix, class report, per-category analysis, and error examples. |
| Communication | Generated result summary, explicit independent sample size, negative synthetic ablation, limitations, and one-time-test policy. |

## Claims deliberately avoided

- production readiness;
- open-vocabulary language understanding;
- causal proof that one architecture is universally superior;
- treating paired rows as independent samples;
- reporting a final-test result before the protocol is frozen.
