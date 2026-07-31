from __future__ import annotations

from src.data import PROJECT_ROOT
from src.final_test_results_v3 import (
    build_final_test_results_manifest,
    canonical_json_bytes,
    file_sha256,
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "dataset_v3"
    / "dataset_v3_final_test_results.json"
)
CHECKSUM_PATH = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "dataset_v3"
    / "dataset_v3_final_test_results.sha256.txt"
)


def main() -> None:
    payload = build_final_test_results_manifest()
    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    MANIFEST_PATH.write_bytes(
        canonical_json_bytes(payload)
    )
    checksum = file_sha256(MANIFEST_PATH)
    CHECKSUM_PATH.write_text(
        f"{checksum}  {MANIFEST_PATH.name}\n",
        encoding="utf-8",
    )

    print()
    print("=== DATASET V3 FINAL TEST RESULTS RECORDED ===")
    print(f"Correct:   {payload['final_reporting']['correct_predictions']}/480")
    print(f"Accuracy:  {payload['final_reporting']['accuracy']:.10f}")
    print(f"Macro F1:  {payload['final_reporting']['macro_f1']:.10f}")
    print("Rerun:     False")
    print("Tuning:    False")
    print("Status: PASS_DATASET_V3_FINAL_TEST_RESULTS_RECORDED")


if __name__ == "__main__":
    main()
