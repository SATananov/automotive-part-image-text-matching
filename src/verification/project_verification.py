from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VERIFICATION_MODULES = (
    "src.verification.development_pipeline",
    "src.verification.real_dataset_foundation",
    "src.verification.sample_intake_workflow",
    "src.verification.first_batch_preparation",
    "src.verification.capture_staging",
    "src.verification.local_capture_import",
    "src.verification.capture_session_readiness",
    "src.verification.capture_dashboard",
    "src.verification.capture_execution",
    "src.verification.review_queue",
    "src.verification.manual_decision_workflow",
    "src.verification.real_dataset_ingestion",
    "src.verification.open_license_dataset",
    "src.verification.external_dataset_integration",
    "src.verification.integrated_training_validation",
    "src.verification.validation_model_improvement",
    "src.verification.final_model_and_evaluation_protocol",
)


def main() -> None:
    failures: list[str] = []

    print("Project verification")
    for module_name in VERIFICATION_MODULES:
        print(f"- {module_name}")
        result = subprocess.run(
            [sys.executable, "-m", module_name],
            cwd=PROJECT_ROOT,
        )
        if result.returncode != 0:
            failures.append(module_name)

    if failures:
        print("Status: FAIL")
        for module_name in failures:
            print(f"- failed: {module_name}")
        raise SystemExit(1)

    print("Status: PASS")


if __name__ == "__main__":
    main()
