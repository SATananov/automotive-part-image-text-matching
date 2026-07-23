from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Callable, Sequence


@dataclass(frozen=True)
class CommandSpec:
    module: str
    description: str
    requires_tensorflow: bool = False


COMMANDS: dict[str, CommandSpec] = {
    "environment": CommandSpec(
        module="src.environment_check",
        description="Show Python, package, and TensorFlow device information.",
        requires_tensorflow=True,
    ),
    "create-development-data": CommandSpec(
        module="src.create_development_dataset",
        description="Regenerate the deterministic development dataset.",
    ),
    "validate-development-data": CommandSpec(
        module="src.validate_development_dataset",
        description="Validate development metadata and image files.",
    ),
    "create-grouped-split": CommandSpec(
        module="src.create_grouped_split",
        description="Regenerate the category-balanced grouped split.",
    ),
    "run-baselines": CommandSpec(
        module="src.run_baseline_models",
        description="Run the majority, text, and image baselines.",
    ),
    "train-text": CommandSpec(
        module="src.train_keras_text_model",
        description="Train and evaluate the Keras text model.",
        requires_tensorflow=True,
    ),
    "train-image": CommandSpec(
        module="src.train_keras_image_model",
        description="Train and evaluate the Keras image model.",
        requires_tensorflow=True,
    ),
    "train-multimodal": CommandSpec(
        module="src.train_multimodal_model",
        description="Train and evaluate the Keras multimodal model.",
        requires_tensorflow=True,
    ),
    "verify-development-pipeline": CommandSpec(
        module="src.verification.development_pipeline",
        description="Verify the Step 008.2 reproducibility safeguards.",
    ),
    "validate-real-data": CommandSpec(
        module="src.validate_real_dataset",
        description=(
            "Validate real-data annotations, approved images, hashes, "
            "and development separation."
        ),
    ),
    "verify-real-dataset-foundation": CommandSpec(
        module="src.verification.real_dataset_foundation",
        description="Verify the Step 009 real-data intake foundation.",
    ),
    "review-real-intake": CommandSpec(
        module="src.review_real_sample_intake",
        description=(
            "Review staged real samples, metadata, image quality, "
            "duplicates, and approval decisions without modifying the dataset."
        ),
    ),
    "apply-real-intake": CommandSpec(
        module="src.apply_real_sample_intake",
        description=(
            "Apply reviewed real-sample approvals and rejections "
            "transactionally."
        ),
    ),
    "verify-sample-intake": CommandSpec(
        module="src.verification.sample_intake_workflow",
        description=(
            "Verify the Step 009.1 sample intake and approval workflow."
        ),
    ),
    "prepare-first-real-batch": CommandSpec(
        module="src.prepare_first_real_batch",
        description=(
            "Validate the balanced first real-data batch plan and scan "
            "captured staging files without changing the live queue."
        ),
    ),
    "dry-run-first-real-batch": CommandSpec(
        module="src.dry_run_first_real_batch",
        description=(
            "Simulate first-batch approvals in temporary storage and prove "
            "that live real-data state is unchanged."
        ),
    ),
    "verify-first-batch-preparation": CommandSpec(
        module="src.verification.first_batch_preparation",
        description=(
            "Verify the Step 009.2 first-batch preparation and dry-run "
            "safeguards."
        ),
    ),
    "stage-first-real-batch-capture": CommandSpec(
        module="src.stage_first_real_batch_capture",
        description=(
            "Normalize local batch_001 captures into staging and generate "
            "a pending review queue draft without changing the live queue."
        ),
    ),
    "verify-capture-staging": CommandSpec(
        module="src.verification.capture_staging",
        description=(
            "Verify the Step 009.3 capture, staging, and review-readiness "
            "safeguards."
        ),
    ),
    "import-first-real-batch": CommandSpec(
        module="src.import_first_real_batch",
        description=(
            "Import descriptively named first-batch photographs from the "
            "local capture inbox into immutable originals storage."
        ),
    ),
    "verify-local-capture-import": CommandSpec(
        module="src.verification.local_capture_import",
        description=(
            "Verify first-batch file naming, capture checklist, and local "
            "import safeguards."
        ),
    ),
    "prepare-first-real-batch-session": CommandSpec(
        module="src.prepare_first_batch_capture_session",
        description=(
            "Generate the operator capture-session worksheet, missing-file "
            "status, and next capture action without changing live data."
        ),
    ),
    "verify-capture-session": CommandSpec(
        module="src.verification.capture_session_readiness",
        description=(
            "Verify first-batch operator guidance and capture-session "
            "preparation safeguards."
        ),
    ),
    "build-first-real-batch-dashboard": CommandSpec(
        module="src.build_first_batch_capture_dashboard",
        description=(
            "Build the first-batch capture dashboard and pipeline progress "
            "snapshot without changing live real-data state."
        ),
    ),
    "verify-capture-dashboard": CommandSpec(
        module="src.verification.capture_dashboard",
        description=(
            "Verify first-batch dashboard, progress tracking, and "
            "immutability safeguards."
        ),
    ),
    "run-first-real-batch-capture-session": CommandSpec(
        module="src.execute_first_batch_capture_session",
        description=(
            "Execute one safe first-batch capture cycle, import available "
            "files, stage valid originals, and update runtime progress."
        ),
    ),
    "refresh-first-real-batch-live-progress": CommandSpec(
        module="src.refresh_first_batch_live_progress",
        description=(
            "Refresh the runtime-only first-batch live progress dashboard "
            "without importing, staging, queueing, or approving files."
        ),
    ),
    "verify-capture-execution": CommandSpec(
        module="src.verification.capture_execution",
        description=(
            "Verify first-batch capture execution, runtime isolation, "
            "rollback, and live progress safeguards."
        ),
    ),
    "activate-first-real-batch-review-queue": CommandSpec(
        module="src.activate_first_batch_review_queue",
        description=(
            "Activate only validated pending first-batch draft rows in the "
            "live review queue without recording decisions."
        ),
    ),
    "prepare-first-real-batch-manual-decisions": CommandSpec(
        module="src.prepare_first_batch_manual_decisions",
        description=(
            "Build and validate the runtime manual-decision workbook without "
            "editing the live queue or approved dataset."
        ),
    ),
    "verify-review-queue": CommandSpec(
        module="src.verification.review_queue",
        description=(
            "Verify review-queue activation, manual decision preparation, "
            "idempotency, and transaction safeguards."
        ),
    ),

    "validate-first-real-batch-manual-decisions": CommandSpec(
        module="src.validate_first_batch_manual_decisions",
        description=(
            "Validate the first-batch manual decision workbook and "
            "create a fingerprinted runtime application plan."
        ),
    ),
    "apply-first-real-batch-manual-decisions": CommandSpec(
        module="src.apply_first_batch_manual_decisions",
        description=(
            "Apply the validated first-batch manual decisions through "
            "the transactional real-sample intake workflow."
        ),
    ),
    "verify-manual-decisions": CommandSpec(
        module="src.verification.manual_decision_workflow",
        description=(
            "Verify manual decision validation, stale-plan blocking, "
            "controlled application, and full rollback safeguards."
        ),
    ),

    "run-first-real-dataset-capture": CommandSpec(
        module="src.run_first_real_dataset_capture",
        description=(
            "Run the safe first real-dataset capture, staging, "
            "review, and manual-decision preparation cycle."
        ),
    ),
    "finalize-first-real-dataset-ingestion": CommandSpec(
        module="src.finalize_first_real_dataset_ingestion",
        description=(
            "Apply validated first-batch decisions and audit "
            "approved real-sample ingestion."
        ),
    ),
    "verify-real-dataset-ingestion": CommandSpec(
        module="src.verification.real_dataset_ingestion",
        description=(
            "Verify first real-dataset capture, ingestion, "
            "recapture, and rollback safeguards."
        ),
    ),

    "collect-open-license-images": CommandSpec(
        module="src.collect_open_license_images",
        description=(
            "Collect open-license automotive-part candidates from "
            "Wikimedia Commons with source and license metadata."
        ),
    ),
    "validate-open-license-images": CommandSpec(
        module="src.validate_open_license_dataset",
        description=(
            "Validate open-license image files, attribution metadata, "
            "hashes, and manual review decisions."
        ),
    ),
    "build-open-license-review-gallery": CommandSpec(
        module="src.build_open_license_review_gallery",
        description=(
            "Build the local HTML gallery for manual review of "
            "open-license image candidates."
        ),
    ),
    "verify-open-license-dataset": CommandSpec(
        module="src.verification.open_license_dataset",
        description=(
            "Verify open-license collection, attribution, review, "
            "validation, and dataset-boundary safeguards."
        ),
    ),

    "integrate-external-dataset": CommandSpec(
        module="src.integrate_external_dataset",
        description=(
            "Build the approved open-license metadata, grouped split, "
            "integrated development split, and locked test manifest."
        ),
    ),
    "validate-external-training-readiness": CommandSpec(
        module="src.validate_external_training_readiness",
        description=(
            "Validate external integration, group isolation, approved-image "
            "provenance, and the locked-test training policy."
        ),
    ),
    "verify-external-dataset-integration": CommandSpec(
        module="src.verification.external_dataset_integration",
        description=(
            "Verify Step 010.2 external integration, grouped split, "
            "training readiness, and test-lock safeguards."
        ),
    ),
    "run-integrated-training-validation": CommandSpec(
        module="src.run_integrated_training_validation",
        description=(
            "Train six integrated baselines and neural models and compare "
            "them on validation without loading the locked test split."
        ),
        requires_tensorflow=True,
    ),
    "verify-integrated-training-validation": CommandSpec(
        module="src.verification.integrated_training_validation",
        description=(
            "Verify integrated training outputs, validation comparison, "
            "and locked-test safeguards."
        ),
    ),
    "run-validation-error-analysis-model-improvement": CommandSpec(
        module=(
            "src.run_validation_error_analysis_and_model_improvement"
        ),
        description=(
            "Analyze integrated validation errors and compare predefined "
            "multimodal improvements without using the locked test split."
        ),
        requires_tensorflow=True,
    ),
    "verify-validation-model-improvement": CommandSpec(
        module="src.verification.validation_model_improvement",
        description=(
            "Verify validation error analysis, controlled experiments, "
            "selection gates, and locked-test safeguards."
        ),
    ),
    "freeze-final-model-evaluation-protocol": CommandSpec(
        module="src.freeze_final_model_and_evaluation_protocol",
        description=(
            "Freeze the selected final model recipe and one-shot evaluation "
            "protocol without opening or authorizing the locked test split."
        ),
    ),
    "verify-final-model-freeze": CommandSpec(
        module="src.verification.final_model_and_evaluation_protocol",
        description=(
            "Verify the final model recipe, evaluation protocol, artifact "
            "fingerprints, and closed locked-test authorization gate."
        ),
    ),
    "build-final-exam-notebook": CommandSpec(
        module="src.build_final_exam_notebook",
        description=(
            "Build and execute the final exam research notebook from "
            "committed train and validation evidence without test access."
        ),
    ),
    "verify-final-exam-notebook": CommandSpec(
        module="src.verification.final_exam_notebook",
        description=(
            "Verify the executed final exam notebook, research narrative, "
            "references, saved outputs, and locked-test safeguards."
        ),
    ),
    "verify-project": CommandSpec(
        module="src.verification.project_verification",
        description=(
            "Run all project integrity and dataset workflow verifications."
        ),
    ),

}

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.project_cli",
        description=(
            "Run project workflows from the repository root without "
            "executing package files by path."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        metavar="COMMAND",
    )

    for command_name, command_spec in COMMANDS.items():
        subparser = subparsers.add_parser(
            command_name,
            help=command_spec.description,
            description=command_spec.description,
        )
        subparser.set_defaults(command_name=command_name)

    return parser


def load_command_module(
    command_name: str,
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> ModuleType:
    try:
        command_spec = COMMANDS[command_name]
    except KeyError as error:
        raise ValueError(
            f"Unknown project command: {command_name}"
        ) from error

    try:
        return importer(command_spec.module)
    except ModuleNotFoundError as error:
        missing_module = error.name or ""

        if (
            command_spec.requires_tensorflow
            and missing_module.split(".")[0] in {"keras", "tensorflow"}
        ):
            raise RuntimeError(
                "This command requires the project environment with "
                "TensorFlow and Keras installed. Activate .venv and run "
                "the command again."
            ) from error

        raise


def run_command(
    command_name: str,
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> None:
    module = load_command_module(
        command_name=command_name,
        importer=importer,
    )

    command_main = getattr(module, "main", None)

    if not callable(command_main):
        raise RuntimeError(
            f"Command module '{module.__name__}' does not define main()."
        )

    command_main()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    command_name = getattr(arguments, "command_name", None)

    if command_name is None:
        parser.print_help()
        return 0

    try:
        run_command(command_name)
    except RuntimeError as error:
        parser.exit(status=1, message=f"Error: {error}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
