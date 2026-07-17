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
    "verify-step-008-2": CommandSpec(
        module="src.verify_step_008_2",
        description="Verify the Step 008.2 reproducibility safeguards.",
    ),
    "validate-real-data": CommandSpec(
        module="src.validate_real_dataset",
        description=(
            "Validate real-data annotations, approved images, hashes, "
            "and development separation."
        ),
    ),
    "verify-step-009": CommandSpec(
        module="src.verify_step_009",
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
    "verify-step-009-1": CommandSpec(
        module="src.verify_step_009_1",
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
    "verify-step-009-2": CommandSpec(
        module="src.verify_step_009_2",
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
    "verify-step-009-3": CommandSpec(
        module="src.verify_step_009_3",
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
    "verify-step-009-4": CommandSpec(
        module="src.verify_step_009_4",
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
    "verify-step-009-5": CommandSpec(
        module="src.verify_step_009_5",
        description=(
            "Verify first-batch operator guidance and capture-session "
            "preparation safeguards."
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
