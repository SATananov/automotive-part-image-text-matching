from src.verify_final_report import (
    EXPECTED_ACCURACY,
    EXPECTED_CORRECT,
    EXPECTED_IMAGES,
    EXPECTED_MACRO_F1,
    EXPECTED_ROWS,
    verify_final_report,
)


def test_final_report_is_executed_and_consistent() -> None:
    summary = verify_final_report()

    assert summary["status"] == "PASS_FINAL_DATASET_V3_REPORT_VERIFIED"
    assert summary["errors"] == 0
    assert summary["training_or_inference_executed"] is False
    assert summary["accuracy"] == EXPECTED_ACCURACY
    assert summary["macro_f1"] == EXPECTED_MACRO_F1
    assert summary["correct_predictions"] == EXPECTED_CORRECT
    assert summary["test_rows"] == EXPECTED_ROWS
    assert summary["test_images"] == EXPECTED_IMAGES
