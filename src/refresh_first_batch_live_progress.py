from __future__ import annotations

from src.execute_first_batch_capture_session import (
    FIRST_BATCH_LIVE_DASHBOARD_PATH,
    refresh_live_progress,
    relative_path,
)


def main() -> None:
    report = refresh_live_progress()
    counts = report["counts"]
    print("First real batch live progress refresh")
    print(f"- Status: {report['status']}")
    print(f"- Readiness: {report['readiness']}")
    print(f"- Overall progress: {report['overall_progress_percent']}%")
    print(f"- Captured: {counts.get('captured', 0)} / {counts.get('planned', 0)}")
    print(f"- Imported: {counts.get('imported', 0)}")
    print(f"- Staged: {counts.get('staged', 0)}")
    print(f"- Review-ready: {counts.get('review_ready', 0)}")
    print(f"- Live dataset unchanged: {report['live_dataset_unchanged']}")
    print(f"- Tracked outputs unchanged: {report['tracked_outputs_unchanged']}")
    print(f"- Live dashboard: {relative_path(FIRST_BATCH_LIVE_DASHBOARD_PATH)}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
