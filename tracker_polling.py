import threading
import time

from frontend_state import record_tracker_event
from predict import predict_one


def _safe_get(value, attr_name, default=None):
    return getattr(value, attr_name, default) if value else default


def _get_display(value, default="unknown"):
    if not value:
        return default

    return (
        getattr(value, "display", None)
        or getattr(value, "key", None)
        or str(value)
    )


def _build_task_dict(issue):
    return {
        "title": getattr(issue, "summary", "") or "",
        "description": getattr(issue, "description", "") or "",
        "acceptance_criteria": "",
        "assignee": _get_display(getattr(issue, "assignee", None)),
        "implementation_days": 0,
        "start_date": str(getattr(issue, "createdAt", ""))[:10] if getattr(issue, "createdAt", None) else None,
        "planned_end_date": str(getattr(issue, "deadline", "")) if getattr(issue, "deadline", None) else None,
        "task_type": _safe_get(getattr(issue, "type", None), "key", "task"),
    }


def _issue_snapshot(issue):
    status = getattr(issue, "status", None)

    return {
        "key": getattr(issue, "key", None),
        "status": _get_display(status),
        "updated_at": str(getattr(issue, "updatedAt", "")),
    }


def _calculate_risk(issue):
    try:
        task_dict = _build_task_dict(issue)
        score, level = predict_one(task_dict)
        if score is not None and issue.description and "Risk prediction:" not in issue.description:
            prediction_text = f"""

        ---
        🤖 Risk prediction:
        Score: {score:.3f}
        Level: {level}
        """
            issue.update(description=(issue.description or "") + prediction_text)
            print(f"  ✅ Описание задачи {issue.key} обновлено")
        return score, level
    except Exception as error:
        print(f"[polling] risk calculation failed for {getattr(issue, 'key', 'unknown')}: {error}")
        return None, "unknown"


def _poll_tracker(client, queue_key="CLIENT", interval_seconds=15, per_page=50):
    print(f"[polling] started for queue {queue_key}, interval {interval_seconds}s")

    previous = {}
    first_run = True

    while True:
        try:
            issues = list(
                client.issues.find(
                    filter={"queue": queue_key},
                    order=["-updated"],
                    per_page=per_page,
                )
            )

            current = {}

            for issue in issues:
                snapshot = _issue_snapshot(issue)
                issue_key = snapshot["key"]

                if not issue_key:
                    continue

                current[issue_key] = snapshot

                if first_run:
                    continue

                old_snapshot = previous.get(issue_key)

                if old_snapshot is None:
                    score, level = _calculate_risk(issue)

                    record_tracker_event(
                        issue_key=issue_key,
                        issue=issue,
                        data={"event": "new_task"},
                        score=score,
                        level=level,
                    )

                    print(f"[polling] new task: {issue_key}")
                    continue

                if old_snapshot.get("status") != snapshot.get("status"):
                    score, level = _calculate_risk(issue)

                    record_tracker_event(
                        issue_key=issue_key,
                        issue=issue,
                        data={"event": "status_changed"},
                        score=score,
                        level=level,
                    )

                    print(f"[polling] status changed: {issue_key}")
                    continue

                if old_snapshot.get("updated_at") != snapshot.get("updated_at"):
                    score, level = _calculate_risk(issue)

                    record_tracker_event(
                        issue_key=issue_key,
                        issue=issue,
                        data={"event": "task_updated"},
                        score=score,
                        level=level,
                    )

                    print(f"[polling] task updated: {issue_key}")

            previous = current
            first_run = False

        except Exception as error:
            print(f"[polling] error: {error}")

        time.sleep(interval_seconds)


def start_tracker_polling(client, queue_key="CLIENT", interval_seconds=15):
    thread = threading.Thread(
        target=_poll_tracker,
        kwargs={
            "client": client,
            "queue_key": queue_key,
            "interval_seconds": interval_seconds,
        },
        daemon=True,
    )

    thread.start()
    return thread
