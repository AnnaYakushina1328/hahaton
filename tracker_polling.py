import threading
import time

from frontend_state import record_tracker_event
from predict import predict_one
from task_analyzer import analyze_tracker_task


LLM_MARKER = "LLM analysis:"

active_polling_threads = {}
active_polling_lock = threading.Lock()


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
        return score, level
    except Exception as error:
        print(f"[polling] risk calculation failed for {getattr(issue, 'key', 'unknown')}: {error}")
        return None, "unknown"


def _analyze_description_with_llm(issue):
    try:
        summary = getattr(issue, "summary", "") or ""
        description = getattr(issue, "description", "") or ""

        result = analyze_tracker_task(summary, description)

        print(f"[polling] llm analysis completed for {getattr(issue, 'key', 'unknown')}")

        return result

    except Exception as error:
        print(f"[polling] llm analysis failed for {getattr(issue, 'key', 'unknown')}: {error}")

        return {
            "clarity_score": "ошибка",
            "clarity_comment": f"Не удалось выполнить LLM-анализ: {error}",
            "risk_level": "желтый",
            "risk_reasons": ["Ошибка LLM-анализа"],
            "recommendations": ["Проверить GEMINI_API_KEY и доступ к Gemini API"],
        }


def _list_to_text(items):
    if not items:
        return "- нет данных"

    if isinstance(items, list):
        return "\n".join(f"- {item}" for item in items)

    return f"- {items}"


def _format_llm_analysis_for_tracker(llm_analysis):
    if not llm_analysis:
        return ""

    clarity_score = llm_analysis.get("clarity_score", "")
    clarity_comment = llm_analysis.get("clarity_comment", "")
    risk_level = llm_analysis.get("risk_level", "")
    risk_reasons = _list_to_text(llm_analysis.get("risk_reasons", []))
    recommendations = _list_to_text(llm_analysis.get("recommendations", []))

    return f"""

---
LLM analysis:

Clarity score: {clarity_score}

Risk level: {risk_level}

Comment:
{clarity_comment}

Risk reasons:
{risk_reasons}

Recommendations:
{recommendations}
"""


def _append_llm_analysis_to_issue(issue, llm_analysis):
    try:
        current_description = getattr(issue, "description", "") or ""

        if LLM_MARKER in current_description:
            return False

        formatted_analysis = _format_llm_analysis_for_tracker(llm_analysis)

        if not formatted_analysis:
            return False

        issue.update(
            description=current_description + formatted_analysis
        )

        print(f"[polling] llm analysis added to tracker issue {getattr(issue, 'key', 'unknown')}")
        return True

    except Exception as error:
        print(f"[polling] failed to update tracker issue {getattr(issue, 'key', 'unknown')}: {error}")
        return False


def _record_tracker_change(issue_key, issue, event_code, queue_key):
    score, level = _calculate_risk(issue)
    llm_analysis = _analyze_description_with_llm(issue)

    event = record_tracker_event(
        issue_key=issue_key,
        issue=issue,
        data={
            "event": event_code,
            "queue": queue_key,
            "llm_analysis": llm_analysis,
        },
        score=score,
        level=level,
    )

    issue_was_updated_by_us = _append_llm_analysis_to_issue(issue, llm_analysis)

    return event, issue_was_updated_by_us


def _poll_tracker_queue(client, queue_key="CLIENT", interval_seconds=30, per_page=50):
    print(f"[polling] started for queue {queue_key}, interval {interval_seconds}s")

    previous = {}
    first_run = True
    self_updated_issues = set()

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
                    _, issue_was_updated_by_us = _record_tracker_change(
                        issue_key=issue_key,
                        issue=issue,
                        event_code="new_task",
                        queue_key=queue_key,
                    )

                    if issue_was_updated_by_us:
                        self_updated_issues.add(issue_key)

                    print(f"[polling] new task in {queue_key}: {issue_key}")
                    continue

                if old_snapshot.get("status") != snapshot.get("status"):
                    _, issue_was_updated_by_us = _record_tracker_change(
                        issue_key=issue_key,
                        issue=issue,
                        event_code="status_changed",
                        queue_key=queue_key,
                    )

                    if issue_was_updated_by_us:
                        self_updated_issues.add(issue_key)

                    print(f"[polling] status changed in {queue_key}: {issue_key}")
                    continue

                if old_snapshot.get("updated_at") != snapshot.get("updated_at"):
                    if issue_key in self_updated_issues:
                        self_updated_issues.remove(issue_key)
                        print(f"[polling] skipped self update in {queue_key}: {issue_key}")
                        continue

                    _, issue_was_updated_by_us = _record_tracker_change(
                        issue_key=issue_key,
                        issue=issue,
                        event_code="task_updated",
                        queue_key=queue_key,
                    )

                    if issue_was_updated_by_us:
                        self_updated_issues.add(issue_key)

                    print(f"[polling] task updated in {queue_key}: {issue_key}")

            previous = current
            first_run = False

        except Exception as error:
            print(f"[polling] error in queue {queue_key}: {error}")

        time.sleep(interval_seconds)


def _prepare_queue_keys(queue_keys):
    if queue_keys is None:
        return []

    if isinstance(queue_keys, str):
        queue_keys = [queue_keys]

    prepared_queue_keys = []
    seen_queue_keys = set()

    for queue_key in queue_keys:
        prepared_queue_key = str(queue_key).strip().upper()

        if not prepared_queue_key or prepared_queue_key in seen_queue_keys:
            continue

        prepared_queue_keys.append(prepared_queue_key)
        seen_queue_keys.add(prepared_queue_key)

    return prepared_queue_keys


def start_tracker_polling(client, queue_keys=None, interval_seconds=30):
    prepared_queue_keys = _prepare_queue_keys(queue_keys)

    if not prepared_queue_keys:
        prepared_queue_keys = ["CLIENT"]

    started_queues = []

    with active_polling_lock:
        for queue_key in prepared_queue_keys:
            existing_thread = active_polling_threads.get(queue_key)

            if existing_thread and existing_thread.is_alive():
                continue

            thread = threading.Thread(
                target=_poll_tracker_queue,
                kwargs={
                    "client": client,
                    "queue_key": queue_key,
                    "interval_seconds": interval_seconds,
                },
                daemon=True,
            )

            thread.start()
            active_polling_threads[queue_key] = thread
            started_queues.append(queue_key)

    if started_queues:
        print(f"[polling] started new queues: {', '.join(started_queues)}")

    active_queues = sorted(active_polling_threads.keys())

    print(f"[polling] active queues: {', '.join(active_queues)}")

    return active_polling_threads


def _queue_discovery_loop(
    client,
    load_queue_keys,
    queue_refresh_interval_seconds=60,
    polling_interval_seconds=30,
):
    while True:
        try:
            queue_keys = load_queue_keys()

            print(f"[queues] refresh found: {', '.join(queue_keys)}")

            start_tracker_polling(
                client=client,
                queue_keys=queue_keys,
                interval_seconds=polling_interval_seconds,
            )

        except Exception as error:
            print(f"[queues] refresh failed: {error}")

        time.sleep(queue_refresh_interval_seconds)


def start_tracker_queue_discovery(
    client,
    load_queue_keys,
    queue_refresh_interval_seconds=60,
    polling_interval_seconds=30,
):
    initial_queue_keys = load_queue_keys()

    start_tracker_polling(
        client=client,
        queue_keys=initial_queue_keys,
        interval_seconds=polling_interval_seconds,
    )

    discovery_thread = threading.Thread(
        target=_queue_discovery_loop,
        kwargs={
            "client": client,
            "load_queue_keys": load_queue_keys,
            "queue_refresh_interval_seconds": queue_refresh_interval_seconds,
            "polling_interval_seconds": polling_interval_seconds,
        },
        daemon=True,
    )

    discovery_thread.start()

    print(f"[queues] auto discovery started, interval {queue_refresh_interval_seconds}s")

    return discovery_thread