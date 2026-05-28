import json
import os
import threading
import time
import urllib.request

from frontend_state import record_tracker_event
from predict import predict_one
from task_analyzer import analyze_tracker_task


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
    summary = getattr(issue, "summary", "") or ""
    description = getattr(issue, "description", "") or ""

    return {
        "title": summary,
        "description": description,
        "acceptance_criteria": "",
        "assignee": _get_display(getattr(issue, "assignee", None)),
        "implementation_days": 0,
        "start_date": str(getattr(issue, "createdAt", ""))[:10] if getattr(issue, "createdAt", None) else None,
        "planned_end_date": str(getattr(issue, "deadline", "")) if getattr(issue, "deadline", None) else None,
        "task_type": _safe_get(getattr(issue, "type", None), "key", "task"),
        "title_len": len(summary),
        "description_len": len(description),
        "acceptance_len": 0,
        "planned_duration_days": 0,
        "start_weekday": 0,
        "planned_end_weekday": 0,
    }


def _issue_snapshot(issue):
    status = getattr(issue, "status", None)

    return {
        "key": getattr(issue, "key", None),
        "status": _get_display(status),
        "updated_at": str(getattr(issue, "updatedAt", "")),
    }


def _normalize_field_value(value):
    if value is None:
        return ""

    return str(
        getattr(value, "display", None)
        or getattr(value, "key", None)
        or value
    ).strip()


def _update_issue_fields_if_changed(issue, updates):
    real_updates = {}

    for field_name, new_value in updates.items():
        current_value = getattr(issue, field_name, None)

        if _normalize_field_value(current_value) != _normalize_field_value(new_value):
            real_updates[field_name] = new_value

    if not real_updates:
        return False

    issue.update(**real_updates)
    return True


def _calculate_risk_and_update_issue(issue):
    try:
        task_dict = _build_task_dict(issue)
        result = predict_one(task_dict)

        if isinstance(result, dict):
            score = result.get("risk_score")
            level = result.get("risk_level")
            risk_of_deadline = result.get("riskOfDeadlineFailure")
            deadline_recs = result.get("deadlineRecommendations")
        else:
            score, level = result
            risk_of_deadline = score
            deadline_recs = ""

        if score is None:
            return None, "unknown", False

        score = round(float(score), 3)

        level_lower = str(level or "").lower()

        if level_lower in ("низкий", "low"):
            export_level = "low"
            risk_value = "низкий"
        elif level_lower in ("средний", "medium"):
            export_level = "medium"
            risk_value = "средний"
        elif level_lower in ("высокий", "high"):
            export_level = "high"
            risk_value = "высокий"
        else:
            export_level = "unknown"
            risk_value = "не рассчитан"

        updates = {}

        if hasattr(issue, "riskOfDeadlineFailure"):
            updates["riskOfDeadlineFailure"] = risk_of_deadline or score

        if hasattr(issue, "failureToMeetDeadlines"):
            updates["failureToMeetDeadlines"] = risk_value

        if hasattr(issue, "deadlineRecommendations") and deadline_recs:
            updates["deadlineRecommendations"] = deadline_recs

        was_updated = _update_issue_fields_if_changed(issue, updates)

        if was_updated:
            print(f"[polling] поля риска обновлены для {getattr(issue, 'key', 'unknown')}")
        else:
            print(f"[polling] поля риска не изменились для {getattr(issue, 'key', 'unknown')}")

        return score, export_level, was_updated

    except Exception as error:
        print(f"[polling] risk calculation failed for {getattr(issue, 'key', 'unknown')}: {error}")
        return None, "unknown", False


def _analyze_description_with_llm(issue):
    try:
        summary = getattr(issue, "summary", "") or ""
        description = getattr(issue, "description", "") or ""

        result = analyze_tracker_task(summary, description)

        print(f"[polling] LLM-анализ завершён для {getattr(issue, 'key', 'unknown')}")

        return result

    except Exception as error:
        print(f"[polling] LLM analysis failed for {getattr(issue, 'key', 'unknown')}: {error}")

        return {
            "clarity_score": "ошибка",
            "clarity_comment": f"Не удалось выполнить LLM-анализ: {error}",
            "risk_level": "желтый",
            "risk_reasons": ["Ошибка LLM-анализа"],
            "missing_sections": [],
            "recommendations": ["Проверить API-ключ и доступность LLM"],
            "quick_fix": "Проверить настройки LLM",
            "description_quality": "Ошибка оценки",
        }


def _update_description_quality_field(issue, llm_analysis):
    try:
        description_quality = (
            llm_analysis.get("description_quality")
            or llm_analysis.get("clarity_score")
            or "Оценка не доступна"
        )

        if not hasattr(issue, "analyzingTheTaskDescription"):
            return False

        was_updated = _update_issue_fields_if_changed(
            issue,
            {
                "analyzingTheTaskDescription": description_quality,
            },
        )

        if was_updated:
            print(
                f"[polling] analyzingTheTaskDescription updated for "
                f"{getattr(issue, 'key', 'unknown')}: {description_quality}"
            )
        else:
            print(
                f"[polling] analyzingTheTaskDescription unchanged for "
                f"{getattr(issue, 'key', 'unknown')}"
            )

        return was_updated

    except Exception as error:
        print(f"[polling] failed to update analyzingTheTaskDescription: {error}")
        return False


def _list_to_text(items):
    if not items:
        return "- нет данных"

    if isinstance(items, list):
        return "\n".join(f"- {item}" for item in items)

    return f"- {items}"


def _format_llm_analysis_comment(issue, llm_analysis):
    issue_key = getattr(issue, "key", "unknown")

    clarity_score = llm_analysis.get("clarity_score", "")
    clarity_comment = llm_analysis.get("clarity_comment", "")
    risk_level = llm_analysis.get("risk_level", "")
    risk_reasons = _list_to_text(llm_analysis.get("risk_reasons", []))
    missing_sections = _list_to_text(llm_analysis.get("missing_sections", []))
    recommendations = _list_to_text(llm_analysis.get("recommendations", []))
    quick_fix = llm_analysis.get("quick_fix", "")
    description_quality = llm_analysis.get("description_quality", "")

    return f"""🤖 LLM-анализ описания задачи {issue_key}

Качество описания: {description_quality}
Понятность задачи: {clarity_score}
Риск по LLM: {risk_level}

Комментарий:
{clarity_comment}

Чего не хватает в описании:
{missing_sections}

Причины риска:
{risk_reasons}

Рекомендации:
{recommendations}

Что исправить в первую очередь:
{quick_fix}
"""


def _add_tracker_comment(issue_key, comment_text):
    token = os.getenv("TOKEN")
    cloud_org_id = os.getenv("CLOUD_ORG_ID")
    org_id = os.getenv("ORG_ID")

    if not token:
        print("[polling] comment was not added: TOKEN is not set")
        return False

    headers = {
        "Authorization": f"OAuth {token}",
        "Content-Type": "application/json",
    }

    if cloud_org_id:
        headers["X-Cloud-Org-ID"] = cloud_org_id
    elif org_id:
        headers["X-Org-ID"] = org_id
    else:
        print("[polling] comment was not added: CLOUD_ORG_ID or ORG_ID is not set")
        return False

    payload = json.dumps(
        {
            "text": comment_text,
        },
        ensure_ascii=False,
    ).encode("utf-8")

    request = urllib.request.Request(
        f"https://api.tracker.yandex.net/v3/issues/{issue_key}/comments",
        data=payload,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            response.read()

        print(f"[polling] LLM-комментарий добавлен к {issue_key}")
        return True

    except Exception as error:
        print(f"[polling] не удалось добавить LLM-комментарий к {issue_key}: {error}")
        return False


def _add_llm_comment_to_issue(issue, llm_analysis):
    issue_key = getattr(issue, "key", None)

    if not issue_key:
        return False

    comment_text = _format_llm_analysis_comment(issue, llm_analysis)

    if not comment_text.strip():
        return False

    return _add_tracker_comment(issue_key, comment_text)


def _record_tracker_change(issue_key, issue, event_code, queue_key):
    score, level, risk_fields_updated = _calculate_risk_and_update_issue(issue)

    llm_analysis = _analyze_description_with_llm(issue)

    quality_field_updated = _update_description_quality_field(issue, llm_analysis)

    comment_added = _add_llm_comment_to_issue(issue, llm_analysis)

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

    issue_was_updated_by_us = risk_fields_updated or quality_field_updated or comment_added

    return event, issue_was_updated_by_us


def _poll_tracker_queue(client, queue_key="CLIENT", interval_seconds=30, per_page=50):
    print(f"[polling] запущена проверка очереди {queue_key}, интервал {interval_seconds} сек.", flush=True)

    previous = {}
    first_run = True
    self_updated_issues = set()

    while True:
        try:
            print(f"[polling] активные очереди {queue_key}...", flush=True)

            issues = list(
                client.issues.find(
                    filter={"queue": queue_key},
                    order=["-updated"],
                    per_page=per_page,
                )
            )

            print(f"[polling] найдено {len(issues)} задач в очереди {queue_key}", flush=True)

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

                    print(f"[polling] новая задача в {queue_key}: {issue_key}")
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

                    print(f"[polling] сек?новая задача в {queue_key}: {issue_key}")
                    continue

                if old_snapshot.get("updated_at") != snapshot.get("updated_at"):
                    if issue_key in self_updated_issues:
                        self_updated_issues.remove(issue_key)
                        print(f"[polling] пропущено собственное обновление в {queue_key}: {issue_key}")
                        continue

                    _, issue_was_updated_by_us = _record_tracker_change(
                        issue_key=issue_key,
                        issue=issue,
                        event_code="task_updated",
                        queue_key=queue_key,
                    )

                    if issue_was_updated_by_us:
                        self_updated_issues.add(issue_key)

                    print(f"[polling] задача обновлена в {queue_key}: {issue_key}")

            previous = current
            first_run = False

        except Exception as error:
            print(f"[polling] ошибка в очереди {queue_key}: {error}", flush=True)

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
        print(f"[polling] запущены новые очереди: {', '.join(started_queues)}")

    active_queues = sorted(active_polling_threads.keys())

    print(f"[polling] активные очереди: {', '.join(active_queues)}")

    return active_polling_threads


def _queue_discovery_loop(
    client,
    load_queue_keys,
    queue_refresh_interval_seconds=87,
    polling_interval_seconds=30,
):
    while True:
        try:
            queue_keys = load_queue_keys()

            print(f"[queues] обновление найдено: {', '.join(queue_keys)}")

            start_tracker_polling(
                client=client,
                queue_keys=queue_keys,
                interval_seconds=polling_interval_seconds,
            )

        except Exception as error:
            print(f"[queues] ошибка обновления очередей: {error}")

        time.sleep(queue_refresh_interval_seconds)


def start_tracker_queue_discovery(
    client,
    load_queue_keys,
    queue_refresh_interval_seconds=87,
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

    print(f"[queues] автообновление очередей запущено, интервал {queue_refresh_interval_seconds} сек.")

    return discovery_thread
