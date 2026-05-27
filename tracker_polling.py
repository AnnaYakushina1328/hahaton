import threading
import time
import json

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


def _calculate_risk_and_update_issue(issue):
    """Рассчитывает риск дедлайна и обновляет ТОЛЬКО поля риск-менеджмента"""
    try:
        task_dict = _build_task_dict(issue)
        result = predict_one(task_dict)
        
        score = result["risk_score"]
        level = result["risk_level"]
        risk_of_deadline = result["riskOfDeadlineFailure"]
        deadline_recs = result["deadlineRecommendations"]
        
        # Обновляем ТОЛЬКО кастомные поля для риск-менеджмента
        try:
            updates = {}
            
            # riskOfDeadlineFailure - риск дедлайна
            if hasattr(issue, 'riskOfDeadlineFailure'):
                updates["riskOfDeadlineFailure"] = risk_of_deadline
            
            # deadlineRecommendations - рекомендации
            if hasattr(issue, 'deadlineRecommendations'):
                updates["deadlineRecommendations"] = deadline_recs
            
            if updates:
                issue.update(**updates)
                print(f"[polling] risk fields updated for {getattr(issue, 'key', 'unknown')}")
        except Exception as e:
            print(f"[polling] failed to update risk fields: {e}")
        
        return score, level
        
    except Exception as error:
        print(f"[polling] risk calculation failed for {getattr(issue, 'key', 'unknown')}: {error}")
        return None, "unknown"


def _analyze_description_with_llm(issue):
    """Анализирует описание задачи через Gemini и возвращает оценку"""
    try:
        summary = getattr(issue, "summary", "") or ""
        description = getattr(issue, "description", "") or ""

        result = analyze_tracker_task(summary, description)
        
        # result теперь словарь с полями, включая description_quality
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
            "description_quality": "Ошибка оценки"
        }


def _update_description_quality_field(issue, llm_analysis):
    """Обновляет поле analyzingTheTaskDescription"""
    try:
        # Берем оценку качества описания из результата Gemini
        description_quality = llm_analysis.get("description_quality", "Оценка не доступна")
        
        if hasattr(issue, 'analyzingTheTaskDescription'):
            issue.update(analyzingTheTaskDescription=description_quality)
            print(f"[polling] description quality field updated for {getattr(issue, 'key', 'unknown')}: {description_quality}")
            return True
    except Exception as e:
        print(f"[polling] failed to update description quality field: {e}")
    
    return False


def _list_to_text(items):
    if not items:
        return "- нет данных"

    if isinstance(items, list):
        return "\n".join(f"- {item}" for item in items)

    return f"- {items}"


def _format_llm_analysis_for_tracker(llm_analysis):
    """Форматирует полный LLM анализ для добавления в description"""
    if not llm_analysis:
        return ""

    clarity_score = llm_analysis.get("clarity_score", "")
    clarity_comment = llm_analysis.get("clarity_comment", "")
    risk_level = llm_analysis.get("risk_level", "")
    risk_reasons = _list_to_text(llm_analysis.get("risk_reasons", []))
    recommendations = _list_to_text(llm_analysis.get("recommendations", []))
    description_quality = llm_analysis.get("description_quality", "")

    return f"""

---
LLM analysis:

Description quality: {description_quality}

Clarity score: {clarity_score}

Risk level: {risk_level}

Comment:
{clarity_comment}

Risk reasons:
{risk_reasons}

Recommendations:
{recommendations}
"""


def _append_llm_analysis_to_description(issue, llm_analysis):
    """Добавляет полный LLM анализ в описание задачи"""
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

        print(f"[polling] llm analysis added to description for {getattr(issue, 'key', 'unknown')}")
        return True

    except Exception as error:
        print(f"[polling] failed to update description: {error}")
        return False


def _record_tracker_change(issue_key, issue, event_code, queue_key):
    # 1. Рассчитываем риск дедлайна и обновляем поля riskOfDeadlineFailure и deadlineRecommendations
    score, level = _calculate_risk_and_update_issue(issue)
    
    # 2. Анализируем описание через Gemini
    llm_analysis = _analyze_description_with_llm(issue)
    
    # 3. Обновляем поле analyzingTheTaskDescription только оценкой качества
    _update_description_quality_field(issue, llm_analysis)
    
    # 4. Добавляем полный LLM анализ в description
    _append_llm_analysis_to_description(issue, llm_analysis)

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

    return event, False


def _poll_tracker_queue(client, queue_key="CLIENT", interval_seconds=30, per_page=50):
    print(f"[polling] started for queue {queue_key}, interval {interval_seconds}s", flush=True)

    previous = {}
    first_run = True
    self_updated_issues = set()

    while True:
        try:
            print(f"[polling] checking queue {queue_key}...", flush=True)
            
            issues = list(
                client.issues.find(
                    filter={"queue": queue_key},
                    order=["-updated"],
                    per_page=per_page,
                )
            )
            
            print(f"[polling] found {len(issues)} issues in queue {queue_key}", flush=True)

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
                    _, _ = _record_tracker_change(
                        issue_key=issue_key,
                        issue=issue,
                        event_code="new_task",
                        queue_key=queue_key,
                    )

                    if issue_key in self_updated_issues:
                        self_updated_issues.remove(issue_key)

                    print(f"[polling] new task in {queue_key}: {issue_key}")
                    continue

                if old_snapshot.get("status") != snapshot.get("status"):
                    _, _ = _record_tracker_change(
                        issue_key=issue_key,
                        issue=issue,
                        event_code="status_changed",
                        queue_key=queue_key,
                    )

                    print(f"[polling] status changed in {queue_key}: {issue_key}")
                    continue

                if old_snapshot.get("updated_at") != snapshot.get("updated_at"):
                    if issue_key in self_updated_issues:
                        self_updated_issues.remove(issue_key)
                        print(f"[polling] skipped self update in {queue_key}: {issue_key}")
                        continue

                    _, _ = _record_tracker_change(
                        issue_key=issue_key,
                        issue=issue,
                        event_code="task_updated",
                        queue_key=queue_key,
                    )

                    print(f"[polling] task updated in {queue_key}: {issue_key}")

            previous = current
            first_run = False

        except Exception as error:
            print(f"[polling] error in queue {queue_key}: {error}", flush=True)

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