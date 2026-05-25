import json
from collections import deque
from datetime import datetime
from queue import Empty, Queue
from threading import Lock

from flask import Response, jsonify, send_from_directory


history = deque(maxlen=100)
subscribers = []
subscribers_lock = Lock()


def _safe_display(value, default="unknown"):
    if not value:
        return default

    return (
        getattr(value, "display", None)
        or getattr(value, "key", None)
        or str(value)
    )


def _detect_trigger(data):
    raw = json.dumps(data or {}, ensure_ascii=False).lower()

    if "status" in raw:
        return "status_changed", "Изменился статус"

    if "create" in raw or "created" in raw or "new" in raw:
        return "new_task", "Новая задача"

    return "task_updated", "Изменение задачи"


def _push_event(event):
    history.appendleft(event)

    with subscribers_lock:
        dead_subscribers = []

        for subscriber in subscribers:
            try:
                subscriber.put_nowait(event)
            except Exception:
                dead_subscribers.append(subscriber)

        for subscriber in dead_subscribers:
            subscribers.remove(subscriber)


def record_tracker_event(issue_key, issue=None, data=None, score=None, level=None, already_processed=False):
    trigger_code, trigger_label = _detect_trigger(data)

    event = {
        "id": f"{issue_key}-{datetime.now().timestamp()}",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trigger_code": trigger_code,
        "trigger_label": trigger_label,
        "issue": issue_key,
        "title": getattr(issue, "summary", None) or "Без названия",
        "status": _safe_display(getattr(issue, "status", None)),
        "assignee": _safe_display(getattr(issue, "assignee", None)),
        "risk_score": round(float(score), 3) if score is not None else None,
        "risk_level": level or "unknown",
        "already_processed": already_processed,
    }

    _push_event(event)
    return event


def register_frontend(app):
    @app.get("/dashboard")
    def dashboard():
        return send_from_directory("frontend", "index.html")

    @app.get("/api/history")
    def api_history():
        return jsonify(list(history))

    @app.get("/events")
    def events():
        subscriber = Queue()

        with subscribers_lock:
            subscribers.append(subscriber)

        def stream():
            try:
                yield "retry: 3000\n\n"

                while True:
                    try:
                        event = subscriber.get(timeout=15)
                        payload = json.dumps(event, ensure_ascii=False)
                        yield f"data: {payload}\n\n"
                    except Empty:
                        yield ": keep-alive\n\n"
            finally:
                with subscribers_lock:
                    if subscriber in subscribers:
                        subscribers.remove(subscriber)

        return Response(stream(), mimetype="text/event-stream")
