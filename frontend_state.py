import csv
import json
from collections import deque
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Lock

from flask import Response, jsonify, send_from_directory
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill


DATA_DIR = Path("data")
HISTORY_FILE = DATA_DIR / "events_history.json"
MAX_HISTORY_SIZE = 1000

history = deque(maxlen=MAX_HISTORY_SIZE)
subscribers = []
subscribers_lock = Lock()


def _load_history_from_file():
    DATA_DIR.mkdir(exist_ok=True)

    if not HISTORY_FILE.exists():
        return

    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as file:
            events = json.load(file)

        if isinstance(events, list):
            for event in events[-MAX_HISTORY_SIZE:]:
                history.append(event)

        print(f"[history] loaded {len(history)} events from {HISTORY_FILE}")
    except Exception as error:
        print(f"[history] failed to load history: {error}")


def _save_history_to_file():
    DATA_DIR.mkdir(exist_ok=True)

    try:
        with HISTORY_FILE.open("w", encoding="utf-8") as file:
            json.dump(list(history), file, ensure_ascii=False, indent=2)
    except Exception as error:
        print(f"[history] failed to save history: {error}")


def _safe_display(value, default="unknown"):
    if not value:
        return default

    return (
        getattr(value, "display", None)
        or getattr(value, "key", None)
        or str(value)
    )


def _detect_trigger(data):
    data = data or {}

    event = str(data.get("event", "")).lower()

    if event in ("new_task", "created", "issue_created", "create"):
        return "new_task", "Новая задача"

    if event in ("status_changed", "issue_status_changed", "status"):
        return "status_changed", "Изменился статус"

    if event in ("task_updated", "updated", "issue_updated", "update"):
        return "task_updated", "Изменение задачи"

    raw = json.dumps(data, ensure_ascii=False).lower()

    if "event.create" in raw or "created" in raw:
        return "new_task", "Новая задача"

    if "status_changed" in raw:
        return "status_changed", "Изменился статус"

    return "task_updated", "Изменение задачи"


def _push_event(event):
    history.appendleft(event)
    _save_history_to_file()

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
    _load_history_from_file()

    @app.get("/dashboard")
    def dashboard():
        return send_from_directory("frontend", "index.html")

    @app.get("/api/history")
    def api_history():
        return jsonify(list(history))

    @app.get("/api/history/export.csv")
    def export_history_csv():
        DATA_DIR.mkdir(exist_ok=True)

        export_file = DATA_DIR / "events_history_export.csv"

        fieldnames = [
            "time",
            "trigger_code",
            "issue",
            "title",
            "status",
            "assignee",
            "risk_score",
            "risk_level",
            "already_processed",
        ]

        with export_file.open("w", encoding="cp1251", newline="") as file:
            file.write("sep=;\n")
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
                delimiter=";",
                lineterminator="\n",
            )
            writer.writeheader()

            for event in history:
                writer.writerow({
                    "time": event.get("time", ""),
                    "trigger_code": event.get("trigger_code", ""),
                    "issue": event.get("issue", ""),
                    "title": event.get("title", ""),
                    "status": event.get("status", ""),
                    "assignee": event.get("assignee", ""),
                    "risk_score": event.get("risk_score", ""),
                    "risk_level": event.get("risk_level", ""),
                    "already_processed": event.get("already_processed", ""),
                })

        return send_from_directory(DATA_DIR, export_file.name, as_attachment=True)

    @app.get("/api/history/export.xlsx")
    def export_history_xlsx():
        DATA_DIR.mkdir(exist_ok=True)

        export_file = DATA_DIR / "events_history_export.xlsx"

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Events History"

        headers = [
            "\u0412\u0440\u0435\u043c\u044f",
            "\u0421\u043e\u0431\u044b\u0442\u0438\u0435",
            "\u0417\u0430\u0434\u0430\u0447\u0430",
            "\u041d\u0430\u0437\u0432\u0430\u043d\u0438\u0435",
            "\u0421\u0442\u0430\u0442\u0443\u0441",
            "\u0418\u0441\u043f\u043e\u043b\u043d\u0438\u0442\u0435\u043b\u044c",
            "Risk score",
            "\u0423\u0440\u043e\u0432\u0435\u043d\u044c \u0440\u0438\u0441\u043a\u0430",
            "\u0423\u0436\u0435 \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u0430\u043d\u043e",
        ]

        trigger_labels = {
            "new_task": "\u041d\u043e\u0432\u0430\u044f \u0437\u0430\u0434\u0430\u0447\u0430",
            "status_changed": "\u0418\u0437\u043c\u0435\u043d\u0438\u043b\u0441\u044f \u0441\u0442\u0430\u0442\u0443\u0441",
            "task_updated": "\u0418\u0437\u043c\u0435\u043d\u0435\u043d\u0438\u0435 \u0437\u0430\u0434\u0430\u0447\u0438",
        }

        risk_labels = {
            "low": "\u041d\u0438\u0437\u043a\u0438\u0439",
            "medium": "\u0421\u0440\u0435\u0434\u043d\u0438\u0439",
            "high": "\u0412\u044b\u0441\u043e\u043a\u0438\u0439",
            "unknown": "\u041d\u0435 \u0440\u0430\u0441\u0441\u0447\u0438\u0442\u0430\u043d",
        }

        sheet.append(headers)

        for event in history:
            trigger_code = event.get("trigger_code", "")
            risk_level = event.get("risk_level", "")

            sheet.append([
                event.get("time", ""),
                trigger_labels.get(trigger_code, trigger_code),
                event.get("issue", ""),
                event.get("title", ""),
                event.get("status", ""),
                event.get("assignee", ""),
                event.get("risk_score", ""),
                risk_labels.get(risk_level, risk_level),
                "\u0414\u0430" if event.get("already_processed") else "\u041d\u0435\u0442",
            ])

        column_widths = {
            "A": 22,
            "B": 24,
            "C": 14,
            "D": 38,
            "E": 22,
            "F": 24,
            "G": 12,
            "H": 18,
            "I": 20,
        }

        for column, width in column_widths.items():
            sheet.column_dimensions[column].width = width

        header_fill = PatternFill("solid", fgColor="DDEBFF")
        header_font = Font(bold=True)
        header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

        for cell in sheet[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = header_alignment

        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)

        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions

        workbook.save(export_file)

        return send_from_directory(DATA_DIR, export_file.name, as_attachment=True)

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
                        event = subscriber.get(timeout=30)
                        payload = json.dumps(event, ensure_ascii=False)
                        yield f"data: {payload}\n\n"
                    except Empty:
                        yield ": keep-alive\n\n"
            finally:
                with subscribers_lock:
                    if subscriber in subscribers:
                        subscribers.remove(subscriber)

        return Response(stream(), mimetype="text/event-stream")