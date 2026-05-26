from frontend_state import register_frontend, record_tracker_event
from flask import Flask, request
from yandex_tracker_client import TrackerClient
from predict import predict_one
import os
from dotenv import load_dotenv
from tracker_polling import start_tracker_polling
import time

app = Flask(__name__)
register_frontend(app)

load_dotenv()

TOKEN = os.getenv("TOKEN")
CLOUD_ORG_ID = os.getenv("CLOUD_ORG_ID")

client = TrackerClient(
    token=TOKEN,
    cloud_org_id=CLOUD_ORG_ID
)

start_tracker_polling(client, queue_key="CLIENT", interval_seconds=30)

@app.route("/webhook", methods=["POST"])
@app.route("/", methods=["GET", "POST"])
def webhook():
    if request.method == "GET":
        return "Webhook server works"

    try:
        data = request.json
        issue_key = data.get("task_key")

        if not issue_key:
            return {"error": "task_key not found"}, 400

        print(f"📥 Обработка задачи: {issue_key}")

        issue = client.issues[issue_key]
        print(f"📄 Задача найдена: {issue.summary}")

        # защита от повторной обработки
        if issue.description and "Risk prediction:" in issue.description:
            print(f"⚠️ Задача {issue_key} уже обработана ранее")
            event = record_tracker_event(
                issue_key=issue_key,
                issue=issue,
                data=data,
                already_processed=True,
            )
            
            return {
                "status": "already processed",
                "event": event,
            }, 200
        
        else: 
            print(f"🔄 Рассчитываем риск для задачи {issue_key}")
            task_dict = {
                "title": issue.summary or "",
                "description": issue.description or "",
                "acceptance_criteria": "",
                "assignee": issue.assignee.display if issue.assignee else "unknown",
                "implementation_days": 0,
                "start_date": issue.createdAt[:10] if issue.createdAt else None,
                "planned_end_date": str(issue.deadline) if getattr(issue, "deadline", None) else None,
                "task_type": issue.type.key if issue.type else "task",
            }

            score, level = predict_one(task_dict)
            print(f"📊 Результат: Score={score:.3f}, Level={level}")

            prediction_text = f"""

---
🤖 Risk prediction:
Score: {score:.3f}
Level: {level}
"""

            issue.update(
                description=(issue.description or "") + prediction_text
            )
            print(f"✅ Описание задачи {issue_key} обновлено")

            event = record_tracker_event(
                issue_key=issue_key,
                issue=issue,
                data=data,
                score=score,
                level=level,
            )

            return {
                "status": "success",
                "issue": issue_key,
                "risk_score": float(score),
                "risk_level": level,
                "event": event,
            }, 200
            
    except Exception as e:
        print(f"❌ Ошибка при обработке вебхука: {e}")
        return {"error": str(e)}, 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)