import json
import urllib.request
from frontend_state import register_frontend, record_tracker_event
from flask import Flask, request
from yandex_tracker_client import TrackerClient
from predict import predict_one
import os
from dotenv import load_dotenv
from tracker_polling import start_tracker_queue_discovery
import time

app = Flask(__name__)
register_frontend(app)

load_dotenv()

TOKEN = os.getenv("TOKEN")
CLOUD_ORG_ID = os.getenv("CLOUD_ORG_ID")
ORG_ID = os.getenv("ORG_ID")


def get_tracker_queue_keys():
    manual_queues = os.getenv("TRACKER_QUEUES", "").strip()

    if manual_queues:
        queues = [
            queue.strip().upper()
            for queue in manual_queues.split(",")
            if queue.strip()
        ]

        print(f"[queues] loaded from env: {', '.join(queues)}")
        return queues

    headers = {
        "Authorization": f"OAuth {TOKEN}",
    }

    if CLOUD_ORG_ID:
        headers["X-Cloud-Org-ID"] = CLOUD_ORG_ID
    elif ORG_ID:
        headers["X-Org-ID"] = ORG_ID

    request = urllib.request.Request(
        "https://api.tracker.yandex.net/v3/queues/?perPage=100",
        headers=headers,
        method="GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            data = response.read().decode("utf-8")

        queues_data = json.loads(data)

        queues = [
            queue.get("key")
            for queue in queues_data
            if queue.get("key")
        ]

        queues = [queue.upper() for queue in queues]

        print(f"[queues] loaded from tracker: {', '.join(queues)}")

        return queues or ["CLIENT"]

    except Exception as error:
        print(f"[queues] failed to load queues from tracker: {error}")
        print("[queues] fallback to CLIENT")

        return ["CLIENT"]


client = TrackerClient(
    token=TOKEN,
    cloud_org_id=CLOUD_ORG_ID
)

start_tracker_queue_discovery(
    client=client,
    load_queue_keys=get_tracker_queue_keys,
    queue_refresh_interval_seconds=60,
    polling_interval_seconds=30,
)


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
        if issue.description and "📝 Анализ описания задачи:" in issue.description:
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

            result = predict_one(task_dict)
            
            print(f"📊 Результат: Score={result['risk_score']:.3f}, Level={result['risk_level']}")
            
            # ТОЛЬКО ЭТО ИДЕТ В DESCRIPTION (только анализ описания)
            description_block = f"""

---
📝 Анализ описания задачи:
{result['analyzingTheTaskDescription']}
"""

            # Обновляем описание задачи (только анализ описания)
            issue.update(
                description=(issue.description or "") + description_block
            )
            print(f"✅ Описание задачи {issue_key} обновлено (добавлен анализ описания)")

            # Обновляем кастомные поля
            try:
                updates = {}
                
                # riskOfDeadlineFailure - сюда пишем Score и Level на русском
                risk_text = f"Оценка риска срыва дедлайна: {result['risk_score']:.3f} (уровень: {result['risk_level']})"
                updates["riskOfDeadlineFailure"] = risk_text
                
                # deadlineRecommendations - сюда пишем рекомендации
                updates["deadlineRecommendations"] = result['deadlineRecommendations']
                
                # analyzingTheTaskDescription - сюда пишем оценку описания
                updates["analyzingTheTaskDescription"] = result['analyzingTheTaskDescription']
                
                issue.update(**updates)
                print(f"✅ Обновлены кастомные поля для задачи {issue_key}")
            except Exception as e:
                print(f"⚠️ Не удалось обновить кастомные поля: {e}")

            event = record_tracker_event(
                issue_key=issue_key,
                issue=issue,
                data=data,
                score=result['risk_score'],
                level=result['risk_level'],
            )

            return {
                "status": "success",
                "issue": issue_key,
                "risk_score": result['risk_score'],
                "risk_level": result['risk_level'],
                "riskOfDeadlineFailure": risk_text,
                "deadlineRecommendations": result['deadlineRecommendations'],
                "analyzingTheTaskDescription": result['analyzingTheTaskDescription'],
                "event": event,
            }, 200
            
    except Exception as e:
        print(f"❌ Ошибка при обработке вебхука: {e}")
        return {"error": str(e)}, 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=False)