from flask import Flask, request
from yandex_tracker_client import TrackerClient
from predict import predict_one
import os
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()

TOKEN = os.getenv("TOKEN")
CLOUD_ORG_ID = os.getenv("CLOUD_ORG_ID")

client = TrackerClient(
    token=TOKEN,
    cloud_org_id=CLOUD_ORG_ID
)

@app.route("/", methods=["GET", "POST"])
def webhook():
    if request.method == "GET":
        return "Webhook server works"

    data = request.json
    issue_key = data.get("task_key")

    if not issue_key:
        return {"error": "task_key not found"}, 400

    issue = client.issues[issue_key]

    # защита от повторной обработки
    if issue.description and "Risk prediction:" in issue.description:
        return {"status": "already processed"}, 200

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

    prediction_text = f"""

---
🤖 Risk prediction:
Score: {score:.3f}
Level: {level}
"""

    issue.update(
        description=(issue.description or "") + prediction_text
    )

    return {
        "status": "success",
        "issue": issue_key,
        "risk_score": float(score),
        "risk_level": level
    }, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)