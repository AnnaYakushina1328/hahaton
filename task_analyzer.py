import json
import os

from google import genai
from google.genai import types


SYSTEM_PROMPT = """
Ты — AI-ассистент риск-менеджмента для Яндекс.Трекера.

Твоя цель — проанализировать задачу, выявить риски и дать рекомендации.

Проанализируй название и описание задачи.

Верни ответ строго в JSON-формате:
{
  "clarity_score": "высокая" | "средняя" | "низкая",
  "clarity_comment": "Краткое обоснование оценки понятности задачи",
  "risk_level": "зеленый" | "желтый" | "красный",
  "risk_reasons": ["список конкретных причин риска"],
  "recommendations": ["список рекомендаций"]
}

Правила:
1. clarity_score = "низкая", если описание короткое, размытое или нет критериев приемки.
2. risk_level = "красный", если задача важная, срочная, но плохо описана.
3. risk_level = "желтый", если есть неопределенность, но задача в целом понятна.
4. risk_level = "зеленый", если задача описана понятно, есть ожидаемый результат и мало рисков.
5. Не используй markdown.
6. Не оборачивай ответ в ```json.
7. Верни только чистый JSON.
"""


def _get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    return genai.Client(api_key=api_key)


def _safe_json_loads(raw_content: str) -> dict:
    content = raw_content.strip()

    if content.startswith("```json"):
        content = content.replace("```json", "", 1).strip()

    if content.startswith("```"):
        content = content.replace("```", "", 1).strip()

    if content.endswith("```"):
        content = content[:-3].strip()

    return json.loads(content)


def analyze_tracker_task(summary: str, description: str) -> dict:
    """
    Анализирует задачу из Яндекс.Трекера через Gemini.
    Возвращает словарь с оценкой понятности, риском и рекомендациями.
    """

    if not summary:
        summary = "Название отсутствует."

    if not description:
        description = "Описание отсутствует."

    user_content = f"""
Название задачи:
{summary}

Описание задачи:
{description}
"""

    try:
        client = _get_gemini_client()

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )

        return _safe_json_loads(response.text)

    except Exception as error:
        return {
            "clarity_score": "ошибка",
            "clarity_comment": f"Не удалось проанализировать задачу: {error}",
            "risk_level": "желтый",
            "risk_reasons": ["Ошибка запроса к LLM"],
            "recommendations": ["Проверить GEMINI_API_KEY и подключение к Gemini API"],
        }