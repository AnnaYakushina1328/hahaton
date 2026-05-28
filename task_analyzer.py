import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


MODELS_TO_TRY = [
    "openrouter/free",
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-flash",
]


SYSTEM_PROMPT = """
Ты — AI-ассистент риск-менеджмента для Яндекс.Трекера.

Твоя задача — оценить качество описания задачи и риски для команды разработки.

Верни ответ строго в JSON-формате:
{
  "clarity_score": "высокая" | "средняя" | "низкая",
  "clarity_comment": "Краткое объяснение оценки понятности задачи",
  "risk_level": "низкий" | "средний" | "высокий",
  "risk_reasons": ["причина риска 1", "причина риска 2"],
  "missing_sections": ["чего не хватает в описании"],
  "recommendations": ["рекомендация 1", "рекомендация 2", "рекомендация 3"],
  "quick_fix": "Что исправить в первую очередь",
  "description_quality": "Отличное описание" | "Хорошее описание" | "Среднее описание" | "Плохое описание"
}

Правила:
1. Если описание короткое, общее или непонятное — ставь "Плохое описание".
2. Если нет критериев готовности, ожидаемого результата или конкретики — риск минимум "средний".
3. Если задача может привести к переделкам из-за неясности — риск "высокий".
4. Не используй markdown.
5. Не оборачивай ответ в ```json.
6. Верни только JSON.
"""


def _get_openrouter_client():
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY не задан в .env")

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=40,
    )


def _extract_json(raw_content):
    content = (raw_content or "").strip()

    if content.startswith("```json"):
        content = content.replace("```json", "", 1).strip()

    if content.startswith("```"):
        content = content.replace("```", "", 1).strip()

    if content.endswith("```"):
        content = content[:-3].strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", content)

    if match:
        json_text = match.group(0)

        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            repaired = _repair_common_json_errors(json_text)
            return json.loads(repaired)

    raise json.JSONDecodeError("JSON object not found", content, 0)


def _repair_common_json_errors(json_text):
    repaired = json_text

    repaired = repaired.replace("\n", "\\n")
    repaired = repaired.replace("\r", "\\r")
    repaired = repaired.replace("\t", "\\t")

    return repaired


def _fallback_analysis(summary, description, error_text=None):
    description = description or ""

    desc_len = len(description.strip())

    has_acceptance_words = any(
        word in description.lower()
        for word in [
            "критер",
            "готов",
            "результат",
            "должно",
            "провер",
            "сценар",
            "ожида",
            "приемк",
            "приёмк",
        ]
    )

    if desc_len < 60:
        clarity_score = "низкая"
        risk_level = "высокий"
        description_quality = "Плохое описание"
    elif desc_len >= 120 and has_acceptance_words:
        clarity_score = "высокая"
        risk_level = "низкий"
        description_quality = "Хорошее описание"
    else:
        clarity_score = "средняя"
        risk_level = "средний"
        description_quality = "Среднее описание"

    clarity_comment = (
        "Описание оценено локальной эвристикой, потому что LLM временно недоступна "
        "или вернула некорректный JSON. Проверьте полноту описания, ожидаемый результат "
        "и критерии готовности."
    )

    if error_text:
        clarity_comment += f" Последняя ошибка LLM: {str(error_text)[:200]}"

    return {
        "clarity_score": clarity_score,
        "clarity_comment": clarity_comment,
        "risk_level": risk_level,
        "risk_reasons": [
            "LLM временно не вернула корректный ответ",
            "Описание может быть неполным или недостаточно конкретным",
        ],
        "missing_sections": [
            "Ожидаемый результат",
            "Критерии готовности",
            "Проверочные сценарии",
        ],
        "recommendations": [
            "Добавить ожидаемый результат выполнения задачи",
            "Описать критерии приемки",
            "Указать, как именно проверить корректность выполнения",
        ],
        "quick_fix": "Добавить критерии готовности и ожидаемый результат.",
        "description_quality": description_quality,
    }


def _normalize_result(result):
    required_defaults = {
        "clarity_score": "средняя",
        "clarity_comment": "Комментарий не получен.",
        "risk_level": "средний",
        "risk_reasons": [],
        "missing_sections": [],
        "recommendations": [],
        "quick_fix": "Уточнить описание задачи.",
        "description_quality": "Среднее описание",
    }

    for key, default_value in required_defaults.items():
        if key not in result or result[key] is None:
            result[key] = default_value

    for key in ("risk_reasons", "missing_sections", "recommendations"):
        if isinstance(result[key], str):
            result[key] = [result[key]]

        if not isinstance(result[key], list):
            result[key] = []

    return result


def analyze_tracker_task(summary: str, description: str) -> dict:
    if not summary or not summary.strip():
        summary = "Название отсутствует"

    if not description or not description.strip():
        description = "Описание отсутствует"

    user_content = f"""
Название задачи:
{summary}

Описание задачи:
{description}
"""

    client = _get_openrouter_client()
    last_error = None

    for model in MODELS_TO_TRY:
        try:
            print(f"[LLM] Пробуем модель: {model}", flush=True)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                temperature=0.1,
                max_tokens=1200,
                response_format={
                    "type": "json_object",
                },
                extra_headers={
                    "HTTP-Referer": "http://localhost:8080",
                    "X-Title": "Tracker Risk Analyzer",
                },
            )

            raw_content = response.choices[0].message.content
            result = _extract_json(raw_content)
            result = _normalize_result(result)

            print(f"[LLM] Успешно с моделью: {model}", flush=True)

            return result

        except Exception as error:
            last_error = error
            print(f"[LLM] Модель {model} не сработала: {error}", flush=True)

    print("[LLM] Все модели недоступны, используется локальная эвристика", flush=True)

    return _fallback_analysis(
        summary=summary,
        description=description,
        error_text=str(last_error) if last_error else None,
    )