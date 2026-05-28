import json
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


MODELS_TO_TRY = [
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-flash",
    "openrouter/free",
]

REQUEST_TIMEOUT_SECONDS = 12
MAX_OUTPUT_TOKENS = 650
RETRY_DELAY_SECONDS = 4


SYSTEM_PROMPT = """
Ты — AI-ассистент риск-менеджмента для Яндекс.Трекера.

Твоя задача — быстро оценить качество описания задачи и риски для команды разработки.

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
1. Отвечай кратко.
2. В списках максимум 3 пункта.
3. Если описание короткое, общее или непонятное — ставь "Плохое описание".
4. Если нет критериев готовности, ожидаемого результата или конкретики — риск минимум "средний".
5. Если задача может привести к переделкам из-за неясности — риск "высокий".
6. Не используй markdown.
7. Не оборачивай ответ в ```json.
8. Верни только JSON.
"""


def _get_openrouter_client():
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY не задан в .env")

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def _extract_json(raw_content):
    content = (raw_content or "").strip()

    if not content:
        raise json.JSONDecodeError("Пустой ответ модели", content, 0)

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

    if not match:
        raise json.JSONDecodeError("JSON object not found", content, 0)

    return json.loads(match.group(0))


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

    if not isinstance(result, dict):
        raise ValueError("Модель вернула не JSON-объект")

    for key, default_value in required_defaults.items():
        if key not in result or result[key] is None:
            result[key] = default_value

    for key in ("risk_reasons", "missing_sections", "recommendations"):
        if isinstance(result[key], str):
            result[key] = [result[key]]

        if not isinstance(result[key], list):
            result[key] = []

        result[key] = result[key][:3]

    return result


def _get_retry_delay(error):
    error_text = str(error)

    match = re.search(r"retry_after_seconds['\"]?:\s*([0-9]+)", error_text)

    if match:
        return min(int(match.group(1)), 30)

    match = re.search(r"Retry-After['\"]?:\s*['\"]?([0-9]+)", error_text)

    if match:
        return min(int(match.group(1)), 30)

    return RETRY_DELAY_SECONDS


def _call_model(client, model, user_content):
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
        max_tokens=MAX_OUTPUT_TOKENS,
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

    return _normalize_result(result)


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
    attempt_round = 1

    while True:
        print(f"[LLM] Круг попыток №{attempt_round}", flush=True)

        for model in MODELS_TO_TRY:
            try:
                print(f"[LLM] Пробуем модель: {model}", flush=True)

                result = _call_model(
                    client=client,
                    model=model,
                    user_content=user_content,
                )

                print(f"[LLM] Успешно с моделью: {model}", flush=True)

                return result

            except json.JSONDecodeError as error:
                print(
                    f"[LLM] Модель {model} вернула некорректный JSON: {error}",
                    flush=True,
                )
                continue

            except Exception as error:
                delay = _get_retry_delay(error)

                print(
                    f"[LLM] Модель {model} не сработала: {error}",
                    flush=True,
                )

                if "429" in str(error):
                    print(
                        f"[LLM] Пойман лимит. Ждём {delay} сек. перед следующей попыткой.",
                        flush=True,
                    )
                    time.sleep(delay)

                continue

        print(
            f"[LLM] Все модели в круге №{attempt_round} не дали валидный результат. "
            f"Ждём {RETRY_DELAY_SECONDS} сек. и пробуем снова.",
            flush=True,
        )

        attempt_round += 1
        time.sleep(RETRY_DELAY_SECONDS)