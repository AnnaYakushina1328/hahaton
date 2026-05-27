import json
import os

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


SYSTEM_PROMPT = """
Ты — AI-ассистент риск-менеджмента для Яндекс.Трекера.

Твоя цель — проанализировать задачу, выявить риски и дать конкретные рекомендации по улучшению.

Проанализируй название и описание задачи.

Верни ответ строго в JSON-формате:
{
  "clarity_score": "высокая" | "средняя" | "низкая",
  "clarity_comment": "Краткое обоснование оценки понятности задачи (2-3 предложения)",
  "risk_level": "зеленый" | "желтый" | "красный",
  "risk_reasons": ["список конкретных причин риска"],
  "missing_sections": ["каких важных разделов не хватает в описании"],
  "recommendations": [
    "конкретная рекомендация №1",
    "конкретная рекомендация №2",
    "конкретная рекомендация №3",
    "конкретная рекомендация №4"
  ],
  "quick_fix": "Что нужно исправить в первую очередь (одно предложение)",
  "description_quality": "Отличное описание" | "Хорошее описание" | "Среднее описание" | "Плохое описание"
}

ПРАВИЛА ОЦЕНКИ ПОНЯТНОСТИ (clarity_score):
1. "низкая" — описание короткое (<100 символов), нет критериев приемки, размытая формулировка, непонятно что делать
2. "средняя" — описание есть, основная идея понятна, но не хватает деталей или критериев приемки
3. "высокая" — чёткое описание, есть ожидаемый результат, есть критерии приемки, понятно когда задача готова

ПРАВИЛА ОЦЕНКИ РИСКОВ (risk_level):
1. "зеленый" — задача понятна, рисков мало, можно брать в работу без доработок
2. "желтый" — есть неопределённость, задача понятна в целом, но нужны уточнения у автора
3. "красный" — задача важная/срочная/критичная, но плохо описана, высокий риск переделок и срыва сроков

КРИТЕРИИ ДЛЯ "КРАСНОГО" РИСКА:
- Задача влияет на критичный функционал ИЛИ есть жесткий дедлайн ИЛИ задача от руководителя
- И при этом clarity_score = "низкая" или "средняя"

ПРАВИЛА ФОРМИРОВАНИЯ РЕКОМЕНДАЦИЙ (recommendations):
1. Если нет критериев приемки → "Добавьте конкретные критерии приемки: что должно работать, как проверить результат"
2. Если описание размытое → "Конкретизируйте задачу: что именно нужно сделать, с какими данными, в каких условиях"
3. Если нет ожидаемого результата → "Опишите ожидаемый результат: как понять, что задача выполнена"
4. Если задача большая и непонятная → "Разбейте задачу на 2-3 более мелкие и понятные подзадачи"
5. Если нет контекста → "Добавьте ссылки на связанные задачи, документы или скриншоты"
6. Если непонятно зачем → "Укажите бизнес-ценность: зачем это нужно, какую проблему решает"

ПРАВИЛА ДЛЯ description_quality:
1. "Отличное описание" — описание подробное (>200 символов), есть четкая формулировка, критерии приемки, ожидаемый результат
2. "Хорошее описание" — описание есть (>100 символов), основная идея понятна, но не хватает деталей
3. "Среднее описание" — описание короткое (50-100 символов), размытая формулировка
4. "Плохое описание" — описание очень короткое (<50 символов) или отсутствует

ВАЖНО:
1. Не используй markdown
2. Не оборачивай ответ в ```json
3. Верни только чистый JSON
4. Никакого пояснительного текста вне JSON
"""


def _get_gemini_client():
    """Создание клиента Gemini"""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Проверь .env файл.")

    return genai.Client(api_key=api_key)


def _safe_json_loads(raw_content: str) -> dict:
    """Безопасный парсинг JSON с очисткой от markdown-оберток"""
    content = raw_content.strip()

    if content.startswith("```json"):
        content = content.replace("```json", "", 1).strip()
    elif content.startswith("```"):
        content = content.replace("```", "", 1).strip()

    if content.endswith("```"):
        content = content[:-3].strip()

    return json.loads(content)


def analyze_tracker_task(summary: str, description: str) -> dict:
    if not summary or summary.strip() == "":
        summary = "Название отсутствует"

    if not description or description.strip() == "":
        description = "Описание отсутствует"

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

        result = _safe_json_loads(response.text)
        
        # Добавляем поле description_quality, если его нет
        if "description_quality" not in result:
            # Определяем по длине описания
            desc_len = len(description)
            if desc_len > 200:
                result["description_quality"] = "Отличное описание"
            elif desc_len > 100:
                result["description_quality"] = "Хорошее описание"
            elif desc_len > 50:
                result["description_quality"] = "Среднее описание"
            else:
                result["description_quality"] = "Плохое описание"
        
        return result

    except Exception as error:
        # Определяем качество по длине описания
        desc_len = len(description)
        if desc_len > 200:
            quality = "Отличное описание"
        elif desc_len > 100:
            quality = "Хорошее описание"
        elif desc_len > 50:
            quality = "Среднее описание"
        else:
            quality = "Плохое описание"
            
        return {
            "clarity_score": "ошибка",
            "clarity_comment": f"Не удалось проанализировать задачу: {error}",
            "risk_level": "желтый",
            "risk_reasons": ["Ошибка запроса к LLM", f"Детали: {error}"],
            "missing_sections": [],
            "recommendations": [
                "Проверить GEMINI_API_KEY в файле .env",
                "Убедиться что API ключ активен",
                "Проверить подключение к интернету",
                "Повторить попытку позже"
            ],
            "quick_fix": "Проверь настройки API и повтори запрос",
            "description_quality": quality
        }