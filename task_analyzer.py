import json
import os
from openai import OpenAI
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
  "risk_level": "низкий" | "средний" | "высокий",
  "risk_reasons": ["список конкретных причин риска"],
  "missing_sections": ["каких важных разделов не хватает в описании"],
  "recommendations": [
    "конкретная рекомендация №1",
    "конкретная рекомендация №2",
    "конкретная рекомендация №3"
  ],
  "quick_fix": "Что нужно исправить в первую очередь (одно предложение)",
  "description_quality": "Отличное описание" | "Хорошее описание" | "Среднее описание" | "Плохое описание"
}

ПРАВИЛА ОЦЕНКИ ПОНЯТНОСТИ (clarity_score):
1. "низкая" — описание короткое (<100 символов), нет критериев приемки, размытая формулировка
2. "средняя" — описание есть, основная идея понятна, но не хватает деталей
3. "высокая" — чёткое описание, есть ожидаемый результат, есть критерии приемки

ПРАВИЛА ОЦЕНКИ РИСКОВ (risk_level):
1. "низкий" — задача понятна, рисков мало, можно брать в работу
2. "средний" — есть неопределённость, нужны уточнения
3. "высокий" — задача плохо описана, высокий риск переделок

ВАЖНО:
1. Не используй markdown
2. Не оборачивай ответ в ```json
3. Верни только чистый JSON
"""

def analyze_tracker_task(summary: str, description: str) -> dict:
    """
    Анализ задачи через OpenRouter (бесплатные модели)
    """
    if not summary or summary.strip() == "":
        summary = "Название отсутствует"

    if not description or description.strip() == "":
        description = "Описание отсутствует"

    user_content = f"""Название задачи: {summary}
Описание задачи: {description}"""

    # Список бесплатных моделей для тестирования
    free_models = [
        "google/gemini-2.5-flash-lite",
        "google/gemini-2.0-flash-lite",
        "mistralai/mistral-7b-instruct:free",
        "microsoft/phi-3-mini-128k-instruct:free",
        "meta-llama/llama-3.2-3b-instruct:free"
    ]
    
    last_error = None
    
    for model in free_models:
        try:
            print(f"Пробуем модель: {model}")
            
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.2,
                max_tokens=1000,
                extra_headers={
                    "HTTP-Referer": "http://localhost:8080",
                    "X-Title": "Tracker Risk Analyzer",
                }
            )
            
            raw_content = response.choices[0].message.content
            
            # Очищаем от markdown
            content = raw_content.strip()
            if content.startswith("```json"):
                content = content.replace("```json", "", 1).strip()
            elif content.startswith("```"):
                content = content.replace("```", "", 1).strip()
            if content.endswith("```"):
                content = content[:-3].strip()
            
            result = json.loads(content)
            
            # Добавляем поле description_quality, если его нет
            if "description_quality" not in result:
                desc_len = len(description)
                if desc_len > 200:
                    result["description_quality"] = "Отличное описание"
                elif desc_len > 100:
                    result["description_quality"] = "Хорошее описание"
                elif desc_len > 50:
                    result["description_quality"] = "Среднее описание"
                else:
                    result["description_quality"] = "Плохое описание"
            
            print(f"✅ Успешно с моделью: {model}")
            return result

        except Exception as error:
            last_error = error
            print(f"❌ Модель {model} не работает: {error}")
            continue
    
    # Если все модели не сработали
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
        "clarity_score": "средняя",
        "clarity_comment": f"Ошибка: ни одна модель не сработала. Последняя ошибка: {str(last_error)[:100]}",
        "risk_level": "средний",
        "risk_reasons": ["Ошибка запроса к OpenRouter API"],
        "missing_sections": [],
        "recommendations": [
            "Проверить OPENROUTER_API_KEY в .env файле",
            "Зайти на openrouter.ai и проверить баланс",
            "Пополнить баланс хотя бы на $1",
            "Или использовать бесплатный Google AI Studio"
        ],
        "quick_fix": "Проверь API ключ или добавь минимальный баланс",
        "description_quality": quality
    }