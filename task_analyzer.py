import json
import os
from openai import OpenAI

# Инициализация клиента
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "твой_ключ_тут"))

SYSTEM_PROMPT = """
Ты — AI-ассистент риск-менеджмента для Яндекс.Трекера. Твоя цель — проанализировать задачу, выявить риски и дать рекомендации.

Ты должен проанализировать название и описание задачи и вернуть ответ строго в формате JSON.
Структура JSON ответа:
{
  "clarity_score": "высокая" | "средняя" | "низкая",
  "clarity_comment": "Краткое обоснование оценки понятности задачи",
  "risk_level": "зеленый" | "желтый" | "красный",
  "risk_reasons": ["список конкретных причин риска, например: размытый дедлайн, нехватка деталей"],
  "recommendations": ["список рекомендаций, например: добавить подзадачи, сменить исполнителя, уточнить стек"]
}

Правила анализа:
1. Оценивай clarity_score как "низкая", если описание состоит из одной фразы или не содержит критериев приемки (Definition of Done).
2. Выставляй "красный" risk_level, если задача кажется критически важной, но описана абстрактно ("починить всё до пятницы").
Выдавай ТОЛЬКО чистый JSON. Не используй markdown-разметку (не пиши ```json ... ```).
"""

def analyze_tracker_task(summary: str, description: str) -> dict:
    """
    Функция для Кристины. Принимает данные из Яндекс.Трекера,
    отправляет в LLM и возвращает распарсенный JSON (словарь).
    """
    if not description:
        description = "Описание отсутствует."
        
    user_content = f"Название задачи: {summary}\nОписание задачи: {description}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Быстрая и дешевая модель, идеально для MVP
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"}, # Жесткий контроль JSON формата
            temperature=0.2 # Низкая температура для стабильного результата
        )
        
        raw_content = response.choices[0].message.content
        return json.loads(raw_content)
        
    except Exception as e:
        # Если что-то упало (нет сети, кончились токены), возвращаем структуру с ошибкой,
        return {
            "clarity_score": "ошибка",
            "clarity_comment": f"Не удалось проанализировать: {str(e)}",
            "risk_level": "желтый",
            "risk_reasons": ["Ошибка запроса к LLM"],
            "recommendations": ["Проверить подключение к API"]
        }