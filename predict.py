import pandas as pd
import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = "deadline_risk_model.joblib"


def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"⚠️ Модель не найдена: {e}")
        return None


def prepare_prediction_data(task_dict):
    """Подготавливает данные для предсказания"""
    df = pd.DataFrame([task_dict])
    
    # Конвертируем даты
    df["start_date_dt"] = pd.to_datetime(df["start_date"], dayfirst=True, errors="coerce")
    df["planned_end_date_dt"] = pd.to_datetime(df["planned_end_date"], dayfirst=True, errors="coerce")
    
    # Вычисляем длительность
    df["planned_duration_days"] = (df["planned_end_date_dt"] - df["start_date_dt"]).dt.days
    df["planned_duration_days"] = df["planned_duration_days"].fillna(0)
    
    # День недели
    df["start_weekday"] = df["start_date_dt"].dt.weekday.fillna(0).astype(int)
    df["planned_end_weekday"] = df["planned_end_date_dt"].dt.weekday.fillna(0).astype(int)
    
    # Длины текстов
    df["title_len"] = df["title"].fillna("").str.len()
    df["description_len"] = df["description"].fillna("").str.len()
    df["acceptance_len"] = df["acceptance_criteria"].fillna("").str.len()
    
    # Объединяем все тексты
    df["text_all"] = df[["title", "description", "acceptance_criteria"]].fillna("").agg(" ".join, axis=1)
    
    # Признаки
    features = [
        "assignee",
        "implementation_days",
        "planned_duration_days",
        "start_weekday",
        "planned_end_weekday",
        "task_type",
        "title_len",
        "description_len",
        "acceptance_len",
        "text_all",
    ]
    
    return df[features]


def get_risk_level(score):
    if score >= 0.70:
        return "высокий"
    elif score >= 0.35:
        return "средний"
    return "низкий"


def generate_recommendations(task_dict, score, level):
    """Генерирует рекомендации по управлению риском срыва дедлайна"""
    recommendations = []
    
    if level == "высокий":
        recommendations.append("🔴 ВЫСОКИЙ РИСК срыва дедлайна! Требуется немедленное внимание.")
        
        planned_duration = task_dict.get("planned_duration_days", 0)
        if planned_duration and planned_duration < 5:
            recommendations.append(f"📅 Рекомендуется увеличить срок реализации с {planned_duration} до {planned_duration + 5} дней")
        
        recommendations.append("👥 Рассмотрите возможность смены исполнителя или привлечения дополнительного ресурса")
        recommendations.append("📊 Назначьте ежедневные проверки статуса выполнения")
        
    elif level == "средний":
        recommendations.append("🟡 СРЕДНИЙ РИСК срыва дедлайна. Рекомендуется усилить контроль.")
        
        planned_duration = task_dict.get("planned_duration_days", 0)
        if planned_duration and planned_duration < 3:
            recommendations.append(f"📅 Увеличьте срок реализации на 2-3 дня")
        
        recommendations.append("📊 Назначьте промежуточные проверки статуса (через 30% и 70% времени)")
        
    else:
        recommendations.append("🟢 НИЗКИЙ РИСК срыва дедлайна. Задача реализуема в срок.")
        recommendations.append("✅ Продолжайте работу в текущем режиме")
    
    return "\n".join(recommendations)


def predict_one(task_dict):
    """
    Основная функция предсказания - ТОЛЬКО риск дедлайна
    
    Returns:
        dict: {
            "risk_score": float,
            "risk_level": str,
            "riskOfDeadlineFailure": str,
            "deadlineRecommendations": str
        }
    """
    try:
        model = load_model()
        if model is None:
            raise Exception("Модель не загружена")
        
        X = prepare_prediction_data(task_dict)
        
        # Реальное предсказание от модели
        score = float(model.predict_proba(X)[0, 1])
        level = get_risk_level(score)
        
        recommendations = generate_recommendations(task_dict, score, level)
        
        return {
            "risk_score": score,
            "risk_level": level,
            "riskOfDeadlineFailure": f"{score:.3f}",
            "deadlineRecommendations": recommendations
        }
        
    except Exception as e:
        print(f"❌ Ошибка в predict_one: {e}")
        return {
            "risk_score": 0.5,
            "risk_level": "средний",
            "riskOfDeadlineFailure": "0.500",
            "deadlineRecommendations": "⚠️ Ошибка модели рисков. Проверьте наличие файла deadline_risk_model.joblib"
        }