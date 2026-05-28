import pandas as pd
import joblib
import numpy as np
from pathlib import Path

RISK_MODEL_PATH = "deadline_risk_model.joblib"
DELAY_MODEL_PATH = "delay_prediction_model.joblib"
REPLACEMENT_MODEL_PATH = "replacement_model.joblib"


def load_models():
    """Загружает все модели"""
    models = {
        "risk": None,
        "delay": None,
        "replacement": None
    }
    try:
        models["risk"] = joblib.load(RISK_MODEL_PATH)
    except:
        print("⚠️ Модель риска не найдена")
    try:
        models["delay"] = joblib.load(DELAY_MODEL_PATH)
    except:
        print("⚠️ Модель прогноза просрочки не найдена")
    try:
        models["replacement"] = joblib.load(REPLACEMENT_MODEL_PATH)
    except:
        print("⚠️ Модель замены не найдена")
    return models


def prepare_prediction_data(task_dict):
    """Подготавливает данные для предсказания"""
    df = pd.DataFrame([task_dict])
    
    # Конвертируем даты
    df["start_date_dt"] = pd.to_datetime(df["start_date"], dayfirst=True, errors="coerce")
    df["planned_end_date_dt"] = pd.to_datetime(df["planned_end_date"], dayfirst=True, errors="coerce")
    
    # Вычисляем длительность
    df["planned_duration_days"] = (df["planned_end_date_dt"] - df["start_date_dt"]).dt.days
    df["planned_duration_days"] = df["planned_duration_days"].fillna(df["implementation_days"])
    
    # День недели
    df["start_weekday"] = df["start_date_dt"].dt.weekday.fillna(0).astype(int)
    df["planned_end_weekday"] = df["planned_end_date_dt"].dt.weekday.fillna(0).astype(int)
    
    # Длины текстов
    df["title_len"] = df["title"].fillna("").str.len()
    df["description_len"] = df["description"].fillna("").str.len()
    
    # acceptance_criteria
    if "acceptance_criteria" in df.columns:
        df["acceptance_len"] = df["acceptance_criteria"].fillna("").str.len()
    else:
        df["acceptance_len"] = 0
    
    # Объединяем все тексты
    text_parts = []
    if "title" in df.columns:
        text_parts.append(df["title"].fillna(""))
    if "description" in df.columns:
        text_parts.append(df["description"].fillna(""))
    if "acceptance_criteria" in df.columns:
        text_parts.append(df["acceptance_criteria"].fillna(""))
    
    if text_parts:
        df["text_all"] = pd.concat(text_parts, axis=1).fillna("").agg(" ".join, axis=1)
    else:
        df["text_all"] = ""
    
    # Оценка сложности
    df["complexity_score"] = df.apply(estimate_complexity_row, axis=1)
    
    # Признаки
    features = [
        "assignee",
        "implementation_days",
        "planned_duration_days",
        "complexity_score",
        "start_weekday",
        "planned_end_weekday",
        "task_type",
        "title_len",
        "description_len",
        "acceptance_len",
        "text_all",
    ]
    
    return df[features]


def estimate_complexity_row(row):
    """Оценка сложности для одной строки"""
    score = 1
    
    days = row.get("planned_duration_days", 0)
    if days > 5:
        score += 1
    if days > 10:
        score += 1
    
    complex_types = ["Backend", "DevOps", "Аналитика", "Интеграция"]
    if row.get("task_type") in complex_types:
        score += 1
    
    desc_len = len(str(row.get("description", "")))
    if desc_len > 500:
        score += 1
    elif desc_len > 200:
        score += 0.5
    
    return min(score, 5)


def get_risk_level(score):
    """Определяет текстовый уровень риска"""
    if score >= 0.70:
        return "высокий"
    elif score >= 0.35:
        return "средний"
    return "низкий"


def get_recommendation_replacement(current_assignee, task_type, replacement_model):
    """Рекомендует замену исполнителя"""
    try:
        # Сначала ищем по типу задачи
        if task_type in replacement_model.get("by_task_type", {}):
            candidates = replacement_model["by_task_type"][task_type]
            for candidate in candidates:
                if candidate != current_assignee:
                    return candidate
        
        # Если нет, берем топовых в целом
        for candidate in replacement_model.get("top_overall", []):
            if candidate != current_assignee:
                return candidate
    except:
        pass
    return None


def get_alternative_assignees(current_assignee, task_type, replacement_model, limit=3):
    """Возвращает топ альтернативных исполнителей"""
    alternatives = []
    try:
        all_assignees = list(replacement_model.get("overall", {}).keys())
        scored = []
        for assignee in all_assignees:
            if assignee == current_assignee:
                continue
            perf = replacement_model["overall"].get(assignee, {})
            score = perf.get("success_rate", 0.5)
            scored.append((score, assignee, perf))
        
        scored.sort(reverse=True)
        
        for score, name, perf in scored[:limit]:
            alternatives.append({
                "name": name,
                "success_rate": round(score, 2),
                "avg_delay": round(perf.get("avg_delay", 0), 1),
                "tasks_count": perf.get("tasks_count", 0)
            })
    except:
        pass
    return alternatives


def predict_one(task_dict):
    """
    Расширенная функция предсказания
    
    Returns:
        dict: {
            "risk_score": float,
            "risk_level": str,
            "riskOfDeadlineFailure": str,
            "predicted_delay_days": float,
            "recommended_extension_days": int,
            "recommended_replacement": str or None,
            "alternative_assignees": list,
            "deadlineRecommendations": str
        }
    """
    # Добавляем отсутствующие поля
    defaults = {
        "acceptance_criteria": "",
        "implementation_days": 0,
        "start_date": None,
        "planned_end_date": None,
        "start_weekday": 0,
        "planned_end_weekday": 0,
        "task_type": "task",
        "priority": "medium",
        "title_len": 0,
        "description_len": 0,
        "acceptance_len": 0,
        "planned_duration_days": 0,
    }
    
    for key, value in defaults.items():
        if key not in task_dict:
            task_dict[key] = value
    
    models = load_models()
    
    # Подготавливаем данные
    X = prepare_prediction_data(task_dict)
    
    # 1. Предсказание риска
    risk_score = 0.5
    if models["risk"]:
        try:
            risk_score = float(models["risk"].predict_proba(X)[0, 1])
        except Exception as e:
            print(f"Ошибка предсказания риска: {e}")
    
    risk_level = get_risk_level(risk_score)
    
    # 2. Предсказание просрочки
    predicted_delay = 0.0
    if models["delay"]:
        try:
            predicted_delay = float(models["delay"].predict(X)[0])
            predicted_delay = max(0, predicted_delay)
        except Exception as e:
            print(f"Ошибка предсказания просрочки: {e}")
    
    # 3. Рекомендация по увеличению дедлайна
    recommended_extension = int(predicted_delay + 2) if predicted_delay > 0 else 0
    
    # 4. Рекомендация по замене исполнителя
    recommended_replacement = None
    if models["replacement"] and risk_level == "высокий":
        recommended_replacement = get_recommendation_replacement(
            task_dict.get("assignee", ""),
            task_dict.get("task_type", "task"),
            models["replacement"]
        )
    
    # 5. Альтернативные исполнители
    alternative_assignees = []
    if models["replacement"]:
        alternative_assignees = get_alternative_assignees(
            task_dict.get("assignee", ""),
            task_dict.get("task_type", "task"),
            models["replacement"]
        )
    
    # 6. Формирование рекомендаций
    recommendations = []
    
    if risk_level == "высокий":
        recommendations.append("🔴 ВЫСОКИЙ РИСК срыва дедлайна! Требуется немедленное внимание.")
    elif risk_level == "средний":
        recommendations.append("🟡 СРЕДНИЙ РИСК срыва дедлайна. Рекомендуется усилить контроль.")
    else:
        recommendations.append("🟢 НИЗКИЙ РИСК срыва дедлайна. Задача реализуема в срок.")
    
    if predicted_delay > 0:
        recommendations.append(f"📊 Прогноз: задача может задержаться на {predicted_delay:.1f} дней")
        if recommended_extension > 0:
            recommendations.append(f"📅 Рекомендуется увеличить дедлайн на {recommended_extension} дней")
    
    if recommended_replacement:
        recommendations.append(f"👥 Рассмотрите замену исполнителя на {recommended_replacement}")
    
    # Проверка на блокирующие задачи
    if task_dict.get("blocked_by"):
        blockers_count = len(task_dict["blocked_by"])
        recommendations.append(f"🚫 Задача заблокирована {blockers_count} задачами. Сначала решите их.")
    
    if not recommendations:
        recommendations.append("✅ Задача в порядке. Продолжайте работу.")
    
    return {
        "risk_score": round(risk_score, 3),
        "risk_level": risk_level,
        "riskOfDeadlineFailure": f"{risk_score:.3f}",
        "predicted_delay_days": round(predicted_delay, 1),
        "recommended_extension_days": recommended_extension,
        "recommended_replacement": recommended_replacement,
        "alternative_assignees": alternative_assignees,
        "deadlineRecommendations": "\n".join(recommendations)
    }