import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from datetime import datetime
import json

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, mean_absolute_error
)

DATA_PATH = "zadachki_dataset(1).csv"
RISK_MODEL_PATH = "deadline_risk_model.joblib"
DELAY_MODEL_PATH = "delay_prediction_model.joblib"
REPLACEMENT_MODEL_PATH = "replacement_model.joblib"
METADATA_PATH = "model_metadata.json"


def prepare_dataset(path):
    """Подготовка данных из CSV"""
    df = pd.read_csv(path)
    
    print(f"📊 Исходные данные: {len(df)} строк")
    
    # Целевые переменные
    df["deadline_failed"] = (df["deviation_days"] > 0).astype(int)  # Бинарный риск
    df["delay_days"] = df["deviation_days"].clip(0, 30)  # Прогноз просрочки (дней)
    
    # Конвертация дат
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
    df["acceptance_len"] = df["acceptance_criteria"].fillna("").str.len()
    
    # Объединяем все тексты
    df["text_all"] = df[["title", "description", "acceptance_criteria"]].fillna("").agg(" ".join, axis=1)
    
    # Оценка сложности
    df["complexity_score"] = df.apply(estimate_complexity, axis=1)
    
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
    
    print(f"   Просроченных задач: {df['deadline_failed'].sum()} ({(df['deadline_failed'].sum()/len(df)*100):.1f}%)")
    print(f"   Средняя просрочка: {df['delay_days'].mean():.1f} дней")
    
    return df[features], df["deadline_failed"], df["delay_days"], df


def estimate_complexity(row):
    """Оценка сложности задачи от 1 до 5"""
    score = 1
    
    # По длительности
    days = row.get("planned_duration_days", 0)
    if days > 5:
        score += 1
    if days > 10:
        score += 1
    
    # По типу задачи
    complex_types = ["Backend", "DevOps", "Аналитика", "Интеграция"]
    if row.get("task_type") in complex_types:
        score += 1
    
    # По объему описания
    desc_len = len(str(row.get("description", "")))
    if desc_len > 500:
        score += 1
    elif desc_len > 200:
        score += 0.5
    
    # По наличию критериев приемки
    if len(str(row.get("acceptance_criteria", ""))) > 100:
        score += 0.5
    
    return min(score, 5)


def build_risk_model():
    """Строит модель для предсказания риска срыва дедлайна"""
    numeric_features = [
        "implementation_days",
        "planned_duration_days",
        "complexity_score",
        "start_weekday",
        "planned_end_weekday",
        "title_len",
        "description_len",
        "acceptance_len",
    ]
    
    cat_features = ["assignee", "task_type"]
    
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ("text", TfidfVectorizer(max_features=200, ngram_range=(1, 2)), "text_all"),
    ])
    
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )),
    ])


def build_delay_model():
    """Строит модель для прогноза просрочки в днях"""
    numeric_features = [
        "implementation_days",
        "planned_duration_days",
        "complexity_score",
        "start_weekday",
        "planned_end_weekday",
        "title_len",
        "description_len",
        "acceptance_len",
    ]
    
    cat_features = ["assignee", "task_type"]
    
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ("text", TfidfVectorizer(max_features=200, ngram_range=(1, 2)), "text_all"),
    ])
    
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )),
    ])


def build_replacement_recommender(df):
    """Строит рекомендательную систему для замены исполнителя"""
    # Анализируем успешность каждого исполнителя
    performance = df.groupby("assignee").agg({
        "deadline_failed": lambda x: 1 - x.mean(),  # успешность (1 - процент просрочек)
        "delay_days": "mean",
        "task_id": "count"
    }).rename(columns={
        "deadline_failed": "success_rate",
        "delay_days": "avg_delay",
        "task_id": "tasks_count"
    })
    
    # Для каждого типа задачи - свои топ-исполнители
    type_performance = df.groupby(["task_type", "assignee"]).agg({
        "deadline_failed": lambda x: 1 - x.mean(),
        "delay_days": "mean"
    }).rename(columns={"deadline_failed": "success_rate", "delay_days": "avg_delay"})
    
    return {
        "overall": performance.to_dict("index"),
        "by_task_type": type_performance.groupby("task_type").apply(
            lambda x: x.nlargest(3, "success_rate").index.get_level_values(1).tolist()
        ).to_dict(),
        "top_overall": performance.nlargest(3, "success_rate").index.tolist()
    }


def train_models():
    """Основная функция обучения всех моделей"""
    print("=" * 60)
    print("🚀 ОБУЧЕНИЕ РАСШИРЕННОЙ МОДЕЛИ РИСКОВ")
    print("=" * 60)
    
    # Загрузка данных
    print("\n📂 Загрузка данных...")
    X, y_risk, y_delay, df = prepare_dataset(DATA_PATH)
    
    # 1. Обучение модели риска
    print("\n🔴 1. Обучение модели риска срыва дедлайна...")
    risk_model = build_risk_model()
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        risk_model, X, y_risk, cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
    )
    
    print("   CV результаты:")
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"     {metric}: {scores[f'test_{metric}'].mean():.3f} +/- {scores[f'test_{metric}'].std():.3f}")
    
    # Обучаем на всех данных
    risk_model.fit(X, y_risk)
    joblib.dump(risk_model, RISK_MODEL_PATH)
    print(f"   ✅ Модель риска сохранена в {RISK_MODEL_PATH}")
    
    # 2. Обучение модели прогноза просрочки
    print("\n⏰ 2. Обучение модели прогноза просрочки...")
    delay_model = build_delay_model()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_delay, test_size=0.2, random_state=42
    )
    delay_model.fit(X_train, y_train)
    
    y_pred = delay_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"   MAE на тесте: {mae:.2f} дня")
    
    joblib.dump(delay_model, DELAY_MODEL_PATH)
    print(f"   ✅ Модель прогноза просрочки сохранена в {DELAY_MODEL_PATH}")
    
    # 3. Построение рекомендательной системы
    print("\n👥 3. Построение рекомендательной системы замены...")
    replacement_recommender = build_replacement_recommender(df)
    joblib.dump(replacement_recommender, REPLACEMENT_MODEL_PATH)
    
    # Сохраняем метаданные
    metadata = {
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "samples_count": len(df),
        "features_count": len(X.columns),
        "risk_cv_scores": {k: float(scores[f'test_{k}'].mean()) for k in ["accuracy", "roc_auc"]},
        "delay_mae": float(mae),
        "top_assignees": replacement_recommender["top_overall"],
        "risk_levels": {
            "low": "< 0.35",
            "medium": "0.35-0.70",
            "high": ">= 0.70"
        }
    }
    
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Модели успешно обучены и сохранены!")
    print(f"📋 Метаданные сохранены в {METADATA_PATH}")
    
    return risk_model, delay_model, replacement_recommender


if __name__ == "__main__":
    train_models()