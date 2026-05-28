import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

DATA_PATH = "zadachki_dataset(1).csv"
MODEL_PATH = "deadline_risk_model.joblib"


def prepare_dataset(path):
    """Подготовка данных из CSV"""
    df = pd.read_csv(path)

    # Целевая переменная - срыв дедлайна
    # deviation_days > 0 значит задача просрочена
    df["deadline_failed"] = (df["deviation_days"] > 0).astype(int)

    # Конвертация дат
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

    return df[features], df["deadline_failed"]


def build_model():
    """Строит pipeline модели"""
    numeric_features = [
        "implementation_days",
        "planned_duration_days",
        "start_weekday",
        "planned_end_weekday",
        "title_len",
        "description_len",
        "acceptance_len",
    ]

    cat_features = [
        "assignee",
        "task_type",
    ]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("text", TfidfVectorizer(max_features=200, ngram_range=(1, 2)), "text_all"),
    ])

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42, C=0.1)),
    ])


def analyze_feature_importance(model, feature_names):
    """Анализирует важность признаков"""
    try:
        # Получаем коэффициенты модели
        coefficients = model.named_steps["model"].coef_[0]
        
        # Для текстовых признаков
        text_vectorizer = model.named_steps["preprocess"].named_transformers_["text"]
        text_features = text_vectorizer.get_feature_names_out()
        
        # Важность признаков
        importance = []
        
        # Собираем все признаки
        for name, coef in zip(feature_names, coefficients):
            importance.append((abs(coef), name, coef))
        
        importance.sort(reverse=True)
        
        print("\n📊 Топ-10 важных признаков (по модулю коэффициента):")
        for i, (abs_coef, name, coef) in enumerate(importance[:10]):
            direction = "⬆️ (увеличивает риск)" if coef > 0 else "⬇️ (уменьшает риск)"
            print(f"  {i+1}. {name}: {coef:.3f} {direction}")
            
    except Exception as e:
        print(f"⚠️ Не удалось проанализировать важность признаков: {e}")


def predict_risk_with_interpretation(task_dict, model):
    """Предсказывает риск с интерпретацией"""
    X = pd.DataFrame([task_dict])
    
    # Вероятность риска
    proba = model.predict_proba(X)[0, 1]
    
    # Интерпретация
    if proba >= 0.7:
        risk_level = "высокий"
        explanation = "Задача имеет высокий риск срыва дедлайна"
    elif proba >= 0.4:
        risk_level = "средний"
        explanation = "Задача имеет средний риск срыва дедлайна"
    else:
        risk_level = "низкий"
        explanation = "Задача имеет низкий риск срыва дедлайна"
    
    return proba, risk_level, explanation


def main():
    print("📊 Загрузка данных...")
    X, y = prepare_dataset(DATA_PATH)
    
    print(f"✅ Данные загружены: {len(X)} строк")
    print(f"   Просроченные задачи (срыв дедлайна): {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    print(f"   Выполненные в срок: {len(y) - sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")

    model = build_model()

    # Cross-validation
    print("\n🔄 5-fold Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
    )

    print("Результаты 5-fold CV:")
    for metric, values in scores.items():
        if metric.startswith("test_"):
            print(f"  {metric.replace('test_', '')}: {values.mean():.3f} +/- {values.std():.3f}")

    # Holdout
    print("\n📊 Holdout validation...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.4).astype(int)  # Изменяем порог для лучшего баланса

    print("Holdout результаты:")
    print(f"  accuracy: {accuracy_score(y_test, preds):.3f}")
    print(f"  precision: {precision_score(y_test, preds):.3f}")
    print(f"  recall: {recall_score(y_test, preds):.3f}")
    print(f"  f1: {f1_score(y_test, preds):.3f}")
    print(f"  roc_auc: {roc_auc_score(y_test, probs):.3f}")
    print(f"  confusion_matrix: {confusion_matrix(y_test, preds).tolist()}")

    # Анализ важности признаков
    print("\n🔍 Анализ важности признаков...")
    model.fit(X, y)
    
    # Получаем названия признаков
    numeric_features = ["implementation_days", "planned_duration_days", "start_weekday", 
                    "planned_end_weekday", "title_len", "description_len", "acceptance_len"]
    cat_encoder = model.named_steps["preprocess"].named_transformers_["cat"]
    cat_features = cat_encoder.get_feature_names_out(["assignee", "task_type"])
    
    feature_names = numeric_features + list(cat_features)
    analyze_feature_importance(model, feature_names)

    # Обучаем на всех данных и сохраняем
    print(f"\n💾 Обучение на всех данных и сохранение модели в {MODEL_PATH}...")
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    print("✅ Модель успешно обучена и сохранена!")


if __name__ == "__main__":
    main()