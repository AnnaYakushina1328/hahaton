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
    df = pd.read_csv(path)

    df["deadline_failed"] = (df["deviation_days"] > 0).astype(int)

    df["start_date_dt"] = pd.to_datetime(df["start_date"], dayfirst=True, errors="coerce")
    df["planned_end_date_dt"] = pd.to_datetime(df["planned_end_date"], dayfirst=True, errors="coerce")

    df["planned_duration_days"] = (df["planned_end_date_dt"] - df["start_date_dt"]).dt.days
    df["start_weekday"] = df["start_date_dt"].dt.weekday
    df["planned_end_weekday"] = df["planned_end_date_dt"].dt.weekday

    df["title_len"] = df["title"].fillna("").str.len()
    df["description_len"] = df["description"].fillna("").str.len()
    df["acceptance_len"] = df["acceptance_criteria"].fillna("").str.len()

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
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])


def risk_level(score):
    if score >= 0.70:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def main():
    X, y = prepare_dataset(DATA_PATH)

    model = build_model()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
    )

    print("5-fold CV:")
    for metric, values in scores.items():
        if metric.startswith("test_"):
            print(metric, round(values.mean(), 3), "+/-", round(values.std(), 3))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    print("\nHoldout:")
    print("accuracy:", round(accuracy_score(y_test, preds), 3))
    print("precision:", round(precision_score(y_test, preds), 3))
    print("recall:", round(recall_score(y_test, preds), 3))
    print("f1:", round(f1_score(y_test, preds), 3))
    print("roc_auc:", round(roc_auc_score(y_test, probs), 3))
    print("confusion_matrix:", confusion_matrix(y_test, preds).tolist())

    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    print("\nSaved:", MODEL_PATH)


if __name__ == "__main__":
    main()
