import pandas as pd
import joblib


MODEL_PATH = "deadline_risk_model.joblib"


def risk_level(score):
    if score >= 0.70:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def prepare_tasks(df):
    df = df.copy()

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

    return df[features]


def predict_file(input_csv, output_csv="predictions.csv"):
    model = joblib.load(MODEL_PATH)

    raw = pd.read_csv(input_csv)
    X = prepare_tasks(raw)

    scores = model.predict_proba(X)[:, 1]

    result = raw.copy()
    result["risk_score"] = scores
    result["risk_level"] = [risk_level(score) for score in scores]

    result.to_csv(output_csv, index=False)
    return result


if __name__ == "__main__":
    predict_file("zadachki_dataset(1).csv")
