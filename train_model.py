from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "BodyFat - Extended.csv"
MODEL_PATH = BASE_DIR / "bodyfat_extended_model.pkl"
FEATURE_NAMES = [
    "sex_encoded",
    "age_years",
    "weight_kg",
    "height_m",
    "abdomen_cm",
    "bmi",
]


def load_dataset():
    required_columns = ["BodyFat", "Sex", "Age", "Weight", "Height", "Abdomen"]
    df = pd.read_csv(DATASET_PATH)
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Kolom dataset tidak lengkap: {', '.join(missing_columns)}")

    df = df[required_columns].copy()
    df["Sex"] = df["Sex"].astype("string").str.strip().str.upper()
    for column in ["BodyFat", "Age", "Weight", "Height", "Abdomen"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    valid_rows = (
        df["Sex"].isin(["M", "F"])
        & df["Age"].between(10, 100)
        & df["Weight"].between(30, 200)
        & df["Height"].between(1.2, 2.5, inclusive="neither")
        & df["Abdomen"].between(40, 150)
        & df["BodyFat"].between(2, 50, inclusive="neither")
    )
    cleaned = df.loc[valid_rows].copy()
    removed_rows = len(df) - len(cleaned)
    if cleaned.empty:
        raise ValueError("Tidak ada baris dataset yang lolos pemeriksaan kualitas.")

    cleaned["sex_encoded"] = cleaned["Sex"].map({"M": 1, "F": 0})
    cleaned["bmi"] = cleaned["Weight"] / (cleaned["Height"] ** 2)
    features = cleaned[
        ["sex_encoded", "Age", "Weight", "Height", "Abdomen", "bmi"]
    ].to_numpy(dtype=np.float64)
    targets = cleaned["BodyFat"].to_numpy(dtype=np.float64)
    sex_labels = cleaned["Sex"].to_numpy()

    return features, targets, sex_labels, removed_rows


def main():
    print(f"Membaca dataset: {DATASET_PATH.name}")
    features, targets, sex_labels, removed_rows = load_dataset()
    print(
        f"Data siap: {len(targets)} baris valid; "
        f"{removed_rows} baris tidak dipakai setelah pemeriksaan kualitas."
    )

    x_train, x_test, y_train, y_test, sex_train, sex_test = train_test_split(
        features,
        targets,
        sex_labels,
        test_size=0.2,
        random_state=42,
        stratify=sex_labels,
    )
    cv_folds = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=43).split(
            x_train, sex_train
        )
    )

    candidates = {
        "Extra Trees Regressor": (
            ExtraTreesRegressor(random_state=42, n_jobs=-1),
            {
                "n_estimators": [200],
                "max_depth": [None, 8],
                "min_samples_leaf": [1, 2, 4],
                "max_features": [0.7, 1.0],
            },
        ),
        "Random Forest Regressor": (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {
                "n_estimators": [200],
                "max_depth": [None, 8],
                "min_samples_leaf": [1, 2, 4],
                "max_features": [0.7, 1.0],
            },
        ),
        "Gradient Boosting Regressor": (
            GradientBoostingRegressor(random_state=42),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.03, 0.05],
                "max_depth": [1, 2, 3],
                "min_samples_leaf": [2, 4],
                "loss": ["huber", "squared_error"],
            },
        ),
        "Hist Gradient Boosting Regressor": (
            HistGradientBoostingRegressor(random_state=42),
            {
                "max_iter": [100, 200],
                "learning_rate": [0.03, 0.05],
                "max_leaf_nodes": [7, 15],
                "l2_regularization": [0, 1],
            },
        ),
    }

    searches = []
    for model_name, (estimator, parameter_grid) in candidates.items():
        search = GridSearchCV(
            estimator=estimator,
            param_grid=parameter_grid,
            scoring="neg_mean_absolute_error",
            cv=cv_folds,
            n_jobs=1,
            refit=True,
        )
        search.fit(x_train, y_train)
        searches.append((model_name, search))
        print(
            f"{model_name}: CV MAE "
            f"{-search.best_score_:.2f} poin persentase"
        )

    best_name, best_search = max(searches, key=lambda item: item[1].best_score_)
    test_predictions = best_search.predict(x_test)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    test_r2 = r2_score(y_test, test_predictions)

    print("\nEvaluasi holdout 20% (tidak dipakai untuk memilih model):")
    print(f"Model terpilih : {best_name}")
    print(f"MAE            : {test_mae:.2f} poin persentase")
    print(f"RMSE           : {test_rmse:.2f} poin persentase")
    print(f"R-squared      : {test_r2:.3f}")
    for sex_code, sex_label in [("M", "Pria"), ("F", "Wanita")]:
        mask = sex_test == sex_code
        if mask.any():
            sex_mae = mean_absolute_error(y_test[mask], test_predictions[mask])
            print(f"MAE {sex_label:6}: {sex_mae:.2f} poin persentase")

    best_model = best_search.best_estimator_
    best_model.fit(features, targets)
    artifact = {
        "version": 1,
        "model": best_model,
        "model_name": best_name,
        "feature_names": FEATURE_NAMES,
        "metrics": {
            "cross_validation_mae": float(-best_search.best_score_),
            "holdout_mae": float(test_mae),
            "holdout_rmse": float(test_rmse),
            "holdout_r2": float(test_r2),
        },
    }
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(artifact, model_file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nModel siap digunakan di app: {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
