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
from sklearn.model_selection import GroupKFold, train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "nhanes_dxa" / "nhanes_dxa_bodyfat.csv"
MODEL_PATH = BASE_DIR / "bodyfat_nhanes_dxa_model.pkl"
FEATURE_NAMES = [
    "sex_encoded",
    "age_years",
    "weight_kg",
    "height_m",
    "abdomen_cm",
    "bmi",
]


def make_candidates():
    return {
        "Extra Trees": lambda: ExtraTreesRegressor(
            n_estimators=250,
            max_features=1.0,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=250,
            max_features=1.0,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": lambda: GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.04,
            max_depth=2,
            min_samples_leaf=4,
            loss="huber",
            random_state=42,
        ),
        "Hist Gradient Boosting": lambda: HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
    }


def weighted_metrics(actual, predicted, sample_weight):
    weight_sum = sample_weight.sum()
    mean_actual = np.average(actual, weights=sample_weight)
    residual_sum = np.sum(sample_weight * np.square(actual - predicted))
    total_sum = np.sum(sample_weight * np.square(actual - mean_actual))
    return {
        "mae": float(np.average(np.abs(actual - predicted), weights=sample_weight)),
        "r2": float(1 - residual_sum / total_sum),
    }


def fit_model(estimator, frame):
    estimator.fit(
        frame[FEATURE_NAMES],
        frame["bodyfat_dxa_pct"],
        sample_weight=frame["sample_weight"],
    )
    return estimator


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset belum tersedia: {DATASET_PATH}\n"
            "Jalankan `python prepare_nhanes_dxa.py` terlebih dahulu."
        )

    data = pd.read_csv(DATASET_PATH)
    expected_imputations = {1, 2, 3, 4, 5}
    if set(data["imputation"].astype(int).unique()) != expected_imputations:
        raise ValueError("Dataset harus memuat kelima versi imputasi NHANES.")
    if data.duplicated(["participant_id", "imputation"]).any():
        raise ValueError("Ditemukan baris peserta/imputasi yang duplikat.")

    participants = data.drop_duplicates("participant_id")
    train_ids, test_ids = train_test_split(
        participants["participant_id"].to_numpy(),
        test_size=0.2,
        random_state=42,
        stratify=participants["sex_encoded"].to_numpy(),
    )
    candidate_scores = {}

    print(
        f"Dataset: {participants.shape[0]} peserta, {data.shape[0]} baris "
        "di lima versi imputasi."
    )
    print(
        "Evaluasi: peserta dipisahkan antar train/test; setiap versi imputasi "
        "dilatih dan dinilai tersendiri."
    )

    for name, factory in make_candidates().items():
        fold_metrics = []
        for imputation in sorted(expected_imputations):
            frame = data[data["imputation"].astype(int).eq(imputation)].copy()
            training = frame[frame["participant_id"].isin(train_ids)].reset_index(
                drop=True
            )
            groups = training["participant_id"].to_numpy()
            splitter = GroupKFold(n_splits=5)
            for fold_train_idx, fold_valid_idx in splitter.split(
                training[FEATURE_NAMES],
                training["bodyfat_dxa_pct"],
                groups=groups,
            ):
                estimator = factory()
                fit_model(estimator, training.iloc[fold_train_idx])
                validation = training.iloc[fold_valid_idx]
                predictions = estimator.predict(validation[FEATURE_NAMES])
                fold_metrics.append(
                    weighted_metrics(
                        validation["bodyfat_dxa_pct"].to_numpy(),
                        predictions,
                        validation["sample_weight"].to_numpy(),
                    )
                )

        score = {
            metric: float(np.mean([result[metric] for result in fold_metrics]))
            for metric in ("mae", "r2")
        }
        candidate_scores[name] = score
        print(
            f"{name:24} CV weighted MAE={score['mae']:.2f}, "
            f"weighted R2={score['r2']:.3f}"
        )

    best_name = min(candidate_scores, key=lambda name: candidate_scores[name]["mae"])
    factory = make_candidates()[best_name]
    holdout_results = []
    for imputation in sorted(expected_imputations):
        frame = data[data["imputation"].astype(int).eq(imputation)].copy()
        training = frame[frame["participant_id"].isin(train_ids)]
        testing = frame[frame["participant_id"].isin(test_ids)]

        estimator = fit_model(factory(), training)
        predictions = estimator.predict(testing[FEATURE_NAMES])
        result = weighted_metrics(
            testing["bodyfat_dxa_pct"].to_numpy(),
            predictions,
            testing["sample_weight"].to_numpy(),
        )
        holdout_results.append(result)

    mean_holdout = {
        metric: float(np.mean([result[metric] for result in holdout_results]))
        for metric in ("mae", "r2")
    }
    std_holdout = {
        metric: float(np.std([result[metric] for result in holdout_results]))
        for metric in ("mae", "r2")
    }
    print(f"\nModel terpilih berdasarkan CV: {best_name}")
    print(
        "Holdout DXA (rata-rata lima imputasi): "
        f"weighted MAE={mean_holdout['mae']:.2f} +/- {std_holdout['mae']:.2f}, "
        f"weighted R2={mean_holdout['r2']:.3f} +/- {std_holdout['r2']:.3f}"
    )

    full_models = []
    for imputation in sorted(expected_imputations):
        frame = data[data["imputation"].astype(int).eq(imputation)]
        full_models.append(fit_model(factory(), frame))

    artifact = {
        "version": 1,
        "model_name": f"{best_name} (ensemble of five NHANES imputations)",
        "target_name": "DXA total body-fat percentage",
        "model": full_models,
        "feature_names": FEATURE_NAMES,
        "metrics": {
            "holdout_weighted_mae_by_imputation": [
                result["mae"] for result in holdout_results
            ],
            "holdout_weighted_r2_by_imputation": [
                result["r2"] for result in holdout_results
            ],
            "holdout_weighted_mae_mean": mean_holdout["mae"],
            "holdout_weighted_r2_mean": mean_holdout["r2"],
        },
        "age_range_years": [18, 69],
        "training_source": "CDC NHANES DXA 1999-2006 public-use data",
    }
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(artifact, model_file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Model DXA tersimpan: {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
