# Model Training for Panel B: Voting Classifier (VC) with RDKit descriptors
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# VotingClassifier Panel B Classification
# using best hyperparameters from SVC, RF, LR, and GBDT runs
#
# Dataset:
#   rdkit_data_12_missing_features.csv
#
# Missing-value handling:
#   SimpleImputer(strategy="mean", keep_empty_features=True)
#   Mean imputation using training-set column means only.
#   Columns that are entirely NaN in training are retained and imputed
#   with 0.0 by SimpleImputer when keep_empty_features=True.
#
# Base models:
#   SVC RBF
#   RandomForestClassifier
#   LogisticRegression ElasticNet with StandardScaler inside its branch
#   GradientBoostingClassifier
#
# Voting:
#   Soft voting, equal weights by default.
#
# Outer split:
#   RepeatedStratifiedKFold
#   n_splits = 5
#   n_repeats = 5
#   total outer test evaluations = 25
#
# Inner validation split:
#   For each outer training fold, 12.5% of the outer training set
#   is used as validation using StratifiedShuffleSplit.
#
# Approximate full-data proportions per fold:
#   Training   = 70%
#   Validation = 10%
#   Test       = 20%
#
# Applicability Domain:
#   Defined only for the final best VotingClassifier selected from
#   the 25 models by lowest validation log-loss.
# ============================================================


# ------------------------------------------------------------
# 1. Run details
# ------------------------------------------------------------

job_id = os.environ.get("SLURM_JOB_ID", "default_job_id")
print(job_id)

print(
    "VotingClassifier Panel B, soft voting, equal weights by default, "
    "using best hyperparameters from SVC, Random Forest, Logistic Regression, "
    "and GBDT artifacts, rdkit, noSMOTE, SimpleImputer mean imputation, "
    "correct RepeatedStratifiedKFold, AD analysis using best selected model"
)

start_time = time.time()
print(start_time)


# ------------------------------------------------------------
# 2. User settings
# ------------------------------------------------------------

RANDOM_STATE = 42

N_SPLITS = 5
N_REPEATS = 5

VALIDATION_SIZE = 0.125

file_path = "/home/ssd6515/Fish/rdkit_data_12_missing_features.csv"

# Put these artifacts in the same directory where this script runs, or change
# ARTIFACT_DIR to the folder containing the four saved best-model .pt files.
ARTIFACT_DIR = "."
OUTPUT_DIR = "."

SVC_ARTIFACT = "best_svc_model_panelb_with_AD.pt"
RF_ARTIFACT = "best_rf_model_mean_imp_with_AD.pt"
LR_ARTIFACT = "best_lr_model_panelb_with_AD.pt"
GBDT_ARTIFACT = "best_gbdt_model_panelb_mean_imp_with_AD.pt"

# Use None for equal weights, or a 4-item list in SVC, RF, LR, GBDT order.
VOTING_WEIGHTS = None
# VOTING_WEIGHTS = [1, 1, 1, 1]

# Applicability-domain settings
AD_K = 5
AD_DISTANCE_PERCENTILE = 95
FEATURE_RANGE_WARNING_THRESHOLD = 0.05

HIGH_CONFIDENCE_THRESHOLD = 0.70
MODERATE_CONFIDENCE_THRESHOLD = 0.50

METADATA_COLUMNS = ["CAS", "QSAR_READY_SMILES", "mol"]


@dataclass(frozen=True)
class VotingHyperparameters:
    svc_gamma: float
    svc_C: float
    rf_n_estimators: int
    rf_max_depth: Optional[int]
    lr_C: float
    lr_l1_ratio: float
    gbdt_n_estimators: int
    gbdt_max_depth: int


def resolve_path(path_value, base_dir):
    path = Path(path_value)

    if path.is_absolute():
        return path

    return Path(base_dir) / path


def load_torch_artifact(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing artifact: {path}\n"
            "Run the corresponding original model script first, or pass the "
            "correct path with --svc-artifact, --rf-artifact, --lr-artifact, "
            "or --gbdt-artifact."
        )

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_weights(weights_value):
    if weights_value is None:
        return None

    if isinstance(weights_value, str):
        weights = [float(value.strip()) for value in weights_value.split(",")]
    else:
        weights = [float(value) for value in weights_value]

    if len(weights) != 4:
        raise ValueError("VOTING_WEIGHTS must contain exactly 4 values: SVC,RF,LR,GBDT.")

    return weights


def require_key(mapping, key, artifact_name):
    if key not in mapping:
        raise KeyError(f"Artifact {artifact_name} is missing required key: {key}")

    return mapping[key]


def validate_shared_array(artifacts, key, label):
    reference = np.asarray(require_key(artifacts["svc"], key, "svc"), dtype=object)

    for name in ["rf", "lr", "gbdt"]:
        candidate = np.asarray(require_key(artifacts[name], key, name), dtype=object)

        if not np.array_equal(reference, candidate):
            raise ValueError(
                f"The {label} in artifact '{name}' do not match the SVC artifact."
            )

    return reference


def normalize_optional_int(value):
    if value is None:
        return None

    if isinstance(value, float) and np.isnan(value):
        return None

    return int(value)


def load_artifacts():
    artifact_dir = Path(ARTIFACT_DIR)

    paths = {
        "svc": resolve_path(SVC_ARTIFACT, artifact_dir),
        "rf": resolve_path(RF_ARTIFACT, artifact_dir),
        "lr": resolve_path(LR_ARTIFACT, artifact_dir),
        "gbdt": resolve_path(GBDT_ARTIFACT, artifact_dir),
    }

    artifacts = {
        name: load_torch_artifact(path)
        for name, path in paths.items()
    }

    feature_names = validate_shared_array(
        artifacts,
        key="feature_names",
        label="feature names",
    )

    class_labels = validate_shared_array(
        artifacts,
        key="class_labels",
        label="class labels",
    )

    class_labels = np.asarray(class_labels, dtype=int)

    svc_hyper = require_key(artifacts["svc"], "best_hyperparameters", "svc")
    rf_hyper = require_key(artifacts["rf"], "best_hyperparameters", "rf")
    lr_hyper = require_key(artifacts["lr"], "best_hyperparameters", "lr")
    gbdt_hyper = require_key(artifacts["gbdt"], "best_hyperparameters", "gbdt")

    hyperparams = VotingHyperparameters(
        svc_gamma=float(require_key(svc_hyper, "gamma", "svc best_hyperparameters")),
        svc_C=float(require_key(svc_hyper, "C", "svc best_hyperparameters")),
        rf_n_estimators=int(require_key(rf_hyper, "n_estimators", "rf best_hyperparameters")),
        rf_max_depth=normalize_optional_int(
            require_key(rf_hyper, "max_depth", "rf best_hyperparameters")
        ),
        lr_C=float(require_key(lr_hyper, "C", "lr best_hyperparameters")),
        lr_l1_ratio=float(require_key(lr_hyper, "l1l2_ratio", "lr best_hyperparameters")),
        gbdt_n_estimators=int(
            require_key(gbdt_hyper, "n_estimators", "gbdt best_hyperparameters")
        ),
        gbdt_max_depth=int(require_key(gbdt_hyper, "max_depth", "gbdt best_hyperparameters")),
    )

    return artifacts, paths, feature_names, class_labels, hyperparams


def load_dataset(data_path, feature_names):
    data = pd.read_csv(data_path)

    if "Class" not in data.columns:
        raise ValueError("Dataset must contain a 'Class' column.")

    missing_features = [
        feature_name
        for feature_name in feature_names
        if feature_name not in data.columns
    ]

    if missing_features:
        raise ValueError(
            "Dataset is missing feature columns required by the artifacts: "
            + ", ".join(missing_features)
        )

    X = data.loc[:, feature_names].to_numpy()
    y = data["Class"].to_numpy()

    return data, X, y


def build_voting_pipeline(hyperparams, random_state, weights):
    svc = SVC(
        kernel="rbf",
        gamma=hyperparams.svc_gamma,
        C=hyperparams.svc_C,
        probability=True,
        random_state=random_state,
    )

    rf = RandomForestClassifier(
        n_estimators=hyperparams.rf_n_estimators,
        max_depth=hyperparams.rf_max_depth,
        random_state=random_state,
        n_jobs=-1,
    )

    lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=hyperparams.lr_C,
                    penalty="elasticnet",
                    l1_ratio=hyperparams.lr_l1_ratio,
                    solver="saga",
                    class_weight="balanced",
                    max_iter=6000,
                    random_state=random_state,
                ),
            ),
        ]
    )

    gbdt = GradientBoostingClassifier(
        n_estimators=hyperparams.gbdt_n_estimators,
        max_depth=hyperparams.gbdt_max_depth,
        learning_rate=0.05,
        random_state=random_state,
    )

    voting_model = VotingClassifier(
        estimators=[
            ("svc", svc),
            ("rf", rf),
            ("lr", lr),
            ("gbdt", gbdt),
        ],
        voting="soft",
        weights=weights,
        n_jobs=None,
    )

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="mean",
                    keep_empty_features=True,
                ),
            ),
            ("vote", voting_model),
        ]
    )


def compute_fold_metrics(y_true, y_pred, class_labels, validation_loss):
    precision_per_class = precision_score(
        y_true,
        y_pred,
        labels=class_labels,
        average=None,
        zero_division=0,
    )
    recall_per_class = recall_score(
        y_true,
        y_pred,
        labels=class_labels,
        average=None,
        zero_division=0,
    )
    f1_per_class = f1_score(
        y_true,
        y_pred,
        labels=class_labels,
        average=None,
        zero_division=0,
    )

    return {
        "validation_loss": validation_loss,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(
            y_true,
            y_pred,
            labels=class_labels,
            average="weighted",
            zero_division=0,
        ),
        "precision": precision_per_class,
        "recall": recall_per_class,
        "f1_not_weighted": f1_per_class,
        "avg_precision": np.mean(precision_per_class),
        "avg_recall": np.mean(recall_per_class),
        "avg_f1_not_weighted": np.mean(f1_per_class),
    }


def summarize_metrics(all_fold_metrics, class_labels):
    all_validation_losses = np.array(
        [fold_metric["validation_loss"] for fold_metric in all_fold_metrics]
    )
    all_accuracies = np.array(
        [fold_metric["accuracy"] for fold_metric in all_fold_metrics]
    )
    all_f1_weighted = np.array(
        [fold_metric["f1_weighted"] for fold_metric in all_fold_metrics]
    )
    all_avg_precision = np.array(
        [fold_metric["avg_precision"] for fold_metric in all_fold_metrics]
    )
    all_avg_recall = np.array(
        [fold_metric["avg_recall"] for fold_metric in all_fold_metrics]
    )
    all_avg_f1_not_weighted = np.array(
        [fold_metric["avg_f1_not_weighted"] for fold_metric in all_fold_metrics]
    )

    all_precision_per_class = np.vstack(
        [fold_metric["precision"] for fold_metric in all_fold_metrics]
    )
    all_recall_per_class = np.vstack(
        [fold_metric["recall"] for fold_metric in all_fold_metrics]
    )
    all_f1_per_class = np.vstack(
        [fold_metric["f1_not_weighted"] for fold_metric in all_fold_metrics]
    )

    final_metrics = {
        "validation_loss_mean": np.mean(all_validation_losses),
        "validation_loss_std": np.std(all_validation_losses),
        "accuracy_mean": np.mean(all_accuracies),
        "accuracy_std": np.std(all_accuracies),
        "f1_weighted_mean": np.mean(all_f1_weighted),
        "f1_weighted_std": np.std(all_f1_weighted),
        "avg_precision_mean": np.mean(all_avg_precision),
        "avg_precision_std": np.std(all_avg_precision),
        "avg_recall_mean": np.mean(all_avg_recall),
        "avg_recall_std": np.std(all_avg_recall),
        "avg_f1_not_weighted_mean": np.mean(all_avg_f1_not_weighted),
        "avg_f1_not_weighted_std": np.std(all_avg_f1_not_weighted),
        "precision_per_class_mean": np.mean(all_precision_per_class, axis=0),
        "precision_per_class_std": np.std(all_precision_per_class, axis=0),
        "recall_per_class_mean": np.mean(all_recall_per_class, axis=0),
        "recall_per_class_std": np.std(all_recall_per_class, axis=0),
        "f1_per_class_mean": np.mean(all_f1_per_class, axis=0),
        "f1_per_class_std": np.std(all_f1_per_class, axis=0),
    }

    all_metrics = {
        "validation_loss_all_folds": all_validation_losses,
        "accuracy_mean": all_accuracies,
        "f1_weighted_mean": all_f1_weighted,
        "avg_precision_mean": all_avg_precision,
        "avg_recall_mean": all_avg_recall,
        "avg_f1_not_weighted_mean": all_avg_f1_not_weighted,
        "precision_per_class_all_folds": all_precision_per_class,
        "recall_per_class_all_folds": all_recall_per_class,
        "f1_per_class_all_folds": all_f1_per_class,
        "validation_loss_final_mean": final_metrics["validation_loss_mean"],
        "validation_loss_final_std": final_metrics["validation_loss_std"],
        "accuracy_final_mean": final_metrics["accuracy_mean"],
        "accuracy_final_std": final_metrics["accuracy_std"],
        "f1_weighted_final_mean": final_metrics["f1_weighted_mean"],
        "f1_weighted_final_std": final_metrics["f1_weighted_std"],
        "avg_precision_final_mean": final_metrics["avg_precision_mean"],
        "avg_precision_final_std": final_metrics["avg_precision_std"],
        "avg_recall_final_mean": final_metrics["avg_recall_mean"],
        "avg_recall_final_std": final_metrics["avg_recall_std"],
        "avg_f1_not_weighted_final_mean": final_metrics["avg_f1_not_weighted_mean"],
        "avg_f1_not_weighted_final_std": final_metrics["avg_f1_not_weighted_std"],
        "precision_per_class_mean": final_metrics["precision_per_class_mean"],
        "precision_per_class_std": final_metrics["precision_per_class_std"],
        "recall_per_class_mean": final_metrics["recall_per_class_mean"],
        "recall_per_class_std": final_metrics["recall_per_class_std"],
        "f1_per_class_mean": final_metrics["f1_per_class_mean"],
        "f1_per_class_std": final_metrics["f1_per_class_std"],
        "class_labels": class_labels,
    }

    return all_metrics, final_metrics


def summarize_repeat_metrics(repeat, split_indices, repeat_fold_metrics, class_labels):
    repeat_all_metrics, repeat_final_metrics = summarize_metrics(
        repeat_fold_metrics,
        class_labels=class_labels,
    )

    return {
        "repeat": repeat,
        "split_indices": split_indices,
        "repeat_all_metrics": repeat_all_metrics,
        "repeat_final_metrics": repeat_final_metrics,
    }


def get_pipeline_classes(model):
    return model.named_steps["vote"].classes_


def build_applicability_domain_reference(
    X_train_imputed,
    feature_names,
    ad_k=5,
    ad_distance_percentile=95,
):
    X_train_imputed = np.asarray(X_train_imputed, dtype=float)

    n_train = X_train_imputed.shape[0]

    if n_train < 3:
        raise ValueError(
            "At least 3 training samples are required to build a meaningful AD reference."
        )

    effective_ad_k = min(ad_k, n_train - 1)

    descriptor_mean = np.mean(X_train_imputed, axis=0)
    descriptor_std = np.std(X_train_imputed, axis=0)
    descriptor_std = np.where(descriptor_std == 0, 1.0, descriptor_std)

    X_train_scaled_for_ad = (X_train_imputed - descriptor_mean) / descriptor_std

    knn_train = NearestNeighbors(
        n_neighbors=effective_ad_k + 1,
        metric="euclidean",
    )
    knn_train.fit(X_train_scaled_for_ad)

    train_distances, train_neighbor_indices = knn_train.kneighbors(
        X_train_scaled_for_ad
    )

    train_knn_mean_distances = train_distances[:, 1:].mean(axis=1)

    ad_distance_threshold = np.percentile(
        train_knn_mean_distances,
        ad_distance_percentile,
    )

    feature_min = np.min(X_train_imputed, axis=0)
    feature_max = np.max(X_train_imputed, axis=0)

    return {
        "ad_method": "standardized RDKit descriptor-space kNN Euclidean distance",
        "ad_k_requested": ad_k,
        "ad_k_effective": effective_ad_k,
        "ad_distance_percentile": ad_distance_percentile,
        "ad_distance_threshold": ad_distance_threshold,
        "descriptor_mean": descriptor_mean,
        "descriptor_std": descriptor_std,
        "feature_min": feature_min,
        "feature_max": feature_max,
        "feature_names": list(feature_names),
        "training_knn_mean_distances": train_knn_mean_distances,
        "training_scaled_features_for_ad": X_train_scaled_for_ad,
        "train_neighbor_indices": train_neighbor_indices,
    }


def evaluate_applicability_domain(
    X_query_imputed,
    ad_reference,
    feature_range_warning_threshold=0.05,
):
    X_query_imputed = np.asarray(X_query_imputed, dtype=float)

    descriptor_mean = ad_reference["descriptor_mean"]
    descriptor_std = ad_reference["descriptor_std"]
    feature_min = ad_reference["feature_min"]
    feature_max = ad_reference["feature_max"]
    X_train_scaled_for_ad = ad_reference["training_scaled_features_for_ad"]
    ad_k = ad_reference["ad_k_effective"]
    ad_distance_threshold = ad_reference["ad_distance_threshold"]

    X_query_scaled_for_ad = (X_query_imputed - descriptor_mean) / descriptor_std

    knn_query = NearestNeighbors(
        n_neighbors=ad_k,
        metric="euclidean",
    )
    knn_query.fit(X_train_scaled_for_ad)

    query_distances, query_neighbor_indices = knn_query.kneighbors(
        X_query_scaled_for_ad
    )

    query_knn_mean_distances = query_distances.mean(axis=1)
    inside_distance_ad = query_knn_mean_distances <= ad_distance_threshold

    outside_feature_range_matrix = (
        (X_query_imputed < feature_min) | (X_query_imputed > feature_max)
    )

    n_features_outside_range = outside_feature_range_matrix.sum(axis=1)
    fraction_features_outside_range = (
        n_features_outside_range / X_query_imputed.shape[1]
    )
    feature_range_warning = (
        fraction_features_outside_range > feature_range_warning_threshold
    )

    ad_status = []

    for inside_distance, range_warning in zip(
        inside_distance_ad,
        feature_range_warning,
    ):
        if inside_distance and not range_warning:
            ad_status.append("Inside AD")
        elif inside_distance and range_warning:
            ad_status.append("Inside distance-based AD, but descriptor-range warning")
        else:
            ad_status.append("Outside AD")

    return {
        "query_knn_mean_distances": query_knn_mean_distances,
        "query_neighbor_indices": query_neighbor_indices,
        "inside_distance_ad": inside_distance_ad,
        "n_features_outside_range": n_features_outside_range,
        "fraction_features_outside_range": fraction_features_outside_range,
        "feature_range_warning": feature_range_warning,
        "outside_feature_range_matrix": outside_feature_range_matrix,
        "ad_status": np.array(ad_status, dtype=object),
    }


def assign_prediction_reliability(
    ad_status,
    max_prediction_probability,
    high_confidence_threshold=0.70,
    moderate_confidence_threshold=0.50,
):
    reliability = []

    for status, max_prob in zip(ad_status, max_prediction_probability):
        if status == "Inside AD" and max_prob >= high_confidence_threshold:
            reliability.append("High")
        elif status == "Inside AD" and max_prob >= moderate_confidence_threshold:
            reliability.append("Moderate")
        else:
            reliability.append("Low")

    return np.array(reliability, dtype=object)


def get_outside_range_feature_names(outside_feature_range_row, feature_names):
    outside_features = [
        feature_name
        for feature_name, is_outside in zip(feature_names, outside_feature_range_row)
        if is_outside
    ]

    if len(outside_features) == 0:
        return ""

    return ";".join(outside_features)


def impute_with_column_means(X, col_means):
    X_array = np.asarray(X, dtype=float).copy()
    missing_rows, missing_cols = np.where(np.isnan(X_array))
    X_array[missing_rows, missing_cols] = col_means[missing_cols]
    return X_array


def build_prediction_dataframe(
    data,
    indices,
    y_true,
    y_pred,
    y_proba,
    class_labels,
    repeat,
    fold,
    split_idx,
):
    max_prediction_probability = np.max(y_proba, axis=1)

    prediction_df = pd.DataFrame(
        {
            "repeat": repeat,
            "fold": fold,
            "split_idx": split_idx,
            "original_dataset_index": indices,
            "true_class": y_true,
            "predicted_class": y_pred,
            "max_prediction_probability": max_prediction_probability,
        }
    )

    insert_metadata_columns(prediction_df, data, indices)

    proba_df = pd.DataFrame(
        y_proba,
        columns=[f"prob_class_{class_label}" for class_label in class_labels],
    )

    return pd.concat(
        [
            prediction_df.reset_index(drop=True),
            proba_df.reset_index(drop=True),
        ],
        axis=1,
    )


def insert_metadata_columns(target_df, data, indices):
    insert_position = 1

    for metadata_column in METADATA_COLUMNS:
        if metadata_column in data.columns:
            target_df.insert(
                insert_position,
                metadata_column,
                data.iloc[indices][metadata_column].to_numpy(),
            )
            insert_position += 1


def save_best_model_without_ad(
    output_dir,
    best_model,
    feature_names,
    class_labels,
    hyperparams,
    artifact_paths,
    weights,
    best_selection,
):
    torch.save(
        {
            "model": best_model,
            "feature_names": feature_names,
            "class_labels": class_labels,
            "best_hyperparameters": hyperparams.__dict__,
            "source_artifacts": {
                name: str(path)
                for name, path in artifact_paths.items()
            },
            "voting": {
                "voting": "soft",
                "weights": weights,
            },
            "best_model_selection": best_selection,
            "train_col_means": best_model.named_steps["imputer"].statistics_,
            "imputer": {
                "type": "sklearn.impute.SimpleImputer",
                "strategy": "mean",
                "keep_empty_features": True,
            },
        },
        output_dir / "best_voting_model_panelb.pt",
    )


def run_applicability_domain_and_save_outputs(
    output_dir,
    data,
    X,
    y,
    feature_names,
    class_labels,
    best_model,
    hyperparams,
    artifact_paths,
    weights,
    best_selection,
):
    best_train_index = best_selection["train_index"]
    best_valid_index = best_selection["valid_index"]
    best_test_index = best_selection["test_index"]

    best_train_feature_raw = X[best_train_index]
    best_test_feature_raw = X[best_test_index]

    best_train_label = y[best_train_index]
    best_test_label = y[best_test_index]

    best_col_means = best_model.named_steps["imputer"].statistics_

    best_train_feature = impute_with_column_means(
        best_train_feature_raw,
        best_col_means,
    )
    best_test_feature = impute_with_column_means(
        best_test_feature_raw,
        best_col_means,
    )

    ad_reference = build_applicability_domain_reference(
        X_train_imputed=best_train_feature,
        feature_names=feature_names,
        ad_k=AD_K,
        ad_distance_percentile=AD_DISTANCE_PERCENTILE,
    )

    ad_summary_df = pd.DataFrame(
        {
            "ad_method": [ad_reference["ad_method"]],
            "ad_k_requested": [ad_reference["ad_k_requested"]],
            "ad_k_effective": [ad_reference["ad_k_effective"]],
            "ad_distance_percentile": [ad_reference["ad_distance_percentile"]],
            "ad_distance_threshold": [ad_reference["ad_distance_threshold"]],
            "training_distance_min": [
                np.min(ad_reference["training_knn_mean_distances"])
            ],
            "training_distance_mean": [
                np.mean(ad_reference["training_knn_mean_distances"])
            ],
            "training_distance_median": [
                np.median(ad_reference["training_knn_mean_distances"])
            ],
            "training_distance_std": [
                np.std(ad_reference["training_knn_mean_distances"])
            ],
            "training_distance_p90": [
                np.percentile(ad_reference["training_knn_mean_distances"], 90)
            ],
            "training_distance_p95": [
                np.percentile(ad_reference["training_knn_mean_distances"], 95)
            ],
            "training_distance_p99": [
                np.percentile(ad_reference["training_knn_mean_distances"], 99)
            ],
            "training_distance_max": [
                np.max(ad_reference["training_knn_mean_distances"])
            ],
            "feature_range_warning_threshold": [FEATURE_RANGE_WARNING_THRESHOLD],
            "high_confidence_threshold": [HIGH_CONFIDENCE_THRESHOLD],
            "moderate_confidence_threshold": [MODERATE_CONFIDENCE_THRESHOLD],
            "best_model_repeat": [best_selection["repeat"]],
            "best_model_fold": [best_selection["fold"]],
            "best_model_split_idx": [best_selection["split_idx"]],
            "best_model_validation_loss": [best_selection["validation_loss"]],
            "best_model_svc_gamma": [hyperparams.svc_gamma],
            "best_model_svc_C": [hyperparams.svc_C],
            "best_model_rf_n_estimators": [hyperparams.rf_n_estimators],
            "best_model_rf_max_depth": [hyperparams.rf_max_depth],
            "best_model_lr_C": [hyperparams.lr_C],
            "best_model_lr_l1_ratio": [hyperparams.lr_l1_ratio],
            "best_model_gbdt_n_estimators": [hyperparams.gbdt_n_estimators],
            "best_model_gbdt_max_depth": [hyperparams.gbdt_max_depth],
            "n_ad_training_chemicals": [best_train_feature.shape[0]],
            "n_features": [best_train_feature.shape[1]],
        }
    )
    ad_summary_df.to_csv(
        output_dir / "voting_panelb_best_model_AD_summary.csv",
        index=False,
    )

    feature_range_df = pd.DataFrame(
        {
            "feature": feature_names,
            "training_min": ad_reference["feature_min"],
            "training_max": ad_reference["feature_max"],
            "training_mean_for_AD_standardization": ad_reference["descriptor_mean"],
            "training_std_for_AD_standardization": ad_reference["descriptor_std"],
            "training_imputation_mean": best_col_means,
        }
    )
    feature_range_df.to_csv(
        output_dir / "voting_panelb_best_model_training_feature_ranges.csv",
        index=False,
    )

    ad_training_distances_df = pd.DataFrame(
        {
            "original_dataset_index": best_train_index,
            "Class": best_train_label,
            "training_knn_mean_distance": ad_reference["training_knn_mean_distances"],
            "inside_training_AD_threshold": (
                ad_reference["training_knn_mean_distances"]
                <= ad_reference["ad_distance_threshold"]
            ),
        }
    )
    insert_metadata_columns(ad_training_distances_df, data, best_train_index)
    ad_training_distances_df.to_csv(
        output_dir / "voting_panelb_best_model_training_AD_distances.csv",
        index=False,
    )

    best_test_pred = best_model.predict(best_test_feature_raw)
    best_test_proba = best_model.predict_proba(best_test_feature_raw)
    best_test_max_proba = np.max(best_test_proba, axis=1)

    best_test_ad_results = evaluate_applicability_domain(
        X_query_imputed=best_test_feature,
        ad_reference=ad_reference,
        feature_range_warning_threshold=FEATURE_RANGE_WARNING_THRESHOLD,
    )

    best_test_reliability = assign_prediction_reliability(
        ad_status=best_test_ad_results["ad_status"],
        max_prediction_probability=best_test_max_proba,
        high_confidence_threshold=HIGH_CONFIDENCE_THRESHOLD,
        moderate_confidence_threshold=MODERATE_CONFIDENCE_THRESHOLD,
    )

    best_test_results_df = pd.DataFrame(
        {
            "original_dataset_index": best_test_index,
            "true_class": best_test_label,
            "predicted_class": best_test_pred,
            "max_prediction_probability": best_test_max_proba,
            "ad_status": best_test_ad_results["ad_status"],
            "prediction_reliability": best_test_reliability,
            "knn_mean_distance": best_test_ad_results["query_knn_mean_distances"],
            "ad_distance_threshold": ad_reference["ad_distance_threshold"],
            "inside_distance_ad": best_test_ad_results["inside_distance_ad"],
            "n_features_outside_training_range": (
                best_test_ad_results["n_features_outside_range"]
            ),
            "fraction_features_outside_training_range": (
                best_test_ad_results["fraction_features_outside_range"]
            ),
            "feature_range_warning": best_test_ad_results["feature_range_warning"],
        }
    )
    insert_metadata_columns(best_test_results_df, data, best_test_index)

    proba_df = pd.DataFrame(
        best_test_proba,
        columns=[f"prob_class_{class_label}" for class_label in class_labels],
    )
    best_test_results_df = pd.concat(
        [
            best_test_results_df.reset_index(drop=True),
            proba_df.reset_index(drop=True),
        ],
        axis=1,
    )

    outside_range_feature_names = []

    for row_idx in range(best_test_ad_results["outside_feature_range_matrix"].shape[0]):
        outside_range_feature_names.append(
            get_outside_range_feature_names(
                best_test_ad_results["outside_feature_range_matrix"][row_idx],
                feature_names,
            )
        )

    best_test_results_df["features_outside_training_range"] = (
        outside_range_feature_names
    )

    best_test_results_df.to_csv(
        output_dir / "voting_panelb_best_model_test_predictions_with_AD.csv",
        index=False,
    )

    n_test = best_test_results_df.shape[0]
    n_inside_distance_ad = int(best_test_results_df["inside_distance_ad"].sum())
    n_outside_distance_ad = n_test - n_inside_distance_ad
    n_feature_range_warning = int(best_test_results_df["feature_range_warning"].sum())

    test_ad_summary_df = pd.DataFrame(
        {
            "n_test_chemicals": [n_test],
            "n_inside_distance_ad": [n_inside_distance_ad],
            "n_outside_distance_ad": [n_outside_distance_ad],
            "percent_inside_distance_ad": [100 * n_inside_distance_ad / n_test],
            "percent_outside_distance_ad": [100 * n_outside_distance_ad / n_test],
            "n_feature_range_warning": [n_feature_range_warning],
            "percent_feature_range_warning": [
                100 * n_feature_range_warning / n_test
            ],
            "n_high_reliability": [
                int((best_test_results_df["prediction_reliability"] == "High").sum())
            ],
            "n_moderate_reliability": [
                int(
                    (best_test_results_df["prediction_reliability"] == "Moderate").sum()
                )
            ],
            "n_low_reliability": [
                int((best_test_results_df["prediction_reliability"] == "Low").sum())
            ],
        }
    )
    test_ad_summary_df.to_csv(
        output_dir / "voting_panelb_best_model_test_AD_summary.csv",
        index=False,
    )

    torch.save(
        {
            "model": best_model,
            "train_col_means": best_col_means,
            "feature_names": feature_names,
            "class_labels": class_labels,
            "best_hyperparameters": hyperparams.__dict__,
            "source_artifacts": {
                name: str(path)
                for name, path in artifact_paths.items()
            },
            "voting": {
                "voting": "soft",
                "weights": weights,
            },
            "best_model_selection": best_selection,
            "applicability_domain": ad_reference,
            "feature_range_warning_threshold": FEATURE_RANGE_WARNING_THRESHOLD,
            "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
            "moderate_confidence_threshold": MODERATE_CONFIDENCE_THRESHOLD,
            "imputer": {
                "type": "sklearn.impute.SimpleImputer",
                "strategy": "mean",
                "keep_empty_features": True,
            },
        },
        output_dir / "best_voting_model_panelb_with_AD.pt",
    )

    return {
        "ad_summary": ad_summary_df,
        "test_ad_summary": test_ad_summary_df,
        "test_predictions_with_ad": best_test_results_df,
    }


def print_final_metrics(final_metrics, class_labels):
    print("\nFinal Overall Metrics across 25 Voting Models (Mean and Std):")
    for metric, value in final_metrics.items():
        if not isinstance(value, np.ndarray):
            print(f"{metric}: {value}")

    print("\nFinal Per-Class Metrics across 25 Voting Models (Mean +/- Std):")
    for class_idx, class_label in enumerate(class_labels):
        print(f"Class {class_label}:")
        print(
            f"  Precision: "
            f"{final_metrics['precision_per_class_mean'][class_idx]:.4f} +/- "
            f"{final_metrics['precision_per_class_std'][class_idx]:.4f}"
        )
        print(
            f"  Recall:    "
            f"{final_metrics['recall_per_class_mean'][class_idx]:.4f} +/- "
            f"{final_metrics['recall_per_class_std'][class_idx]:.4f}"
        )
        print(
            f"  F1 Score:  "
            f"{final_metrics['f1_per_class_mean'][class_idx]:.4f} +/- "
            f"{final_metrics['f1_per_class_std'][class_idx]:.4f}"
        )


# ------------------------------------------------------------
# 3. Start single-flowing HPC execution
# ------------------------------------------------------------

print("Starting Panel B VotingClassifier run")
print("Random state:", RANDOM_STATE)
print("Data path:", file_path)
print("Artifact dir:", ARTIFACT_DIR)
print("Output dir:", OUTPUT_DIR)

output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

weights = parse_weights(VOTING_WEIGHTS)

artifacts, artifact_paths, feature_names, class_labels, hyperparams = load_artifacts()

print("\nLoaded best hyperparameters:")
print(hyperparams)
print("Voting weights:", weights if weights is not None else "equal")

data, X, y = load_dataset(file_path, feature_names)

print("\nShape of concatenated data:", X.shape)
print("Shape of target labels:", y.shape)
print("Class labels from artifacts:", class_labels)

dataset_class_labels = np.unique(y)
if not np.array_equal(np.asarray(class_labels), np.asarray(dataset_class_labels)):
    raise ValueError(
        "Class labels from artifacts do not match np.unique(y) in dataset. "
        f"Artifacts: {class_labels}; dataset: {dataset_class_labels}"
    )

rskf = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE,
)
all_outer_splits = list(rskf.split(X, y))

expected_n_splits = N_SPLITS * N_REPEATS
print("Total number of outer splits:", len(all_outer_splits))
print("Expected number of outer splits:", expected_n_splits)

if len(all_outer_splits) != expected_n_splits:
    raise ValueError(
        f"Expected {expected_n_splits} splits, but got {len(all_outer_splits)}."
    )

all_fold_metrics = []
all_fold_predictions = []
repeat_metrics_list = []

global_best_val_loss = np.inf
global_best_model = None
global_best_selection = None

for repeat in range(N_REPEATS):
    print("\n============================================================")
    print("repeat:", repeat)
    print("============================================================")

    repeat_fold_metrics = []
    repeat_split_indices = []

    for fold in range(N_SPLITS):
        split_idx = repeat * N_SPLITS + fold
        print("\n  Fold:", fold)

        train_valid_index, test_index = all_outer_splits[split_idx]

        X_train_valid = X[train_valid_index]
        y_train_valid = y[train_valid_index]

        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE + split_idx,
        )

        inner_train_pos, valid_pos = next(
            sss.split(X_train_valid, y_train_valid)
        )

        train_index = train_valid_index[inner_train_pos]
        valid_index = train_valid_index[valid_pos]

        train_feature_raw = X[train_index]
        train_label = y[train_index]

        valid_feature_raw = X[valid_index]
        valid_label = y[valid_index]

        test_feature_raw = X[test_index]
        test_label = y[test_index]

        print("    Training data shape:", train_feature_raw.shape)
        print("    Validation data shape:", valid_feature_raw.shape)
        print("    Test data shape:", test_feature_raw.shape)

        model = build_voting_pipeline(
            hyperparams=hyperparams,
            random_state=RANDOM_STATE + split_idx,
            weights=weights,
        )

        model.fit(train_feature_raw, train_label)

        fitted_classes = get_pipeline_classes(model)
        if not np.array_equal(fitted_classes, class_labels):
            raise ValueError(
                "Fitted VotingClassifier classes do not match artifact class labels. "
                f"Fitted: {fitted_classes}; artifacts: {class_labels}"
            )

        valid_proba = model.predict_proba(valid_feature_raw)
        valid_loss = log_loss(
            valid_label,
            valid_proba,
            labels=class_labels,
        )

        test_pred = model.predict(test_feature_raw)
        test_proba = model.predict_proba(test_feature_raw)

        fold_metrics = compute_fold_metrics(
            y_true=test_label,
            y_pred=test_pred,
            class_labels=class_labels,
            validation_loss=valid_loss,
        )

        all_fold_metrics.append(fold_metrics)
        repeat_fold_metrics.append(fold_metrics)
        repeat_split_indices.append(split_idx)

        fold_prediction_df = build_prediction_dataframe(
            data=data,
            indices=test_index,
            y_true=test_label,
            y_pred=test_pred,
            y_proba=test_proba,
            class_labels=class_labels,
            repeat=repeat,
            fold=fold,
            split_idx=split_idx,
        )
        all_fold_predictions.append(fold_prediction_df)

        print(f"    Validation log-loss: {valid_loss:.6f}")
        print(f"    Test accuracy: {fold_metrics['accuracy']:.4f}")
        print(f"    Test weighted F1: {fold_metrics['f1_weighted']:.4f}")

        if valid_loss < global_best_val_loss:
            global_best_val_loss = valid_loss
            global_best_model = model
            global_best_selection = {
                "repeat": repeat,
                "fold": fold,
                "split_idx": split_idx,
                "validation_loss": valid_loss,
                "train_index": train_index,
                "valid_index": valid_index,
                "test_index": test_index,
            }

    repeat_metrics_list.append(
        summarize_repeat_metrics(
            repeat=repeat,
            split_indices=repeat_split_indices,
            repeat_fold_metrics=repeat_fold_metrics,
            class_labels=class_labels,
        )
    )

all_metrics, final_metrics = summarize_metrics(all_fold_metrics, class_labels)

print("\n25 Fold Validation Log-Loss Results:")
print(all_metrics["validation_loss_all_folds"])
print("\n25 Fold Accuracy Results:")
print(all_metrics["accuracy_mean"])
print("\n25 Fold F1 Weighted Results:")
print(all_metrics["f1_weighted_mean"])
print("\n25 Fold Average Precision Results:")
print(all_metrics["avg_precision_mean"])
print("\n25 Fold Average Recall Results:")
print(all_metrics["avg_recall_mean"])
print("\n25 Fold Average F1 Results:")
print(all_metrics["avg_f1_not_weighted_mean"])
print("\n25 Fold Per-Class Precision Results:")
print(all_metrics["precision_per_class_all_folds"])
print("\n25 Fold Per-Class Recall Results:")
print(all_metrics["recall_per_class_all_folds"])
print("\n25 Fold Per-Class F1 Results:")
print(all_metrics["f1_per_class_all_folds"])

print_final_metrics(final_metrics, class_labels)

with open(output_dir / "results_voting_panelb_final_metrics.pkl", "wb") as f:
    pickle.dump(all_metrics, f)

with open(output_dir / "results_voting_panelb_repeat.pkl", "wb") as f:
    pickle.dump(repeat_metrics_list, f)

all_predictions_df = pd.concat(all_fold_predictions, axis=0, ignore_index=True)
all_predictions_df.to_csv(
    output_dir / "voting_panelb_all_fold_test_predictions.csv",
    index=False,
)

print(
    "\nBest Overall VotingClassifier selected from all 25 outer evaluations:"
    f"\n  repeat = {global_best_selection['repeat']}"
    f"\n  fold = {global_best_selection['fold']}"
    f"\n  split_idx = {global_best_selection['split_idx']}"
    f"\n  validation_loss = {global_best_selection['validation_loss']}"
)

save_best_model_without_ad(
    output_dir=output_dir,
    best_model=global_best_model,
    feature_names=feature_names,
    class_labels=class_labels,
    hyperparams=hyperparams,
    artifact_paths=artifact_paths,
    weights=weights,
    best_selection=global_best_selection,
)

ad_outputs = run_applicability_domain_and_save_outputs(
    output_dir=output_dir,
    data=data,
    X=X,
    y=y,
    feature_names=feature_names,
    class_labels=class_labels,
    best_model=global_best_model,
    hyperparams=hyperparams,
    artifact_paths=artifact_paths,
    weights=weights,
    best_selection=global_best_selection,
)

print("\nBest VotingClassifier held-out test set AD summary:")
print(ad_outputs["test_predictions_with_ad"]["ad_status"].value_counts(dropna=False))
print("\nBest VotingClassifier held-out test set reliability summary:")
print(
    ad_outputs["test_predictions_with_ad"]["prediction_reliability"].value_counts(
        dropna=False
    )
)

print("\nSaved outputs to:", output_dir.resolve())
print("Saved final metrics: results_voting_panelb_final_metrics.pkl")
print("Saved repeat metrics: results_voting_panelb_repeat.pkl")
print("Saved all-fold predictions: voting_panelb_all_fold_test_predictions.csv")
print("Saved best model: best_voting_model_panelb.pt")
print("Saved best model with AD: best_voting_model_panelb_with_AD.pt")
print("Saved AD test predictions: voting_panelb_best_model_test_predictions_with_AD.csv")

end_time = time.time()
execution_time = (end_time - start_time) / 60
print(f"\nTotal execution time: {execution_time:.2f} minutes")