# Model Training for Panel B: Support Vector Classifier (SVC) with RDKit descriptors
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, log_loss
from sklearn.neighbors import NearestNeighbors
import os
import time
import pickle
import torch


# ============================================================
# SVC Panel B Classification with Correct RepeatedStratifiedKFold
# + Applicability Domain Analysis
#
# Dataset:
#   rdkit_data_12_missing_features.csv
#
# Missing-value handling:
#   Mean imputation using training-set column means only.
#   The same training means are applied to validation and test sets.
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
# No ECFP, no SMOTE
#
# Applicability Domain:
#   Defined only for the final best model selected from the 25 models.
#   AD training set = training subset used to fit the selected best model.
#
# AD method:
#   1. Standardized RDKit descriptor-space kNN Euclidean distance
#   2. Descriptor min-max range check
#   3. Prediction probability confidence category
#
# Important:
#   AD does NOT stop prediction.
#   All test chemicals are still predicted.
#   AD only flags prediction reliability.
# ============================================================


# ------------------------------------------------------------
# Helper functions: missing-value imputation
# ------------------------------------------------------------

def compute_train_column_means(X_train):
    """
    Compute column means from the training set only.
    If a column is entirely NaN in training, replace that mean with 0.0.
    """
    X_train = np.asarray(X_train, dtype=float)

    train_col_means = np.nanmean(X_train, axis=0)
    train_col_means = np.where(np.isnan(train_col_means), 0.0, train_col_means)

    return train_col_means


def mean_impute_with_given_column_means(X, col_means):
    """
    Impute missing values in X using precomputed column means.
    These means should come from the training set only.
    """
    X = np.asarray(X, dtype=float).copy()

    missing_rows, missing_cols = np.where(np.isnan(X))
    X[missing_rows, missing_cols] = col_means[missing_cols]

    return X


# ------------------------------------------------------------
# Helper functions: applicability domain
# ------------------------------------------------------------

def build_applicability_domain_reference(
    X_train_imputed,
    feature_names,
    ad_k=5,
    ad_distance_percentile=95,
):
    """
    Build applicability-domain reference values from the training set
    used by the selected best model.

    AD is based on standardized RDKit descriptor-space kNN Euclidean distance.

    Parameters
    ----------
    X_train_imputed : array-like, shape (n_train, n_features)
        Training descriptor matrix after imputation.

    feature_names : list
        Feature names corresponding to descriptor columns.

    ad_k : int
        Number of nearest neighbors used for AD distance.

    ad_distance_percentile : float
        Percentile of training kNN mean distances used as AD threshold.

    Returns
    -------
    ad_reference : dict
        Dictionary containing AD reference statistics and thresholds.
    """
    X_train_imputed = np.asarray(X_train_imputed, dtype=float)

    n_train = X_train_imputed.shape[0]

    if n_train < 3:
        raise ValueError(
            "At least 3 training samples are required to build a meaningful AD reference."
        )

    effective_ad_k = min(ad_k, n_train - 1)

    descriptor_mean = np.mean(X_train_imputed, axis=0)
    descriptor_std = np.std(X_train_imputed, axis=0)

    # Prevent division by zero for constant descriptors.
    descriptor_std = np.where(descriptor_std == 0, 1.0, descriptor_std)

    X_train_scaled_for_ad = (X_train_imputed - descriptor_mean) / descriptor_std

    # For training compounds, use k + 1 because the nearest neighbor is the compound itself.
    knn_train = NearestNeighbors(
        n_neighbors=effective_ad_k + 1,
        metric="euclidean",
    )

    knn_train.fit(X_train_scaled_for_ad)

    train_distances, train_neighbor_indices = knn_train.kneighbors(
        X_train_scaled_for_ad
    )

    # Remove self-distance in first column.
    train_knn_mean_distances = train_distances[:, 1:].mean(axis=1)

    ad_distance_threshold = np.percentile(
        train_knn_mean_distances,
        ad_distance_percentile,
    )

    feature_min = np.min(X_train_imputed, axis=0)
    feature_max = np.max(X_train_imputed, axis=0)

    ad_reference = {
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
    }

    return ad_reference


def evaluate_applicability_domain(
    X_query_imputed,
    ad_reference,
    feature_range_warning_threshold=0.05,
):
    """
    Evaluate whether query chemicals are inside or outside the applicability domain.

    Parameters
    ----------
    X_query_imputed : array-like, shape (n_query, n_features)
        Query descriptor matrix after imputation using best-model training means.

    ad_reference : dict
        AD reference dictionary from build_applicability_domain_reference().

    feature_range_warning_threshold : float
        Fraction of features allowed outside training min-max range before issuing
        a descriptor-range warning.

    Returns
    -------
    ad_results : dict
        AD results for each query chemical.
    """
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

    below_min = X_query_imputed < feature_min
    above_max = X_query_imputed > feature_max

    outside_feature_range_matrix = below_min | above_max

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

    ad_results = {
        "query_knn_mean_distances": query_knn_mean_distances,
        "query_neighbor_indices": query_neighbor_indices,
        "inside_distance_ad": inside_distance_ad,
        "n_features_outside_range": n_features_outside_range,
        "fraction_features_outside_range": fraction_features_outside_range,
        "feature_range_warning": feature_range_warning,
        "outside_feature_range_matrix": outside_feature_range_matrix,
        "ad_status": np.array(ad_status, dtype=object),
    }

    return ad_results


def assign_prediction_reliability(
    ad_status,
    max_prediction_probability,
    high_confidence_threshold=0.70,
    moderate_confidence_threshold=0.50,
):
    """
    Assign a simple user-facing reliability category using AD status and
    predicted class probability.

    Reliability rules:
      High:
        Inside AD and max probability >= 0.70

      Moderate:
        Inside AD and 0.50 <= max probability < 0.70

      Low:
        Outside AD, descriptor-range warning, or max probability < 0.50
    """
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
    """
    Convert a boolean outside-range row into a semicolon-separated list of feature names.
    """
    outside_features = [
        feature_name
        for feature_name, is_outside in zip(feature_names, outside_feature_range_row)
        if is_outside
    ]

    if len(outside_features) == 0:
        return ""

    return ";".join(outside_features)


# ------------------------------------------------------------
# 1. Run details
# ------------------------------------------------------------

job_id = os.environ.get("SLURM_JOB_ID", "default_job_id")
print(job_id)

print(
    "gamma_values = [0.001,0.002,0.003,0.004,0.005,0.006,0.0061,0.0062,0.0063,"
    "0.0064,0.0065,0.0066,0.0067,0.0068,0.0069,0.007,0.0071,0.0072,0.0073,"
    "0.0074,0.0075,0.0076,0.0077,0.0078,0.0079,0.008,0.0081,0.0082,0.0083,"
    "0.0084,0.00841,0.00842,0.00843,0.00844,0.00845,0.00846,0.00847,0.00848,"
    "0.00849,0.0085,0.0086,0.0087,0.0088,0.0089,0.009,0.01,0.011,0.012,"
    "0.013,0.014,0.015,0.016,0.0161,0.0162,0.0163,0.0164,0.0165,0.0166,"
    "0.0167,0.0168,0.0169,0.017,0.018,0.019,0.02,0.021,0.022,0.023,0.024,"
    "0.025,0.026,0.027,0.028,0.029,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1], "
    "C_values = [30,31,32,33,33.1,33.2,33.3,33.4,33.5,33.6,33.7,33.8,33.81,"
    "33.82,33.83,33.84,33.85,33.86,33.87,33.88,33.89,33.9,34,35,36,37,38,"
    "39,40,41,41.1,41.2,41.3,41.4,41.5,41.6,41.7,41.8,41.9,42,42.1,42.3,"
    "42.4,42.5,42.6,42.7,42.8,42.9,43,44,45,46,46.1,46.2,46.3,46.4,46.5,"
    "46.51,46.52,46.53,46.54,46.55,46.56,46.57,46.58,46.59,46.6,46.61,"
    "46.62,46.7,46.8,46.9,47,47.1,47.2,47.3,47.4,47.5,47.6,47.7,47.8,"
    "47.9,48,49,50], rdkit, noSMOTE, mean_imputation, correct RepeatedStratifiedKFold, "
    "AD analysis using best selected model"
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

# Applicability-domain settings
AD_K = 5
AD_DISTANCE_PERCENTILE = 95
FEATURE_RANGE_WARNING_THRESHOLD = 0.05

HIGH_CONFIDENCE_THRESHOLD = 0.70
MODERATE_CONFIDENCE_THRESHOLD = 0.50


# ------------------------------------------------------------
# 3. Hyperparameter grid
# ------------------------------------------------------------

gamma_values = [
    0.001, 0.002, 0.003, 0.004, 0.005,
    0.006, 0.0061, 0.0062, 0.0063, 0.0064,
    0.0065, 0.0066, 0.0067, 0.0068, 0.0069,
    0.007, 0.0071, 0.0072, 0.0073, 0.0074,
    0.0075, 0.0076, 0.0077, 0.0078, 0.0079,
    0.008, 0.0081, 0.0082, 0.0083, 0.0084,
    0.00841, 0.00842, 0.00843, 0.00844,
    0.00845, 0.00846, 0.00847, 0.00848,
    0.00849, 0.0085, 0.0086, 0.0087,
    0.0088, 0.0089, 0.009, 0.01, 0.011,
    0.012, 0.013, 0.014, 0.015,
    0.016, 0.0161, 0.0162, 0.0163,
    0.0164, 0.0165, 0.0166, 0.0167,
    0.0168, 0.0169, 0.017, 0.018, 0.019,
    0.02, 0.021, 0.022, 0.023, 0.024,
    0.025, 0.026, 0.027, 0.028, 0.029,
    0.03, 0.04, 0.05, 0.06, 0.07,
    0.08, 0.09, 0.1,
]

C_values = [
    30, 31, 32, 33, 33.1, 33.2, 33.3, 33.4,
    33.5, 33.6, 33.7, 33.8, 33.81, 33.82,
    33.83, 33.84, 33.85, 33.86, 33.87, 33.88,
    33.89, 33.9, 34, 35, 36, 37, 38, 39,
    40, 41, 41.1, 41.2, 41.3, 41.4, 41.5,
    41.6, 41.7, 41.8, 41.9, 42, 42.1, 42.3,
    42.4, 42.5, 42.6, 42.7, 42.8, 42.9, 43,
    44, 45, 46, 46.1, 46.2, 46.3, 46.4,
    46.5, 46.51, 46.52, 46.53, 46.54, 46.55,
    46.56, 46.57, 46.58, 46.59, 46.6, 46.61,
    46.62, 46.7, 46.8, 46.9, 47, 47.1, 47.2,
    47.3, 47.4, 47.5, 47.6, 47.7, 47.8,
    47.9, 48, 49, 50,
]


# ------------------------------------------------------------
# 4. Load dataset
# ------------------------------------------------------------

data = pd.read_csv(file_path)

Class = data["Class"].to_numpy()

features = data.drop(columns=["CAS", "QSAR_READY_SMILES", "mol", "Class"])
feature_names = features.columns.tolist()

X_numpy = features.to_numpy()

concatenated_data_woFP = X_numpy

class_labels = np.unique(Class)

print("Shape of concatenated data:", concatenated_data_woFP.shape)
print("Shape of target labels:", Class.shape)
print("Feature names:", feature_names)
print("Class labels:", class_labels)

X = concatenated_data_woFP
y = Class


# ------------------------------------------------------------
# 5. Containers for all results
# ------------------------------------------------------------

all_fold_metrics = []
all_fold_predictions = []

repeat_metrics_list = []

repeat_best_models = []
repeat_best_imputation_means = []
repeat_best_val_losses = []
repeat_best_hyperparams = []

# Store indices for repeat-best models.
repeat_best_train_indices = []
repeat_best_valid_indices = []
repeat_best_test_indices = []
repeat_best_split_indices = []

# Global best model across all 25 outer evaluations.
global_best_val_loss = np.inf
global_best_model = None
global_best_col_means = None
global_best_hyperparams = None
global_best_repeat = None
global_best_fold = None
global_best_split_idx = None
global_best_train_index = None
global_best_valid_index = None
global_best_test_index = None


# ------------------------------------------------------------
# 6. Create all RepeatedStratifiedKFold splits correctly
# ------------------------------------------------------------

rskf = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE,
)

# IMPORTANT:
# Create the complete list of 25 splits once.
# This prevents accidentally restarting the split generator.
all_outer_splits = list(rskf.split(X, y))

expected_n_splits = N_SPLITS * N_REPEATS

print("Total number of outer splits:", len(all_outer_splits))
print("Expected number of outer splits:", expected_n_splits)

if len(all_outer_splits) != expected_n_splits:
    raise ValueError(
        f"Expected {expected_n_splits} splits, but got {len(all_outer_splits)}."
    )


# ------------------------------------------------------------
# 7. Repeated stratified cross-validation
# ------------------------------------------------------------

fold_counter = 0

for repeat in range(N_REPEATS):
    print("\n============================================================")
    print("repeat:", repeat)
    print("============================================================")

    repeat_training_losses = []
    repeat_training_scores = []
    repeat_validation_losses = []
    repeat_accuracies = []
    repeat_f1_weighted = []

    repeat_precision = []
    repeat_recall = []
    repeat_f1_not_weighted = []

    repeat_avg_precision_list = []
    repeat_avg_recall_list = []
    repeat_avg_f1_not_weighted_list = []

    repeat_predictions = []

    repeat_best_val_loss = np.inf
    repeat_best_model = None
    repeat_best_col_means = None
    repeat_best_hyper = None
    repeat_best_train_index = None
    repeat_best_valid_index = None
    repeat_best_test_index = None
    repeat_best_split_idx = None

    for k in range(N_SPLITS):
        print("\n  Batch:", k)

        split_idx = repeat * N_SPLITS + k

        train_valid_index, test_index = all_outer_splits[split_idx]

        # ----------------------------------------------------
        # Outer split:
        # train_valid_index = approximately 80%
        # test_index        = approximately 20%
        # Both are stratified.
        # ----------------------------------------------------

        X_train_valid = X[train_valid_index]
        y_train_valid = y[train_valid_index]

        # ----------------------------------------------------
        # Inner split:
        # From the outer train_valid set, make train/validation.
        #
        # VALIDATION_SIZE = 0.125
        # 12.5% of 80% = 10% of full dataset
        # 87.5% of 80% = 70% of full dataset
        # ----------------------------------------------------

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

        print("    Training data shape before SMOTE:", train_feature_raw.shape)
        print("    Training labels shape before SMOTE:", train_label.shape)
        print("    Validation data shape:", valid_feature_raw.shape)
        print("    Validation labels shape:", valid_label.shape)
        print("    Test data shape:", test_feature_raw.shape)
        print("    Test labels shape:", test_label.shape)

        print(
            "    Train class counts:",
            dict(zip(*np.unique(train_label, return_counts=True))),
        )
        print(
            "    Validation class counts:",
            dict(zip(*np.unique(valid_label, return_counts=True))),
        )
        print(
            "    Test class counts:",
            dict(zip(*np.unique(test_label, return_counts=True))),
        )

        # ----------------------------------------------------
        # Mean imputation using training-set column means only
        # ----------------------------------------------------

        train_col_means = compute_train_column_means(train_feature_raw)

        train_feature = mean_impute_with_given_column_means(
            train_feature_raw,
            train_col_means,
        )
        valid_feature = mean_impute_with_given_column_means(
            valid_feature_raw,
            train_col_means,
        )
        test_feature = mean_impute_with_given_column_means(
            test_feature_raw,
            train_col_means,
        )

        print("train_feature NAN count after imputation:", np.isnan(train_feature).sum())
        print("valid_feature NAN count after imputation:", np.isnan(valid_feature).sum())
        print("test_feature NAN count after imputation:", np.isnan(test_feature).sum())

        # ----------------------------------------------------
        # Hyperparameter search
        # ----------------------------------------------------

        best_valid_loss = np.inf
        best_model = None
        best_pred = None
        best_g = None
        best_c = None

        for gamma in gamma_values:
            for C in C_values:
                model = SVC(
                    kernel="rbf",
                    gamma=gamma,
                    C=C,
                    probability=True,
                    random_state=RANDOM_STATE + split_idx,
                )

                model.fit(train_feature, train_label)

                try:
                    valid_loss = log_loss(
                        valid_label,
                        model.predict_proba(valid_feature),
                        labels=class_labels,
                    )
                except Exception as e:
                    print("    Error computing validation log_loss:", e)
                    valid_loss = np.inf

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_model = model
                    best_pred = model.predict(test_feature)
                    best_g = gamma
                    best_c = C

        # ----------------------------------------------------
        # Update repeat-best model
        # ----------------------------------------------------

        if best_valid_loss < repeat_best_val_loss:
            repeat_best_val_loss = best_valid_loss
            repeat_best_model = best_model
            repeat_best_col_means = train_col_means
            repeat_best_hyper = (best_g, best_c)
            repeat_best_train_index = train_index
            repeat_best_valid_index = valid_index
            repeat_best_test_index = test_index
            repeat_best_split_idx = split_idx

        # ----------------------------------------------------
        # Update global best model across all 25 outer evaluations
        # ----------------------------------------------------

        if best_valid_loss < global_best_val_loss:
            global_best_val_loss = best_valid_loss
            global_best_model = best_model
            global_best_col_means = train_col_means
            global_best_hyperparams = (best_g, best_c)
            global_best_repeat = repeat
            global_best_fold = k
            global_best_split_idx = split_idx
            global_best_train_index = train_index
            global_best_valid_index = valid_index
            global_best_test_index = test_index

        # ----------------------------------------------------
        # Compute metrics for the current fold
        # ----------------------------------------------------

        if best_model is not None:
            accuracy = accuracy_score(test_label, best_pred)

            precision = precision_score(
                test_label,
                best_pred,
                labels=class_labels,
                average=None,
                zero_division=0,
            )

            recall = recall_score(
                test_label,
                best_pred,
                labels=class_labels,
                average=None,
                zero_division=0,
            )

            f1_not_weighted = f1_score(
                test_label,
                best_pred,
                labels=class_labels,
                average=None,
                zero_division=0,
            )

            f1_weighted = f1_score(
                test_label,
                best_pred,
                labels=class_labels,
                average="weighted",
                zero_division=0,
            )

            training_score = best_model.score(train_feature, train_label)

            training_loss = log_loss(
                train_label,
                best_model.predict_proba(train_feature),
                labels=class_labels,
            )

            avg_precision = np.mean(precision)
            avg_recall = np.mean(recall)
            avg_f1_not_weighted = np.mean(f1_not_weighted)

        else:
            accuracy = None
            precision = None
            recall = None
            f1_not_weighted = None
            f1_weighted = None
            training_score = None
            training_loss = None
            avg_precision = None
            avg_recall = None
            avg_f1_not_weighted = None

        # ----------------------------------------------------
        # Store fold-level results
        # ----------------------------------------------------

        repeat_training_losses.append(training_loss)
        repeat_training_scores.append(training_score)
        repeat_validation_losses.append(best_valid_loss)
        repeat_accuracies.append(accuracy)
        repeat_f1_weighted.append(f1_weighted)

        repeat_precision.append(precision)
        repeat_recall.append(recall)
        repeat_f1_not_weighted.append(f1_not_weighted)

        repeat_predictions.append(best_pred)

        repeat_avg_precision_list.append(avg_precision)
        repeat_avg_recall_list.append(avg_recall)
        repeat_avg_f1_not_weighted_list.append(avg_f1_not_weighted)

        fold_metrics = {
            "repeat": repeat,
            "fold": k,
            "split_idx": split_idx,

            "training_loss": training_loss,
            "training_score": training_score,
            "validation_loss": best_valid_loss,
            "accuracy": accuracy,

            "precision": precision,
            "recall": recall,
            "f1_weighted": f1_weighted,
            "f1_not_weighted": f1_not_weighted,

            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1_not_weighted": avg_f1_not_weighted,

            "best_gamma": best_g,
            "best_c": best_c,

            "train_index": train_index,
            "valid_index": valid_index,
            "test_index": test_index,

            "train_col_means": train_col_means,
        }

        all_fold_metrics.append(fold_metrics)
        all_fold_predictions.append(best_pred)

        print(f"    Fold {k} metrics:")
        print(f"    Best Gamma: {best_g}, Best C: {best_c}")
        print(f"      Training Loss: {training_loss}")
        print(f"      Training Score: {training_score}")
        print(f"      Validation Loss: {best_valid_loss}")
        print(f"      Accuracy: {accuracy}")
        print(f"      Precision per class: {precision}")
        print(f"      Avg Precision: {avg_precision:.4f}")
        print(f"      Recall per class: {recall}")
        print(f"      Avg Recall: {avg_recall:.4f}")
        print(f"      F1 Weighted: {f1_weighted}")
        print(f"      F1 per class: {f1_not_weighted}")
        print(f"      Avg F1: {avg_f1_not_weighted:.4f}")

        fold_counter += 1

    # --------------------------------------------------------
    # Save repeat-best model and hyperparameters
    # --------------------------------------------------------

    repeat_best_models.append(repeat_best_model)
    repeat_best_imputation_means.append(repeat_best_col_means)
    repeat_best_val_losses.append(repeat_best_val_loss)
    repeat_best_hyperparams.append(repeat_best_hyper)

    repeat_best_train_indices.append(repeat_best_train_index)
    repeat_best_valid_indices.append(repeat_best_valid_index)
    repeat_best_test_indices.append(repeat_best_test_index)
    repeat_best_split_indices.append(repeat_best_split_idx)

    print(
        f"\nrepeat {repeat} Best Model Hyperparameters: "
        f"Gamma = {repeat_best_hyper[0]}, C = {repeat_best_hyper[1]}"
    )

    # --------------------------------------------------------
    # Compute repeat-level aggregated metrics
    # --------------------------------------------------------

    repeat_training_losses = np.array(repeat_training_losses)
    repeat_training_scores = np.array(repeat_training_scores)
    repeat_validation_losses = np.array(repeat_validation_losses)
    repeat_accuracies = np.array(repeat_accuracies)
    repeat_f1_weighted = np.array(repeat_f1_weighted)

    repeat_precision_arr = (
        np.vstack(repeat_precision)
        if repeat_precision[0] is not None
        else None
    )

    repeat_recall_arr = (
        np.vstack(repeat_recall)
        if repeat_recall[0] is not None
        else None
    )

    repeat_f1_not_weighted_arr = (
        np.vstack(repeat_f1_not_weighted)
        if repeat_f1_not_weighted[0] is not None
        else None
    )

    repeat_avg_precision_mean = np.mean(repeat_avg_precision_list)
    repeat_avg_precision_std = np.std(repeat_avg_precision_list)

    repeat_avg_recall_mean = np.mean(repeat_avg_recall_list)
    repeat_avg_recall_std = np.std(repeat_avg_recall_list)

    repeat_avg_f1_not_weighted_mean = np.mean(repeat_avg_f1_not_weighted_list)
    repeat_avg_f1_not_weighted_std = np.std(repeat_avg_f1_not_weighted_list)

    print(f"\nrepeat {repeat} Aggregated Metrics:")
    print(
        "  Training Loss - Mean: {:.4f}, Std: {:.4f}".format(
            repeat_training_losses.mean(),
            repeat_training_losses.std(),
        )
    )
    print(
        "  Training Score - Mean: {:.4f}, Std: {:.4f}".format(
            repeat_training_scores.mean(),
            repeat_training_scores.std(),
        )
    )
    print(
        "  Validation Loss - Mean: {:.4f}, Std: {:.4f}".format(
            repeat_validation_losses.mean(),
            repeat_validation_losses.std(),
        )
    )
    print(
        "  Accuracy - Mean: {:.4f}, Std: {:.4f}".format(
            repeat_accuracies.mean(),
            repeat_accuracies.std(),
        )
    )
    print(
        "  F1 Weighted - Mean: {:.4f}, Std: {:.4f}".format(
            repeat_f1_weighted.mean(),
            repeat_f1_weighted.std(),
        )
    )

    if repeat_precision_arr is not None:
        print(
            "  Precision per class - Mean: {}, Std: {}".format(
                np.mean(repeat_precision_arr, axis=0),
                np.std(repeat_precision_arr, axis=0),
            )
        )

    print(
        "  Avg Precision over folds: Mean: {:.4f}, Std: {:.4f}".format(
            repeat_avg_precision_mean,
            repeat_avg_precision_std,
        )
    )

    if repeat_recall_arr is not None:
        print(
            "  Recall per class - Mean: {}, Std: {}".format(
                np.mean(repeat_recall_arr, axis=0),
                np.std(repeat_recall_arr, axis=0),
            )
        )

    print(
        "  Avg Recall over folds: Mean: {:.4f}, Std: {:.4f}".format(
            repeat_avg_recall_mean,
            repeat_avg_recall_std,
        )
    )

    if repeat_f1_not_weighted_arr is not None:
        print(
            "  F1 per class - Mean: {}, Std: {}".format(
                np.mean(repeat_f1_not_weighted_arr, axis=0),
                np.std(repeat_f1_not_weighted_arr, axis=0),
            )
        )

    print(
        "  Avg F1 over folds: Mean: {:.4f}, Std: {:.4f}".format(
            repeat_avg_f1_not_weighted_mean,
            repeat_avg_f1_not_weighted_std,
        )
    )

    repeat_metrics = {
        "training_loss_mean": repeat_training_losses.mean(),
        "training_loss_std": repeat_training_losses.std(),

        "training_score_mean": repeat_training_scores.mean(),
        "training_score_std": repeat_training_scores.std(),

        "validation_loss_mean": repeat_validation_losses.mean(),
        "validation_loss_std": repeat_validation_losses.std(),

        "accuracy_mean": repeat_accuracies.mean(),
        "accuracy_std": repeat_accuracies.std(),

        "f1_weighted_mean": repeat_f1_weighted.mean(),
        "f1_weighted_std": repeat_f1_weighted.std(),

        "precision_mean": (
            np.mean(repeat_precision_arr, axis=0)
            if repeat_precision_arr is not None
            else None
        ),
        "precision_std": (
            np.std(repeat_precision_arr, axis=0)
            if repeat_precision_arr is not None
            else None
        ),

        "recall_mean": (
            np.mean(repeat_recall_arr, axis=0)
            if repeat_recall_arr is not None
            else None
        ),
        "recall_std": (
            np.std(repeat_recall_arr, axis=0)
            if repeat_recall_arr is not None
            else None
        ),

        "f1_not_weighted_mean": (
            np.mean(repeat_f1_not_weighted_arr, axis=0)
            if repeat_f1_not_weighted_arr is not None
            else None
        ),
        "f1_not_weighted_std": (
            np.std(repeat_f1_not_weighted_arr, axis=0)
            if repeat_f1_not_weighted_arr is not None
            else None
        ),

        "avg_precision_mean": repeat_avg_precision_mean,
        "avg_precision_std": repeat_avg_precision_std,

        "avg_recall_mean": repeat_avg_recall_mean,
        "avg_recall_std": repeat_avg_recall_std,

        "avg_f1_not_weighted_mean": repeat_avg_f1_not_weighted_mean,
        "avg_f1_not_weighted_std": repeat_avg_f1_not_weighted_std,

        "predictions": repeat_predictions,
        "class_labels": class_labels,
    }

    repeat_metrics_list.append(repeat_metrics)


# ------------------------------------------------------------
# 8. Save fold metrics and repeat-level metrics
# ------------------------------------------------------------

with open("results_svc_panelb_repeat_svc.pkl", "wb") as f:
    pickle.dump(
        {
            "all_fold_metrics": all_fold_metrics,
            "all_fold_predictions": all_fold_predictions,
            "repeat_metrics_list": repeat_metrics_list,
            "repeat_best_hyperparams": repeat_best_hyperparams,
            "repeat_best_val_losses": repeat_best_val_losses,
            "repeat_best_train_indices": repeat_best_train_indices,
            "repeat_best_valid_indices": repeat_best_valid_indices,
            "repeat_best_test_indices": repeat_best_test_indices,
            "repeat_best_split_indices": repeat_best_split_indices,
            "class_labels": class_labels,
        },
        f,
    )


# ------------------------------------------------------------
# 9. Compute final overall metrics across all 25 models
# ------------------------------------------------------------

all_accuracies = np.array([fm["accuracy"] for fm in all_fold_metrics])
all_f1_weighted = np.array([fm["f1_weighted"] for fm in all_fold_metrics])
all_avg_precision = np.array([fm["avg_precision"] for fm in all_fold_metrics])
all_avg_recall = np.array([fm["avg_recall"] for fm in all_fold_metrics])
all_avg_f1_not_weighted = np.array(
    [fm["avg_f1_not_weighted"] for fm in all_fold_metrics]
)

all_precision_per_class = np.vstack(
    [fm["precision"] for fm in all_fold_metrics]
)

all_recall_per_class = np.vstack(
    [fm["recall"] for fm in all_fold_metrics]
)

all_f1_per_class = np.vstack(
    [fm["f1_not_weighted"] for fm in all_fold_metrics]
)

print("\n25 Fold Accuracy Results:")
print(all_accuracies)

print("\n25 Fold F1 Weighted Results:")
print(all_f1_weighted)

print("\n25 Fold Average Precision Results:")
print(all_avg_precision)

print("\n25 Fold Average Recall Results:")
print(all_avg_recall)

print("\n25 Fold Average F1 Results:")
print(all_avg_f1_not_weighted)

print("\n25 Fold Per-Class Precision Results:")
print(all_precision_per_class)

print("\n25 Fold Per-Class Recall Results:")
print(all_recall_per_class)

print("\n25 Fold Per-Class F1 Results:")
print(all_f1_per_class)

final_metrics = {
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

print("\nFinal Overall Metrics across 25 Models (Mean and Std):")
for metric, value in final_metrics.items():
    print(f"{metric}: {value}")

print("\nFinal Per-Class Metrics across 25 Models (Mean +/- Std):")
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
# 10. Save final metrics
# ------------------------------------------------------------

all_metrics = {
    "accuracy_mean": all_accuracies,
    "f1_weighted_mean": all_f1_weighted,
    "avg_precision_mean": all_avg_precision,
    "avg_recall_mean": all_avg_recall,
    "avg_f1_not_weighted_mean": all_avg_f1_not_weighted,

    "precision_per_class_all_folds": all_precision_per_class,
    "recall_per_class_all_folds": all_recall_per_class,
    "f1_per_class_all_folds": all_f1_per_class,

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

with open("results_svc_panelb_final_metrics_svc.pkl", "wb") as f:
    pickle.dump(all_metrics, f)


# ------------------------------------------------------------
# 11. Select and save the overall best model
# ------------------------------------------------------------

best_overall_model = global_best_model
best_overall_col_means = global_best_col_means
best_overall_hyper = global_best_hyperparams
best_overall_train_index = global_best_train_index
best_overall_valid_index = global_best_valid_index
best_overall_test_index = global_best_test_index

print(
    f"\nBest Overall Model selected from all 25 outer evaluations:"
    f"\n  repeat = {global_best_repeat}"
    f"\n  fold = {global_best_fold}"
    f"\n  split_idx = {global_best_split_idx}"
    f"\n  validation_loss = {global_best_val_loss}"
    f"\n  Gamma = {best_overall_hyper[0]}"
    f"\n  C = {best_overall_hyper[1]}"
)

# Save model and imputation means together.
# New prediction data must first be mean-imputed using train_col_means.
torch.save(
    {
        "model": best_overall_model,
        "train_col_means": best_overall_col_means,
        "feature_names": feature_names,
        "class_labels": class_labels,
        "best_hyperparameters": {
            "gamma": best_overall_hyper[0],
            "C": best_overall_hyper[1],
        },
        "best_model_selection": {
            "repeat": global_best_repeat,
            "fold": global_best_fold,
            "split_idx": global_best_split_idx,
            "validation_loss": global_best_val_loss,
        },
        "train_index": best_overall_train_index,
        "valid_index": best_overall_valid_index,
        "test_index": best_overall_test_index,
    },
    "best_svc_model_panelb.pt",
)

print("Best overall SVC model and imputation means saved as best_svc_model_panelb.pt")


# ------------------------------------------------------------
# 12. Applicability Domain analysis for selected best model
# ------------------------------------------------------------

print("\n============================================================")
print("Applicability Domain Analysis for Selected Best SVC Model")
print("============================================================")

# AD training set = exact training subset used to fit the selected best model.
best_train_feature_raw = X[best_overall_train_index]
best_train_label = y[best_overall_train_index]

best_valid_feature_raw = X[best_overall_valid_index]
best_valid_label = y[best_overall_valid_index]

best_test_feature_raw = X[best_overall_test_index]
best_test_label = y[best_overall_test_index]

best_train_feature = mean_impute_with_given_column_means(
    best_train_feature_raw,
    best_overall_col_means,
)

best_valid_feature = mean_impute_with_given_column_means(
    best_valid_feature_raw,
    best_overall_col_means,
)

best_test_feature = mean_impute_with_given_column_means(
    best_test_feature_raw,
    best_overall_col_means,
)

# Build AD reference from the best model's training set only.
ad_reference = build_applicability_domain_reference(
    X_train_imputed=best_train_feature,
    feature_names=feature_names,
    ad_k=AD_K,
    ad_distance_percentile=AD_DISTANCE_PERCENTILE,
)

print("\nAD method:", ad_reference["ad_method"])
print("Requested AD k:", ad_reference["ad_k_requested"])
print("Effective AD k:", ad_reference["ad_k_effective"])
print("AD distance percentile:", ad_reference["ad_distance_percentile"])
print("AD distance threshold:", ad_reference["ad_distance_threshold"])

print("\nTraining kNN mean distance summary:")
print("  Min:", np.min(ad_reference["training_knn_mean_distances"]))
print("  Mean:", np.mean(ad_reference["training_knn_mean_distances"]))
print("  Median:", np.median(ad_reference["training_knn_mean_distances"]))
print("  Std:", np.std(ad_reference["training_knn_mean_distances"]))
print("  90th percentile:", np.percentile(ad_reference["training_knn_mean_distances"], 90))
print("  95th percentile:", np.percentile(ad_reference["training_knn_mean_distances"], 95))
print("  99th percentile:", np.percentile(ad_reference["training_knn_mean_distances"], 99))
print("  Max:", np.max(ad_reference["training_knn_mean_distances"]))

# Save AD summary values for manuscript reporting.
ad_summary_df = pd.DataFrame(
    {
        "ad_method": [ad_reference["ad_method"]],
        "ad_k_requested": [ad_reference["ad_k_requested"]],
        "ad_k_effective": [ad_reference["ad_k_effective"]],
        "ad_distance_percentile": [ad_reference["ad_distance_percentile"]],
        "ad_distance_threshold": [ad_reference["ad_distance_threshold"]],
        "training_distance_min": [np.min(ad_reference["training_knn_mean_distances"])],
        "training_distance_mean": [np.mean(ad_reference["training_knn_mean_distances"])],
        "training_distance_median": [np.median(ad_reference["training_knn_mean_distances"])],
        "training_distance_std": [np.std(ad_reference["training_knn_mean_distances"])],
        "training_distance_p90": [np.percentile(ad_reference["training_knn_mean_distances"], 90)],
        "training_distance_p95": [np.percentile(ad_reference["training_knn_mean_distances"], 95)],
        "training_distance_p99": [np.percentile(ad_reference["training_knn_mean_distances"], 99)],
        "training_distance_max": [np.max(ad_reference["training_knn_mean_distances"])],
        "feature_range_warning_threshold": [FEATURE_RANGE_WARNING_THRESHOLD],
        "high_confidence_threshold": [HIGH_CONFIDENCE_THRESHOLD],
        "moderate_confidence_threshold": [MODERATE_CONFIDENCE_THRESHOLD],
        "best_model_repeat": [global_best_repeat],
        "best_model_fold": [global_best_fold],
        "best_model_split_idx": [global_best_split_idx],
        "best_model_validation_loss": [global_best_val_loss],
        "best_model_gamma": [best_overall_hyper[0]],
        "best_model_C": [best_overall_hyper[1]],
        "n_ad_training_chemicals": [best_train_feature.shape[0]],
        "n_features": [best_train_feature.shape[1]],
    }
)

ad_summary_df.to_csv("svc_panelb_best_model_AD_summary.csv", index=False)

print("\nSaved AD summary to: svc_panelb_best_model_AD_summary.csv")

# Save min and max feature ranges of the best model's training set.
feature_range_df = pd.DataFrame(
    {
        "feature": feature_names,
        "training_min": ad_reference["feature_min"],
        "training_max": ad_reference["feature_max"],
        "training_mean_for_AD_standardization": ad_reference["descriptor_mean"],
        "training_std_for_AD_standardization": ad_reference["descriptor_std"],
        "training_imputation_mean": best_overall_col_means,
    }
)

feature_range_df.to_csv("svc_panelb_best_model_training_feature_ranges.csv", index=False)

print("Saved training feature min/max ranges to: svc_panelb_best_model_training_feature_ranges.csv")

# Save training-set kNN AD distances.
ad_training_distances_df = pd.DataFrame(
    {
        "original_dataset_index": best_overall_train_index,
        "Class": best_train_label,
        "training_knn_mean_distance": ad_reference["training_knn_mean_distances"],
        "inside_training_AD_threshold": (
            ad_reference["training_knn_mean_distances"]
            <= ad_reference["ad_distance_threshold"]
        ),
    }
)

if "CAS" in data.columns:
    ad_training_distances_df.insert(
        1,
        "CAS",
        data.iloc[best_overall_train_index]["CAS"].to_numpy(),
    )

if "QSAR_READY_SMILES" in data.columns:
    ad_training_distances_df.insert(
        2,
        "QSAR_READY_SMILES",
        data.iloc[best_overall_train_index]["QSAR_READY_SMILES"].to_numpy(),
    )

ad_training_distances_df.to_csv(
    "svc_panelb_best_model_training_AD_distances.csv",
    index=False,
)

print("Saved training AD distances to: svc_panelb_best_model_training_AD_distances.csv")


# ------------------------------------------------------------
# 13. Apply selected best SVC model + AD to its held-out test set
# ------------------------------------------------------------

print("\n============================================================")
print("Best SVC Model Test Set Predictions with AD Flags")
print("============================================================")

# Predict every test chemical.
# AD does not remove any chemical from prediction.
best_test_pred = best_overall_model.predict(best_test_feature)
best_test_proba = best_overall_model.predict_proba(best_test_feature)
best_test_max_proba = np.max(best_test_proba, axis=1)

# Get predicted class probability column names.
proba_column_names = [
    f"prob_class_{class_label}"
    for class_label in best_overall_model.classes_
]

# Evaluate AD for every test chemical.
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
        "original_dataset_index": best_overall_test_index,
        "true_class": best_test_label,
        "predicted_class": best_test_pred,
        "max_prediction_probability": best_test_max_proba,
        "ad_status": best_test_ad_results["ad_status"],
        "prediction_reliability": best_test_reliability,
        "knn_mean_distance": best_test_ad_results["query_knn_mean_distances"],
        "ad_distance_threshold": ad_reference["ad_distance_threshold"],
        "inside_distance_ad": best_test_ad_results["inside_distance_ad"],
        "n_features_outside_training_range": best_test_ad_results["n_features_outside_range"],
        "fraction_features_outside_training_range": best_test_ad_results["fraction_features_outside_range"],
        "feature_range_warning": best_test_ad_results["feature_range_warning"],
    }
)

if "CAS" in data.columns:
    best_test_results_df.insert(
        1,
        "CAS",
        data.iloc[best_overall_test_index]["CAS"].to_numpy(),
    )

if "QSAR_READY_SMILES" in data.columns:
    insert_position = 2 if "CAS" in best_test_results_df.columns else 1
    best_test_results_df.insert(
        insert_position,
        "QSAR_READY_SMILES",
        data.iloc[best_overall_test_index]["QSAR_READY_SMILES"].to_numpy(),
    )

# Add class probabilities.
proba_df = pd.DataFrame(
    best_test_proba,
    columns=proba_column_names,
)

best_test_results_df = pd.concat(
    [
        best_test_results_df.reset_index(drop=True),
        proba_df.reset_index(drop=True),
    ],
    axis=1,
)

# Add list of descriptors outside training range.
outside_range_feature_names = []

for row_idx in range(best_test_ad_results["outside_feature_range_matrix"].shape[0]):
    outside_range_feature_names.append(
        get_outside_range_feature_names(
            best_test_ad_results["outside_feature_range_matrix"][row_idx],
            feature_names,
        )
    )

best_test_results_df["features_outside_training_range"] = outside_range_feature_names

best_test_results_df.to_csv(
    "svc_panelb_best_model_test_predictions_with_AD.csv",
    index=False,
)

print("Saved best-model test predictions with AD to: svc_panelb_best_model_test_predictions_with_AD.csv")

print("\nBest-model held-out test set AD summary:")
print(best_test_results_df["ad_status"].value_counts(dropna=False))
print("\nBest-model held-out test set reliability summary:")
print(best_test_results_df["prediction_reliability"].value_counts(dropna=False))

n_test = best_test_results_df.shape[0]
n_inside_distance_ad = int(best_test_results_df["inside_distance_ad"].sum())
n_outside_distance_ad = n_test - n_inside_distance_ad
n_feature_range_warning = int(best_test_results_df["feature_range_warning"].sum())

print(f"\nNumber of best-model test chemicals: {n_test}")
print(f"Inside distance-based AD: {n_inside_distance_ad}")
print(f"Outside distance-based AD: {n_outside_distance_ad}")
print(f"Descriptor-range warnings: {n_feature_range_warning}")

# Save compact test AD summary.
test_ad_summary_df = pd.DataFrame(
    {
        "n_test_chemicals": [n_test],
        "n_inside_distance_ad": [n_inside_distance_ad],
        "n_outside_distance_ad": [n_outside_distance_ad],
        "percent_inside_distance_ad": [100 * n_inside_distance_ad / n_test],
        "percent_outside_distance_ad": [100 * n_outside_distance_ad / n_test],
        "n_feature_range_warning": [n_feature_range_warning],
        "percent_feature_range_warning": [100 * n_feature_range_warning / n_test],
        "n_high_reliability": [
            int((best_test_results_df["prediction_reliability"] == "High").sum())
        ],
        "n_moderate_reliability": [
            int((best_test_results_df["prediction_reliability"] == "Moderate").sum())
        ],
        "n_low_reliability": [
            int((best_test_results_df["prediction_reliability"] == "Low").sum())
        ],
    }
)

test_ad_summary_df.to_csv(
    "svc_panelb_best_model_test_AD_summary.csv",
    index=False,
)

print("Saved best-model test AD summary to: svc_panelb_best_model_test_AD_summary.csv")


# ------------------------------------------------------------
# 14. Save final best SVC model with AD reference
# ------------------------------------------------------------

torch.save(
    {
        "model": best_overall_model,
        "train_col_means": best_overall_col_means,
        "feature_names": feature_names,
        "class_labels": class_labels,
        "best_hyperparameters": {
            "gamma": best_overall_hyper[0],
            "C": best_overall_hyper[1],
        },
        "best_model_selection": {
            "repeat": global_best_repeat,
            "fold": global_best_fold,
            "split_idx": global_best_split_idx,
            "validation_loss": global_best_val_loss,
        },
        "train_index": best_overall_train_index,
        "valid_index": best_overall_valid_index,
        "test_index": best_overall_test_index,

        # Applicability-domain reference
        "applicability_domain": ad_reference,
        "feature_range_warning_threshold": FEATURE_RANGE_WARNING_THRESHOLD,
        "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
        "moderate_confidence_threshold": MODERATE_CONFIDENCE_THRESHOLD,
    },
    "best_svc_model_panelb_with_AD.pt",
)

print("Best overall SVC model with AD saved as best_svc_model_panelb_with_AD.pt")


# ------------------------------------------------------------
# 15. Execution time
# ------------------------------------------------------------

end_time = time.time()
execution_time = (end_time - start_time) / 60
print(f"Total execution time: {execution_time:.2f} minutes")