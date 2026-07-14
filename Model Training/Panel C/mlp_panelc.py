# Model Training for Panel C: Multi-layer Perceptron (MLP) with ECFP molecular fingerprint
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, log_loss
import os
import time
import pickle
import torch


# ============================================================
# ECFP + MLP Panel C Classification
# with Correct RepeatedStratifiedKFold
#
# Dataset:
#   bcf_data.csv
#
# Features:
#   ECFP/Morgan fingerprints
#   radius = 2
#   fpSize = 2048
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
# No SMOTE
# ============================================================


# ------------------------------------------------------------
# 1. Run details
# ------------------------------------------------------------

job_id = os.environ.get("SLURM_JOB_ID", "default_job_id")
print(job_id)

print(
    "hidden_layer_sizes_values = [(50,), (100,), (200,), (50, 50), (100, 100)], "
    "alpha_values = [0.0001, 0.001, 0.01, 0.1], "
    "learning_rate_init_values = [0.001, 0.01, 0.05, 0.1], "
    "learning_rate_values = ['constant', 'invscaling', 'adaptive'], "
    "mlp_t2_panelc_nosmote, correct RepeatedStratifiedKFold"
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

hidden_layer_sizes_values = [
    (50,),
    (100,),
    (200,),
    (50, 50),
    (100, 100),
]

alpha_values = [
    0.0001,
    0.001,
    0.01,
    0.1,
]

learning_rate_init_values = [
    0.001,
    0.01,
    0.05,
    0.1,
]

learning_rate_values = [
    "constant",
    "invscaling",
    "adaptive",
]

file_path = "/home/ssd6515/Fish/bcf_data.csv"


# ------------------------------------------------------------
# 3. Load dataset
# ------------------------------------------------------------

data = pd.read_csv(file_path)

SMILES = data["SMILES"].to_numpy()
Class = data["Class"].to_numpy()

class_labels = np.unique(Class)


# ------------------------------------------------------------
# 4. Generate ECFP fingerprints
# ------------------------------------------------------------

print("start generating ECFP")

FP = []
ONS_index = []

morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)

for i, sm in enumerate(SMILES):
    if isinstance(sm, str) and "N" in sm:
        ONS_index.append(i)

    mol = Chem.MolFromSmiles(sm)

    if mol is None:
        raise ValueError(f"Invalid SMILES at row {i}: {sm}")

    fp = morgan_gen.GetFingerprint(mol)

    # Convert RDKit ExplicitBitVect to numpy array of 0/1 values.
    FP.append(np.array(fp, dtype=int))

FP = np.array(FP)

print("finish generating ECFP")


# ------------------------------------------------------------
# 5. Define feature matrix and target vector
# ------------------------------------------------------------

concatenated_data_woFP = FP

print("Shape of concatenated data:", concatenated_data_woFP.shape)
print("Shape of target labels:", Class.shape)
print("Class labels:", class_labels)

X = concatenated_data_woFP
y = Class


# ------------------------------------------------------------
# 6. Containers for all results
# ------------------------------------------------------------

all_fold_metrics = []
all_fold_predictions = []

repeat_metrics_list = []

repeat_best_models = []
repeat_best_val_losses = []
repeat_best_hyperparams = []


# ------------------------------------------------------------
# 7. Create all RepeatedStratifiedKFold splits correctly
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
# 8. Repeated stratified cross-validation
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
    repeat_best_hyper = None

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

        train_feature = X[train_index]
        train_label = y[train_index]

        valid_feature = X[valid_index]
        valid_label = y[valid_index]

        test_feature = X[test_index]
        test_label = y[test_index]

        print("    Training data shape before SMOTE:", train_feature.shape)
        print("    Training labels shape before SMOTE:", train_label.shape)
        print("    Validation data shape:", valid_feature.shape)
        print("    Validation labels shape:", valid_label.shape)
        print("    Test data shape:", test_feature.shape)
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
        # Hyperparameter search
        # ----------------------------------------------------

        best_valid_loss = np.inf
        best_model = None
        best_pred = None

        best_hls = None
        best_alpha = None
        best_lr_init = None
        best_lr = None

        for hls in hidden_layer_sizes_values:
            for alpha in alpha_values:
                for lr_init in learning_rate_init_values:
                    for lr in learning_rate_values:
                        model = MLPClassifier(
                            hidden_layer_sizes=hls,
                            alpha=alpha,
                            learning_rate_init=lr_init,
                            learning_rate=lr,
                            max_iter=1000,
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

                            best_hls = hls
                            best_alpha = alpha
                            best_lr_init = lr_init
                            best_lr = lr

        # ----------------------------------------------------
        # Update repeat-best model
        # ----------------------------------------------------

        if best_valid_loss < repeat_best_val_loss:
            repeat_best_val_loss = best_valid_loss
            repeat_best_model = best_model
            repeat_best_hyper = (
                best_hls,
                best_alpha,
                best_lr_init,
                best_lr,
            )

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

            "best_hidden_layer_sizes": best_hls,
            "best_alpha": best_alpha,
            "best_learning_rate_init": best_lr_init,
            "best_learning_rate": best_lr,

            "train_index": train_index,
            "valid_index": valid_index,
            "test_index": test_index,
        }

        all_fold_metrics.append(fold_metrics)
        all_fold_predictions.append(best_pred)

        print(f"    Fold {k} metrics:")
        print(
            "    Best hyperparameters: "
            f"hidden_layer_sizes: {best_hls}, "
            f"alpha: {best_alpha}, "
            f"learning_rate_init: {best_lr_init}, "
            f"learning_rate: {best_lr}"
        )
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
    repeat_best_val_losses.append(repeat_best_val_loss)
    repeat_best_hyperparams.append(repeat_best_hyper)

    print(
        f"\nrepeat {repeat} Best Model Hyperparameters: "
        f"hidden_layer_sizes = {repeat_best_hyper[0]}, "
        f"alpha = {repeat_best_hyper[1]}, "
        f"learning_rate_init = {repeat_best_hyper[2]}, "
        f"learning_rate = {repeat_best_hyper[3]}"
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
# 9. Save fold metrics, predictions, and repeat-level metrics
# ------------------------------------------------------------

with open("results_mlp_t2panela_repeat.pkl", "wb") as f:
    pickle.dump(
        {
            "all_fold_metrics": all_fold_metrics,
            "all_fold_predictions": all_fold_predictions,
            "repeat_metrics_list": repeat_metrics_list,
            "repeat_best_hyperparams": repeat_best_hyperparams,
            "class_labels": class_labels,
            "ONS_index": ONS_index,
        },
        f,
    )


# ------------------------------------------------------------
# 10. Compute final overall metrics across all 25 models
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
# 11. Save final metrics
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

with open("results_mlp_t2panela_final_metrics.pkl", "wb") as f:
    pickle.dump(all_metrics, f)


# ------------------------------------------------------------
# 12. Select the overall best model
# ------------------------------------------------------------

best_repeat_index = np.argmin(repeat_best_val_losses)

best_overall_model = repeat_best_models[best_repeat_index]
best_repeat_hyper = repeat_best_hyperparams[best_repeat_index]

print(
    f"\nBest Overall Model from repeat {best_repeat_index}: "
    f"hidden_layer_sizes = {best_repeat_hyper[0]}, "
    f"alpha = {best_repeat_hyper[1]}, "
    f"learning_rate_init = {best_repeat_hyper[2]}, "
    f"learning_rate = {best_repeat_hyper[3]}"
)

# To save the best overall model, uncomment this:
# torch.save(best_overall_model, "best_mlp_model_t2panela.pt")
# print("Best overall MLP model saved as best_mlp_model_t2panela.pt")


# ------------------------------------------------------------
# 13. Execution time
# ------------------------------------------------------------

end_time = time.time()
execution_time = (end_time - start_time) / 60
print(f"Total execution time: {execution_time:.2f} minutes")