# Model Training for Panel D: Logistic Regression with MACCS molecular fingerprint
import pandas as pd
import numpy as np
from utility import Kfold
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, log_loss
import os
import time
import pickle
import torch  # For saving model with torch.save

job_id = os.environ.get('SLURM_JOB_ID', 'default_job_id')
print(job_id)


# Measure start time
start_time = time.time()
print(start_time)

# Load the csv file. Refer to RDKit Data Extraction/Generate_RDKit_Features.ipynb for details on how this dataset was fetched.
file_path = 'bcf_data.csv'
data = pd.read_csv(file_path)

# Convert SMILES column to NumPy array
SMILES = data['SMILES'].to_numpy()

# Generate MACCS fingerprints for all molecules
print('Start generating MACCS fingerprints')

FP = []
ONS_index = []  # Indices where the molecule contains N, O, or S

for i, sm in enumerate(SMILES):
    if 'N' in sm:  # or 'O' in sm or 'S' in sm:
        ONS_index.append(i)
    
    mol = Chem.MolFromSmiles(sm)
    
    if mol is not None:  # Ensure valid molecule
        fp = MACCSkeys.GenMACCSKeys(mol)  # Generate MACCS fingerprint
        FP.append(fp)

# Convert list of fingerprints to a NumPy array
FP = np.array([list(fp) for fp in FP])
FP = FP[:, 1:]  # Remove first column (bit 0)

print('Finished generating MACCS fingerprints')

#target variable
Class = data['Class'].to_numpy()

concatenated_data_woFP = FP

print("Shape of concatenated data:", concatenated_data_woFP.shape)
print("Shape of target labels:", Class.shape)


n_sample = len(concatenated_data_woFP)
total_id = np.arange(n_sample)

# Lists to store metrics and predictions for all 25 models (5 repeats x 5 folds)
all_fold_metrics = []      # each element is a dict of metrics from one fold
all_fold_predictions = []  # raw predictions (as arrays) from each fold

# Also store repeat-level metrics (means and stds per repeat) in a list
repeat_metrics_list = []
# To store the best model from each repeat (lowest validation loss among the 5 folds)
repeat_best_models = []
repeat_best_val_losses = []
repeat_best_hyperparams = []  # To store best_c and best_ratio per repeat

# Loop over repeats (for stability, 5 repeats → 25 models total)
for repeat in range(5):
    print('repeat:', repeat)
    np.random.shuffle(total_id)
    train_split_index, test_split_index = Kfold(len(concatenated_data_woFP), 5)
    splits = 5

    # Lists to store metrics per fold in this repeat
    repeat_training_losses = []
    repeat_training_scores = []
    repeat_validation_losses = []
    repeat_accuracies = []
    repeat_f1_weighted = []
    repeat_precision = []       # list of arrays (per fold)
    repeat_recall = []          # list of arrays (per fold)
    repeat_f1_not_weighted = [] # list of arrays (per fold)

    # Lists for the averaged metrics (across classes) per fold
    repeat_avg_precision_list = []
    repeat_avg_recall_list = []
    repeat_avg_f1_not_weighted_list = []

    # Also store predictions for each fold in the repeat
    repeat_predictions = []

    # Variables to track the best model for this repeat
    repeat_best_val_loss = np.inf
    repeat_best_model = None
    repeat_best_hyper = None
    patience = 6
    patience_counter = 0

    for k in range(splits):
        print('  Batch:', k)
        # Split indices for train, validation, and test
        train_index = train_split_index[k][:int(len(train_split_index[k]) * 0.875)]
        valid_index = train_split_index[k][int(len(train_split_index[k]) * 0.875):]
        test_index = test_split_index[k]
        
        # Map IDs to features/labels
        train_id = [total_id[i] for i in train_index]
        valid_id = [total_id[i] for i in valid_index]
        test_id = [total_id[i] for i in test_index]
        
        train_feature = np.array([concatenated_data_woFP[i] for i in train_id])
        train_label = np.array([Class[i] for i in train_id])
        print("    Training data shape before SMOTE:", train_feature.shape)
        print("    Training labels shape before SMOTE:", train_label.shape)
        
        valid_feature = np.array([concatenated_data_woFP[i] for i in valid_id])
        valid_label = np.array([Class[i] for i in valid_id])
        test_feature = np.array([concatenated_data_woFP[i] for i in test_id])
        test_label = np.array([Class[i] for i in test_id])
        
        # Define hyperparameters to tune
        C_pool = [0.1, 1.0, 10.0]
        l1l2_ratio = [0.25, 0.5, 0.75]

        best_valid_loss = np.inf
        best_model = None
        best_pred = None
        best_c = None
        best_r = None
        # Iterate over each combination of C and penalty
        for ratio in l1l2_ratio:        
            for c in C_pool:
                model = LogisticRegression(C=c, penalty='elasticnet', l1_ratio=ratio, 
                                        solver='saga', class_weight='balanced', 
                                        multi_class='multinomial', max_iter=100)
                model.fit(train_feature, train_label)
                training_score = model.score(train_feature, train_label)

                # Compute training log loss
                train_loss = log_loss(train_label, model.predict_proba(train_feature)) 
                                
                try:
                    valid_loss = log_loss(valid_label, model.predict_proba(valid_feature))
                except Exception as e:
                    print("    Error computing log_loss:", e)
                    valid_loss = np.inf
                
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    patience_counter = 0
                    best_model = model
                    best_pred = model.predict(test_feature)
                    test_score = model.score(test_feature, test_label)
                    best_c = c
                    best_r = ratio
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at batch {k}")
                    break
        
        # Update repeat-best model if current fold's valid loss is lower
        if best_valid_loss < repeat_best_val_loss:
            repeat_best_val_loss = best_valid_loss
            repeat_best_model = best_model
            repeat_best_hyper = (best_c, best_r)
        
        # Compute metrics for the current fold
        if best_model is not None:
            accuracy = accuracy_score(test_label, best_pred)
            precision = precision_score(test_label, best_pred, average=None)
            recall = recall_score(test_label, best_pred, average=None)
            f1_not_weighted = f1_score(test_label, best_pred, average=None)
            f1_weighted = f1_score(test_label, best_pred, average='weighted')
            training_score = best_model.score(train_feature, train_label)
            training_loss = log_loss(train_label, best_model.predict_proba(train_feature)) 

            # Compute average metrics over classes for this fold
            avg_precision = np.mean(precision) if precision is not None else None
            avg_recall = np.mean(recall) if recall is not None else None
            avg_f1_not_weighted = np.mean(f1_not_weighted) if f1_not_weighted is not None else None
            
        else:
            accuracy = precision = recall = f1_not_weighted = f1_weighted = training_score = training_loss = None
            avg_precision = avg_recall = avg_f1_not_weighted = None

        # Append fold-level metrics to repeat lists
        repeat_training_losses.append(training_loss)
        repeat_training_scores.append(training_score)
        repeat_validation_losses.append(best_valid_loss)
        repeat_accuracies.append(accuracy)
        repeat_f1_weighted.append(f1_weighted)
        repeat_precision.append(precision)
        repeat_recall.append(recall)
        repeat_f1_not_weighted.append(f1_not_weighted)
        repeat_predictions.append(best_pred)
        # Append the averaged metrics for this fold
        repeat_avg_precision_list.append(avg_precision)
        repeat_avg_recall_list.append(avg_recall)
        repeat_avg_f1_not_weighted_list.append(avg_f1_not_weighted)

        # Also store these fold metrics in our overall list (for 25 models)
        fold_metrics = {
            'training_loss': training_loss,
            'training_score': training_score,
            'validation_loss': best_valid_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_weighted': f1_weighted,
            'f1_not_weighted': f1_not_weighted,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1_not_weighted': avg_f1_not_weighted,
            'best_c': best_c,
            'best_ratio': best_r
        }
        all_fold_metrics.append(fold_metrics)
        all_fold_predictions.append(best_pred)
        
        print(f"    Fold {k} metrics:")
        # Print the best hyperparameters for this batch
        print(f"    Best C: {best_c}, Best l1l2_Ratio: {best_r}")
        print(f"      Training Loss: {training_loss}, Training Score: {training_score}")
        print(f"      Validation Loss: {best_valid_loss}")
        print(f"      Accuracy: {accuracy}")
        print(f"      Precision: {precision}")
        print(f"      Avg Precision: {avg_precision:.4f}")
        print(f"      Recall: {recall}")
        print(f"      Avg Recall: {avg_recall:.4f}")
        print(f"      F1 Weighted: {f1_weighted}")
        print(f"      F1 (Not Weighted): {f1_not_weighted}")
        print(f"      Avg F1 (Not Weighted): {avg_f1_not_weighted:.4f}")

    # Save the best model of the current repeat along with its hyperparameters
    repeat_best_models.append(repeat_best_model)
    repeat_best_val_losses.append(repeat_best_val_loss)
    repeat_best_hyperparams.append(repeat_best_hyper)
    
    print(f"repeat {repeat} Best Model Hyperparameters: C = {repeat_best_hyper[0]}, l1l2_Ratio = {repeat_best_hyper[1]}")
    
    # Compute the repeat-level averaged metrics (over the 5 folds)
    repeat_avg_precision_mean = np.mean(repeat_avg_precision_list)
    repeat_avg_precision_std = np.std(repeat_avg_precision_list)
    repeat_avg_recall_mean = np.mean(repeat_avg_recall_list)
    repeat_avg_recall_std = np.std(repeat_avg_recall_list)
    repeat_avg_f1_not_weighted_mean = np.mean(repeat_avg_f1_not_weighted_list)
    repeat_avg_f1_not_weighted_std = np.std(repeat_avg_f1_not_weighted_list)

    # Convert repeat lists to numpy arrays for computing stats
    repeat_training_losses = np.array(repeat_training_losses)
    repeat_training_scores = np.array(repeat_training_scores)
    repeat_validation_losses = np.array(repeat_validation_losses)
    repeat_accuracies = np.array(repeat_accuracies)
    repeat_f1_weighted = np.array(repeat_f1_weighted)
    
    # For precision, recall, and f1 (not weighted) stack the arrays (each is per fold)
    repeat_precision_arr = np.vstack(repeat_precision) if repeat_precision[0] is not None else None
    repeat_recall_arr = np.vstack(repeat_recall) if repeat_recall[0] is not None else None
    repeat_f1_not_weighted_arr = np.vstack(repeat_f1_not_weighted) if repeat_f1_not_weighted[0] is not None else None
    
    print(f"repeat {repeat} Aggregated Metrics:")
    print("  Training Loss - Mean: {:.4f}, Std: {:.4f}".format(repeat_training_losses.mean(), repeat_training_losses.std()))
    print("  Training Score - Mean: {:.4f}, Std: {:.4f}".format(repeat_training_scores.mean(), repeat_training_scores.std()))
    print("  Validation Loss - Mean: {:.4f}, Std: {:.4f}".format(repeat_validation_losses.mean(), repeat_validation_losses.std()))
    print("  Accuracy - Mean: {:.4f}, Std: {:.4f}".format(repeat_accuracies.mean(), repeat_accuracies.std()))
    print("  F1 Weighted - Mean: {:.4f}, Std: {:.4f}".format(repeat_f1_weighted.mean(), repeat_f1_weighted.std()))
    if repeat_precision_arr is not None:
        print("  Precision - Mean: {}, Std: {}".format(np.mean(repeat_precision_arr, axis=0), np.std(repeat_precision_arr, axis=0)))
    
    print("  Avg Precision over folds: Mean: {:.4f}, Std: {:.4f}".format(repeat_avg_precision_mean, repeat_avg_precision_std))
    
    if repeat_recall_arr is not None:
        print("  Recall - Mean: {}, Std: {}".format(np.mean(repeat_recall_arr, axis=0), np.std(repeat_recall_arr, axis=0)))
    
    print("  Avg Recall over folds: Mean: {:.4f}, Std: {:.4f}".format(repeat_avg_recall_mean, repeat_avg_recall_std))

    if repeat_f1_not_weighted_arr is not None:
        print("  F1 (Not Weighted) - Mean: {}, Std: {}".format(np.mean(repeat_f1_not_weighted_arr, axis=0), np.std(repeat_f1_not_weighted_arr, axis=0)))
    
    print("  Avg F1 (Not Weighted) over folds: Mean: {:.4f}, Std: {:.4f}".format(repeat_avg_f1_not_weighted_mean, repeat_avg_f1_not_weighted_std))

    # Store aggregated metrics for this repeat
    repeat_metrics = {
        'training_loss_mean': repeat_training_losses.mean(),
        'training_loss_std': repeat_training_losses.std(),
        'training_score_mean': repeat_training_scores.mean(),
        'training_score_std': repeat_training_scores.std(),
        'validation_loss_mean': repeat_validation_losses.mean(),
        'validation_loss_std': repeat_validation_losses.std(),
        'accuracy_mean': repeat_accuracies.mean(),
        'accuracy_std': repeat_accuracies.std(),
        'f1_weighted_mean': repeat_f1_weighted.mean(),
        'f1_weighted_std': repeat_f1_weighted.std(),
        'precision_mean': np.mean(repeat_precision_arr, axis=0) if repeat_precision_arr is not None else None,
        'precision_std': np.std(repeat_precision_arr, axis=0) if repeat_precision_arr is not None else None,
        'recall_mean': np.mean(repeat_recall_arr, axis=0) if repeat_recall_arr is not None else None,
        'recall_std': np.std(repeat_recall_arr, axis=0) if repeat_recall_arr is not None else None,
        'f1_not_weighted_mean': np.mean(repeat_f1_not_weighted_arr, axis=0) if repeat_f1_not_weighted_arr is not None else None,
        'f1_not_weighted_std': np.std(repeat_f1_not_weighted_arr, axis=0) if repeat_f1_not_weighted_arr is not None else None,
        'avg_precision_mean': repeat_avg_precision_mean,
        'avg_precision_std': repeat_avg_precision_std,
        'avg_recall_mean': repeat_avg_recall_mean,
        'avg_recall_std': repeat_avg_recall_std,
        'avg_f1_not_weighted_mean': repeat_avg_f1_not_weighted_mean,
        'avg_f1_not_weighted_std': repeat_avg_f1_not_weighted_std,
        'predictions': repeat_predictions  # list of predictions (one per fold)
    }
    repeat_metrics_list.append(repeat_metrics)

# Save all fold (25 models) metrics and predictions as well as repeat-level metrics
with open('results_lr_t2panelb_repeat.pkl', 'wb') as f:
    pickle.dump({
        'all_fold_metrics': all_fold_metrics,
        'all_fold_predictions': all_fold_predictions,
        'repeat_metrics_list': repeat_metrics_list,
        'repeat_best_hyperparams': repeat_best_hyperparams  # best hyperparameters per repeat
    }, f)

# ----------------------------------------
# Compute final overall metrics across all 25 models

all_accuracies = np.array([fm['accuracy'] for fm in all_fold_metrics])
all_f1_weighted = np.array([fm['f1_weighted'] for fm in all_fold_metrics])
all_avg_precision = np.array([fm['avg_precision'] for fm in all_fold_metrics])
all_avg_recall = np.array([fm['avg_recall'] for fm in all_fold_metrics])
all_avg_f1_not_weighted = np.array([fm['avg_f1_not_weighted'] for fm in all_fold_metrics])

print("25 Fold Accuracy Results:")
print(all_accuracies)

print("25 Fold F1 Weighted Results:")
print(all_f1_weighted)

print("25 Fold Average Precision Results:")
print(all_avg_precision)

print("25 Fold Average Recall Results:")
print(all_avg_recall)

print("25 Fold Average F1 (Not Weighted) Results:")
print(all_avg_f1_not_weighted)

# Compute overall mean and standard deviation
final_metrics = {
    'accuracy_mean': np.mean(all_accuracies),
    'accuracy_std': np.std(all_accuracies),
    'f1_weighted_mean': np.mean(all_f1_weighted),
    'f1_weighted_std': np.std(all_f1_weighted),
    'avg_precision_mean': np.mean(all_avg_precision),
    'avg_precision_std': np.std(all_avg_precision),
    'avg_recall_mean': np.mean(all_avg_recall),
    'avg_recall_std': np.std(all_avg_recall),
    'avg_f1_not_weighted_mean': np.mean(all_avg_f1_not_weighted),
    'avg_f1_not_weighted_std': np.std(all_avg_f1_not_weighted)
}

print('\nFinal Overall Metrics across 25 Models (Mean and Std):')
for metric, value in final_metrics.items():
    print(f'{metric}: {value}')

all_metrics = {
    'accuracy_mean': all_accuracies,
    'f1_weighted_mean': all_f1_weighted,
    'avg_precision_mean': all_avg_precision,
    'avg_recall_mean': all_avg_recall,
    'avg_f1_not_weighted_mean': all_avg_f1_not_weighted,
}

with open('results_lr_t2panelb_final_metrics.pkl', 'wb') as f:
    pickle.dump(all_metrics, f)

# ----------------------------------------
# Select the overall best model from the 5 repeat-best models
# Here we choose the one with the lowest validation loss.
best_repeat_index = np.argmin(repeat_best_val_losses)
best_overall_model = repeat_best_models[best_repeat_index]
best_repeat_hyper = repeat_best_hyperparams[best_repeat_index]
print(f"Best Overall Model from repeat {best_repeat_index}: C = {best_repeat_hyper[0]}, l1l2_ratio = {best_repeat_hyper[1]}")
# Save the best overall model as a .pt file using torch.save
torch.save(best_overall_model, 'best_lr_model.pt')
print("Best overall lr model saved as best_lr_model.pt")
# ----------------------------------------

end_time = time.time()
execution_time = (end_time - start_time) / 60
print(f"Total execution time: {execution_time:.2f} minutes")