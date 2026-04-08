# ChemClassifier

ChemClassifier is a machine learning workflow for chemical bioconcentration classification. The repository includes:

- data preparation code for generating RDKit descriptors from a source chemical dataset
- model training scripts for four feature panels
- a feature-importance notebook for interpreting descriptor-based models

The code is organized so you can move from environment setup to training runs and result files with minimal manual changes.

## Project Structure

```text
ChemClassifier/
|-- Environment/
|   `-- environment.yml
|-- RDKit Data Exraction/
|   |-- Generate_RDKit_Features.ipynb
|   `-- rdkit_data.csv
|-- Model Training/
|   |-- utility.py
|   |-- Panel A/
|   |-- Panel B/
|   |-- Panel C/
|   `-- Panel D/
|-- Feature Importance/
|   |-- FeatureImportance.ipynb
|   |-- all_feature_importances_panela.pkl
|   `-- all_feature_importances_panelb.pkl
`-- README.md
```

## What Each Panel Represents

- `Panel A`: Model Training for Dragon descriptor subset from `bcf_data.csv`
- `Panel B`: Model Training for RDKit descriptor from `rdkit_data.csv`
- `Panel C`: Model Training for ECFP (Morgan fingerprint) generated from SMILES in `bcf_data.csv`
- `Panel D`: Model Training for MACCS fingerprint generated from SMILES in `bcf_data.csv`

## Setup

### 1. Create the Conda environment

The project environment is stored in [Environment/environment.yml](Environment/environment.yml).

```powershell
conda env create -f .\Environment\environment.yml
conda activate toxvenv
```

This environment includes the main dependencies used by the project, including:

- Python 3.12
- RDKit
- scikit-learn
- pandas and numpy
- PyTorch
- shap
- Jupyter

## Data Preparation

### Input files used by the project

There are two main training inputs:

- `bcf_data.csv`: Obtained from Grisoni, F., Consonni, V., Vighi, M., Villa, S., Todeschini, R., 2016. Investigating the mechanisms of bioconcentration through QSAR classification trees. Environ Int 88, 198-205. https://doi.org/10.1016/j.envint.2015.12.024.
- `rdkit_data.csv`: Refer to [RDKit Data Exraction/Generate_RDKit_Features.ipynb](https://github.com/ssd6515/ChemClassifier/blob/04a5890030ec530ccc8bbb63a3fb868826b6cef9/RDKit%20Data%20Exraction/Generate_RDKit_Features.ipynb) for details on how this dataset was generated. This dataset is provided at [RDKit Data Extraction/rdkit_data.csv](https://github.com/ssd6515/ChemClassifier/blob/930873a20aa58e59a4fe97e2f4701d74c6e7fb5e/RDKit%20Data%20Exraction/rdkit_data.csv)

## Model Training

### Panel A

- `gbrt_panela.py`: Gradient Boosting Decision Trees (GBDT) with Dragon descriptors
- `lr_panela.py`: Logistic Regression with Dragon descriptors
- `rf_panela.py`: Random Forest (RF) with Dragon descriptors
- `svc_panela.py`: Support Vector Classifier (SVC) with Dragon descriptors

### Panel B

- `gbrt_panelb.py`: Gradient Boosting Decision Trees (GBDT) with RDKit descriptors
- `lr_panelb.py`: Logistic Regression with RDKit descriptors
- `rf_panelb.py`: Random Forest (RF) with RDKit descriptors
- `svc_panelb.py`: Support Vector Classifier (SVC) with RDKit descriptors

### Panel C

- `gbrt_panelc.py`: Gradient Boosting Decision Trees (GBDT) with ECFP molecular fingerprint
- `lr_panelc.py`: Logistic Regression with ECFP molecular fingerprint
- `mlp_panelc.py`: Multi-layer Perceptron (MLP) with ECFP molecular fingerprint
- `rf_panelc.py`: Random Forest Classifier with ECFP molecular fingerprint
- `svc_panelc.py`: Support Vector Classifier (SVC) with ECFP molecular fingerprint

### Panel D

- `gbrt_paneld.py`: Gradient Boosting Decision Trees (GBDT) with MACCS molecular fingerprint
- `lr_paneld.py`: Logistic Regression with MACCS molecular fingerprint
- `rf_paneld.py`: Random Forest Classifier with MACCS molecular fingerprint
- `svc_paneld.py`: Support Vector Machine with MACCS molecular fingerprint

## What The Training Scripts Do

Each training script follows the same general workflow:

1. load the panel-specific input data
2. build the feature matrix
3. run 5 repeats of 5-fold cross-validation
4. tune model hyperparameters on a validation split
5. evaluate test-fold performance
6. save fold-level and summary metrics
7. save the best overall trained model

The main evaluation outputs reported by the scripts are:

- accuracy
- weighted F1
- precision
- recall
- precision, recall, and F1 across 3 bioconcentration mechanism classes

## Output Files

Running a training script produces following files.

Typical outputs include:

- `results_*_repeat.pkl`: fold-level and repeat-level metrics
- `results_*_final_metrics.pkl`: arrays of final metrics across all 25 trained models
- `best_*.pt`: saved best model

Examples:

- `results_rf_panela_repeat.pkl`
- `results_rf_panela_final_metrics.pkl`
- `best_rf_model.pt`

Some Gradient Boosting scripts also save feature importance files, for example:

- `all_feature_importances_panela.pkl`
- `all_feature_importances_panelb.pkl`

## Feature Importance Analysis

The notebook [Feature Importance/FeatureImportance.ipynb](https://github.com/ssd6515/ChemClassifier/blob/930873a20aa58e59a4fe97e2f4701d74c6e7fb5e/Feature%20Importance/FeatureImportance.ipynb) is used to inspect descriptor importance outputs generated by the model training runs.

The repository already contains:

- [Feature Importance/all_feature_importances_panela.pkl](https://github.com/ssd6515/ChemClassifier/blob/930873a20aa58e59a4fe97e2f4701d74c6e7fb5e/Feature%20Importance/all_feature_importances_panela.pkl)
- [Feature Importance/all_feature_importances_panelb.pkl](https://github.com/ssd6515/ChemClassifier/blob/930873a20aa58e59a4fe97e2f4701d74c6e7fb5e/Feature%20Importance/all_feature_importances_panelb.pkl)

## Reproducing Results

If you want to reproduce a full set of results, a practical order is:

1. create and activate the Conda environment
2. prepare or obtain `bcf_data.csv`
3. run [RDKit Data Exraction/Generate_RDKit_Features.ipynb](https://github.com/ssd6515/ChemClassifier/blob/04a5890030ec530ccc8bbb63a3fb868826b6cef9/RDKit%20Data%20Exraction/Generate_RDKit_Features.ipynb) if you need to regenerate `rdkit_data.csv`
4. copy the needed CSV into the panel folder you plan to train
5. run the desired panel scripts
6. optionally open the feature-importance notebook for interpretation

## Notes And Caveats

- `bcf_data.csv` is referenced by many scripts, but it is not currently included in this repository.
- `rdkit_data.csv` is included under `RDKit Data Exraction`.
- The scripts print progress and metrics to the console and store serialized results as pickle files.
