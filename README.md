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

- `bcf_data.csv`: used by Panels A, C, and D
- `rdkit_data.csv`: used by Panel B

### How `rdkit_data.csv` is generated

The notebook [RDKit Data Exraction/Generate_RDKit_Features.ipynb](/C:/Users/Shashwat/Documents/1PhDEnvironmentalHealthDataScience/ChemClassifier/RDKit%20Data%20Exraction/Generate_RDKit_Features.ipynb) starts from a source Excel file (`bcf_data.xlsx`), retrieves or standardizes SMILES, builds RDKit molecules, and exports `rdkit_data.csv`.

The repository already includes one generated file at [RDKit Data Exraction/rdkit_data.csv](/C:/Users/Shashwat/Documents/1PhDEnvironmentalHealthDataScience/ChemClassifier/RDKit%20Data%20Exraction/rdkit_data.csv).

### Expected columns

The training scripts expect the following:

- `bcf_data.csv` must include `Class`
- `bcf_data.csv` must also include the Dragon descriptor columns used in Panel A
- `bcf_data.csv` must include `SMILES` for Panels C and D
- `rdkit_data.csv` must include `CAS`, `QSAR_READY_SMILES`, `mol`, and `Class`

### Important file placement note

The training scripts use relative paths such as `bcf_data.csv` and `rdkit_data.csv`, and they also import `utility.py` from the parent `Model Training` folder. Because of that, the easiest way to run the scripts is:

1. stay inside the `Model Training` directory
2. set `PYTHONPATH` to that directory
3. copy the required CSV file into the panel folder you want to run

For example:

- copy `bcf_data.csv` into `Model Training/Panel A`, `Model Training/Panel C`, or `Model Training/Panel D`
- copy `rdkit_data.csv` into `Model Training/Panel B`

## How To Run The Models

### PowerShell setup for training

From the repository root:

```powershell
cd ".\Model Training"
$env:PYTHONPATH = (Get-Location).Path
```

That environment variable is needed because the panel scripts import `utility.py` with:

```python
from utility import Kfold
```

### Run a single model

Example: Panel A Random Forest

```powershell
Copy-Item "..\bcf_data.csv" ".\Panel A\bcf_data.csv"
python ".\Panel A\rf_panela.py"
```

Example: Panel B Random Forest

```powershell
Copy-Item "..\RDKit Data Exraction\rdkit_data.csv" ".\Panel B\rdkit_data.csv"
python ".\Panel B\rf_panelb.py"
```

Example: Panel C MLP

```powershell
Copy-Item "..\bcf_data.csv" ".\Panel C\bcf_data.csv"
python ".\Panel C\mlp_panelc.py"
```

Example: Panel D SVC

```powershell
Copy-Item "..\bcf_data.csv" ".\Panel D\bcf_data.csv"
python ".\Panel D\svc_paneld.py"
```

### Available training scripts

#### Panel A

- `gbrt_panela.py`
- `lr_panela.py`
- `rf_panela.py`
- `svc_panela.py`

#### Panel B

- `gbrt_panelb.py`
- `lr_panelb.py`
- `rf_panelb.py`
- `svc_panelb.py`

#### Panel C

- `gbrt_panelc.py`
- `lr_panelc.py`
- `mlp_panelc.py`
- `rf_panelc.py`
- `svc_panelc.py`

#### Panel D

- `gbrt_paneld.py`
- `lr_paneld.py`
- `rf_paneld.py`
- `svc_paneld.py`

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
- class-wise precision
- class-wise recall
- class-wise F1
- average precision, recall, and F1 across classes

## Output Files

Running a training script produces files in the same panel folder as the script.

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

The notebook [Feature Importance/FeatureImportance.ipynb](/C:/Users/Shashwat/Documents/1PhDEnvironmentalHealthDataScience/ChemClassifier/Feature%20Importance/FeatureImportance.ipynb) is used to inspect descriptor importance outputs generated by the model training runs.

The repository already contains:

- [Feature Importance/all_feature_importances_panela.pkl](/C:/Users/Shashwat/Documents/1PhDEnvironmentalHealthDataScience/ChemClassifier/Feature%20Importance/all_feature_importances_panela.pkl)
- [Feature Importance/all_feature_importances_panelb.pkl](/C:/Users/Shashwat/Documents/1PhDEnvironmentalHealthDataScience/ChemClassifier/Feature%20Importance/all_feature_importances_panelb.pkl)

## Reproducing Results

If you want to reproduce a full set of results, a practical order is:

1. create and activate the Conda environment
2. prepare or obtain `bcf_data.csv`
3. run [RDKit Data Exraction/Generate_RDKit_Features.ipynb](/C:/Users/Shashwat/Documents/1PhDEnvironmentalHealthDataScience/ChemClassifier/RDKit%20Data%20Exraction/Generate_RDKit_Features.ipynb) if you need to regenerate `rdkit_data.csv`
4. copy the needed CSV into the panel folder you plan to train
5. set `PYTHONPATH` to the `Model Training` folder
6. run the desired panel scripts
7. inspect the generated `.pkl` and `.pt` files
8. optionally open the feature-importance notebook for interpretation

## Notes And Caveats

- `bcf_data.csv` is referenced by many scripts, but it is not currently included in this repository.
- `rdkit_data.csv` is included under `RDKit Data Exraction`, but Panel B scripts expect it in their working folder unless you edit the script path.
- The current scripts are written as standalone files and do not accept command-line arguments for input or output paths.
- Several scripts save output files with names inherited from earlier experiments, so some filenames in Panels C and D still contain labels such as `t2panela` or `t2panelb`.
- The scripts print progress and metrics to the console and store serialized results as pickle files.

## Suggested Next Improvement

If you plan to share this project more broadly, the biggest usability improvement would be to refactor the training scripts so they accept:

- `--input`
- `--output-dir`
- `--panel`
- `--model`

That would remove the current need to copy CSV files into panel folders and manually set `PYTHONPATH`.
