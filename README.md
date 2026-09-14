# CW-MultiTarget-ML
This repository contains the main code for the multi-target machine learning framework developed to predict the effluent concentrations of COD, NH₄⁺-N, and NO₃⁻-N in constructed wetlands.

## Overview

The framework uses a multi-output Gradient Boosting Decision Tree (GBDT) model to predict multiple effluent quality indicators simultaneously. SHAP-based feature contribution analysis is further used to investigate the relative contribution of input variables to model predictions.

The repository also includes the code used for independent experimental validation of the trained model.

## Repository structure

```text
CW-MultiTarget-ML/
├── README.md
├── requirements.txt
├── multi_output_GBDT.py
├── SHAP_analysis.py
└── external_validation.py
```

### Main scripts

* `multi_output_GBDT.py`: Training and evaluation of the multi-output GBDT model.
* `SHAP_analysis.py`: SHAP-based model interpretation and feature contribution analysis.
* `external_validation.py`: Independent experimental validation using data not included in the model training dataset.

## Data availability

The dataset used in this study was compiled from previously published literature and experimental data and is not included in this repository.

Researchers who require the compiled dataset for academic purposes may contact the corresponding author.

## Environment

The analyses were conducted using Python 3.12.

The main Python packages used in this study include:

* NumPy
* pandas
* scikit-learn
* XGBoost
* LightGBM
* SHAP
* Matplotlib
* Optuna
* Joblib

The exact package versions used in the analysis are provided in `requirements.txt`.

## Usage

1. Install Python 3.12.

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Prepare the required input data according to the format used in the scripts.

4. Run the corresponding Python script for model training, model interpretation, or independent experimental validation.

## Notes

The trained model files, raw experimental data, intermediate results, and generated figures are not included in this repository. Local file paths used during the original analysis should be adapted to the user's own directory structure.
