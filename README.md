# Liquid Neural Network for Predicting Common Bile Duct Stone Recurrence after ERCP (ercp-cbds-lnn)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)](https://scikit-learn.org/)

This repository provides the official implementation, trained model weights, preprocessing pipelines, and clinical data dictionary for the research paper:

> **Development and Validation of a Liquid Neural Network-Based Prediction Model and Web-Based Application for Stone Recurrence After Endoscopic Retrograde Cholangiopancreatography**  
> *BMC Medical Informatics and Decision Making* (Under Review).

---

## 📋 Overview & Highlights

- **Clinical Problem**: Common bile duct stone (CBDS) recurrence is a frequent and serious late complication following endoscopic stone clearance via ERCP.
- **Model Architecture**: A continuous-time **Liquid Neural Network (LNN)** based on Liquid Time-Constant (LTC) neurons (*Hasani et al., AAAI 2021*), utilizing shared-weight numerical Euler ordinary differential equation (ODE) integration steps.
- **Predictors (6 features)**: Selected via LASSO $\lambda_{1se}$ from 19 candidate predictors:
  1. `PAD`: Periampullary diverticulum (0 = No, 1 = Yes)
  2. `Stone_count`: Number of stones (1 = 1, 2 = 2, 3 = $\ge$ 3)
  3. `Stone_diameter`: Maximum stone diameter (mm)
  4. `CBD_diameter`: Common bile duct diameter (mm)
  5. `EST`: Endoscopic sphincterotomy (0 = No, 1 = Yes)
  6. `CBDA`: Common bile duct angulation (0 = $> 145^\circ$, 1 = $\le 145^\circ$)
- **Performance**:
  - **Internal Validation Cohort** ($n = 225$): **AUROC = 0.927** (95% CI: 0.88–0.96), Brier score = 0.086.
  - **Independent External Test Set** ($n = 152$): **AUROC = 0.878** (95% CI: 0.809–0.938), Accuracy = 83.55%, Sensitivity = 78.57%, Specificity = 84.68% at the locked cutoff.
- **Decision Cutoff**: Locked at **0.198**, obtained by maximizing the Youden index on out-of-fold cross-validation predictions within the training cohort and applied unchanged to the validation and external test sets.

---

## 📂 Repository Structure

```text
ercp-cbds-lnn/
├── README.md                 # Complete documentation and quickstart guide
├── LICENSE                   # MIT Open Source License
├── requirements.txt          # Strictly pinned dependencies matching manuscript
├── data_dictionary.csv       # Clinical definition & encoding for all 19 candidate variables
├── model/
│   ├── lnn_weights.pt        # Trained PyTorch LNN state_dict
│   ├── imputer.joblib        # Training-set-fitted IterativeImputer (BayesianRidge)
│   ├── scaler.joblib         # Training-set-fitted MinMaxScaler
│   └── config.json           # Model configuration, threshold, and feature list
├── src/
│   ├── ltc_model.py          # PyTorch definition of the LTCCell architecture
│   ├── predict.py            # End-to-end inference and risk stratification script
│   └── train.py              # Full training pipeline script
└── example/
    └── synthetic_input.csv   # Synthesized mock clinical cohort (50 patients)
```

---

## ⚙️ Installation

We recommend using a Python 3.10+ virtual environment:

```bash
# Clone or extract repository
git clone https://github.com/your-username/ercp-cbds-lnn.git
cd ercp-cbds-lnn

# Install dependencies (strictly pinned)
pip install -r requirements.txt
```

---

## 🚀 Quickstart & Inference

### 1. Batch Prediction on a CSV File
Run inference on a batch of patients (e.g., using the provided synthetic sample):

```bash
python src/predict.py --input example/synthetic_input.csv --output example/synthetic_predictions.csv
```

Output preview (`example/synthetic_predictions.csv`):
```text
Patient_ID,...,PAD,Stone_count,Stone_diameter,CBD_diameter,EST,CBDA,Predicted_Recurrence_Probability,Binary_Prediction_Threshold_0.198,Risk_Stratification
SYNTH_001,...,1,3,17.1,13.4,1,0,0.1792,0,Lower Risk (< 0.198)
SYNTH_002,...,0,1,11.7,13.0,1,0,0.0141,0,Lower Risk (< 0.198)
SYNTH_004,...,0,3,12.7,19.3,1,1,0.7818,1,Higher Risk (>= 0.198)
```
The output file keeps every column of the input (all 19 candidate variables are present in the synthetic
sample) and appends the three prediction columns shown above; 13 of the 50 synthetic patients fall above
the locked threshold.

### 2. Single Patient CLI Evaluation
Evaluate an individual patient directly via command line arguments:

```bash
python src/predict.py --pad 1 --stone_count 3 --stone_diameter 15.0 --cbd_diameter 14.0 --est 1 --cbda 1
```

Console Output:
```text
=======================================================
          LNN CBDS Recurrence Risk Assessment
=======================================================
Predicted Recurrence Probability: 88.12%
Decision Cutoff Threshold       : 0.198
Risk Stratification Category    : Higher Risk (>= 0.198)
=======================================================
```

### 3. Python API Integration
```python
from src.predict import LNNPredictor
import pandas as pd

predictor = LNNPredictor()
df = pd.read_csv("example/synthetic_input.csv")
predictions = predictor.predict_dataframe(df)
print(predictions[["Predicted_Recurrence_Probability", "Risk_Stratification"]].head())
```

---

## 📖 Data Dictionary

A comprehensive data dictionary for all 19 candidate predictors and the primary outcome is provided in [`data_dictionary.csv`](data_dictionary.csv). It details:
- Clinical domain, variable name, Chinese name, and measurement timing.
- Exact permissible ranges, units, and numerical encodings.
- Measurement protocols (e.g., standardized cholangiographic angles and diameters).

---

## 🔒 Ethical Compliance & Data Privacy

- **IRB Approval**: The study protocol was approved by the Ethics Committee of the Changshu Hospital Affiliated to Soochow University (institutional review board approval number: **L202502142**).
- **Patient Privacy**: Under the retrospective non-interventional ethical waiver, raw individual patient-level clinical records cannot be publicly disseminated.
- **Synthetic Data**: To enable immediate code execution, transparency, and reproducibility without compromising patient privacy, this repository includes `example/synthetic_input.csv`. This synthesized dataset preserves identical column headers, data formats, and plausible clinical ranges without containing any real patient identities.
- **Data Access**: De-identified clinical data can be requested from the corresponding author upon reasonable research request and institutional data transfer agreement.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
