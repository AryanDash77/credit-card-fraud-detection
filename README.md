# Credit Card Fraud Detection System

An end-to-end machine learning system to detect fraudulent credit card 
transactions using XGBoost, SMOTE, and SHAP explainability.

## 📊 Dataset
- 284,807 transactions with only 0.17% fraud (highly imbalanced)
- Source: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## 🔬 Project Structure
- `eda.ipynb` — Exploratory Data Analysis & Preprocessing
- `models.ipynb` — Model Training & Evaluation
- `shap.ipynb` — SHAP Explainability

## ⚙️ Tech Stack
Python, Pandas, Scikit-learn, XGBoost, imbalanced-learn, SHAP, Streamlit

## 📈 Results

| Model | Fraud Recall | AUC |
|---|---|---|
| Logistic Regression (baseline) | 0.64 | 0.957 |
| Random Forest + SMOTE | 0.81 | 0.969 |
| XGBoost + SMOTE | **0.89** | **0.979** |

## 🔑 Key Techniques
- **SMOTE** to handle extreme class imbalance (0.17% fraud)
- **XGBoost** gradient boosting for superior fraud detection
- **SHAP** for individual transaction explainability
- Precision-Recall curves as primary evaluation metric

## 🚀 How to Run
1. Download dataset from Kaggle and place in project folder
2. Run `eda.ipynb` → `models.ipynb` → `shap.ipynb` in order
3. Launch app: `streamlit run app.py`