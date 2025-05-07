# 📊 Tech Moms Application Model: Final Report & Production Insights

---

## 🧾 Executive Summary

### 🎯 Goal
To deploy a production-ready predictive model that assists Tech Moms program administrators in identifying applicants most likely to be accepted—maximizing both efficiency and fairness in the selection process.

### 📏 Metrics
- **F1 Score** (primary metric for imbalanced classes)
- **Recall & Precision**
- **Accuracy & ROC AUC**
- **Calibration MAE**

### 🔍 Key Findings
- **Complement Naive Bayes** with PCA + SMOTE offers best balance of recall, precision, and calibration.
- Final model identifies ~50% more true positives than baseline.
- Probability calibration MAE improved to **0.1175**.

### ⚠️ Risks & Assumptions
- Assumes past acceptance decisions reflect future criteria.
- Limitations include potential **bias** in demographic features and **small dataset size** (n = 1732).
- Despite SMOTE, class imbalance may still influence precision.

---

## 🧪 Step-by-Step Model Walkthrough

---

### 🧭 Exploratory Data Analysis (EDA)

#### 📌 Variables of Interest
- `education`, `employment`, `race`, `ethnicity`, `children`
- `has_computer`, `computer_type`
- `month`, `quarter`, `is_weekend`

#### 🧼 Outlier Handling
- No significant numeric outliers found. Categorical levels were reviewed.

#### 🩺 Imputation
- **Mode imputation** for categorical values
- Binary flags to preserve data integrity when necessary

---

## 🤖 Final Model Pipeline

### 1. Model Selection
- Benchmarked 8 models using unified metrics
- **Complement Naive Bayes** chosen for highest F1 and calibrated recall

### 2. Pipeline Implementation
- **PCA** reduced 37 features to 22 (95.4% variance retained)
- **SMOTE** balanced the class distribution
- **Complement Naive Bayes** classifier used for final predictions

### 3. Evaluation Results
| Metric       | Value  |
|--------------|--------|
| F1 Score     | 0.508  |
| Recall       | 0.500  |
| Precision    | 0.432  |
| ROC AUC      | 0.601  |
| Calibration MAE | 0.1175 |

### 4. Prediction & Inference
- Applied model to **2024 applicant data**
- Produced both **class labels** and **probability scores**
- Used **permutation importance** for explainability

---

## 📈 Visual Insights

- **Confusion Matrix**: Improved TP/FP balance
- **Calibration Curve**: Almost ideal alignment
- **Probability Histogram**: Clear class separation
- **Feature Importance**: Top contributors visualized

---

## 🚀 Production Readiness

### 🔄 Ongoing Validation
- Track model drift and performance decay
- Implement weekly F1 score monitoring
- Run A/B tests comparing manual vs. model-assisted review

### 🏭 Enterprise Deployment Steps
1. **Serialize model**: `final_model_pipeline.pkl`
2. Wrap in REST API using **Flask** or **FastAPI**
3. Create **batch scoring** utility for CSV uploads
4. Auto-generate cohort-level insights

### 🌐 Public Accessibility
- Host on [GitHub](https://github.com/)
- Deploy interactive app on [Streamlit](https://streamlit.io/) or [Hugging Face Spaces](https://huggingface.co/spaces)
- Connect to **Google Forms** or **Airtable** for non-technical use

---

## 📎 Technical Appendix

### 🧰 Libraries Used
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- `imbalanced-learn` (SMOTE)
- `joblib`, `pickle` (model serialization)

### 📂 Files and Resources
| File                          | Description                                 |
|-------------------------------|---------------------------------------------|
| `tech_moms_preprocessed.csv` | Cleaned dataset                             |
| `final_model_pipeline.pkl`   | Serialized production-ready model pipeline  |
| `final_model_metrics.json`   | Final model performance metrics             |
| `model_feature_columns.pkl`  | Feature list used in modeling               |

---
