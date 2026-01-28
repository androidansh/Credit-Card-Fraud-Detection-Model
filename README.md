# 💳 Credit Card Fraud Detection System  
### Hybrid Anomaly Detection + Supervised Learning (Isolation Forest + XGBoost)

---

## 📌 Project Overview

This project implements an **industry-style Credit Card Fraud Detection System** using a **hybrid machine learning architecture**:

- **Isolation Forest** for anomaly detection  
- **XGBoost Classifier** for final fraud prediction  

The system is designed to handle **highly imbalanced real-world transaction data** and prioritizes **fraud recall** while maintaining strong overall performance.

---

## 🧠 Problem Statement

Credit card fraud datasets are:
- Extremely imbalanced (fraud < 1%)
- High-dimensional
- Non-linear in behavior

Traditional models struggle to detect rare fraud patterns reliably.  
This project solves that using **unsupervised anomaly detection combined with supervised classification**.

---

## 🏗️ Architecture (Industry-Style)

**Raw Transactions**
       **↓**
**Feature Engineering**
       **↓**
**Scaling (StandardScaler)**
       **↓**
**Isolation Forest**
       **↓**
**Anomaly Features**
**(anomaly_score, is_outlier)**
       **↓**
**XGBoost Classifier**
       **↓**
**Fraud Probability**
       **↓**
**Threshold-Based Decision**

---

## 📊 Dataset Description

The dataset contains real-world credit card transaction records.

### Final Features Used

**| Feature          |         Description                         |**
|------------------|---------------------------------------------|
| amount           | Transaction amount                          |
| distance         | Distance between cardholder & merchant (km) |
| is_weekend       | Weekend flag (0/1)                          |
| hour             | Hour of transaction                         |
| day_of_week      | Day of week (0–6)                           |
| category_encoded | Encoded merchant category                   |
| anomaly_score    | Isolation Forest anomaly score              |
| is_outlier       | Binary anomaly flag                         |

Target:
- `**is_fraud` (0 = Normal, 1 = Fraud)**

---

## ⚙️ Models Used

### 🔹 Isolation Forest
- Detects abnormal transaction patterns
- Generates anomaly-based features
- Contamination tuned to real fraud rate (~2%)

### 🔹 XGBoost Classifier
- Handles class imbalance effectively
- Learns complex non-linear patterns
- Uses anomaly features as strong fraud signals

---

## 📈 Model Evaluation

### Classification Report

```
Fraud Recall: 95%
Fraud Precision: 26%
Overall Accuracy: 98%
ROC-AUC Score: 0.997
```

### Key Insight:
- **Recall is prioritized** over precision
- Missing fraud is costlier than false alerts
- Threshold tuning is applied instead of default 0.5

---

## 🎯 Threshold Optimization

Final fraud decision is made using a **custom probability threshold (0.8)** to:
- Reduce false positives
- Maintain high fraud recall

```python
prediction = (fraud_probability >= 0.8).astype(int)
```

---

## 🧪 Testing & Validation

Synthetic test transactions were created to validate:
- Normal consumer behavior
- Fraud-like abnormal behavior

The model assigns:
- Low fraud probability to normal patterns
- High fraud probability to abnormal transactions

---

## 🧠 Key Learnings

- Hybrid models outperform single-model approaches in fraud detection
- Anomaly detection enhances supervised learning
- Threshold tuning is critical in imbalanced problems
- Accuracy alone is misleading for fraud problems

---

## 📌 Future Improvements

- SHAP-based model explainability
- Real-time transaction scoring API
- Online learning for concept drift
- Cost-sensitive loss optimization

---

## 📄 Conclusion

This project demonstrates a **production-ready fraud detection pipeline** aligned with real-world banking systems.  
The hybrid approach ensures **robust detection**, **scalability**, and **business-aligned decision-making**.

---

## 👤 Author

**Anshuman Gupta**  
Machine Learning | Data Science  


