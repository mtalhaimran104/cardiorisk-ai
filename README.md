# CardioRisk AI: Heart Disease Prediction

A machine learning project that predicts heart disease risk from clinical features using Python and scikit-learn.

## Problem

Heart disease is a major health concern, and identifying potential risk patterns in clinical data can support further medical assessment. This project explores machine learning models to classify whether a patient has heart disease based on clinical features.

## Dataset

- **Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **File:** `heart_disease_uci.csv`
- **Size:** 920 patient records × 16 columns in the project dataset.
- **Target:** The `num` column is converted into a binary target: `0` = no heart disease, `1` = heart disease present (original values 1–4).
- **Features include:** Age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG results, maximum heart rate, exercise-induced angina, ST depression, slope, number of major vessels, and thalassemia.

## Approach

1. **Exploratory Data Analysis (EDA):** Examined the dataset structure, clinical features, target distribution, and data-quality issues.
2. **Preprocessing:** Prepared clinical features and investigated missing values and data-quality issues before model training.
3. **Models compared:** Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Decision Tree, Random Forest, Gradient Boosting, AdaBoost, Gaussian Naive Bayes, XGBoost, and LightGBM.
4. **Model selection:** Used `GridSearchCV` with five-fold `StratifiedKFold` cross-validation to compare models and tune hyperparameters.
5. **Evaluation:** Evaluated model performance using classification metrics and confusion matrices.

## Results

The following are the recorded cross-validation accuracy results from model selection.

| Model | 5-Fold CV Accuracy |
|---|---:|
| Logistic Regression | 59.25% |
| XGBoost | 58.63% |
| LightGBM | 58.32% |

**Best model:** Logistic Regression, selected based on its highest recorded cross-validation accuracy. The selected hyperparameter was `C = 0.1`.

*Note: These are model-selection results, not a claim of final test-set performance. Final binary-classification precision, recall, F1-score, and test accuracy should be added after verifying the latest evaluation output.*

## Key Takeaways

- Accuracy alone is insufficient for evaluating a medical classification model.
- Recall for the heart-disease class is particularly important because false negatives can mean that patients with heart disease are incorrectly classified as low risk.
- Comparing multiple algorithms with stratified cross-validation provides a more systematic approach to model selection.
- Clinical datasets can contain missing values, mixed feature types, and other data-quality challenges that affect model performance.

## Limitations

This is an educational machine learning project and **not a medical diagnostic tool**. It has not been established as clinically reliable and should not be used to make medical decisions.

The model's performance is limited by the dataset, its representativeness, and the evaluation performed. Independent external validation and further clinical assessment would be required before any real-world use.

## How to Run

```bash
git clone https://github.com/mtalhaimran104/cardiorisk-ai.git
cd cardiorisk-ai
jupyter notebook
```

Open `heart_disease_prediction.ipynb` to explore the analysis, preprocessing, model training, and evaluation.

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · XGBoost · LightGBM · Jupyter Notebook
