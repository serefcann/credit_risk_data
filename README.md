# credit_risk_data
# Home Credit Risk Prediction Project

This project focuses on predicting home credit risk using data from the Home Credit Default Risk dataset. It includes steps such as database connection, data preprocessing, feature engineering, and training machine learning models to evaluate performance.

## 📌 Project Overview

The goal is to predict the probability of a client defaulting on a loan. The project explores different feature representations and model performances to determine which engineered dataset yields better predictive power.

## ⚙️ Technologies & Libraries

- **Python**
- **MySQL** – for storing and querying data
- **Pandas**, **NumPy** – data manipulation
- **Matplotlib**, **Seaborn** – data visualization
- **Scikit-learn** – preprocessing and modeling
- **PolynomialFeatures** – feature engineering
- **Pipeline**, **ColumnTransformer** – modular preprocessing

## 📂 Data Handling

- Connected multiple tables from MySQL to retrieve raw data.
- Merged datasets using common keys (such as `SK_ID_CURR`) for comprehensive analysis.
- Pulled and loaded data into Pandas DataFrames for further processing.

## 🧹 Data Preprocessing

- Imputed missing values using `SimpleImputer` (strategy: median).
- Applied `StandardScaler` for feature scaling.
- Created new features via domain knowledge and polynomial interactions.
- Handled outliers and NaNs carefully before modeling.

## 🧠 Feature Engineering

1. **Polynomial Features**:
    - Used `PolynomialFeatures` on selected continuous variables (`EXT_SOURCE_1`, `EXT_SOURCE_2`, `EXT_SOURCE_3`, `DAYS_BIRTH`) up to degree 3.
    - Analyzed correlation of new features with the target.

2. **Domain Knowledge Features**:
    - `CREDIT_INCOME_PERCENT` = `AMT_CREDIT` / `AMT_INCOME_TOTAL`
    - `DAYS_EMPLOYED_PERCENT` = `DAYS_EMPLOYED` / `DAYS_BIRTH`
    - `INCOME_PER_PERSON_FAMILY` = `AMT_INCOME_TOTAL` / `CNT_CHILDREN` + 1
    - `ANNUITY_INCOME_PERCENT` = `AMT_ANNUITY` / `AMT_INCOME_TOTAL` 
    - `CREDIT_TERM` = `AMT_ANNUITY` / `AMT_CREDIT`

## 🔍 Exploratory Data Analysis

- Plotted distributions and correlations between features and the target.
- Visualized KDE plots for EXT_SOURCE variables.
- Grouped age features into bins to analyze default rates.

## 🤖 Modeling

### Logistic Regression

- Built a baseline model using `LogisticRegression`.
- Applied hyperparameter tuning (`max_iter`).
- Evaluated model using ROC AUC score.

### Random Forest Classifier

- Trained `RandomForestClassifier` to improve non-linear prediction.
- Compared with logistic regression in terms of performance.

## ✅ Results

- Compared three datasets:  
  - **Original (`app_train`)**  
  - **Polynomial Features (`poly_train`)**  
  - **Domain Features (`app_train_domain`)**

- Evaluated performance using:
  - ROC AUC
  - Feature correlation with target
  - Model interpretability

## 🏁 Conclusion

- **Domain Feature set** provided the best trade-off between performance and interpretability.
- Polynomial features increased complexity but did not significantly improve accuracy.
- Logistic Regression performed well on scaled, clean data, while Random Forest captured complex relationships.

---




