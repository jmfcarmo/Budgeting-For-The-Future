# Budgeting for the Future: Predicting Community Preferences with Machine Learning

This project was developed as part of a Kaggle-based academic challenge focused on participatory budgeting. Using predictive modeling techniques, we forecasted the lifestyle preferences of residents in a fictional municipality, Mining City, to help local government allocate public funds more effectively.

## 📌 Problem Statement

Participatory budgeting empowers citizens to influence how public funds are spent. The city council of Mining City wants to improve this process by understanding the population’s preferences and priorities. Using survey and demographic data, the goal was to classify each citizen into one of several lifestyle categories:

- Travel Enthusiast  
- Health-Conscious  
- Adventure Seeker  
- Fitness Enthusiast  
- Investor  

We approached this as a **multiclass classification** problem, evaluated using the **weighted F1 score**.

---

## 📊 Dataset

We received two datasets:  
- `train.csv` (with features + target)  
- `test.csv` (features only, for final submission)

Features included:
- Demographics: age, country, city, etc.  
- Financial indicators: investment risk, portfolio value, charity donations  
- Behavioral scores: health consciousness, tech savviness, entertainment engagement, etc.  
- Target variable: `lifestyle_type`

---

## 🧪 Methodology

### 1. Exploratory Data Analysis (EDA)
- Handled missing values and outliers
- Explored feature distributions and class imbalance
- Investigated correlations and feature importance

### 2. Preprocessing
- Date parsing
- Coherence check
- Outliers treatment
- Feature engineering (new variables and one-hot encoding)
- Feature scaling (MinMax Scaling)

### 3. Feature Selection
  - Spearman Correlation
  - ANOVA
  - Chi-Squared Score
  - RFE
  - Lasso Regression

### 4. Modeling
Algorithms tested:
  - Logistic Regression
  - Gaussian Naive Bayes
  - KNN Classifier
  - Ball Tree
  - KD Tree
  - Decision Trees
  - Bagging
  - Random Forests
  - Boosting (AdaBoost, Gradient Boosting)
  - Stacking
  - Neural Networks
  - Support Vector Machines
    
  Final model selected based on validation F1 score

---

## 📈 Results

- Final model: **Neural Networks** using **MinMax** scaling method
- Achieved **F1-weighted score of 0.78506** on validation set  
- Submitted predictions on Kaggle test set in CSV format

---

## 📦 Tools & Libraries

- Data manipulation: `pandas`, `numpy`
- Visualization: `matplotlib`, `seaborn`, `graphviz`, `tree`
- Preprocessing: `LabelEncoder`, `MinMaxScaler`, `KNNImputer`
- Feature selection: `RFE`, `SelectKBest`, `f_classif`
- Modeling:  
  - Linear models: `LogisticRegression`, `LassoCV`  
  - Tree-based models: `DecisionTreeClassifier`, `RandomForestClassifier`, `BaggingClassifier`, `AdaBoostClassifier`  
  - Others: `KNeighborsClassifier`, `GaussianNB`, `MLPClassifier`
- Model evaluation: `confusion_matrix`, `classification_report`, `roc_auc_score`, `f1_score`, etc.
- Tuning: `RandomizedSearchCV`, `KFold`, `train_test_split`
- Statistical analysis: `spearmanr`, `chi2_contingency`
- Utilities: `datetime`, `warnings`, `time`

---

## 🧠 Key Takeaways

- Successful multiclass classification in a policy-driven context
- Feature importance revealed financial wellness and risk preferences as key predictors
- Demonstrated practical application of ML in social good / public sector governance

---

## 📁 Project Structure

```bash
.
├── data/                    # Contains train/test datasets
├── notebook/                # Jupyter notebooks with full analysis
├── output/                  # CSV submission file
├── README.md                # Project overview
