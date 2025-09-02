

***

# Credit Risk Analysis Using Machine Learning

## Overview
This project analyzes **credit risk** using machine learning models to estimate the Probability of Default (PD) for loan applications. By predicting PD, lenders can minimize defaults and associated losses, helping prevent financial crises like that of 2008.[1]

## Problem Statement
Approving loans without scientific evaluation increases default risk and can destabilize financial systems. Loss from loan default is calculated as:
$$
\text{Expected Loss} = \text{PD} \times \text{EAD} \times \text{LGD}
$$
- **PD**: Probability of Default  
- **EAD**: Exposure at Default  
- **LGD**: Loss Given Default[1]

This notebook focuses on **PD** prediction using the German Credit Data.

## Dataset
- **Source**: German Credit Dataset
- **Features**:
    - Age
    - Sex
    - Job
    - Housing
    - Saving accounts
    - Checking account
    - Credit amount
    - Duration
    - Purpose
    - Risk (Target: 'good' or 'bad')

## Libraries Used
- pandas, numpy for data manipulation
- matplotlib, seaborn, plotly for visualization
- scikit-learn for machine learning models and evaluation
- xgboost for gradient boosting models

## Workflow

### 1. Data Preparation
- Load and inspect the dataset
- Data cleaning and EDA (Exploratory Data Analysis)

### 2. Modeling
Multiple machine learning algorithms are evaluated:

| Algorithm                       | Description |
|---------------------------------|-------------|
| Logistic Regression              | Baseline linear model for classification |
| Decision Tree Classifier         | Tree-based model capturing non-linear relationships |
| Random Forest Classifier         | Ensemble of decision trees for better generalization |
| K-Nearest Neighbors (KNN)        | Instance-based learning using distance metrics |
| Linear Discriminant Analysis     | Dimensionality reduction with class separation |
| Gaussian Naive Bayes             | Probabilistic classifier assuming feature independence |
| Support Vector Classifier (SVC)  | Finds optimal hyperplane for classification |
| XGBoost Classifier               | Gradient boosting model with high predictive power |

### 3. Evaluation
- Metrics: **Accuracy**, **F1 score**, **Classification report**, **Confusion matrix**
- Cross-validation and parameter tuning using GridSearchCV

### 4. Results
| Model                  | Train Accuracy | Test Accuracy |
|------------------------|---------------|--------------|
| SVC                    | 0.8259        | 0.8261       |
| Decision Tree          | 1.0000        | 0.7526       |
| Random Forest          | 1.0000        | 0.8259       |
| Gaussian Naive Bayes   | 0.8120        | 0.8110       |

#### Top Feature Importances (Random Forest)
| Feature           | Importance |
|-------------------|------------|
| Credit amount     | 0.198      |
| Age               | 0.143      |
| Duration          | 0.135      |
| Checking account  | 0.124      |

## Usage
1. Install required libraries (see notebook imports).
2. Download the German Credit Data CSV from Kaggle.
3. Run the notebook to replicate model training and evaluation.
4. Review feature importances and model performance.

## Notes
- The notebook demonstrates both data exploration and model comparison.
- Results may vary depending on random states and parameter choices.[1]

## License
This notebook is intended for educational purposes and demonstration. Please check dataset usage terms for compliance.[1]

