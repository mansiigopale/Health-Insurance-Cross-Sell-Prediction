# **Health Insurance Cross-Sell Prediction 🚗🏥**

## **📌 Project Overview**

The insurance industry depends heavily on data-driven decision-making. This project focuses on predicting whether health insurance policyholders are also likely to purchase vehicle insurance. Identifying these potential customers helps insurance companies optimize marketing efforts, reduce costs, and increase revenue through effective cross-selling strategies.

By applying machine learning classification models, we analyze customer demographics and policy details to make accurate predictions.

## **🎯 Objectives**

The main goal of this project is to build a predictive model that classifies customers based on their likelihood of purchasing vehicle insurance. To achieve this, the project involves:

* Performing Exploratory Data Analysis (EDA) to uncover patterns in the data.

* Cleaning and preprocessing data to make it suitable for machine learning.

* Building and comparing multiple classification algorithms such as Logistic Regression, Random Forest, and XGBoost.

* Evaluating performance using accuracy, precision, recall, F1-score, and ROC-AUC.

## **📊 Dataset**

* **Source:** Provided dataset (TRAIN-HEALTH INSURANCE CROSS SELL PREDICTION.csv)

* **Size:** ~380,000 rows and 12 features

* **Features:** Customer demographics, policy details, and vehicle-related attributes

* **Target variable:** Response

* **1** → Customer is interested in vehicle insurance

* **0** → Customer is not interested

This large dataset allows the model to learn meaningful insights about which customers are most likely to respond positively to cross-sell offers.




## **🔧 Tech Stack & Libraries**

Python for implementation

* **Data Analysis:** Pandas, NumPy

* **Visualization:** Matplotlib, Seaborn for charts and plots

**Machine Learning:**

* **scikit-learn** → Logistic Regression, Random Forest

* **XGBoost** → Gradient boosting model for improved accuracy

* **Model Tuning:** GridSearchCV, RandomizedSearchCV for hyperparameter optimization

  

## **📂 Project Workflow**

* Data Preprocessing

* Encoded categorical variables

* Standardized numerical features for scaling

* Checked for multicollinearity using Variance Inflation Factor (VIF)

**Exploratory Data Analysis (EDA)**

* Plotted distributions to understand customer demographics

* Analyzed relationships between features and insurance response

* Used heatmaps and correlation analysis

**Model Building**

* Logistic Regression → Baseline model for interpretability

* Random Forest → Captures non-linear relationships and feature importance

* XGBoost → Boosting algorithm to maximize predictive power

**Model Evaluation**

* Compared models using Accuracy, Precision, Recall, F1-score, and ROC-AUC

* Visualized Confusion Matrix and ROC curves for better understanding




  ## **🚀 Results & Insights**

* Logistic Regression gave a good baseline but struggled with recall.

* Random Forest and XGBoost significantly improved performance, making them better suited for identifying potential customers.

* The model can guide the insurance company to target customers more effectively, increasing the chances of successful cross-selling.



## **📌 Future Enhancements**

* Balance the dataset using SMOTE or other resampling techniques.

* Deploy the final model with Flask/Streamlit for real-world use.

* Explore ensemble models and deep learning approaches for further improvement.

## **✨ Author** - 
Manasi Gopale

  

