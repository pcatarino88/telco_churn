# Predicting and Simulating Churn for a Telco Player

## Overview
This project builds an end-to-end **customer churn prediction and simulation framework** for a Telco player.  
The objectives are to:
1. **Predict customer churn** using machine learning  
2. Understand **key churn drivers** across customer segments  
3. Provide a **what-if simulator** to estimate the impact of different retention strategies (e.g. contract changes)

---

## Project Structure
The project is organized into five notebooks, following a clear machine learning lifecycle:

1. **Merge & Split** – data merging and train/validation/test split  
2. **EDA & Segmentation** – exploratory data analysis and customer segmentation  
3. **Data Preparation** – data cleaning, feature engineering, and transformations  
4. **Feature Selection & Modeling** – baseline model, Logistic Regression, and **LightGBM (final selected model)**  
5. **Deployment** – model packaging and preparation for application usage  

Reusable functions are implemented under the `utils/` directory and encapsulated in a trained pipeline saved as:  
`models/best_churn_pipeline.pkl`

---

## Model Performance
The final LightGBM model was evaluated using a fixed decision threshold of **0.60**.

**Test set performance:**
- **ROC AUC:** 0.906  
- **Accuracy:** 0.832  
- **Recall:** 0.778  
- **F1 Score:** 0.711  
- **F2 Score:** 0.750  

Results are consistent across training, validation, and test sets, indicating good model generalization.

---

## Data
The dataset used in this project is **illustrative** and was obtained in an **academic environment**.  
It does not represent real customer data.

---

## Streamlit App
An interactive Streamlit application was developed in `app.py`.

The app functions as a **churn impact simulator**, allowing users to:
- Modify selected model inputs (e.g. contract duration)
- Observe the **expected positive or negative impact on churn probability**
- Explore probability distribution shifts under different scenarios

---

## Tech Stack
- **Python**
- **Machine Learning:** scikit-learn, LightGBM, SHAP, SMOTE  
- **Data Handling:** pandas, numpy  
- **Visualization:** matplotlib, seaborn  
- **Deployment:** Streamlit  

*(List not exhaustive)*