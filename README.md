Machine-Learning-Group-Project

This project includes data analysis and machine learning workflows using Jupyter notebooks. It features raw and processed datasets, Python scripts, and results visualization for predictive modeling and performance evaluation.


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

1 . Motivation

The primary objective of this project is to predict mental illness using patient demographic and survey data. It explores different machine learning models to evaluate which approach provides the most reliable predictions, with a focus on practical implementation and interpretability.


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

2 . Dataset Info

Source: NYS OMH PCS 2019 (Patient Characteristics Survey)

Files included:

Patient_Characteristics_Survey__PCS___2019.csv

NYSOMH_PCS2019_DataDictionary.pdf

NYSOMH_PCS2019_Overview.pdf

Attributes: Demographic data, diagnostic categories, survey responses, etc.

Size: ~11,000 rows × 12 features

Target: Predict presence/absence of mental illness


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


3 . Technologies Used 


Languages: Python

Tools & Libraries:

Jupyter Notebook

NumPy, Pandas, Matplotlib, Seaborn

Scikit-learn

Flask / Streamlit (Web App)

Git & GitHub (Version Control)

ML Models:

Logistic Regression

Random Forest

Support Vector Machine (SVM)

XGBoost


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


How to Run ( Steps are included below )

1 . Clone the repository

git clone https://github.com/aarjunvarma/Machine-Learning-Group-Project.git
cd Machine-Learning-Group-Project

2 . Set up a virtual environment (optional but recommended)

pip install -r requirements.txt

3 . Run the notebook

jupyter notebook Notebooks/MI_MACHINE_LEARNING_PROJECT_FINAL.ipynb

4 . (Optional) Launch the web app

cd "Web App"
streamlit run webapp_file_copy.py

