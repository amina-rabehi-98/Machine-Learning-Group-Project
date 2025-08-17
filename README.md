Machine-Learning-Group-Project

A collaborative project showcasing data preprocessing, exploratory analysis, predictive modeling, and evaluation using Jupyter notebooks and Python scripts. Includes raw and processed datasets, reproducible workflows, and visualizations of results.

This project explores the application of machine learning to real-world healthcare data in order to predict mental illness and substance use disorders. Using the 2019 Patient Characteristics Survey (PCS) dataset, we analyze demographic, social, and medical factors, apply data preprocessing and feature engineering, and evaluate multiple predictive models. The work combines data analysis, model comparison, and web deployment to provide insights into mental health prediction and demonstrate practical machine learning workflows.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

1 . Motivation

Mental health disorders represent a growing global challenge, affecting millions of individuals each year. Early identification and accurate prediction of such conditions can significantly improve prevention strategies, patient care, and resource allocation.

This project was developed as part of our MSc program, in the context of a school project, to explore how machine learning models can be applied to real-world health data. By analyzing the 2019 Patient Characteristics Survey (PCS) dataset, we aim to:

- Investigate relationships between demographic, social, and medical factors.
- Build predictive models for mental illness and substance use disorders.
- Compare the performance of different machine learning algorithms (Logistic Regression, Decision Tree, Random Forest) with and without data balancing techniques (SMOTE).
- Demonstrate the feasibility of deploying such models in a simple and accessible web application (via Streamlit).

Our ultimate motivation, within this academic project, is to combine data engineering, analytics, and data science skills to provide insights that could support decision-making in healthcare contexts.


------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

2 . Dataset Info

- Description: This dataset comes from the 2019 Patient Characteristics Survey (PCS). It contains demographic, medical, and social information about patients receiving mental health services across different programs.

File name: Patient_Characteristics_Survey__PCS___2019.csv
Size: 196,102 rows × 76 columns
Format: CSV (originally provided in Excel, converted for analysis)


🔑 Key Variables
Sociodemographic data: Age Group, Sex, Transgender, Sexual Orientation, Hispanic Ethnicity, Race, Living Situation, Household Composition, Education Status, Employment Status
Mental health and developmental disorders: Mental Illness, Serious Mental Illness, Intellectual Disability, Autism Spectrum, Other Developmental Disability
Substance use disorders: Alcohol Related Disorder, Drug Substance Disorder, Opioid Related Disorder
Chronic medical conditions: Diabetes, Obesity, Stroke, Cancer, Hyperlipidemia, High Blood Pressure, Heart Attack, Pulmonary Asthma
Lifestyle and habits: Cannabis Recreational Use, Smokes, Received Smoking Counseling
Social and insurance coverage: Medicaid Insurance, Medicare Insurance, Private Insurance, Public Assistance Cash Program

🎯Target: Predict presence/absence of mental illness


Source: NYS OMH PCS 2019 (Patient Characteristics Survey)
Files included:
- Patient_Characteristics_Survey__PCS___2019.csv
- NYSOMH_PCS2019_DataDictionary.pdf
- NYSOMH_PCS2019_Overview.pdf




------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


3 . Technologies Used 
Programming Language: Python 3.x
Data Analysis & ML: pandas, NumPy, scikit-learn, imbalanced-learn (SMOTE)
Ml model : Logistic Regression, Random Forest, Decision Tree
Visualization: Matplotlib, Seaborn
Model Deployment: Streamlit
Environment: Jupyter Notebook, GitHub
Version Control: Git

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Authors
 
- Lyna Mouhoubi– MSc Data Engineering - DSTI School of engineering – [GitHub](https://github.com/lyna-username)
- Nirusa Jegaseelan – MSc Data Analytics - DSTI School of engineering – [GitHub](https://github.com/Nirusa04)
- Amina Rabehi – MSc Data Analytics - DSTI School of engineering – [GitHub](https://github.com/nom-username)
- Arjun Chintham – MSc Data Science - DSTI School of engineering – [GitHub](https://github.com/nom-username)


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

