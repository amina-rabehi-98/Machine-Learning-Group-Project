import pandas as pd
import pickle
import streamlit as st
import numpy as np

st.set_page_config(page_title="Mental Illness Prediction App", layout="centered")

st.title("Mental Illness Prediction App")
st.write("""
Upload your CSV file and get predictions on mental illness based on a **Random Forest** model.
""")

st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

# ✅ Charger le modèle Random Forest
with open("C:/Users/nirus/Documents/ML_app_2/ment_ill_clf.pkl", "rb") as f:
    rf_model = pickle.load(f)




# Placeholder à remplacer par NaN
placeholder_dict = {
    'Age Group': ['UNKNOWN'], 'Sex': ['UNKNOWN'], 'Transgender': ['UNKNOWN', "CLIENT DIDN'T ANSWER"],
    'Sexual Orientation': ['CLIENT DID NOT ANSWER', 'UNKNOWN'], 'Hispanic Ethnicity': ['UNKNOWN'],
    'Race': ['UNKNOWN RACE'], 'Living Situation': ['UNKNOWN'], 'Household Composition': ['UNKNOWN'],
    'Religious Preference': ['DATA NOT AVAILABLE'], 'Veteran Status': ['UNKNOWN'],
    'Employment Status': ['UNKNOWN EMPLOYMENT STATUS'], 'Number Of Hours Worked Each Week': ['UNKNOWN EMPLOYMENT HOURS'],
    'Education Status': ['UNKNOWN'], 'Special Education Services': ['UNKNOWN'], 'Intellectual Disability': ['UNKNOWN'],
    'Autism Spectrum': ['UNKNOWN'], 'Other Developmental Disability': ['UNKNOWN'],
    'Alcohol Related Disorder': ['UNKNOWN'], 'Drug Substance Disorder': ['UNKNOWN'],
    'Opioid Related Disorder': ['UNKNOWN'], 'Mobility Impairment Disorder': ['UNKNOWN'],
    'Hearing Impairment': ['UNKNOWN'], 'Visual Impairment': ['UNKNOWN'], 'Speech Impairment': ['UNKNOWN'],
    'Hyperlipidemia': ['UNKNOWN'], 'High Blood Pressure': ['UNKNOWN'], 'Diabetes': ['UNKNOWN'],
    'Obesity': ['UNKNOWN'], 'Heart Attack': ['UNKNOWN'], 'Stroke': ['UNKNOWN'], 'Other Cardiac': ['UNKNOWN'],
    'Pulmonary Asthma': ['UNKNOWN'], 'Alzheimer or Dementia': ['UNKNOWN'], 'Kidney Disease': ['UNKNOWN'],
    'Liver Disease': ['UNKNOWN'], 'Endocrine Condition': ['UNKNOWN'], 'Neurological Condition': ['UNKNOWN'],
    'Traumatic Brain Injury': ['UNKNOWN'], 'Joint Disease': ['UNKNOWN'], 'Cancer': ['UNKNOWN'],
    'Other Chronic Med Condition': ['UNKNOWN'], 'No Chronic Med Condition': ['UNKNOWN'],
    'Cannabis Recreational Use': ['UNKNOWN'], 'Cannabis Medicinal Use': ['UNKNOWN'], 'Smokes': ['UNKNOWN'],
    'Received Smoking Medication': ['UNKNOWN'], 'Received Smoking Counseling': ['UNKNOWN'],
    'Serious Mental Illness': ['UNKNOWN'], 'Alcohol 12m Service': ['UNKNOWN'], 'Opioid 12m Service': ['UNKNOWN'],
    'Drug/Substance 12m Service': ['UNKNOWN'], 'Principal Diagnosis Class': ['UNKNOWN'],
    'Additional Diagnosis Class': ['UNKNOWN'], 'SSI Cash Assistance': ['UNKNOWN'],
    'SSDI Cash Assistance': ['UNKNOWN'], 'Veterans Disability Benefits': ['UNKNOWN'],
    'Veterans Cash Assistance': ['UNKNOWN'], 'Public Assistance Cash Program': ['UNKNOWN'],
    'Other Cash Benefits': ['UNKNOWN'], 'Medicaid and Medicare Insurance': ['UNKNOWN'], 'No Insurance': ['UNKNOWN'],
    'Medicaid Insurance': ['UNKNOWN'], 'Medicaid Managed Insurance': ['UNKNOWN'], 'Medicare Insurance': ['UNKNOWN'],
    'Private Insurance': ['UNKNOWN'], 'Child Health Plus Insurance': ['UNKNOWN'], 'Other Insurance': ['UNKNOWN'],
    'Criminal Justice Status': ['UNKNOWN']
}

# One-Hot Encoding
one_hot = [
    "Region Served", "Sex", "Hispanic Ethnicity", "Race", "Transgender",
    "Special Education Services", "Household Composition", "Sexual Orientation",
    "Religious Preference", "Additional Diagnosis Class",
    "Principal Diagnosis Class", "Program Category"
]

# Ordinal Encoding
ordinal_cols = {
    "Living Situation": ["INSTITUTIONAL SETTING", "OTHER LIVING SITUATION", "PRIVATE RESIDENCE"],
    "Education Status": ["NO FORMAL EDUCATION", "PRE-K TO FIFTH GRADE", "OTHER", "MIDDLE SCHOOL TO HIGH SCHOOL", "SOME COLLEGE", "COLLEGE OR GRADUATE DEGREE"],
    "Employment Status": ["NOT IN LABOR FORCE:UNEMPLOYED AND NOT LOOKING FOR WORK", "UNEMPLOYED, LOOKING FOR WORK", "NON-PAID/VOLUNTEER", "EMPLOYED"],
    "Age Group": ["CHILD", "ADULT"],
    "Number Of Hours Worked Each Week": ["NOT APPLICABLE", "01-14 HOURS", "15-34 HOURS", "35 HOURS OR MORE"]
}



# Final features attendues par le modèle
final_columns = ['Age Group', 'Living Situation', 'Employment Status',
       'Number Of Hours Worked Each Week', 'Education Status', 'Intellectual Disability',
       'Other Developmental Disability', 'Alcohol Related Disorder',
       'Drug Substance Disorder', 'Mobility Impairment Disorder',
       'Hearing Impairment', 'Visual Impairment', 'Hyperlipidemia',
       'High Blood Pressure', 'Diabetes', 'Obesity', 'Heart Attack', 'Stroke',
       'Other Cardiac', 'Pulmonary Asthma', 'Alzheimer or Dementia',
       'Kidney Disease', 'Liver Disease', 'Endocrine Condition',
       'Neurological Condition', 'Traumatic Brain Injury', 'Joint Disease',
       'Cancer', 'Other Chronic Med Condition', 'No Chronic Med Condition',
       'Unknown Chronic Med Condition', 'Cannabis Recreational Use',
       'Cannabis Medicinal Use', 'Smokes', 'Received Smoking Medication',
       'Received Smoking Counseling', 'Serious Mental Illness',
       'Alcohol 12m Service', 'Drug/Substance 12m Service',
       'SSI Cash Assistance', 'SSDI Cash Assistance',
       'Veterans Cash Assistance', 'Public Assistance Cash Program',
       'Other Cash Benefits', 'Medicaid and Medicare Insurance',
       'No Insurance', 'Unknown Insurance Coverage', 'Medicaid Insurance',
       'Medicaid Managed Insurance', 'Medicare Insurance', 'Private Insurance',
       'Child Health Plus Insurance', 'Region Served_LONG ISLAND REGION',
       'Region Served_NEW YORK CITY REGION', 'Region Served_WESTERN REGION',
       'Sex_MALE', 'Hispanic Ethnicity_YES, HISPANIC/LATINO',
       'Race_WHITE ONLY', 'Transgender_YES, TRANSGENDER',
       'Special Education Services_NOT APPLICABLE',
       'Household Composition_LIVES ALONE',
       'Household Composition_NOT APPLICABLE',
       'Sexual Orientation_LESBIAN OR GAY', 'Sexual Orientation_OTHER',
       'Sexual Orientation_STRAIGHT OR HETEROSEXUAL',
       'Religious Preference_I CONSIDER MYSELF SPIRITUAL, BUT NOT RELIGIOUS',
       'Religious Preference_I DO NOT HAVE A FORMAL RELIGION, NOR AM I A SPIRITUAL PERSON',
       'Additional Diagnosis Class_NO ADDITIONAL DIAGNOSIS',
       'Additional Diagnosis Class_NOT MI - DEVELOPMENTAL DISORDERS',
       'Additional Diagnosis Class_NOT MI - ORGANIC MENTAL DISORDER',
       'Additional Diagnosis Class_SUBSTANCE-RELATED AND ADDICTIVE DISORDERS',
       'Principal Diagnosis Class_NOT MI - DEVELOPMENTAL DISORDERS',
       'Principal Diagnosis Class_NOT MI - ORGANIC MENTAL DISORDER',
       'Principal Diagnosis Class_NOT MI - OTHER',
       'Principal Diagnosis Class_SUBSTANCE-RELATED AND ADDICTIVE DISORDERS',
       'Program Category_INPATIENT', 'Program Category_OUTPATIENT',
       'Program Category_RESIDENTIAL', 'Program Category_SUPPORT'
] 

# --- Traitement et prédiction ---
# --- Traitement et prédiction ---
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file, delimiter=';', on_bad_lines='skip')
    st.write(input_df.head())

    if "Three Digit Residence Zip Code" in input_df.columns:
        input_df["Three Digit Residence Zip Code"] = (
            input_df["Three Digit Residence Zip Code"]
            .astype(str)
            .str.extract(r'(\d{3})')[0]
        )
        input_df["Three Digit Residence Zip Code"] = pd.to_numeric(
            input_df["Three Digit Residence Zip Code"], errors='coerce'
        ).fillna(0).astype(int)

    # Nettoyage des placeholders
    for col, values in placeholder_dict.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].replace(values, np.nan)

    # Imputation
    for col in input_df.select_dtypes(include='object').columns:
        input_df[col].fillna(input_df[col].mode()[0], inplace=True)
    for col in input_df.select_dtypes(include=np.number).columns:
        input_df[col].fillna(input_df[col].median(), inplace=True)

    # YES/NO → 1/0
    for col in input_df.columns:
        values = input_df[col].astype(str).str.strip().str.upper()
        input_df[col] = values.replace({
            "YES": 1,
            "NO": 0,
            "NOT APPLICABLE": 0
        })

    # TRUE/FALSE → 1/0
    for col in input_df.columns:
        values = input_df[col].astype(str).str.strip().str.lower()
        if set(values.dropna().unique()).issubset({"true", "false"}):
            input_df[col] = values.map({"true": 1, "false": 0})

    # Ordinal Encoding
    for col, categories in ordinal_cols.items():
        if col in input_df.columns:
            mapping = {cat: i for i, cat in enumerate(categories)}
            input_df[col] = input_df[col].map(mapping)

    # One-Hot Encoding
    input_df = pd.get_dummies(input_df, columns=one_hot, drop_first=True)

    # Supprimer colonnes inutiles
    cols_to_drop = ['Survey Year', 'Preferred Language']
    input_df.drop(columns=[col for col in cols_to_drop if col in input_df.columns], inplace=True)

    # Colonnes manquantes
    for col in final_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Réordonner les colonnes
    input_df = input_df[final_columns]

    # Nettoyage final
    input_df = input_df.fillna(0)
    input_df = input_df.astype(float)

    # Vérification finale
    non_numeric_cols = input_df.select_dtypes(include='object').columns
    if len(non_numeric_cols) > 0:
        st.error(f"Erreur : les colonnes suivantes ne sont pas numériques : {list(non_numeric_cols)}")
        st.stop()

    #  Prédiction avec Random Forest
    prediction = rf_model.predict(input_df)
    input_df['Prediction'] = prediction

    st.subheader("Prédictions :")
    st.write(input_df[['Prediction']])

else:
    st.info("Veuillez importer un fichier CSV pour commencer.")