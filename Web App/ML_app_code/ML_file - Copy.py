import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score

# --- Configuration ---
warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

# --- Chargement du dataset ---
df = pd.read_csv("c:/Users/nirus/Documents/ML_app_2/Patient_Characteristics_Survey__PCS___2019.csv", sep=';', on_bad_lines='skip')

df = df.drop(columns=['Survey Year', 'Three Digit Residence Zip Code'])

print("\nColumns after dropping:")
print(df.columns.tolist())


df.dtypes.value_counts()



# List of placeholder values that actually mean "missing"
placeholders = [
    'UNKNOWN', 'UNKNOWN RACE', 'UNKNOWN EMPLOYMENT STATUS',
    'UNKNOWN EMPLOYMENT HOURS', 'UNKNOWN INSURANCE COVERAGE',
    'UNKNOWN CHRONIC MED CONDITION', 'DATA NOT AVAILABLE',
    'CLIENT DID NOT ANSWER', "CLIENT DIDN'T ANSWER"
]
df.replace(placeholders, np.nan, inplace=True)
#on les remplace par NaN


## II- EDA

#Target vizualisation

df['Mental Illness'] = df['Mental Illness'].where(
    df['Mental Illness'].isna(),
    df['Mental Illness'].str.upper().str.strip()
)


df["Mental Illness"] = df["Mental Illness"].map({
    'YES': 1,
    'NO': 0,
    '1': 1,
    '0': 0
})

df["Mental Illness"] = df["Mental Illness"].astype('Int64')


#Correction de potentiel incoherence 
mask = (
    (df['Mental Illness'] == 1) &
    (df['Serious Mental Illness'] != 'YES') &
    (df['Principal Diagnosis Class'] != 'MENTAL ILLNESS') &
    (df['Additional Diagnosis Class'] != 'MENTAL ILLNESS')
)

df.loc[mask, 'Mental Illness'] = 0
print(f"{mask.sum()} valeurs corrigées à 0 dans 'Mental Illness' car non justifiées")


df['Mental Illness'].value_counts(dropna=False)

# Vizualization de nos données qualitatifs/ catégorielles (object)

#rendre en MAJ sans toucher les NaN
for col in df.select_dtypes('object'):
    df[col] = df[col].where(df[col].isna(), df[col].str.upper())



# Création d'ensemble thématiques :

#ensemble socio-demographique
socio_cols = [
    'Age Group', 'Sex', 'Transgender', 'Sexual Orientation',
    'Hispanic Ethnicity', 'Race', 'Preferred Language',
    'Religious Preference', 'Region Served'
    ]

#ensemble travail/etude
stuwork_cols = [ 'Education Status', 'Special Education Services',
    'Employment Status', 'Number Of Hours Worked Each Week' ]


#Ensemble troubles neuro
neurodev_cols =['Intellectual Disability', 'Autism Spectrum',
    'Other Developmental Disability', 'Neurological Condition',
    'Speech Impairment', 'Hearing Impairment', 'Visual Impairment',
    'Mobility Impairment Disorder', 'Traumatic Brain Injury']


#ensemble addictions
addiction_cols =['Alcohol Related Disorder', 'Drug Substance Disorder',
    'Opioid Related Disorder' ,'Cannabis Medicinal Use', 'Smokes']



#ensemble pathologie
chronic_disease_cols = [
    'Obesity', 'Diabetes', 'Cancer'
]

#statut judiciaire
justice_cols = ['Criminal Justice Status']



# III- HANDLING MISSING VALUES

#Eviter les bug
df.replace(['NAN', 'NaN', 'nan', 'None', 'NONE', '[nan]'], np.nan, inplace=True)


#ETAPE 1 : Imputation simple par le mode (très peu de NaN)
cols_faible_nan = ['Sex', 'Age Group', 'Preferred Language']
for col in cols_faible_nan:
    mode_val = df[col].mode(dropna=True)[0]
    df[col] = df[col].fillna(mode_val)
    print(f"{col} imputé par mode : {mode_val}")


# Étape 2 : Fonction d’imputation par regroupement logique
def imputer_par_groupe(df, col, group_cols):
    df_copy = df.copy()
    # Créer une table avec les valeurs les plus fréquentes par groupe
    group_df = df_copy.dropna(subset=[col]).groupby(group_cols)[col].agg(lambda x: x.value_counts().index[0]).reset_index()
    group_df.rename(columns={col: 'valeur_majoritaire'}, inplace=True)

    # Joindre pour récupérer la valeur imputée
    df_copy = df_copy.merge(group_df, on=group_cols, how='left')

    # Remplacer les NaN avec la valeur du groupe
    mask = df_copy[col].isna() & df_copy['valeur_majoritaire'].notna()
    df_copy.loc[mask, col] = df_copy.loc[mask, 'valeur_majoritaire']

    # Nettoyage
    df_copy.drop(columns='valeur_majoritaire', inplace=True)
    print(f"{col} imputé par groupe {group_cols} — {mask.sum()} valeurs remplies, reste {df_copy[col].isna().sum()} NaN")
    return df_copy


# Étape 3 : Colonnes imputées par groupe logique
grouped_imputations = {
    'Sexual Orientation': ['Age Group', 'Sex', 'Region Served'],
    'Employment Status': ['Education Status', 'Age Group'],
    'Obesity': ['Age Group', 'Sex', 'Diabetes'],
    'Transgender': ['Sex'],
    'Race': ['Region Served', 'Hispanic Ethnicity'],
    'Cannabis Medicinal Use': ['Age Group', 'Alcohol Related Disorder'],
    'Religious Preference': ['Age Group', 'Sex'],
    'Living Situation': ['Region Served'],
    'Household Composition': ['Living Situation', 'Age Group'],
}

for col, grp in grouped_imputations.items():
    df = imputer_par_groupe(df, col, grp)



# ÉTAPE 4 - Imputation par cascade intelligente
def imputation_en_cascade(df, col, groupings):
    df_copy = df.copy()
    total_filled = 0
    for group_cols in groupings:
        table = df_copy.dropna(subset=[col]).groupby(group_cols)[col].agg(lambda x: x.value_counts().index[0]).reset_index()
        table.rename(columns={col: 'valeur_majoritaire'}, inplace=True)
        df_copy = df_copy.merge(table, on=group_cols, how='left')
        mask = df_copy[col].isna() & df_copy['valeur_majoritaire'].notna()
        df_copy.loc[mask, col] = df_copy.loc[mask, 'valeur_majoritaire']
        filled_now = mask.sum()
        total_filled += filled_now
        df_copy.drop(columns='valeur_majoritaire', inplace=True)
        print(f"{col} imputé par groupe {group_cols} — {filled_now} valeurs remplies, reste {df_copy[col].isna().sum()} NaN")
        if df_copy[col].isna().sum() == 0:
            break
    return df_copy


# ETAPE 5 : Colonnes à traiter par cascade élargie
cascading_imputations = {
    'Employment Status': [['Education Status', 'Age Group'], ['Age Group']],
    'Obesity': [['Age Group', 'Sex', 'Diabetes'], ['Age Group', 'Sex'], ['Age Group']],
    'Race': [['Region Served', 'Hispanic Ethnicity'], ['Region Served']],
    'Cannabis Medicinal Use': [['Age Group', 'Alcohol Related Disorder'], ['Age Group']]
}

for col, cascades in cascading_imputations.items():
    df = imputation_en_cascade(df, col, cascades)


def imputation_en_cascade(df, col, groupings):
    df_copy = df.copy()
    total_filled = 0
    for group_cols in groupings:
        grouped = df_copy[df_copy[col].notna()].groupby(group_cols)[col].agg(pd.Series.mode).reset_index()
        grouped.rename(columns={col: 'valeur_majoritaire'}, inplace=True)
        df_copy = df_copy.merge(grouped, on=group_cols, how='left')
        mask = df_copy[col].isna() & df_copy['valeur_majoritaire'].notna()
        df_copy.loc[mask, col] = df_copy.loc[mask, 'valeur_majoritaire']
        filled_now = mask.sum()
        total_filled += filled_now
        df_copy.drop(columns='valeur_majoritaire', inplace=True)
        print(f"{col} imputé par groupe {group_cols} — {filled_now} valeurs remplies, reste {df_copy[col].isna().sum()} NaN")
        if df_copy[col].isna().sum() == 0:
            break
    return df_copy


cascading_imputations = {
    'Hispanic Ethnicity': [['Race', 'Region Served'], ['Region Served']],
    'Education Status': [['Age Group', 'Employment Status'], ['Age Group']],
    'Veteran Status': [['Age Group', 'Sex'], ['Age Group']],
    'Special Education Services': [['Education Status', 'Age Group'], ['Age Group']],
    'Number Of Hours Worked Each Week': [['Employment Status'], ['Age Group']],
    'Intellectual Disability': [['Age Group', 'Sex']],
    'Autism Spectrum': [['Age Group', 'Sex']],
    'Other Developmental Disability': [['Age Group', 'Sex']],
    'Alcohol Related Disorder': [['Age Group', 'Sex']],
    'Drug Substance Disorder': [['Age Group', 'Sex']],
    'Opioid Related Disorder': [['Age Group', 'Sex']],
    'Mobility Impairment Disorder': [['Age Group', 'Sex']],
    'Hearing Impairment': [['Age Group', 'Sex']],
    'Visual Impairment': [['Age Group', 'Sex']],
    'Speech Impairment': [['Age Group', 'Sex']],
    'Hyperlipidemia': [['Age Group', 'Sex']],
    'High Blood Pressure': [['Age Group', 'Sex']],
    'Diabetes': [['Age Group', 'Sex']],
    'Heart Attack': [['Age Group', 'Sex']],
    'Stroke': [['Age Group', 'Sex']],
    'Other Cardiac': [['Age Group', 'Sex']],
    'Pulmonary Asthma': [['Age Group', 'Sex']],
    'Alzheimer or Dementia': [['Age Group', 'Sex']],
    'Kidney Disease': [['Age Group', 'Sex']],
    'Liver Disease': [['Age Group', 'Sex']],
    'Endocrine Condition': [['Age Group', 'Sex']],
    'Neurological Condition': [['Age Group', 'Sex']],
    'Traumatic Brain Injury': [['Age Group', 'Sex']],
    'Joint Disease': [['Age Group', 'Sex']],
    'Cancer': [['Age Group', 'Sex']],
    'Other Chronic Med Condition': [['Age Group', 'Sex']],
    'No Chronic Med Condition': [['Age Group', 'Sex']],
    'Cannabis Recreational Use': [['Age Group', 'Drug Substance Disorder'], ['Age Group']],
    'Smokes': [['Age Group', 'Drug Substance Disorder'], ['Age Group']],
    'Received Smoking Medication': [['Age Group', 'Smokes']],
    'Received Smoking Counseling': [['Age Group', 'Smokes']],
    'Alcohol 12m Service': [['Age Group', 'Alcohol Related Disorder']],
    'Opioid 12m Service': [['Age Group', 'Opioid Related Disorder']],
    'Drug/Substance 12m Service': [['Age Group', 'Drug Substance Disorder']],
    'SSI Cash Assistance': [['Age Group', 'Employment Status']],
    'SSDI Cash Assistance': [['Age Group', 'Employment Status']],
    'Veterans Disability Benefits': [['Age Group', 'Veteran Status']],
    'Veterans Cash Assistance': [['Age Group', 'Veteran Status']],
    'Public Assistance Cash Program': [['Age Group', 'Employment Status']],
    'Other Cash Benefits': [['Age Group', 'Employment Status']],
    'Medicaid and Medicare Insurance': [['Age Group', 'Employment Status']],
    'No Insurance': [['Age Group', 'Employment Status']],
    'Medicaid Insurance': [['Age Group', 'Employment Status']],
    'Medicaid Managed Insurance': [['Age Group', 'Employment Status']],
    'Medicare Insurance': [['Age Group', 'Employment Status']],
    'Private Insurance': [['Age Group', 'Employment Status']],
    'Child Health Plus Insurance': [['Age Group', 'Employment Status']],
    'Other Insurance': [['Age Group', 'Employment Status']],
    'Criminal Justice Status': [['Age Group', 'Sex']]
}

# Application des imputations
for col, cascades in cascading_imputations.items():
    df = imputation_en_cascade(df, col, cascades)


#Addictions for Children
for col in addiction_cols :
    df.loc[(df['Age Group'] == 'CHILD') & (df[col].isna()), col] = 'NO'
#Work
df.loc[(df['Age Group'] == 'CHILD') &
(df['Employment Status'].isna()), 'Employment Status'] = 'NOT APPLICABLE'
df.loc[(df['Age Group'] == 'CHILD') &
(df['Number Of Hours Worked Each Week'].isna()), 'Number Of Hours Worked Each Week'] = 'NOT APPLICABLE'

#Social help
cols_aides = [
    'SSI Cash Assistance', 'SSDI Cash Assistance', 'Veterans Disability Benefits',
    'Veterans Cash Assistance', 'Public Assistance Cash Program', 'Other Cash Benefits'
]
for col in cols_aides:
    df.loc[(df['Age Group'] == 'CHILD') &
    (df[col].isna()), col] = 'NO'

#Assurances
df.loc[(df['Age Group'] == 'CHILD') &
(df['Child Health Plus Insurance'].isna()), 'Child Health Plus Insurance'] = 'YES'
assurances = [
    'Medicaid and Medicare Insurance', 'No Insurance', 'Medicaid Insurance',
    'Medicaid Managed Insurance', 'Medicare Insurance', 'Private Insurance', 'Other Insurance'
]

for col in assurances:
    df.loc[(df['Age Group'] == 'CHILD') &
     (df[col].isna()), col] = 'NO'

#identity

df.loc[(df['Age Group'] == 'CHILD') &
  (df['Transgender'].isna()), 'Transgender'] = 'NO, NOT TRANSGENDER'

df.loc[(df['Age Group'] == 'CHILD') &
  (df['Sexual Orientation'].isna()), 'Sexual Orientation'] = 'OTHER'

df.loc[(df['Age Group'] == 'CHILD') &
  (df['Religious Preference'].isna()), 'Religious Preference'] = 'I DO NOT HAVE A FORMAL RELIGION, NOR AM I A SPIRITUAL PERSON'





for col in df.select_dtypes(include='bool').columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(False, inplace=True)
        print(f"'{col}' (bool) filled with False")



#utiliser les autres colonnes
df['Serious Mental Illness'] = df['Serious Mental Illness'].astype(str).str.upper().str.strip()
df['Principal Diagnosis Class'] = df['Principal Diagnosis Class'].astype(str).str.upper().str.strip()
df['Additional Diagnosis Class'] = df['Additional Diagnosis Class'].astype(str).str.upper().str.strip()

# Imputer par 1
df.loc[
    df['Mental Illness'].isna() & (
        (df['Principal Diagnosis Class'] == 'MENTAL ILLNESS') |
        (df['Additional Diagnosis Class'] == 'MENTAL ILLNESS') |
        (df['Serious Mental Illness'] == 'YES')
    ),
    'Mental Illness'
] = 1

# Imputer par NO
df.loc[
    df['Mental Illness'].isna() & (
        df['Principal Diagnosis Class'].str.startswith('NOT MI') |
        df['Additional Diagnosis Class'].str.startswith('NOT MI') |
        (df['Serious Mental Illness'] == 'NO')
    ),
    'Mental Illness'
] = 0 

# Pour les derniers cas ambigus, on choisit par défaut NO
df['Mental Illness'] = df['Mental Illness'].fillna(0)



print(df['Mental Illness'].value_counts(dropna=False))


#Drop 'Prefered Langage' Columns
if 'Preferred Language' in df.columns:
    df.drop(columns=['Preferred Language'], inplace=True)
    print("Preferred Language Deleted")


# Liste des colonnes concernées par les valeurs multiples mal encodées (ex: '[NO, YES]')
colonnes_multiples = [
    'SSI Cash Assistance', 'SSDI Cash Assistance', 'Veterans Disability Benefits',
    'Veterans Cash Assistance', 'Public Assistance Cash Program', 'Other Cash Benefits'
]

# Fonction de nettoyage
for col in colonnes_multiples:
    df[col] = df[col].apply(lambda x: x if isinstance(x, str) and x in ['YES', 'NO'] else np.nan)
    print(f"{col} — Valeurs corrigées")

# Imputation simple par le mode 
for col in colonnes_multiples:
    mode_val = df[col].mode(dropna=True)[0]
    df[col] = df[col].fillna(mode_val)  # Pas de inplace ici
    print(f"{col} rempli par le mode : {mode_val}")


#Eviter les bug
df.replace(['NAN', 'NaN', 'nan', 'None', 'NONE', '[nan]'], np.nan, inplace=True)


# Harmonisation finale de cohérence entre les 3 colonnes de diagnostic

# Si Principal = MENTAL ILLNESS ou Additional = MENTAL ILLNESS → Mental = 1
df.loc[
    (df['Mental Illness'].isna()) &
    (
        (df['Principal Diagnosis Class'] == 'MENTAL ILLNESS') |
        (df['Additional Diagnosis Class'] == 'MENTAL ILLNESS')
    ),
    'Mental Illness'
] = 1

# Si Principal commence par NOT MI ou Additional commence par NOT MI → Mental = 0
df.loc[
    (df['Mental Illness'].isna()) &
    (
        df['Principal Diagnosis Class'].str.startswith('NOT MI', na=False) |
        df['Additional Diagnosis Class'].str.startswith('NOT MI', na=False)
    ),
    'Mental Illness'
] = 0

# Si Mental = 1 et Serious est NaN → Serious = YES
df.loc[(df['Mental Illness'] == 1) & (df['Serious Mental Illness'].isna()), 'Serious Mental Illness'] = 'YES'
# Si Mental = 0 et Serious est NaN → Serious = NO
df.loc[(df['Mental Illness'] == 0) & (df['Serious Mental Illness'].isna()), 'Serious Mental Illness'] = 'NO'

# Si Mental = 1 et Principal est NaN → Principal = MENTAL ILLNESS
df.loc[(df['Mental Illness'] == 1) & (df['Principal Diagnosis Class'].isna()), 'Principal Diagnosis Class'] = 'MENTAL ILLNESS'
# Si Mental = 0 et Principal est NaN → Principal = NOT MI - OTHER
df.loc[(df['Mental Illness'] == 0) & (df['Principal Diagnosis Class'].isna()), 'Principal Diagnosis Class'] = 'NOT MI - OTHER'

# Si Mental = 1 et Additional est NaN → Additional = MENTAL ILLNESS
df.loc[(df['Mental Illness'] == 1) & (df['Additional Diagnosis Class'].isna()), 'Additional Diagnosis Class'] = 'MENTAL ILLNESS'
# Si Mental = 0 et Additional est NaN → Additional = NO ADDITIONAL DIAGNOSIS
df.loc[(df['Mental Illness'] == 0) & (df['Additional Diagnosis Class'].isna()), 'Additional Diagnosis Class'] = 'NO ADDITIONAL DIAGNOSIS'



#Check missing values
print(df.isna().sum())

print(df['Mental Illness'].value_counts(dropna=False))



for col in df.columns:
    print(f"\n{col}:\n", df[col].value_counts(dropna=False))

df_cleaned = df


df_cleaned.shape



# IV - FEATURE ENGINEERING
import pandas as pd

# --- One-Hot Encoding ---
one_hot = [
    "Region Served",
    "Sex",
    "Hispanic Ethnicity",
    "Race",
    "Transgender",
    "Special Education Services",
    "Household Composition",
    "Sexual Orientation",
    "Religious Preference",
    "Additional Diagnosis Class",
    "Principal Diagnosis Class",
    "Program Category",
]

# One-Hot Encoding
df_OneHot = pd.get_dummies(df_cleaned, columns=one_hot, drop_first=True)
df_OneHot


print(df_OneHot.isna().sum())

print(df_OneHot.columns)

# --- Ordinal encoding ---
# Filtrer les lignes où Education Status est 'OTHER'
df_other = df[df["Education Status"] == "OTHER"]

# Liste des colonnes à visualiser
cols_to_plot = ["Age Group", "Sex", "Region Served", "Employment Status", "Special Education Services"]

# Créer un subplot pour chaque colonne
plt.figure(figsize=(16, 14))
for i, col in enumerate(cols_to_plot, 1):
    plt.subplot(3, 2, i)
    sns.countplot(data=df_other, x=col, order=df_other[col].value_counts().index, palette="Set2")
    plt.title(f"Distribution de {col} pour Education Status = 'OTHER'")
    plt.xticks(rotation=45)
    plt.tight_layout()
plt.show()

# les personnes ayant répondus "other" dans "education stautus" sont en majorité des adultes, non actif, vivant dans la region de NY dont le speciale education services n'est pas applicable.




import matplotlib.pyplot as plt
import seaborn as sns

# Filtrer uniquement les personnes "NOT IN LABOR FORCE:UNEMPLOYED AND NOT LOOKING FOR WORK"
df_unemployed = df[
    df["Employment Status"].str.upper() == "NOT IN LABOR FORCE:UNEMPLOYED AND NOT LOOKING FOR WORK"
]

# Créer un graphique en barres avec Seaborn
plt.figure(figsize=(12, 6))
sns.countplot(
    data=df_unemployed,
    x="Education Status",
    order=df_unemployed["Education Status"].value_counts().index,
    palette="Set3"
)

# Personnaliser le graphique
plt.title("Nombre de personnes 'NOT IN LABOR FORCE' par niveau d'éducation", fontsize=14)
plt.xlabel("Niveau d'éducation")
plt.ylabel("Nombre de personnes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# a titre de comparaison, les personnes non actif a un niveau d'education middle school to highschool. donc other sera juste avant cette valeur


# Filtrer les lignes où Education Status est 'OTHER'
df_other = df[df["Education Status"] == "NO FORMAL EDUCATION"]

# Liste des colonnes à visualiser
cols_to_plot = ["Age Group", "Sex", "Region Served", "Employment Status", "Special Education Services"]

# Créer un subplot pour chaque colonne
plt.figure(figsize=(16, 14))
for i, col in enumerate(cols_to_plot, 1):
    plt.subplot(3, 2, i)
    sns.countplot(data=df_other, x=col, order=df_other[col].value_counts().index, palette="Set2")
    plt.title(f"Distribution de {col} pour Education Status = 'NO FORMAL EDUCATION'")
    plt.xticks(rotation=45)
    plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Filtrer uniquement les personnes avec "NO FORMAL EDUCATION"
df_no_education = df[
    df["Education Status"].str.upper() == "NO FORMAL EDUCATION"
]

# Créer un graphique en barres avec Seaborn
plt.figure(figsize=(12, 6))
sns.countplot(
    data=df_no_education,
    x="Employment Status",
    order=df_no_education["Employment Status"].value_counts().index,
    palette="Set2"
)

# Personnaliser le graphique
plt.title("Distribution of Employment Status among people without formal education", fontsize=14)
plt.xlabel("Employment Status")
plt.ylabel("Nombre de personnes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Copier le DataFrame précédant
df_Ordinal1 = df_OneHot.copy()

# --- Ordinal Encoding ---
ordinal_cols = {
    "Living Situation": [
        "INSTITUTIONAL SETTING",  # less autonomous
        "OTHER LIVING SITUATION",  # unknown / temporary
        "PRIVATE RESIDENCE"   # stable and autonomous
    ],
    "Education Status": [
        "NO FORMAL EDUCATION", #jamais scolarisé
        "PRE-K TO FIFTH GRADE",# primaire
        "OTHER", # inconnu / atypique => regarder graphique précédent
        "MIDDLE SCHOOL TO HIGH SCHOOL", # secondaire
        "SOME COLLEGE", # post secondaire
        "COLLEGE OR GRADUATE DEGREE",  # diplôme universitaire

    ],
    "Employment Status": [
        "NOT IN LABOR FORCE:UNEMPLOYED AND NOT LOOKING FOR WORK",# non actif
        "UNEMPLOYED, LOOKING FOR WORK",# non employé et cherche a être actif
        "NON-PAID/VOLUNTEER", #actif non payé
        "EMPLOYED" #actif payé
    ],
    "Age Group": [
        "CHILD",
        "ADULT"
    ],
    "Number Of Hours Worked Each Week": [
        "NOT APPLICABLE", #0
        "01-14 HOURS",
        "15-34 HOURS",
        "35 HOURS OR MORE"
    ]
}

# Appliquer l'encodage ordinal dans le nouveau DataFrame
for col, categories in ordinal_cols.items():
    mapping = {cat: i for i, cat in enumerate(categories)}
    df_Ordinal1[col] = df_Ordinal1[col].map(mapping)

df_Ordinal1


print(df_Ordinal1.isna().sum())

df_Ordinal1["Education Status"].unique()

print(df_Ordinal1["Medicaid Managed Insurance"].unique())


#nous regroupons les "not applicable" avec "no" dans la colonne "Medicaid Managed Insurance"

df_Ordinal1["Medicaid Managed Insurance"] = df_Ordinal1["Medicaid Managed Insurance"].replace("NOT APPLICABLE", "NO")


#convertir les TRUE FALSE en 0 et 1
df_TrueFalse = df_Ordinal1.copy()  # ou ton DataFrame cible

for col in df_TrueFalse.columns:
    col_str = df_TrueFalse[col].astype(str).str.strip().str.lower()
    unique_vals = set(col_str.dropna().unique())
    if unique_vals.issubset({"true", "false"}):
        df_TrueFalse[col] = col_str.map({"true": 1, "false": 0})

df_TrueFalse


#convertir les yes no
dfYesNo = df_TrueFalse.copy()  # ou ton DataFrame cible

for col in dfYesNo.columns:
    col_str = dfYesNo[col].astype(str).str.strip().str.upper()
    unique_vals = set(col_str.dropna().unique())
    if unique_vals.issubset({"YES", "NO"}):
        dfYesNo[col] = col_str.map({"YES": 1, "NO": 0})

dfYesNo


# Voir les colonnes en type string (object ou string)
colonnes_string = dfYesNo.select_dtypes(include=["object", "string"]).columns
print(colonnes_string)


print(df['Mental Illness'].value_counts())


print(df['Serious Mental Illness'].value_counts())


print(df['SSI Cash Assistance'].value_counts())

##Matrice de correlation
# Option pour afficher jusqu'à 100 lignes dans la console
pd.set_option('display.max_rows', 100)

# Calcul de la matrice de corrélation sur toutes les colonnes numériques
correlation_matrix = dfYesNo.corr()
# Extraction des corrélations avec la colonne 'Mental Illness'
correlations_with_mental = correlation_matrix["Mental Illness"]

# Tri par ordre décroissant
correlations_with_mental_sorted = correlations_with_mental.sort_values(ascending=False)

# Affichage complet (jusqu'à 100 lignes)
print(correlations_with_mental_sorted)


#V- MACHINE LEARNING PHASE
#Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df_train, df_test = train_test_split(dfYesNo,test_size = 0.2) # 20% go into testing and 80% on training
len(dfYesNo) # tell me the size of the data frame

len(df_train) # tell me the size of the data frame of training

len(df_test) # tell me the size of the data frame of training

print(dfYesNo.Sex_MALE.mean())
print(df_train.Sex_MALE.mean())
print(df_test.Sex_MALE.mean())

print(dfYesNo["Mental Illness"].mean())
print(df_train["Mental Illness"].mean())
print(df_test["Mental Illness"].mean())

# Sélectionner les colonnes souhaitées
X_train = df_train.loc[:, [
    'Age Group', 'Living Situation', 'Employment Status',
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
]].values

# Cible
y_train = df_train['Mental Illness'].values


# get the values of the columns for the test data
X_test = df_test.loc[:, [
        'Age Group', 'Living Situation', 'Employment Status',
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
       'Program Category_RESIDENTIAL', 'Program Category_SUPPORT']].values

#cible
y_test = df_test['Mental Illness'].values


# Try an ensemble classifier: Random Forest
rf_model = RandomForestClassifier()


rf_model.fit(X=X_train,y=y_train)

y_test_predicted_rf = rf_model.predict(X_test)

report_rf = classification_report(y_pred=y_test_predicted_rf,y_true=y_test)



# --- Sauvegarde ---
with open("ment_ill_clf.pkl", "wb") as f:
    pickle.dump(rf_model, f)

print(" Modèle sauvegardé sous 'ment_ill_clf.pkl'")
