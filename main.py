import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Dataset
LOAN_DATA_CSV = './datasets/loan3000.csv'

# Criar dataframe
loan_data = pd.read_csv(LOAN_DATA_CSV)

# Criacao de cópia para efeitos de comparação
loan_data_copy = loan_data

# Intantiate Label Encoder
label_encoder = LabelEncoder()

# Aplicar label encoder
for col in loan_data.columns:
    if loan_data[col].dtype == 'object':
        loan_data[col] = label_encoder.fit_transform(loan_data[col])

# Printar tipos
print(loan_data.dtypes)

# Printar 5 primeiros registros com titulo da coluna
print(loan_data.head())

