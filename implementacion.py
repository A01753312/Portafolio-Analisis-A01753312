# 1. importar librerias
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

#2. cargar dataset
dataset = pd.read_csv(
    "figure_skating_dataset.csv"
)

X = dataset.drop(
    columns=["Qualified"]
)

Y = dataset["Qualified"]

print("Forma del dataset:", dataset.shape)

#3. separar test 

X_temp, Xtest, Y_temp, Ytest = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42,
    stratify=Y
)
# 4. separar train y validation 

Xtrain, Xval, Ytrain, Yval = train_test_split(
    X_temp,
    Y_temp,
    test_size=0.25,
    random_state=42,
    stratify=Y_temp
)

print("\n--- División del Dataset ---")

print(
    "Entrenamiento:",
    len(Xtrain)
)

print(
    "Validación:",
    len(Xval)
)

print(
    "Prueba:",
    len(Xtest)
)


print("\nDistribución Train:")
print(
    Ytrain.value_counts().sort_index()
)

print("\nDistribución Validation:")
print(
    Yval.value_counts().sort_index()
)

print("\nDistribución Test:")
print(
    Ytest.value_counts().sort_index()
)

