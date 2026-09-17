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

#5. Modelo Inicial

modelo_inicial = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

modelo_inicial.fit(
    Xtrain,
    Ytrain
)

print("\nModelo inicial entrenado correctamente.")

#6. Predicciones train y validation
pred_train = modelo_inicial.predict(Xtrain)
pred_val = modelo_inicial.predict(Xval)

#7. Evaluación del modelo inicial

# TRAIN
accuracy_train = accuracy_score(
    Ytrain,
    pred_train
)

precision_train = precision_score(
    Ytrain,
    pred_train
)

recall_train = recall_score(
    Ytrain,
    pred_train
)

f1_train = f1_score(
    Ytrain,
    pred_train
)


# VALIDATION
accuracy_val = accuracy_score(
    Yval,
    pred_val
)

precision_val = precision_score(
    Yval,
    pred_val
)

recall_val = recall_score(
    Yval,
    pred_val
)

f1_val = f1_score(
    Yval,
    pred_val
)

print("\n--- MODELO INICIAL ---")

print("\nTRAIN")
print("Accuracy:", round(accuracy_train, 4))
print("Precision:", round(precision_train, 4))
print("Recall:", round(recall_train, 4))
print("F1 Score:", round(f1_train, 4))

print("\nVALIDATION")
print("Accuracy:", round(accuracy_val, 4))
print("Precision:", round(precision_val, 4))
print("Recall:", round(recall_val, 4))
print("F1 Score:", round(f1_val, 4))


# Diferencia Train - Validation
diferencia_accuracy = (
    accuracy_train - accuracy_val
)

print(
    "\nDiferencia Accuracy Train - Validation:",
    round(diferencia_accuracy, 4)
)

#8. Gráfica train vs validation

metricas = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

resultados_train = [
    accuracy_train,
    precision_train,
    recall_train,
    f1_train
]

resultados_val = [
    accuracy_val,
    precision_val,
    recall_val,
    f1_val
]

x = range(len(metricas))
ancho = 0.35

plt.figure(figsize=(8, 5))

barras_train = plt.bar(
    [i - ancho / 2 for i in x],
    resultados_train,
    width=ancho,
    label="Train"
)

barras_val = plt.bar(
    [i + ancho / 2 for i in x],
    resultados_val,
    width=ancho,
    label="Validation"
)

plt.xticks(x, metricas)
plt.ylabel("Valor")
plt.ylim(0, 1.08)

plt.title(
    "Modelo inicial: Train vs Validation"
)

plt.legend()

# Valores sobre las barras
for barra, valor in zip(
    barras_train,
    resultados_train
):
    plt.text(
        barra.get_x() + barra.get_width() / 2,
        valor + 0.015,
        f"{valor:.4f}",
        ha="center",
        fontsize=9
    )

for barra, valor in zip(
    barras_val,
    resultados_val
):
    plt.text(
        barra.get_x() + barra.get_width() / 2,
        valor + 0.015,
        f"{valor:.4f}",
        ha="center",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    "modelo_inicial_train_vs_validation.png",
    dpi=300
)

plt.close()

print(
    "\nGráfica generada:"
    "\n- modelo_inicial_train_vs_validation.png"
)

# 9. Análisis de complejidad
profundidades = [
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    None
]

accuracy_train_depth = []
accuracy_val_depth = []


for profundidad in profundidades:

    modelo_depth = RandomForestClassifier(
        n_estimators=100,
        max_depth=profundidad,
        random_state=42
    )

    modelo_depth.fit(
        Xtrain,
        Ytrain
    )

    pred_train_depth = modelo_depth.predict(
        Xtrain
    )

    pred_val_depth = modelo_depth.predict(
        Xval
    )

    acc_train = accuracy_score(
        Ytrain,
        pred_train_depth
    )

    acc_val = accuracy_score(
        Yval,
        pred_val_depth
    )

    accuracy_train_depth.append(
        acc_train
    )

    accuracy_val_depth.append(
        acc_val
    )


# Mostrar resultados
print(
    "\n--- ANÁLISIS DE MAX_DEPTH ---"
)

for profundidad, train_acc, val_acc in zip(
    profundidades,
    accuracy_train_depth,
    accuracy_val_depth
):

    print(
        f"max_depth={profundidad} | "
        f"Train={train_acc:.4f} | "
        f"Validation={val_acc:.4f}"
    )

# 9.1 Gráfica de complejidad


etiquetas_depth = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "8",
    "10",
    "None"
]

plt.figure(
    figsize=(9, 5)
)

plt.plot(
    etiquetas_depth,
    accuracy_train_depth,
    marker="o",
    label="Train"
)

plt.plot(
    etiquetas_depth,
    accuracy_val_depth,
    marker="o",
    label="Validation"
)

plt.xlabel(
    "Profundidad máxima del árbol"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "Efecto de max_depth sobre Train y Validation"
)

plt.ylim(
    0.5,
    1.05
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "curva_complejidad_max_depth.png",
    dpi=300
)

plt.close()


print(
    "\nGráfica generada:"
    "\n- curva_complejidad_max_depth.png"
)

# 10. Modelo regularizado

modelo_regularizado = RandomForestClassifier(
    n_estimators=100,
    max_depth=2,
    random_state=42
)

modelo_regularizado.fit(
    Xtrain,
    Ytrain
)


# Predicciones
pred_train_reg = modelo_regularizado.predict(
    Xtrain
)

pred_val_reg = modelo_regularizado.predict(
    Xval
)

#11. Evaluación del modelo regularizado
# TRAIN
accuracy_train_reg = accuracy_score(
    Ytrain,
    pred_train_reg
)

precision_train_reg = precision_score(
    Ytrain,
    pred_train_reg
)

recall_train_reg = recall_score(
    Ytrain,
    pred_train_reg
)

f1_train_reg = f1_score(
    Ytrain,
    pred_train_reg
)


# VALIDATION
accuracy_val_reg = accuracy_score(
    Yval,
    pred_val_reg
)

precision_val_reg = precision_score(
    Yval,
    pred_val_reg
)

recall_val_reg = recall_score(
    Yval,
    pred_val_reg
)

f1_val_reg = f1_score(
    Yval,
    pred_val_reg
)


print("\n--- MODELO REGULARIZADO ---")

print("\nTRAIN")
print("Accuracy:", round(accuracy_train_reg, 4))
print("Precision:", round(precision_train_reg, 4))
print("Recall:", round(recall_train_reg, 4))
print("F1 Score:", round(f1_train_reg, 4))

print("\nVALIDATION")
print("Accuracy:", round(accuracy_val_reg, 4))
print("Precision:", round(precision_val_reg, 4))
print("Recall:", round(recall_val_reg, 4))
print("F1 Score:", round(f1_val_reg, 4))

print(
    "\nDiferencia Accuracy Train - Validation:",
    round(
        accuracy_train_reg - accuracy_val_reg,
        4
    )
)

# 12. Comparación antes y después

metricas_comparacion = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

modelo_inicial_val = [
    accuracy_val,
    precision_val,
    recall_val,
    f1_val
]

modelo_regularizado_val = [
    accuracy_val_reg,
    precision_val_reg,
    recall_val_reg,
    f1_val_reg
]

x = range(len(metricas_comparacion))
ancho = 0.35

plt.figure(figsize=(8, 5))

barras_inicial = plt.bar(
    [i - ancho / 2 for i in x],
    modelo_inicial_val,
    width=ancho,
    label="Modelo inicial"
)

barras_regularizado = plt.bar(
    [i + ancho / 2 for i in x],
    modelo_regularizado_val,
    width=ancho,
    label="Modelo regularizado"
)

plt.xticks(
    x,
    metricas_comparacion
)

plt.ylabel("Valor")

plt.ylim(
    0,
    1.08
)

plt.title(
    "Validation: modelo inicial vs regularizado"
)

plt.legend()


for barra, valor in zip(
    barras_inicial,
    modelo_inicial_val
):
    plt.text(
        barra.get_x() + barra.get_width() / 2,
        valor + 0.015,
        f"{valor:.4f}",
        ha="center",
        fontsize=9
    )


for barra, valor in zip(
    barras_regularizado,
    modelo_regularizado_val
):
    plt.text(
        barra.get_x() + barra.get_width() / 2,
        valor + 0.015,
        f"{valor:.4f}",
        ha="center",
        fontsize=9
    )


plt.tight_layout()

plt.savefig(
    "comparacion_regularizacion.png",
    dpi=300
)

plt.close()
# Evaluación final en test

# Predicciones del modelo inicial
pred_test_inicial = modelo_inicial.predict(
    Xtest
)

# Predicciones del modelo regularizado
pred_test_reg = modelo_regularizado.predict(
    Xtest
)


# MODELO INICIAL

accuracy_test_inicial = accuracy_score(
    Ytest,
    pred_test_inicial
)

precision_test_inicial = precision_score(
    Ytest,
    pred_test_inicial
)

recall_test_inicial = recall_score(
    Ytest,
    pred_test_inicial
)

f1_test_inicial = f1_score(
    Ytest,
    pred_test_inicial
)


# MODELO REGULARIZADO

accuracy_test_reg = accuracy_score(
    Ytest,
    pred_test_reg
)

precision_test_reg = precision_score(
    Ytest,
    pred_test_reg
)

recall_test_reg = recall_score(
    Ytest,
    pred_test_reg
)

f1_test_reg = f1_score(
    Ytest,
    pred_test_reg
)


print("\n--- RESULTADOS FINALES EN TEST ---")

print("\nMODELO INICIAL")
print("Accuracy:", round(accuracy_test_inicial, 4))
print("Precision:", round(precision_test_inicial, 4))
print("Recall:", round(recall_test_inicial, 4))
print("F1 Score:", round(f1_test_inicial, 4))

print("\nMODELO REGULARIZADO")
print("Accuracy:", round(accuracy_test_reg, 4))
print("Precision:", round(precision_test_reg, 4))
print("Recall:", round(recall_test_reg, 4))
print("F1 Score:", round(f1_test_reg, 4))

# matriz de confusión
matriz_test_inicial = confusion_matrix(
    Ytest,
    pred_test_inicial
)

matriz_test_reg = confusion_matrix(
    Ytest,
    pred_test_reg
)

print("\nMatriz de confusión - Modelo inicial:")
print(matriz_test_inicial)

print("\nMatriz de confusión - Modelo regularizado:")
print(matriz_test_reg)

#14. Train vs Validation vs Test
conjuntos = [
    "Train",
    "Validation",
    "Test"
]

accuracy_final = [
    accuracy_train_reg,
    accuracy_val_reg,
    accuracy_test_reg
]

precision_final = [
    precision_train_reg,
    precision_val_reg,
    precision_test_reg
]

recall_final = [
    recall_train_reg,
    recall_val_reg,
    recall_test_reg
]

f1_final = [
    f1_train_reg,
    f1_val_reg,
    f1_test_reg
]

x = range(len(conjuntos))
ancho = 0.20

plt.figure(figsize=(10, 5))

plt.bar(
    [i - 1.5 * ancho for i in x],
    accuracy_final,
    width=ancho,
    label="Accuracy"
)

plt.bar(
    [i - 0.5 * ancho for i in x],
    precision_final,
    width=ancho,
    label="Precision"
)

plt.bar(
    [i + 0.5 * ancho for i in x],
    recall_final,
    width=ancho,
    label="Recall"
)

plt.bar(
    [i + 1.5 * ancho for i in x],
    f1_final,
    width=ancho,
    label="F1 Score"
)

plt.xticks(
    x,
    conjuntos
)

plt.ylabel("Valor")
plt.ylim(0, 1.05)

plt.title(
    "Modelo regularizado: Train vs Validation vs Test"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "train_validation_test_regularizado.png",
    dpi=300
)

plt.close()

modelos = [
    "Inicial",
    "Regularizado"
]

brechas = [
    accuracy_train - accuracy_val,
    accuracy_train_reg - accuracy_val_reg
]

plt.figure(figsize=(6, 5))

barras = plt.bar(
    modelos,
    brechas
)

plt.ylabel(
    "Diferencia de Accuracy"
)

plt.title(
    "Brecha Train-Validation antes y después de regularizar"
)

plt.ylim(
    0,
    0.08
)

for barra, valor in zip(
    barras,
    brechas
):
    plt.text(
        barra.get_x() + barra.get_width() / 2,
        valor + 0.002,
        f"{valor:.4f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "brecha_train_validation_regularizacion.png",
    dpi=300
)

plt.close()

# comparación de matrices de confusión
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

matrices = [
    (matriz_test_inicial, "Modelo Inicial"),
    (matriz_test_reg, "Modelo Regularizado")
]

for ax, (matriz, titulo) in zip(axes, matrices):

    imagen = ax.imshow(matriz)

    ax.set_title(titulo)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No calificó (0)", "Calificó (1)"])

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["No calificó (0)", "Calificó (1)"])

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                matriz[i][j],
                ha="center",
                va="center",
                fontsize=13
            )

fig.colorbar(imagen, ax=axes, shrink=0.8)

plt.tight_layout()
plt.savefig("matrices_confusion_analisis.png", dpi=300)
plt.close()