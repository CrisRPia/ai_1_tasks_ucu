# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
from sklearn.utils import Bunch
import typer
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    r2_score,
)
from sklearn.datasets import load_breast_cancer

from data import get_data

app = typer.Typer()


@app.command()
def boston():
    # After fill by bfill, we have one row still with na. It's just one, let's
    # get rid of it.
    boston = get_data("HousingData.csv").fillna(method="bfill").dropna()

    x = boston.drop("MEDV", axis=1)
    y = boston["MEDV"]

    print(f"\n📊 X tiene forma: {x.shape}")
    print(f"📊 y tiene forma: {y.shape}")
    print("🎯 Queremos predecir: Precio de casas en miles de USD")
    print(f"📈 Precio mínimo: ${y.min():.1f}k, Precio máximo: ${y.max():.1f}k")

    # === ENTRENAR MODELO DE REGRESIÓN LINEAL ===

    # 1. Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    print(f"📊 Datos de entrenamiento: {X_train.shape[0]} casas")
    print(f"📊 Datos de prueba: {X_test.shape[0]} casas")

    # 2. Crear y entrenar el modelo
    modelo_regresion = LinearRegression()
    _ = modelo_regresion.fit(X_train, y_train)

    print("✅ Modelo entrenado!")

    # 3. Hacer predicciones
    predicciones = modelo_regresion.predict(X_test)

    print(f"\n🔮 Predicciones hechas para {len(predicciones)} casas")

    # 4. Evaluar qué tan bueno es el modelo con MÚLTIPLES MÉTRICAS
    mae = mean_absolute_error(y_test, predicciones)
    mse = mean_squared_error(y_test, predicciones)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predicciones)

    # Calcular MAPE manualmente
    mape = np.mean(np.abs((y_test - predicciones) / y_test)) * 100

    print("\n📈 MÉTRICAS DE EVALUACIÓN:")
    print(f"   📊 MAE (Error Absoluto Medio): ${mae:.2f}k")
    print(f"   📊 MSE (Error Cuadrático Medio): {mse:.2f}")
    print(f"   📊 RMSE (Raíz del Error Cuadrático): ${rmse:.2f}k")
    print(f"   📊 R² (Coeficiente de determinación): {r2:.3f}")
    print(f"   📊 MAPE (Error Porcentual Absoluto): {mape:.1f}%")

    print("\n🔍 INTERPRETACIÓN:")
    print(f"   💰 En promedio nos equivocamos por ${mae:.2f}k (MAE)")
    print(f"   📈 El modelo explica {r2 * 100:.1f}% de la variabilidad (R²)")
    print(f"   📊 Error porcentual promedio: {mape:.1f}% (MAPE)")

    # 5. Comparar algunas predicciones reales vs predichas
    print("\n🔍 EJEMPLOS (Real vs Predicho):")
    for i in range(5):
        real = y_test.iloc[i]
        predicho = predicciones[i]
        print(f"   Casa {i + 1}: Real ${real:.1f}k vs Predicho ${predicho:.1f}k")


@app.command()
def medic():
    # === CARGAR DATOS DE DIAGNÓSTICO DE CÁNCER ===

    # 1. Cargar el dataset de cáncer de mama (que viene con sklearn)
    cancer_data = load_breast_cancer()
    assert isinstance(cancer_data, Bunch)

    # 2. Convertir a DataFrame para verlo mejor
    X_cancer = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
    y_cancer = cancer_data.target  # 0 = maligno, 1 = benigno

    print("🏥 DATASET: Breast Cancer (Diagnóstico)")
    print(f"   📊 Pacientes: {X_cancer.shape[0]}")
    print(f"   📊 Características: {X_cancer.shape[1]}")
    print("   🎯 Objetivo: Predecir si tumor es benigno (1) o maligno (0)")

    # 3. Ver balance de clases
    casos_malignos = (y_cancer == 0).sum()
    casos_benignos = (y_cancer == 1).sum()

    print("\n📊 DISTRIBUCIÓN:")
    print(f"   ❌ Casos malignos: {casos_malignos}")
    print(f"   ✅ Casos benignos: {casos_benignos}")

    # === ENTRENAR MODELO DE CLASIFICACIÓN ===

    # 1. Dividir datos en entrenamiento y prueba
    X_train_cancer, X_test_cancer, y_train_cancer, y_test_cancer = train_test_split(
        X_cancer, y_cancer, test_size=0.2, random_state=42
    )

    print(f"📊 Datos de entrenamiento: {X_train_cancer.shape[0]} pacientes")
    print(f"📊 Datos de prueba: {X_test_cancer.shape[0]} pacientes")

    # 2. Crear y entrenar modelo de regresión logística
    modelo_clasificacion = LogisticRegression(max_iter=5000, random_state=42)
    _ = modelo_clasificacion.fit(X_train_cancer, y_train_cancer)

    print("✅ Modelo de clasificación entrenado!")

    # 3. Hacer predicciones
    predicciones_cancer = modelo_clasificacion.predict(X_test_cancer)

    # 4. Evaluar con MÚLTIPLES MÉTRICAS de clasificación
    exactitud = accuracy_score(y_test_cancer, predicciones_cancer)
    precision = precision_score(y_test_cancer, predicciones_cancer)
    recall = recall_score(y_test_cancer, predicciones_cancer)
    f1 = f1_score(y_test_cancer, predicciones_cancer)

    print("\n📈 MÉTRICAS DE CLASIFICACIÓN:")
    print(f"   🎯 Exactitud (Accuracy): {exactitud:.3f} ({exactitud * 100:.1f}%)")
    print(f"   🎯 Precisión (Precision): {precision:.3f} ({precision * 100:.1f}%)")
    print(f"   🎯 Recall (Sensibilidad): {recall:.3f} ({recall * 100:.1f}%)")
    print(f"   🎯 F1-Score: {f1:.3f}")

    # Mostrar matriz de confusión de forma simple
    matriz_confusion = confusion_matrix(y_test_cancer, predicciones_cancer)
    print("\n🔢 MATRIZ DE CONFUSIÓN:")
    print(f"   📊 {matriz_confusion}")
    print("   📋 [Verdaderos Negativos, Falsos Positivos]")
    print("   📋 [Falsos Negativos, Verdaderos Positivos]")

    # Reporte detallado
    print("\n📋 REPORTE DETALLADO:")
    print(
        classification_report(y_test_cancer, predicciones_cancer, target_names=["Maligno", "Benigno"])
    )

    print("\n🔍 INTERPRETACIÓN MÉDICA:")
    print(
        f"   🩺 Precision: De los casos que predecimos como benignos, {precision * 100:.1f}% lo son realmente"
    )
    print(
        f"   🩺 Recall: De todos los casos benignos reales, detectamos {recall * 100:.1f}%"
    )
    print(f"   🩺 F1-Score: Balance general entre precision y recall: {f1:.3f}")

    # 5. Ver ejemplos específicos
    print("\n🔍 EJEMPLOS (Real vs Predicho):")
    for i in range(5):
        real = "Benigno" if y_test_cancer[i] == 1 else "Maligno"
        predicho = "Benigno" if predicciones_cancer[i] == 1 else "Maligno"
        print(f"   Paciente {i + 1}: Real: {real} vs Predicho: {predicho}")


if __name__ == "__main__":
    app()
