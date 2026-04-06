import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import os

def ventas_por_año(data: pd.DataFrame) -> pd.DataFrame: 
    """Agrupa ventas globales por año y elimina los últimos 5 años más recientes"""
    annual_sales = data.groupby('Year')['Global_Sales'].sum().reset_index()
    annual_sales = annual_sales.sort_values('Year')
    annual_sales = annual_sales.iloc[:-5]
    return annual_sales

def lineal_regression(annual_sales: pd.DataFrame) -> LinearRegression:
    """Entrena el modelo de regresión lineal"""
    X = annual_sales[['Year']]
    y = annual_sales['Global_Sales']
    
    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    print("Resultados del Modelo de Regresión Lineal")
    print("----------------------------------------")
    print("Coeficientes:", model.coef_[0])
    print("Intercepto:", model.intercept_)
    print(f"Coeficiente de determinación R²: {r2:.3f}")

    return model

def graficar_modelo(annual_sales: pd.DataFrame, model: LinearRegression) -> None:
    X = annual_sales[['Year']]
    y = annual_sales['Global_Sales']
    y_pred = model.predict(X)

    plt.figure(figsize=(10,6))
    plt.scatter(X, y, color='blue', label='Ventas reales')
    plt.plot(X, y_pred, color='red', label='Regresión lineal')
    plt.title('Ventas Globales de Videojuegos por Año')
    plt.xlabel('Año')
    plt.ylabel('Ventas Globales (millones)')
    plt.legend()
    plt.grid(True)
    plt.show()

def predecir_futuro(model: LinearRegression) -> None:
    """Predice ventas futuras usando el modelo entrenado"""
    años_futuros = pd.DataFrame({'Year': np.arange(2015, 2023)})
    predicciones = model.predict(años_futuros)
    resultados = pd.DataFrame({'Año': años_futuros['Year'], 'Ventas_Previstas': predicciones})
    print("\nPredicción de ventas para años futuros:")
    print(resultados)

def main():
    data = pd.read_csv("Practica 1/videogamesales/videogamesales_clean.csv")

    # 🔧 AGREGADO (solución al error)
    data['Year'] = pd.to_datetime(data['Year'], errors='coerce')
    data['Year'] = data['Year'].dt.year
    data = data.dropna(subset=['Year'])

    annual_sales = ventas_por_año(data)
    modelo = lineal_regression(annual_sales)
    graficar_modelo(annual_sales, modelo)
    predecir_futuro(modelo)

if __name__ == "__main__":
    main()