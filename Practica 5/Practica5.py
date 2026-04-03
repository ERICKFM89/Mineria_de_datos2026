import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Lectura de datos
data = pd.read_csv("videogamesales/videogamesales_clean.csv")


def regression_ventas_anio(data: pd.DataFrame) -> None: 
    """Ventas por a través de los años"""
    
    # Características
    X = annual_sales[['Year']]
    y = annual_sales['Global_Sales']

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar modelo
    model = LinearRegression()
    model.fit(X, y)

    # Predicciones
    y_pred = model.predict(X)

    # Coeficientes
    coef = model.coef_[0]
    intercept = model.intercept_
    r2 = r2_score(y, y_pred)

    # Evaluación
    print("Resultados del Modelo de Regresión Lineal Ventas por cada año")
    print("------------------------------------------------------------------")
    print("Coeficientes:", coef)
    print("Intercepto:", intercept)
    print(f"Coeficiente de determinación R²: {r2:.3f}")
    
    
    # Grafica 1, sin regresión
    plt.figure(figsize=(7, 4))
    sns.scatterplot(x='Year', y='Global_Sales', data=annual_sales, color='blue')
    plt.title('Evolución de Ventas Globales por Año')
    plt.xlabel('Año de Lanzamiento')
    plt.ylabel('Ventas Globales Totales (millones)')
    plt.grid(True)
    plt.show()
    
    # Grafica 2, con regresión
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Year', y='Global_Sales', data=annual_sales, color='blue', label='Datos reales')
    plt.plot(annual_sales['Year'], y_pred, color='red')
    plt.title('Ventas Globales Anuales de Videojuegos (1980-2020)')
    plt.xlabel('Año de Lanzamiento')
    plt.ylabel('Ventas Globales Totales (millones)')
    plt.grid(True)
    plt.show()
    
    
    
    
def regression_EU_NA(data: pd.DataFrame) -> None: 
    """Relación entre las ventas de Europa y Norteamerica"""
    
    # Seleccionar características
    features = ['NA_Sales']
    X = data[features]
    y = data['EU_Sales']

    # Crear y entrenar modelo
    model = LinearRegression()
    model.fit(X, y)

    # Predecir y calcular métricas
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    # Evaluación del modelo
    print("\n\nResultados de Regresión Lineal Ventas Europa con Norteamerica")
    print("------------------------------------------------------------------")
    print(f"R²: {r2:.4f}")
    print("Coeficientes:", model.coef_)
    print("Intercepto: ", model.intercept_)
    
    # Gráfica 3
    plt.figure(figsize=(10, 6))
    sns.regplot(x=X['NA_Sales'], y=y, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.title('Relación entre Ventas en Norteamérica y Europa', pad=20)
    plt.xlabel('Ventas EEUU (millones)')
    plt.ylabel('Ventas Europa (millones)')
    plt.grid(True)
    plt.show()

# Ventas por año
annual_sales = data.groupby('Year')['Global_Sales'].sum().reset_index()
regression_ventas_anio(annual_sales)

# Ventas entre Europa y Norteamerica
regression_EU_NA(data)