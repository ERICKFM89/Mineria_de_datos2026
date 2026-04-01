import pandas as pd
import numpy as np

# Lectura de datos
data = pd.read_csv("Practica 1/videogamesales/videogamesales_clean.csv")


def mostrar_info_general(data: pd.DataFrame) -> None: 
    """Muestra información sobre el dataset"""
    print("--- Información General ---")
    print(data.info())
    print(data.describe())
    

# VENTAS
def ventas_por_genero(data: pd.DataFrame) -> pd.DataFrame:    
    """Muestra las ventas globales por género de videojuego"""
    ventas_genero = data.groupby('Genre')['Global_Sales'].aggregate(['count', 'mean', 'sum', 'min', 'max']).sort_values(by='sum', ascending=False)
    ventas_genero.columns = ['Total_Juegos', 'Promedio_Ventas', 'Ventas_Totales (millones)', 'Venta_Mínima', 'Venta_Máxima']
    return ventas_genero

def ventas_por_plataforma(data: pd.DataFrame) -> pd.DataFrame: 
    """Muestra las ventas globales por plataforma/consola"""
    ventas_plataforma = data.groupby('Platform')['Global_Sales'].aggregate(['count', 'mean', 'sum', 'min', 'max']).sort_values(by='sum', ascending=False)
    ventas_plataforma.columns = ['Total_Juegos', 'Promedio_Ventas', 'Ventas_Totales(millones)', 'Venta_Mínima', 'Venta_Máxima']
    return ventas_plataforma 

def ventas_por_publisher(data: pd.DataFrame) -> pd.DataFrame: 
    """Muestra las ventas globales por publisher"""
    ventas_publi = data.groupby('Publisher')['Global_Sales'].aggregate(['count', 'mean', 'sum', 'min', 'max']).sort_values(by='sum', ascending=False)
    ventas_publi.columns = ['Total_Juegos', 'Promedio_Ventas', 'Ventas_Totales(millones)', 'Venta_Mínima', 'Venta_Máxima']
    return ventas_publi.head(20)

def ventas_por_region(data: pd.DataFrame) -> pd.Series: 
    """Muestra las por region"""
    ventas_region = data[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum()
    return ventas_region

def ventas_por_anio(data: pd.DataFrame) -> pd.Series: 
    """Muestra las ventas por año"""
    venta_year = data.groupby('Year', as_index=False)['Global_Sales'].sum()
    return venta_year

# -------------------------------

# Conteo
def juegos_por_anio(data: pd.DataFrame) -> pd.Series:
    """Cuenta la cantidad de videojuegos en ese año"""
    return data['Year'].value_counts().sort_index()

def cantidad_genero(data: pd.DataFrame) -> pd.Series: 
    """Cuenta la cantidad de videojuegos en ese género"""
    return data['Genre'].value_counts()

def cantidad_consola(data: pd.DataFrame) -> pd.Series: 
    """Cuenta la cantidad de videojuegos en esa consola"""
    return data['Platform'].value_counts()


def top_15_juegos(data: pd.DataFrame) -> pd.Series: 
    """Top 15 de videojuegos más comprados"""
    top = data.groupby(['Name', 'Platform'])['Global_Sales'].aggregate(['max']).sort_values(by='max', ascending=False)
    top.columns = ['Venta']
    return top.head(15)

def correlacion_ventas(data: pd.DataFrame) -> pd.Series: 
    """Muestra la correlación entre ventas por regiones"""
    return data[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Global_Sales']].corr()


def main():

    print("----------------VENTAS---------------")
    # Análisis
    # Ventas Globales por género
    print(ventas_por_genero(data))
    print("\n")
    
    # Ventas Globales por plataforma
    print(ventas_por_plataforma(data))
    print("\n")
    
    # Ventas Globales por pubisher
    print(ventas_por_publisher(data))
    print("\n")
    
    # Ventas por región
    print(ventas_por_region(data))
    print("\n")
    
    # Ventas Globales por año
    print(ventas_por_anio(data))


    print("\n-------------Conteo---------------")
    
    # Cantidad de videojuegos por año
    print(juegos_por_anio(data))
    print("\n")
    
    # Cantidad de videojuegos por género
    print(cantidad_genero(data))
    print("\n")
    
    # Cantidad de videojuegos por consola
    print(cantidad_consola(data))
    print("\n")

    # Top 15 videojuegos
    print(top_15_juegos(data))
    print("\n")
    
    # Correlación entre regiones
    print(correlacion_ventas(data))
    print("\n")
    

if __name__ == "__main__":
    main()