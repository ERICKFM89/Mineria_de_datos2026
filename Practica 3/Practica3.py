import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Lectura de datos
data = pd.read_csv("Practica 1/videogamesales/videogamesales_clean.csv")

def graficar_conteo_por_categoria(data: pd.DataFrame) -> None: 
    """Genera gráficos de barras para el conteo de juegos por categoría"""
    
    nombres_espanol = {
    'Platform': 'Plataforma',
    'Genre': 'Género',
    'Year': 'Año'
    }
    
    categorias =  ['Platform', 'Genre', 'Year']
    
    for col in categorias:
        plt.figure(figsize=(12, 4))
        data[col].value_counts().plot(kind='bar')
        plt.title(f'Cantidad de juegos producidos por {nombres_espanol[col]}')
        plt.xlabel(nombres_espanol[col])
        plt.ylabel('Número de juegos')
        plt.show()



def graficar_ventas_por_año(data: pd.DataFrame) -> None: 
    """Muestra las ventas globales por año"""
    venta_year = data.groupby('Year', as_index=False)['Global_Sales'].sum()
    
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=venta_year, x='Year', y='Global_Sales')
    plt.title('Ventas globales totales por año')
    plt.ylabel('Ventas Globales (millones)')
    plt.show()
    


def graficar_top_plataformas(data: pd.DataFrame) -> None: 
    """Muestra las plataformas con más ventas"""
    top_n=10
    ventas_plataforma = data.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    ventas_plataforma.plot(kind='barh', color='skyblue')
    plt.title(f'Top {top_n} Plataformas por Ventas Globales (en millones)')
    plt.xlabel('Ventas Totales (Millones USD)')
    plt.ylabel('Plataforma')
    plt.grid(linestyle='--')
    plt.show()
    
def graficar_top_publisher(data: pd.DataFrame) -> None: 
    """Muestra las marcas/publisher con más ventas"""
    top_n=10
    ventas_publi = data.groupby('Publisher')['Global_Sales'].sum().sort_values(ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    ventas_publi.plot(kind='barh', color='skyblue')
    plt.title(f'Top {top_n} Publisher por Ventas Globales (en millones)')
    plt.xlabel('Ventas Totales (Millones USD)')
    plt.ylabel('Empresas')
    plt.grid(linestyle='--')
    plt.show()
    
    

def graficar_distribucion_ventas_genero(data: pd.DataFrame) -> None: 
    """Muestra la distribución de ventas por género"""
    ventas_genero = data.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)
    
    # Diagrama de torta
    plt.figure(figsize=(8, 8))
    plt.pie(ventas_genero, labels=ventas_genero.index, autopct='%1.1f%%')
    plt.title('Distribución de Ventas por Género')
    plt.show()
    
    # Gráfico de cajas
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Genre', y='Global_Sales', data=data)
    plt.xticks(rotation=90)
    plt.title('Distribución de Ventas por Género')
    plt.xlabel('Género')
    plt.ylabel('Ventas Totales (Millones USD)')
    plt.ylim(0, 2)
    plt.show()



def graficar_ventas_por_region(data: pd.DataFrame) -> None: 
    """Muestra las ventas por región y género"""
    ventas_region = data.groupby('Genre')[['NA_Sales', 'EU_Sales', 'JP_Sales']].sum()
    
    ventas_region.plot(kind='bar', figsize=(10, 6))
    plt.title('Ventas por Región')
    plt.ylabel('Ventas (Millones USD)')
    plt.xticks(rotation=45)
    plt.grid(linestyle='--')
    plt.show()


def mostrar_correlacion_ventas(data: pd.DataFrame) -> None: 
    """Muestra la matriz de correlación entre ventas por regiones"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(data[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Global_Sales']].corr(), annot=True, cmap='coolwarm')
    plt.title("Matriz de correlación entre ventas")
    plt.show()
    
    
    

def main():
    
    # Análisis de conteo de juegos
    graficar_conteo_por_categoria(data)
    
    # Ventas por año
    graficar_ventas_por_año(data)
    
    # Top plataformas
    graficar_top_plataformas(data)
    
    # Top publisher
    graficar_top_publisher(data)
    
    # Distribución por género
    graficar_distribucion_ventas_genero(data)
    
    # Ventas por región
    graficar_ventas_por_region(data)
    
    # Correlación entre ventas
    mostrar_correlacion_ventas(data)

if __name__ == "__main__":
    main()