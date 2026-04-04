import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


def preparar_datos(df, features):
    """Escala de datos"""
    data = df[features]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data


def aplicar_kmeans(df, scaled_data, n_clusters=4):
    """Aplica KMeans, asigna los clusters al DataFrame y retorna el modelo."""
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    df['Cluster'] = clusters
    return df, kmeans


def analizar_y_graficar(df, features):
    """Grafica los resultados."""
    # Clusters
    cluster_analysis = df.groupby('Cluster')[features].mean()
    print("Análisis de Clústeres:\n", cluster_analysis)

    # Grafica1
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='NA_Sales', y='EU_Sales', hue='Cluster', data=df, palette='plasma')
    plt.title('Visualización de Clústeres (NA Sales vs EU Sales)')
    plt.show()
    
    # Grafica2
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='NA_Sales', y='EU_Sales', hue='Cluster', data=df, palette='plasma')

    # Marcar los centroides con una 'X'
    plt.scatter(cluster_analysis['NA_Sales'], 
                cluster_analysis['EU_Sales'], 
                color='red', marker='X', s=100, label='Centroides')

    # Zoom en la gráfica (ajustar según rango de tus datos)
    plt.ylim(-1,15) 
    plt.xlim(-1,10) 

    plt.title('Cluster Visualization (NA Sales vs EU Sales)')
    plt.legend()
    plt.show()



def main():
    # Lectura de datos
    df = pd.read_csv("Practica 1/videogamesales/videogamesales_clean.csv")
    features = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']

    # Funciones
    datos_escalados = preparar_datos(df, features)
    df, modelo = aplicar_kmeans(df, datos_escalados)
    analizar_y_graficar(df, features)
    

if __name__ == "__main__":
    main()