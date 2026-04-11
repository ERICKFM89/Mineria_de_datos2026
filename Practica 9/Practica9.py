import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def generar_nube_palabras(data: pd.DataFrame) -> None: 
    texto = ' '.join(data["Name"].dropna().astype(str).tolist())

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(texto)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Nube de palabras - {"Name"}', fontsize=20)
    plt.show()

def main():
    data = pd.read_csv("Practica 1/videogamesales/videogamesales_clean.csv")
    generar_nube_palabras(data)

if __name__ == "__main__":
    main()