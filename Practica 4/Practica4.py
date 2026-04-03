import pandas as pd
import numpy as np

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Lectura de datos
data = pd.read_csv("Practica 1/videogamesales/videogamesales_clean.csv")

def analizar_diferencias_genero(data):
    # Modelo ANOVA
    modl = ols('Global_Sales ~ Genre', data=data).fit()
    anova_df = sm.stats.anova_lm(modl, typ=2)

    valor_p = anova_df["PR(>F)"].iloc[0]

    if valor_p < 0.05:
        print("Hay diferencias significativas entre géneros")
        print(anova_df)

        # Prueba post-hoc de Tukey
        tukey = pairwise_tukeyhsd(endog=data["Global_Sales"],
                                    groups=data["Genre"],
                                    alpha=0.05)

        print("\nResultados de la prueba de Tukey:")
        print(tukey)

    else:
        print("No hay diferencias significativas entre géneros")
        
        
analizar_diferencias_genero(data)