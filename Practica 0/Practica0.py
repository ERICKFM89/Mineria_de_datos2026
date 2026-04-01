import opendatasets as od

dataset_link = "https://www.kaggle.com/datasets/gregorut/videogamesales"
od.download(dataset_link)

import os
os.chdir("./videogamesales")
os.listdir()

import pandas as pd
data = pd.read_csv("vgsales.csv")
print(data)