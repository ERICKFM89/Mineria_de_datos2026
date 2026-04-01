import pandas as pd
data = pd.read_csv("videogamesales/vgsales.csv")

# Change Nan for Not specified in Publisher
data['Publisher'] = data['Publisher'].fillna('Not specified')

# Convert to datetime
data['Year'] = pd.to_datetime(data['Year'], format='%Y')

# DELETE null data
data.dropna(subset=['Year'], inplace=True)

print(data)
data.to_csv("videogamesales/videogamesales_clean.csv", index=False)