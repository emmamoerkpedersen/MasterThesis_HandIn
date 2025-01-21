import pandas as pd
import numpy as np

data = pd.read_csv('/Users/emmamork/Downloads/Vandkemi - Hydrometri_20250121_145426.csv', sep =';')
df = pd.DataFrame(data)
df['x-koordinat'] = df['x-koordinat'].str.replace(',', '.')
df['y-koordinat'] = df['y-koordinat'].str.replace(',', '.')


df.to_excel("output.xlsx", index=False)
df.to_csv("output.csv", index=False)