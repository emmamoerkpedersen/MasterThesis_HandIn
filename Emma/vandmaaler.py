import pandas as pd
import numpy as np

data = pd.read_csv('/Users/emmamork/Downloads/Vandkemi - Hydrometri_20250121_145426.csv', sep =';')
df = pd.DataFrame(data)

df.to_excel("output.xlsx")