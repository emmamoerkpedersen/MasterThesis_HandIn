import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



original_data = pd.read_pickle("data_utils/Sample data/original_data.pkl")
print(f'This is sum of original data: {original_data["21006845"]["rainfall"].sum()}')

plt.plot(original_data["21006845"]["rainfall"].index, original_data["21006845"]["rainfall"].values)
plt.show()