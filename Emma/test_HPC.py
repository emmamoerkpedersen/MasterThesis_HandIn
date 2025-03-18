import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



original_data = np.random.randint(0, 100, size=(1000, 1000))
print(f'This is sum of original data: {original_data.sum()}')

plt.plot(original_data)
plt.show()