# import numpy as np
# import matplotlib.pyplot as plt

# # Fixing random state for reproducibility
# np.random.seed(19680801)


# N = 50
# x = np.random.rand(N)
# y = np.random.rand(N)
# colors = np.random.rand(N)
# area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

# plt.scatter(x, y, s=area, c=colors, alpha=0.5)
# plt.savefig("image.png",dpi=120) 


import requests
from datetime import timedelta, datetime
import pandas as pd
import joblib
import numpy as np

MODEL = joblib.load('best_gs_pipeline.pkl')
df = pd.read_csv("test.csv")
data = df.iloc[:,:-1]
predicted = MODEL.predict(data)
print(predicted)
np.savetxt("predicted.csv", predicted)
