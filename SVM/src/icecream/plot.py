import hym.DecisionTree as dt
import matplotlib.pyplot as plt

df = dt.load_df('./icecream.xlsx')

arr = df.to_numpy()

plt.scatter(arr[:, 0], arr[:, 1])
plt.show()
