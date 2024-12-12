import hym.DecisionTree as dt
import matplotlib.pyplot as plt

df = dt.load_df('./icecream.xlsx')

arr = df.to_numpy()

plt.figure(figsize=(20, 20))
plt.scatter(arr[:, 0], arr[:, 1])
plt.savefig('./icecream.png', dpi=500)
plt.close()
