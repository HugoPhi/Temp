import hym.DecisionTree as dt
import matplotlib.pyplot as plt


df = dt.load_df('./housing.xlsx')

arr = df.to_numpy()
print(arr.shape)
title_name = df.columns.tolist()


for i in range(arr.shape[1] - 1):
    plt.scatter(arr[:, i], arr[:, -1])
    plt.ylabel(f'{title_name[-1]}')
    plt.xlabel(f'{title_name[i]}')
    plt.savefig(f'./{title_name[i]}.png')
    plt.close()
