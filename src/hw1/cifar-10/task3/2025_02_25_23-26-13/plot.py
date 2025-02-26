import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# 读取数据
test_data = pd.read_csv('test.csv')
valid_data = pd.read_csv('valid.csv')

# 提取K值
test_data['k'] = test_data['model'].str.extract('(\d+)').astype(int)
valid_data['k'] = valid_data['model'].str.extract('(\d+)').astype(int)

# 设置绘图风格
sns.set(style="whitegrid", font="Noto Serif SC")

# 创建一个横向排列的三子图
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. 准确率随K值的变化
axes[0].plot(test_data['k'], test_data['accuracy'], label='Test Accuracy', marker='o')
axes[0].plot(valid_data['k'], valid_data['accuracy_mean'], label='Validation Accuracy', marker='^')
axes[0].fill_between(valid_data['k'],
                     valid_data['accuracy_mean'] - valid_data['accuracy_std'],
                     valid_data['accuracy_mean'] + valid_data['accuracy_std'],
                     alpha=0.2)
axes[0].set_xlabel('K Value')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Accuracy vs K Value')
axes[0].legend(loc='lower left')

# 添加局部放大图（准确率）
axins0 = inset_axes(axes[0], width="60%", height="30%", loc='upper right', borderpad=2)
axins0.plot(test_data['k'], test_data['accuracy'], label='Test Accuracy', marker='o')
axins0.plot(valid_data['k'], valid_data['accuracy_mean'], label='Validation Accuracy', marker='^')
axins0.fill_between(valid_data['k'],
                    valid_data['accuracy_mean'] - valid_data['accuracy_std'],
                    valid_data['accuracy_mean'] + valid_data['accuracy_std'],
                    alpha=0.2)
axins0.set_xlim(2, 20)  # 设置X轴范围
axins0.set_ylim(0.36, 0.385)  # 设置Y轴范围
axins0.set_xticks(range(2, 21, 1))  # 细化刻度
mark_inset(axes[0], axins0, loc1=1, loc2=2, fc="none", ec="0.5")


# 2. 平均召回率随K值的变化
axes[1].plot(test_data['k'], test_data['avg_recall'], label='Test Average Recall', marker='o')
axes[1].plot(valid_data['k'], valid_data['avg_recall_mean'], label='Validation Average Recall', marker='^')
axes[1].fill_between(valid_data['k'],
                     valid_data['avg_recall_mean'] - valid_data['avg_recall_std'],
                     valid_data['avg_recall_mean'] + valid_data['avg_recall_std'],
                     alpha=0.2)
axes[1].set_xlabel('K Value')
axes[1].set_ylabel('Average Recall')
axes[1].set_title('Average Recall vs K Value')
axes[1].legend(loc='lower left')


# 添加局部放大图（平均召回率）
axins1 = inset_axes(axes[1], width="60%", height="30%", loc='upper right', borderpad=2)
axins1.plot(test_data['k'], test_data['avg_recall'], label='Test Average Recall', marker='o')
axins1.plot(valid_data['k'], valid_data['avg_recall_mean'], label='Validation Average Recall', marker='^')
axins1.fill_between(valid_data['k'],
                    valid_data['avg_recall_mean'] - valid_data['avg_recall_std'],
                    valid_data['avg_recall_mean'] + valid_data['avg_recall_std'],
                    alpha=0.2)
axins1.set_xlim(2, 20)  # 设置X轴范围
axins1.set_ylim(0.36, 0.385)  # 设置Y轴范围
axins1.set_xticks(range(2, 21, 1))  # 细化刻度
mark_inset(axes[1], axins1, loc1=1, loc2=2, fc="none", ec="0.5")


# 3. 验证集和测试集的测试时间随K值的变化
axes[2].plot(test_data['k'], test_data['testing time'], label='Test Testing Time', marker='o')
axes[2].plot(valid_data['k'], valid_data['testing time_mean'], label='Validation Testing Time', marker='^')
axes[2].fill_between(valid_data['k'],
                     valid_data['testing time_mean'] - valid_data['testing time_std'],
                     valid_data['testing time_mean'] + valid_data['testing time_std'],
                     alpha=0.2, color='orange')
axes[2].set_xlabel('K Value')
axes[2].set_ylabel('Testing Time (seconds)')
axes[2].set_title('Testing Time vs K Value')
axes[2].legend(loc='lower right')  # 调整图例位置

# 添加局部放大图
axins = inset_axes(axes[2], width="40%", height="40%", loc='upper left', borderpad=2)  # 创建局部放大图
axins.plot(test_data['k'], test_data['testing time'], label='Test Testing Time', marker='o', linestyle='-', linewidth=2)
axins.plot(valid_data['k'], valid_data['testing time_mean'], label='Validation Testing Time', marker='^', linestyle='--', linewidth=2)
axins.fill_between(valid_data['k'],
                   valid_data['testing time_mean'] - valid_data['testing time_std'],
                   valid_data['testing time_mean'] + valid_data['testing time_std'],
                   alpha=0.2, color='orange')

# 设置局部放大图的显示范围（例如，放大K值在10到20之间的区域）
axins.set_xlim(10, 20)  # 设置X轴范围
axins.set_ylim(26, 40)  # 设置Y轴范围

# 添加放大区域的标记
mark_inset(axes[2], axins, loc1=3, loc2=4, fc="none", ec="0.5")  # 标记放大区域

# 调整布局
plt.tight_layout()
plt.savefig('./plots.png', dpi=1000)
