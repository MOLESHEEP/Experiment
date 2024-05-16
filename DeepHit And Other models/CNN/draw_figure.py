
from math import log
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 生成一些示例数据
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
df = pd.read_csv('heat_map.csv')
x = df['x0']
y = df['x1']
z_true = 2*x+y
#z_true = log(5) * (-(x **2+y **2)/(2*0.5**2))
z = df['pre']

# 绘制散点图
sc1 = axes[0].scatter(x, y, c=z_true, cmap='viridis')
sc2 = axes[1].scatter(x, y, c=z, cmap='viridis')
# 添加颜色条
plt.colorbar(sc1, ax=axes[0])
plt.colorbar(sc2, ax=axes[1])

# 设置标题和标签
#plt.title('COX prediction')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')

# 调整子图之间的间距
plt.tight_layout()
# 显示图形
plt.show()