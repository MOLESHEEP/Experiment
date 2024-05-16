
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 生成一些示例数据
df = pd.read_csv('heatmap.csv')
x = df['x0']
y = df['x1']
z = df['true_h']

# 绘制散点图
plt.scatter(x, y, c=z, cmap='viridis')

# 添加颜色条
plt.colorbar()

# 设置标题和标签
plt.title('Scatter Plot with Color')
plt.xlabel('X')
plt.ylabel('Y')

# 显示图形
plt.show()