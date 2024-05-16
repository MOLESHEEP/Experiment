from math import log
import torch
from sklearn.model_selection import train_test_split
from networks import DeepSurv
from utils import read_config
import numpy as np
import pandas as pd

datatype = 'Guassian'
df = pd.read_csv('custom_data\Guassian\GuassianData.csv')
X = df.iloc[:,1:-2].values
x_train,x_test = train_test_split(X,test_size=0.7,random_state=1)
ini_file = 'configs\gaussian.ini'
config = read_config(ini_file)
model = DeepSurv(config['network'])
model.load_state_dict(torch.load('logs_Guassian\models\gaussian.ini.pth')['model'])
model.eval()
#x_test = torch.from_numpy(x_test.to_numpy(dtype='object'))
with torch.no_grad():
    pred = model(torch.Tensor(x_test))
    pred = pred.numpy()
    df = pd.concat([pd.DataFrame(x_test),pd.DataFrame(pred)],axis=1)
    df.columns = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','pred']
    if datatype == 'Linear':
        df['true_h'] = df['x0']+ 2 * df['x1']
    elif datatype == 'Guassian':
        df['true_h'] = log(5) * (-(df['x0'] **2+df['x1'] **2)/(2*0.5**2))
    df.to_csv('predict_result.csv')

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 生成一些示例数据
x = df['x0']
y = df['x1']
#z = df['true_h']
z = df['pred']


# 绘制散点图
plt.scatter(x, y, c=z, cmap='viridis',s=80)

# 添加颜色条
plt.colorbar()

# 设置标题和标签
plt.title('Scatter Plot with Color')
plt.xlabel('X')
plt.ylabel('Y')

# 显示图形
plt.show()

