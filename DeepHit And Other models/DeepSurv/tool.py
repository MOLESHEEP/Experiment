import random
import h5py
from numpy import source
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np 
import shutil

from torch import float32, int32


def generateH5(path,split_size = 0.3):
    # 读取CSV文件并将它们合并为一个DataFrame
    csv_file = path  # 用你的实际文件名替换这里
    df = pd.read_csv(csv_file)

    train,test = train_test_split(df,test_size=split_size,random_state=random.randint(0,50),shuffle=True)
    # 创建一个HDF5文件并将数据写入其中
    with h5py.File(r'data\nasafull\nasafull.h5', 'w') as hf:
        hf.create_dataset('test/e', data=test['e'].astype(np.int32))
        hf.create_dataset('test/hr', data=test['hr'].astype(np.float32))
        hf.create_dataset('test/t', data=test['t'].astype(np.float32))
        hf.create_dataset('test/x', data=test.iloc[:,1:-2].astype(np.float32))
        hf.create_dataset('train/e', data=train['e'].astype(np.int32))
        hf.create_dataset('train/hr', data=train['hr'].astype(np.float32))
        hf.create_dataset('train/t', data=train['t'].astype(np.float32))
        hf.create_dataset('train/x', data=train.iloc[:,1:-2].astype(np.float32))
        hf.create_dataset('valid/e', data=test['e'].astype(np.int32))
        hf.create_dataset('valid/hr', data=test['hr'].astype(np.float32))
        hf.create_dataset('valid/t', data=test['t'].astype(np.float32))
        hf.create_dataset('valid/x', data=test.iloc[:,1:-2].astype(np.float32))
        hf.create_dataset('viz/e', data=df['e'].astype(np.int32))
        hf.create_dataset('viz/hr', data=df['hr'].astype(np.float32))
        hf.create_dataset('viz/t', data=df['t'].astype(np.float32))
        hf.create_dataset('viz/x', data=df.iloc[:,1:-2].astype(np.float32))

    # file_path = r'C:\Experiment\DeepHit And Other models\DeepSurv\data\nasafull\nasafull.h5'
    # f = h5py.File(file_path, 'r')
    
    # print("Objects in the HDF5 file:")
    # for name in f:
    #     print(name)
    #     for dataset_name in f[name]:
    #         dataset = f[name][dataset_name]
    #         print(dataset_name, dataset.shape,dataset.dtype)  # 打印数据集名称和形状
    #         # 进一步操作和查看数据集内容
    
    # f.close()

# if __name__ == "__main__":
#     generateH5()
