from turtle import color
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from scoring import concordance_index
from sklearn.metrics import brier_score_loss
def readData(Xpath,Ypath,test_radio = 0.5):
    X = pd.read_csv(Xpath)
    y = pd.read_csv(Ypath)
    seed = random.randint(1,100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_radio, random_state=2333)
    return X_train, X_test, y_train, y_test


def MP_Base_Data(x_train,y_train,EVAL_TIME):
    df_x = x_train
    df_y = y_train
    #df = y_train
    df_y['label'] = (df_y['event_time']<=EVAL_TIME).astype(int)
    #df_y = pd.concat([df_y,df],ignore_index=True)
    #df_x = pd.concat([df_x,x_train],ignore_index=True)
    return df_x,df_y

Xpath = r'data\Linear\cleaned_features_final.csv'
Ypath = r'data\Linear\label.csv'

input_dim = 32
hidden_node = 16
instance = 140
result = []
for i in range(1):
    X_train, X_test, y_train, y_test = readData(Xpath = Xpath,Ypath=Ypath,test_radio=0.7)
    model = Sequential()
    model.add(Dense(input_dim, activation='relu', input_shape=(X_train.shape[1],)))  # 第一层，32个隐藏单元
    model.add(Dense(hidden_node, activation='relu'))  # 第二层，16个隐藏单元
    model.add(Dense(1, activation='sigmoid'))  # 输出层，一个输出单元，使用sigmoid激活函数

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    plt.step(1,0,where='post',color = 'b')
    for time in range(int(y_train['event_time'].min()+1),480):
        print(f"#################TIME AT : {time} #####################")
        y_train_new = y_train
        y_train_new['label'] = (y_train_new['event_time']<=time).astype(int)
        model.fit(X_train, y_train_new['label'], epochs=10, batch_size=32, validation_split=0.2)
        y_pred = model.predict(X_train.iloc[instance-1:instance])
        result.append([time,y_pred[0]])
        plt.step(time,y_pred[0],where = 'post',color = 'b')
    plt.show()
    # c_index = concordance_index(y_test['event_time'],y_pred,y_test['label'])
    # BS = brier_score_loss(y_true = y_test['label'],y_prob= y_pred)

    # print('c_index = ',c_index)
    # print('Brier Score = ',BS)
    # result.append([c_index,BS])
pd.DataFrame(result).to_csv('CNN_LINE_0.7.csv')
# pd.DataFrame(X_test).to_csv('heat_map.csv')
# pd.DataFrame(y_pred).to_csv('CNNresult.csv')