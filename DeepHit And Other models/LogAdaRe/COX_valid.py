import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import brier_score,concordance_index_censored
from scoring import concordance_index
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

x = pd.read_csv(r'data\Guassian\cleaned_features_final.csv')
# scaler_features = MinMaxScaler()
# normalized_features = scaler_features.fit_transform(x)
# x = pd.DataFrame(normalized_features, columns=x.columns)
print(x)
y = pd.read_csv(r'data\Guassian\label.csv')

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=2333)

y_new = y_train.iloc[:,:2]
y_new = y_new.to_numpy()
print(x_test)
aux = [(e1,e2) for e1,e2 in y_new]
y_new = np.array(aux,dtype=[('Status','?'),('Survival_in_days','<f8')])
#******************************COX************************************
COX_estimator = CoxPHSurvivalAnalysis(alpha=0.5,n_iter=200).fit(x_train,y_new)
predicted= COX_estimator.predict(x_test)
beta = COX_estimator.coef_

# 基准生存函数（这里只是示例，实际应用时需要根据具体情况计算）
def baseline_survival_function(t):
    return np.exp(-0.02 * t)  # 一个简单的示例函数，实际应用时需要根据具体情况设定

# 给定一个时间点 t 和协变量向量 x
t = 12
pred = []
for x in x_test.values:
    # 计算相对风险
    relative_risk = np.exp(np.dot(x, beta))

    # 计算生存概率
    baseline_survival = baseline_survival_function(t)
    survival_probability = baseline_survival ** relative_risk
    pred.append(survival_probability)
    print(survival_probability)

C_index_val = concordance_index(y_test['event_time'],pred,y_test['Label'])
print(C_index_val)