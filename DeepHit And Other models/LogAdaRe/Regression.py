from turtle import color
from matplotlib.lines import lineStyles
import pandas as pd
from lifelines import LogLogisticFitter
from scipy.__config__ import show
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2
from sklearn.metrics import brier_score_loss
from lifelines import KaplanMeierFitter
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from scoring import concordance_index

#fig, axes = plt.subplots(3, 3, figsize=(13.5, 7.5))

# #Metabric
# x = pd.read_csv("data\METABRIC\cleaned_features_final.csv")
# y = pd.read_csv("data\METABRIC\label.csv")
# lifeline = pd.read_csv("data\METABRIC\lifeline.csv")

#IoT
#x = pd.read_csv("LogAdaRe\data\IoT\cleaned_features_final.csv")
#y = pd.read_csv("LogAdaRe\data\IoT\label.csv")
#lifeline = pd.read_csv("LogAdaRe\data\IoT\lifeline.csv")

result = []
x = pd.read_csv(r'data\NASAt_ALL\cleaned_features_final.csv')
y = pd.read_csv(r'data\NASAt_ALL\label.csv')

for i in range(1):
    #NASA


    X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=i)

    print('**************************LOG-logistic*****************************')
    #LOG-logistic
    llr = LogLogisticFitter()
    llr.fit(y_train['time'],y_train['label'])
    y_pred = [llr.predict(T) for T in y_test['time']]
    llr.plot_survival_function(ci_show=True)
    c_val = concordance_index(y_time=y_test["time"], y_pred=y_pred, y_event=y_test["label"])
    BS_score = brier_score_loss(y_true=y_test["label"], y_prob=y_pred)
    y_pred = pd.DataFrame(y_pred)
    y_pred.columns = ['prob']
    y_test = y_test.reset_index()
    data_L = pd.concat([y_test,y_pred],axis=1)
    data_L = data_L.sort_values(by="time",ascending=True)
    data_L.to_csv("./Log_lifeline.csv")
    print(f'Logistic : C-index {round(c_val, 3)}')
    print('Brier Score:',BS_score)
    plt.ylim(0,1)
    plt.xlim(0,380)
    plt.show()

    print('**************************MP-logistic*****************************')
    #MP-logistic
    lr = LogisticRegression(penalty="l2", C=10, solver="liblinear", max_iter=1000,random_state=None)
    model_log = lr.fit(X_train,y_train['label'])
    y_pred = model_log.predict(X_test)
    c_val = concordance_index(y_time=y_test["event_time"], y_pred=y_pred, y_event=y_test["label"])
    BS_score = brier_score_loss(y_true=y_test["label"], y_prob=y_pred)
    y_pred = pd.DataFrame(y_pred)
    y_pred.columns = ['prob']
    y_test = y_test.reset_index()
    data_L = pd.concat([y_test,y_pred],axis=1)
    data_L = data_L.sort_values(by="event_time",ascending=True)
    data_L.to_csv("./Log_lifeline.csv")
    print(f'Logistic : C-index {round(c_val, 3)}')
    print('Log_Score:',model_log.score(X_test,y_test['label']))
    print('Brier Score:',BS_score)
    result.append(['Log',i,c_val,BS_score])

    

    print('******************************Adaboost******************************')
    #Adaboost
    ada = AdaBoostRegressor(n_estimators=100, learning_rate=0.5, random_state=37)
    model_ada = ada.fit(X_train,y_train['label'])
    y_pred = model_ada.predict(X_test)

    c_val = concordance_index(y_time=y_test["event_time"], y_pred=y_pred, y_event=y_test["label"])
    BS_score = brier_score_loss(y_true=y_test["label"], y_prob=y_pred)
    print(f'Ada : C-index {round(c_val, 3)}')
    print('Ada_Score:',model_ada.score(X_test,y_test['label']))
    print('Brier Score:',BS_score)
    result.append(['Ada',i,c_val,BS_score])

    print('***************************Random forest****************************')
    #Random forest
    RF = RandomForestRegressor(random_state=37)
    model_RF = RF.fit(X_train,y_train['label'])
    y_pred = model_RF.predict(X_test)
    c_val = concordance_index(y_time=y_test["event_time"], y_pred=y_pred, y_event=y_test["label"])
    BS_score = brier_score_loss(y_true=y_test["label"], y_prob=y_pred)
    print(f'RF : C-index {round(c_val, 3)}')
    print('RF_Score:',model_RF.score(X_test,y_test['label']))
    print('Brier Score:',BS_score)
    result.append(['RF',i,c_val,BS_score])


# print('***************************km****************************')
kmf = KaplanMeierFitter()
kmf.fit(y['event_time'], event_observed=y['label'])
kmf.plot_survival_function(ci_show=False)  # km 生存概率


result = pd.DataFrame(result)
result.columns = ['models','Cycle','C-index','BS']
result.to_csv('.\LARresult.csv')

