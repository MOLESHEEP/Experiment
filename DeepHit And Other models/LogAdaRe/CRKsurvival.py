import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sksurv.ensemble import RandomSurvivalForest
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis
from scoring import concordance_index
from sksurv.metrics import brier_score,concordance_index_censored
from sklearn.model_selection import train_test_split
result = []

instance = 90
estimate_ins = 1 #245

for i in range(1):
    #NASA
    x = pd.read_csv(r'data\Guassian\cleaned_features_final.csv')
    y = pd.read_csv(r'data\Guassian\label.csv')

    # x_shuffle = x.sample(frac=1,random_state=i).reset_index(drop=True)
    # y_shuffle = y.sample(frac=1,random_state=i).reset_index(drop=True)

    # train_ratio = 0.8
    # train_size = int(train_ratio * len(x_shuffle))

    # x_train = x.iloc[:train_size]
    # y_train = y.iloc[:train_size]
    # x_test  = x.iloc[train_size:]
    # y_test  = y.iloc[train_size:]

    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.7,random_state=i)

    y_new = y_test.iloc[:,:2]
    y_new = y_new.to_numpy()
    print(x_test)
    aux = [(e1,e2) for e1,e2 in y_new]
    y_new = np.array(aux,dtype=[('Status','?'),('Survival_in_days','<f8')])
    #***********************RandomSurvivalForestModel*************************
    rsf = RandomSurvivalForest(n_estimators=500).fit(x_test,y_new)
    #改变instance
    chf_funs = rsf.predict_survival_function(x_train.iloc[instance - 1: instance])
    #####################
    predicted= rsf.predict(x_test)
    pre = pd.DataFrame(predicted)
    x_test.to_csv('heat_map.csv')
    pre.to_csv('heat_map1.csv')
    C_index_val = concordance_index_censored(y_test['Label'],y_test['event_time'],predicted)
    print("RSF",C_index_val,"##############################")
    #BS
    survs = rsf.predict_survival_function(x_train)
    preds = [fn(estimate_ins) for fn in survs]
    dex = [(e1,e2) for e1,e2 in y_test.iloc[:,:2].to_numpy()]
    yi = np.array(aux,dtype=[('Status','?'),('Survival_in_days','<f8')])
    #times,brier_score_val = brier_score(yi,yi,preds,estimate_ins)
    #result.append(['RSF',i,C_index_val[0],brier_score_val[0]])
    #Plot
    for fn in chf_funs:
        fn.x,fn.y = np.append(0,fn.x),np.append(1,fn.y)
        plt.step(fn.x,fn(fn.x),where="post",label = 'RSF')
    #///////////////////////////////////////////////////////////////////////////

    #******************************COX************************************
    #COX_estimator = CoxPHSurvivalAnalysis(alpha=0.5).fit(x_test,y_new)
    # predicted= COX_estimator.predict(x_test)
    # pre = pd.DataFrame(predicted)
    
    # C_index_val = concordance_index_censored(y_test['Label'],y_test['event_time'],predicted)
    # x_test.to_csv('heat_map.csv')
    # pre.to_csv('heat_map1.csv')
    # #改变instance
    # cox_surv_funs = COX_estimator.predict_survival_function(x_train.iloc[instance-1:instance])
    # #############
    # #BS
    # survs = COX_estimator.predict_survival_function(x_test)
    # preds = [fn(estimate_ins) for fn in survs]
    # dex = [(e1,e2) for e1,e2 in y_test.iloc[:,:2].to_numpy()]
    # yi = np.array(dex,dtype=[('Status','?'),('Survival_in_days','<f8')])
    # times,brier_score_val = brier_score(yi,yi,preds,estimate_ins)
    # print("COX",C_index_val,"##############################")
    # result.append(['COX',i,C_index_val[0],brier_score_val[0]])

    # for fn in cox_surv_funs:
    #     fn.x,fn.y = np.append(0,fn.x),np.append(1,fn.y)
    #     plt.step(fn.x,fn(fn.x),where="post",label = 'COX')
    #/////////////////////////////////////////////////////////////////////

#***********************DeepHit Survival Function*************************
# pred = pd.read_csv(r'data\Deephit\deep_hit_pred.csv')
# plt.step(pred.index,pred.iloc[:,4-1],where = 'post',label = 'DeepHit',linestyle='--')

# #******************************KM curve************************************
# time, survival_prob = kaplan_meier_estimator(y['Label'],y['event_time'])
# time, survival_prob = np.append(0, time), np.append(1, survival_prob)
# plt.step(time, survival_prob, where = "post",label = 'KM')
#//////////////////////////////////////////////////////////////////////////

# result = pd.DataFrame(result)
# result.columns = ['models','Cycle','C-index','BS']
# result.to_csv('.\CRresult.csv')
# plt.ylim(0, 1.1)
# plt.xlim(0,1)
# plt.legend()
# plt.show()