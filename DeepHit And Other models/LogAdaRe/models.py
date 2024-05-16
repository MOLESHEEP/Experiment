from cProfile import label
from math import log
from mimetypes import init
from operator import index
import time
from turtle import color
from unittest import result
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sksurv.ensemble import RandomSurvivalForest

from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis
from lifelines import WeibullAFTFitter
from sksurv.metrics import brier_score,concordance_index_censored
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from scoring import concordance_index

def WeibullAFT(x_train,y_train,x_test,y_test,instance):
    weibull_model = WeibullAFTFitter(penalizer=0.01)
    df = pd.concat([x_train,y_train['event_time'],y_train['label']],axis=1)
    weibull_model.fit(df,duration_col = 'event_time',event_col = 'label')
    predicted_survival_functions = weibull_model.predict_survival_function(x_test)
    preds = weibull_model.predict_median(x_test)
    predicted_probabilities = 1 - predicted_survival_functions.values
    c_index = concordance_index(y_time=y_test['event_time'], y_pred= predicted_probabilities[100], y_event=y_test['label'])
    brier = brier_score_loss(y_test['label'],y_prob=predicted_probabilities[100])
    print(f"#########WeibullAFT######## : C_INDEX : {c_index}  Brier_Score: {brier}")
    chf_funs = weibull_model.predict_survival_function(x_train.iloc[instance - 1: instance])
    return c_index,brier,chf_funs,preds

def RSF(x_train,y_train,x_test,y_test,estimate_ins,instance):
    rsf = RandomSurvivalForest().fit(x_train,y_train)
    chf_funs = rsf.predict_survival_function(x_train.iloc[instance - 1: instance])
    predicted= rsf.predict(x_test)
    C_index_val = concordance_index_censored(y_test['Label'],y_test['event_time'],predicted)
    C_val = concordance_index(y_time=y_test["event_time"], y_pred = predicted, y_event=y_test["label"])
    survs = rsf.predict_survival_function(x_test)
    preds = [fn(estimate_ins) for fn in survs]
    y_test = dataTostatus(y_test)
    aux = [(e1,e2) for e1,e2 in y_test]
    yi = np.array(aux,dtype=[('Status','?'),('Survival_in_days','<f8')])
    times,brier_score_val = brier_score(yi,yi,preds,estimate_ins)
    print(f"#########RSF######## : C_INDEX : {C_index_val[0]}({C_val})  Brier_Score: {brier_score_val[0]}")
    return C_index_val,brier_score_val,chf_funs,predicted,C_val

def CoxModel(x_train,y_train,x_test,y_test,estimate_ins,instance):
    COX_estimator = CoxPHSurvivalAnalysis(alpha=0.5).fit(x_train,y_train)
    predicted= COX_estimator.predict(x_test)
    C_index_val = concordance_index_censored(y_test['Label'],y_test['event_time'],predicted)
    C_val = concordance_index(y_time=y_test["event_time"], y_pred = predicted, y_event=y_test["label"])
    #brier_score_val = time_dependent_brier_score(predicted,y_test['event_time'],y_test['Label'])
    cox_surv_funs = COX_estimator.predict_survival_function(x_train.iloc[instance-1:instance])
    survs = COX_estimator.predict_survival_function(x_test)
    preds = [fn(estimate_ins) for fn in survs]
    y_test = dataTostatus(y_test)
    dex = [(e1,e2) for e1,e2 in y_test]
    yi = np.array(dex,dtype=[('Status','?'),('Survival_in_days','<f8')])
    #stimes,brier_score_val = brier_score(yi,yi,preds,estimate_ins)
    #print(f"#########COX######## : C_INDEX : {C_index_val[0]}({C_val})  Brier_Score: {brier_score_val}")
    return C_index_val,cox_surv_funs,predicted,C_val

def Logistic(x_train,y_train,x_test,y_test):
    lr = LogisticRegression(penalty="l2", C=10, solver="liblinear", max_iter=1000,random_state=None)
    model_log = lr.fit(x_train,y_train['label'])
    y_pred = model_log.predict(x_test)
    y_pred_prob = model_log.predict_proba(x_test)
    c_val = concordance_index(y_time=y_test["event_time"], y_pred=y_pred_prob[:,1], y_event=y_test["label"])
    BS_score = brier_score_loss(y_true=y_test["label"], y_prob=y_pred_prob[:,1])
    print(f"#########Logistic######## : C_INDEX : {c_val}  Brier_Score: {BS_score}")
    return c_val,BS_score,y_pred_prob[:,1]

def Adaboost(x_train,y_train,x_test,y_test):
    base_regressor = DecisionTreeClassifier()
    ada = AdaBoostClassifier(estimator=base_regressor,n_estimators=100, learning_rate=0.5, random_state=37)
    model_ada = ada.fit(x_train,y_train['label'])
    y_pred = model_ada.predict(x_test)
    y_pred_prob = model_ada.predict_proba(x_test)
    c_val = concordance_index(y_time=y_test["event_time"], y_pred=y_pred_prob[:,1], y_event=y_test["label"])
    BS_score = brier_score_loss(y_true=y_test["label"], y_prob=y_pred_prob[:,1])
    print(f"#########Adaboost######## : C_INDEX : {c_val}  Brier_Score: {BS_score}")
    return c_val,BS_score,y_pred_prob[:,1]

def RF(x_train,y_train,x_test,y_test):
    #Random forest
    RF = RandomForestClassifier(random_state=37)
    model_RF = RF.fit(x_train,y_train['label'])
    y_pred = model_RF.predict(x_test)
    y_pred_prob = model_RF.predict_proba(x_test)
    c_val = concordance_index(y_time=y_test["event_time"], y_pred=y_pred_prob[:,1], y_event=y_test["label"])
    BS_score = brier_score_loss(y_true=y_test["label"], y_prob=y_pred_prob[:,1])
    print(f"#########Random forest######## : C_INDEX : {c_val}  Brier_Score: {BS_score}")
    return c_val,BS_score,y_pred_prob[:,1]

def plotKM(y_train):
    time, survival_prob = kaplan_meier_estimator(y_train['Label'],y_train['event_time'])
    time, survival_prob = np.append(0, time), np.append(1, survival_prob)
    plt.step(time, survival_prob, where = "post",label = 'KM')


def plotStep(chf_funs,model_type = 'RSF'):
    for fn in chf_funs:
        fn.x,fn.y = np.append(0,fn.x),np.append(1,fn.y)
        plt.step(fn.x,fn(fn.x),where="post",label = model_type)

def dataTostatus(y):
    y = y.iloc[:,:2].to_numpy()
    aux = [(e1,e2) for e1,e2 in y]
    y = np.array(aux,dtype=[('Status','?'),('Survival_in_days','<f8')])
    return y

def MP_Base_Data(x_train,y_train,EVAL_TIME):
    df_x = x_train
    df_y = y_train
    #df = y_train
    df_y['label'] = (df_y['event_time']<=EVAL_TIME).astype(int)
    #df_y = pd.concat([df_y,df],ignore_index=True)
    #df_x = pd.concat([df_x,x_train],ignore_index=True)
    return df_x,df_y

def plotMbBased(x_train,y_train,x_test):
    log_line = [1]
    ada_line = [1]
    rf_line  = [1]
    for time in range(y['event_time'].min()+1,480):
        print(f"#################TIME AT : {time} #####################")
        y_train_new = y_train
        y_train_new['label'] = (y_train_new['event_time']<=time).astype(int)
        log_model = LogisticRegression(penalty="l2", C=10, solver="liblinear", max_iter=1000,random_state=37).fit(x_train,y_train_new['label'])
        ada_model = AdaBoostClassifier(estimator=DecisionTreeClassifier(),n_estimators=100, learning_rate=0.5, random_state=37).fit(x_train,y_train_new['label'])
        rf_model  = RandomForestClassifier(random_state=37).fit(x_train,y_train_new)
        log_prob = log_model.predict_proba(x_test)
        ada_prob = ada_model.predict_proba(x_test)
        rf_prob  = rf_model.predict_proba(x_test)
        print(f"Logistic Probability : {log_prob[0,0]}")
        print(f"Adaboost Probability : {ada_prob[0,0]}")
        print(f"Random Forest Probability : {rf_prob[2][0,0]}")
        log_line.append(log_prob[0,0])
        ada_line.append(ada_prob[0,0])
        rf_line.append(rf_prob[2][0,0])
    sequences = [0]
    sequences[1:] = list(range(y['event_time'].min()+1,480))
    sequences.append(500) 
    log_line.append(0)
    ada_line.append(0)
    rf_line.append(0)
    plt.step(sequences,log_line,where = 'post',label = 'Logistic',color = 'black')
    plt.step(sequences,ada_line,where = 'post',label = 'Adaboost',color = 'g')
    plt.step(sequences,rf_line,where = 'post',label = 'RF',color = 'b')
    pred = pd.read_csv(r'data\Deephit\deep_hit_pred.csv')
    plt.step(pred.index,pred.iloc[:,9],where = 'post',label = 'DeepHit',linestyle='--',color = 'r')
    plt.xlim(0,500)
    plt.legend()
    plt.show()
        
if __name__ == "__main__":
    
    x = pd.read_csv(r'data\NASA_ALL\cleaned_features_final.csv')
    y = pd.read_csv(r'data\NASA_ALL\label.csv')
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.7,random_state=1)
    instance = 140
    # plotKM(y)
    # plotMbBased(x_train,y_train,x_train.iloc[instance-1:instance])
    y_train_stat = dataTostatus(y_train)
    # instance = 140
    # estimate_ins = 0.2 #245
    # log_c_index,log_bs,predicted= Logistic(x_train,y_train,x_test,y_test)
    # pd.DataFrame(x_test).to_csv('heat_map.csv')
    # pd.DataFrame(predicted).to_csv('heat_map1.csv')
    # result = []
    # c_result = []
    # b_result = []
    #for i in range(1):
        #print(f"##################### epoch : {i} #####################")
        # x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.7,random_state=i)
        # log_c_index,log_bs= Logistic(x_train,y_train,x_test,y_test)
        # ada_c_index,ada_bs = Adaboost(x_train,y_train,x_test,y_test)
        # rf_c_index,rf_bs = RF(x_train,y_train,x_test,y_test)
        #plotKM(y)

        # y_train_stat = dataTostatus(y_train)
        #instance = 140
    estimate_ins = 245 #245
        # rsf_c_index,rsf_bs,chf_funs,predicted,rsf_c = RSF(x_train,y_train_stat,x_test,y_test,estimate_ins,instance)
        # plotStep(chf_funs,model_type = 'RSF')
    cox_c_index,chf_funs,predicted,cox_c =CoxModel(x_train,y_train_stat,x_test,y_test,estimate_ins,instance)
        # plotStep(chf_funs,model_type = 'COX')
        # W_c_index,W_bs,chf_funs = WeibullAFT(x_train, y_train, x_test,y_test,instance)
        # plt.step(np.append(0,chf_funs.index),np.append(1,chf_funs.values),where="post",label = 'AFT')
        # plotKM(y_train,y_test)
        # pred = pd.read_csv(r'data\Deephit\deep_hit_pred.csv')
        # plt.step(pred.index,pred.iloc[:,3],where = 'post',label = 'DeepHit',linestyle='--',color = 'r')
        # plt.xlim(0,500)
        # plt.legend()
        # plt.show()

        # c_result.append([log_c_index,ada_c_index,rf_c_index,rsf_c_index,cox_c_index,W_c_index])
        # b_result.append([log_bs,ada_bs,rf_bs,rsf_bs[0],cox_bs[0],W_bs])
    # c_df = pd.DataFrame(c_result)
    # c_df.columns = ['log','ada','rf','rsf','cox','WAFT']
    # b_df = pd.DataFrame(b_result)
    # b_df.columns = ['log','ada','rf','rsf','cox','WAFT']
    # c_df.to_csv('c_index_result.csv')
    # b_df.to_csv('bs_result.csv')

    #这段代码是用来画DeepSurv和CNN的
    # df = pd.read_csv('survival_fun_0.7.csv')
    # plt.step(df['time'],df['deepsurv'],where = 'post',label = 'Deepsurv')
    # plt.step(df['time'],df['cnn'],where = 'post',label = 'CNN')
    # pred = pd.read_csv(r'data\Deephit\deep_hit_pred.csv')
    # plt.step(pred.index,pred.iloc[:,9],where = 'post',label = 'DeepHit',linestyle='--',color = 'r')
    # plotKM(y)
    # plt.xlim(0,500)
    # plt.legend()
    # plt.show()