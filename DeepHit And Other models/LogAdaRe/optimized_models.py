import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sksurv.ensemble import RandomSurvivalForest
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.functions import StepFunction

from lifelines import WeibullAFTFitter

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from scoring import time_dependent_concordance_index,time_depend_weight_brier_score,MAE


def WeibullAFT(x_train,y_train,x_test):
    weibull_model = WeibullAFTFitter(penalizer=0.01)
    n_samples = len(x_test)
    df = pd.concat([x_train,y_train['event_time'],y_train['label']],axis=1)
    weibull_model.fit(df,duration_col = 'event_time',event_col = 'label')
    sf = weibull_model.predict_survival_function(x_test)
    cf = weibull_model.predict_cumulative_hazard(x_test)
    surv_funs = np.empty(n_samples, dtype=object)
    chf_funs = np.empty(n_samples, dtype=object)
    for i in range(n_samples):
        surv_funs[i] = StepFunction(sf.iloc[:,0].values,sf.iloc[:,i].values)
        chf_funs[i] = StepFunction(cf.iloc[:,0].values,cf.iloc[:,i].values)
    return surv_funs,chf_funs

def RSF(x_train,y_train,x_test):
    y_train = dataTostatus(y_train)
    rsf = RandomSurvivalForest().fit(x_train,y_train)
    surv_funs = rsf.predict_survival_function(x_test)
    chf_funs= rsf.predict_cumulative_hazard_function(x_test)
    return surv_funs,chf_funs

def CoxModel(x_train,y_train,x_test):
    y_train = dataTostatus(y_train)
    COX_estimator = CoxPHSurvivalAnalysis(alpha=0.5).fit(x_train,y_train)
    chf_funs = COX_estimator.predict_cumulative_hazard_function(x_test)
    surv_funs = COX_estimator.predict_survival_function(x_test)
    return surv_funs,chf_funs


def plotKM(y_train):
    time, survival_prob = kaplan_meier_estimator(y_train['Label'],y_train['event_time'])
    time, survival_prob = np.append(0, time), np.append(1, survival_prob)
    plt.step(time, survival_prob, where = "post",label = 'KM')


def plotStep(sf, model_type='RSF'):
    #for fn in surv_funs:
    # sf.x = np.append(0, sf.x)
    # sf.y = np.append(1, sf.y)
    plt.step(sf.x, sf(sf.x), where="post", label=model_type)

def dataTostatus(y):
    y = y.iloc[:,:2].to_numpy()
    aux = [(e1,e2) for e1,e2 in y]
    y = np.array(aux,dtype=[('Status','?'),('Survival_in_days','<f8')])
    return y


def BasicMLMethods(x_train,y_train,x_test,time_points = 10):
    n_samples = len(x_test)
    t_min = y_train['event_time'].min()
    t_max = y_train['event_time'].max()
    log_surv_pred = np.empty((time_points,n_samples))
    ada_surv_pred = np.empty((time_points,n_samples))
    rf_surv_pred = np.empty((time_points,n_samples))
    index  = 0
    print(len(np.arange(t_min,t_max,(t_max-t_min)/time_points)))
    for time in np.arange(t_min,t_max,(t_max-t_min)/time_points):
        if index >= time_points:
            break
        print(f"#################TIME AT : {time} #####################")
        y_train_new = y_train.copy()
        y_train_new['label'] = (y_train_new['event_time']<=time).astype(int)
        if y_train_new['label'].max() - y_train_new['label'].min():
            log_model = LogisticRegression(penalty="l2", C=10, solver="liblinear", max_iter=1000,random_state=37).fit(x_train,y_train_new['label'])
            ada_model = AdaBoostClassifier(estimator=DecisionTreeClassifier(),n_estimators=100, learning_rate=0.5, random_state=37).fit(x_train,y_train_new['label'])
            rf_model  = RandomForestClassifier(random_state=37).fit(x_train,y_train_new['label'])
            log_prob = log_model.predict_proba(x_test)[:,0]
            ada_prob = ada_model.predict_proba(x_test)[:,0]
            rf_prob  = rf_model.predict_proba(x_test)[:,0]
        elif y_train_new['label'].all() == 0:
            log_prob = np.array([0] * n_samples)
            ada_prob = np.array([0] * n_samples)
            rf_prob  = np.array([0] * n_samples)
        elif y_train_new['label'].all() == 1:
            log_prob = np.array([1] * n_samples)
            ada_prob = np.array([1] * n_samples)
            rf_prob  = np.array([1] * n_samples)
        log_surv_pred[index] = log_prob
        ada_surv_pred[index] = ada_prob
        rf_surv_pred[index]  = rf_prob
        index += 1
    uniq_time = np.arange(t_min,t_max,(t_max-t_min)/(time_points))
    uniq_time = uniq_time[:time_points]
    log_surv_funs = np.empty(n_samples, dtype=object)
    ada_surv_funs = np.empty(n_samples, dtype=object)
    rf_surv_funs = np.empty(n_samples, dtype=object)
    for s in range(n_samples):
        log_surv_funs[s] = StepFunction(uniq_time,log_surv_pred[:,s])
        ada_surv_funs[s] = StepFunction(uniq_time,ada_surv_pred[:,s])
        rf_surv_funs[s] = StepFunction(uniq_time,rf_surv_pred[:,s])
    return log_surv_funs,ada_surv_funs,rf_surv_funs


if __name__ == "__main__":
    x = pd.read_csv(r'data\GompertzLinear\cleaned_features_final.csv')
    y = pd.read_csv(r'data\GompertzLinear\label.csv')
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)
    cox_surv_funs,chf_funs = CoxModel(x_train,y_train,x_test)
    # AFT_surv_funs,chf_funs = WeibullAFT(x_train,y_train,x_test)
    # rsf_surv_funs,chf_funs = RSF(x_train,y_train,x_test)
    log_surv_funs,ada_surv_funs,rf_surv_funs = BasicMLMethods(x_train,y_train,x_test,time_points = 100)
    plotStep(log_surv_funs[0],model_type='Log')
    plotStep(ada_surv_funs[0],model_type='Ada')
    plotStep(rf_surv_funs[0],model_type='RF')
    plotStep(cox_surv_funs[0],model_type='cox')
    plt.legend()
    # cox_score,score_list = MAE(cox_surv_funs,AFT_surv_funs)
    # cox_score,score_list = MAE(cox_surv_funs,rsf_surv_funs)
    cox_score,score_list = MAE(cox_surv_funs,log_surv_funs)
    cox_score,score_list = MAE(cox_surv_funs,ada_surv_funs)
    cox_score,score_list = MAE(cox_surv_funs,rf_surv_funs)
    #log_surv_funs,ada_surv_funs,rf_surv_funs = BasicMLMethods(x_train,y_train,x_test,time_points = 5)
    # print(log_surv_funs)
    # time_vary_c_index,mean_ctd_index = time_dependent_concordance_index(surv_funs,y_test['event_time'],y_test['label'])
    # brier_scores,mean_brier_score = time_depend_weight_brier_score(surv_funs,y_test['event_time'],y_test['label'])
    # plotStep(log_surv_funs[0],model_type='Log')
    # plotStep(ada_surv_funs[0],model_type='Ada')
    # plotStep(rf_surv_funs[0],model_type='RF')
    plt.show()