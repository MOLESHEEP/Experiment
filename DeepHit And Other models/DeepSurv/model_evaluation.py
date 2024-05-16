from math import exp
from statistics import mean
import torch
from sklearn.model_selection import train_test_split
from networks import DeepSurv
from scoring import time_depend_weight_brier_score,time_dependent_concordance_index,MAE
from utils import read_config
import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.functions import StepFunction
import matplotlib.pyplot as plt
import os
import shutil


def dataTostatus(y):
    y = y.iloc[:,:2].to_numpy()
    aux = [(e1,e2) for e1,e2 in y]
    y = np.array(aux,dtype=[('Status','?'),('Survival_in_days','<f8')])
    return y

def CoxModel(x_train,y_train,x_test):
    y_train = dataTostatus(y_train)
    COX_estimator = CoxPHSurvivalAnalysis(alpha=0.5).fit(x_train,y_train)
    surv_funs = COX_estimator.predict_survival_function(x_test)
    baseline_survival = COX_estimator.baseline_survival_
    return surv_funs,baseline_survival

#TODO：其实可以直接算出生存函数，这里放之后完成
def trueSurvival(baseline_survival,x_test,lamda_t = 1,shape_paramater = 2,datatype = 1):
    hazard = 2*x_test['x0'] + x_test['x1']
    hazard = np.exp(hazard.values)
    surv_funs = np.empty(len(hazard),dtype=object)
    for i in range(len(hazard)):
        if datatype == 1:
            surv_funs[i] = StepFunction(baseline_survival.x,np.exp(-1*lamda_t * baseline_survival.x * hazard[i]))
        elif datatype == 2:
            surv_funs[i] = StepFunction(baseline_survival.x,np.exp(-1*lamda_t * baseline_survival.x ** shape_paramater* hazard[i]))
        else:
            surv_funs[i] = StepFunction(baseline_survival.x,np.exp(-1*lamda_t/shape_paramater * np.exp(baseline_survival.x * shape_paramater) - 1) * hazard[i])
    return surv_funs

def DeepSurv_Survival_funs(baseline_survival,x_test,ini_pth = r'configs\GompertzLinear.ini', model_pth = r'logs_Guassian\models\GompertzLinear.ini.pth'):
    ini_file = ini_pth
    config = read_config(ini_file)
    model = DeepSurv(config['network'])
    model.load_state_dict(torch.load(model_pth)['model'])
    model.eval()
    #加载模型
    with torch.no_grad():
        pred = model(torch.Tensor(x_test))
        pred = pred.numpy()
    h_x = pred.flatten()
    surv_funs = np.empty(len(h_x),dtype=object)
    for i,risk in enumerate(h_x):
        surv_funs[i] = StepFunction(baseline_survival.x,baseline_survival.y ** exp(risk))
    return surv_funs



def Model_Evaluation(models_dir,config_dir,ini_file,model_name):
    x = pd.read_csv(rf'custom_data\{model_name}\cleaned_features_final.csv')
    y = pd.read_csv(rf'custom_data\{model_name}\label.csv')
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3,random_state=1)
    cox_surv_funs,baseline_survival = CoxModel(x_train,y_train,x_test)
    deepsurv_surv_funs =  DeepSurv_Survival_funs(baseline_survival,x_test.to_numpy(),ini_pth = os.path.join(config_dir, ini_file),model_pth=os.path.join(models_dir, ini_file.split('\\')[-1]+'.pth'))
    mae,mae_list = MAE(cox_surv_funs,deepsurv_surv_funs)
    time_vary_c_index,mean_ctd_index = time_dependent_concordance_index(deepsurv_surv_funs,y_time=y_test['event_time'],y_event=y_test['label'])
    brier_scores,mean_brier_score = time_depend_weight_brier_score(deepsurv_surv_funs,y_time=y_test['event_time'],y_event=y_test['label'])
    copy_file_if_not_exist(models_dir,config_dir,ini_file,rf'bestmodel/{model_name}',mae,mean_brier_score,mean_ctd_index)
    print(f'mean brier score : {mean_brier_score}  mean ctd-index : {mean_ctd_index}  MAE : {mae}')
    return mean_ctd_index,mean_brier_score,mae


def copy_file_if_not_exist(models_dir,configs_dir,ini_file, destination_folder,MAE,BS,c_index):
    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # 拼接目标文件路径
    score_path = os.path.join(destination_folder, 'MAE.csv')
    model_path = os.path.join(models_dir, ini_file.split('\\')[-1]+'.pth')
    ini_path = os.path.join(configs_dir, ini_file)
    # 检查文件是否存在
    if not os.path.exists(score_path):
        # 文件不存在，进行复制
        shutil.copy(model_path, os.path.join(destination_folder, ini_file.split('\\')[-1]+'.pth'))
        shutil.copy(ini_path, os.path.join(destination_folder, ini_file))
        pd.DataFrame({'MAE' : [MAE],'BS' : [BS],'C_index' : [c_index]}).to_csv(score_path)
        print(f"SCORE NOT EXIST, ALL FILES COPY TO '{destination_folder}' ")
    elif os.path.exists(score_path):
        df = pd.read_csv(score_path)
        if MAE < df['MAE'].iloc[0]:
            shutil.copy(model_path, os.path.join(destination_folder, ini_file.split('\\')[-1]+'.pth'))
            shutil.copy(ini_path, os.path.join(destination_folder, ini_file))
            pd.DataFrame({'MAE' : [MAE],'BS' : [BS],'C_index' : [c_index]}).to_csv(score_path)
            print(f"SCORE UPDATE , ALL FILES UPDATE TO '{destination_folder}' ")


if __name__ == '__main__':

    x = pd.read_csv(r'custom_data\GompertzLinear\cleaned_features_final.csv')
    y = pd.read_csv(r'custom_data\GompertzLinear\label.csv')
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3,random_state=1)
    cox_surv_funs,baseline_survival = CoxModel(x_train,y_train,x_test)
    surv_funs = trueSurvival(baseline_survival,x_test,lamda_t = 1,shape_paramater = 2,datatype = 1)
    # deepsurv_surv_funs =  DeepSurv_Survival_funs(baseline_survival,x_test.to_numpy())
    # mae,mae_list = MAE(cox_surv_funs,deepsurv_surv_funs)
    # time_vary_c_index,mean_ctd_index = time_dependent_concordance_index(deepsurv_surv_funs,y_time=y_test['event_time'],y_event=y_test['label'])
    # brier_scores,mean_brier_score = time_depend_weight_brier_score(deepsurv_surv_funs,y_time=y_test['event_time'],y_event=y_test['label'])

    instance = 18
    plt.step(cox_surv_funs[instance].x,cox_surv_funs[instance](cox_surv_funs[instance].x),where='post',label = 'COX')
    plt.step(surv_funs[instance].x,surv_funs[instance](surv_funs[instance].x),where='post',label = 'true')
    # plt.step(deepsurv_surv_funs[instance].x,deepsurv_surv_funs[instance](deepsurv_surv_funs[instance].x),where='post',label = 'DeepSurv')
    plt.legend()
    plt.show()
    # # print(f"mean brier_score : {mean_brier_score}  mean_C_index : {mean_ctd_index} ")

    # logs_dir = 'logs_Guassian'
    # models_dir = os.path.join(logs_dir, 'models')
    # configs_dir = 'configs'
    # Model_Evaluation(models_dir,configs_dir,'GompertzLinear.ini','GompertzLinear')