from math import exp, log
import torch
from sklearn.model_selection import train_test_split
from networks import DeepSurv
from utils import read_config
import numpy as np
import pandas as pd
from sksurv.linear_model import CoxPHSurvivalAnalysis
import matplotlib.pyplot as plt

df = pd.read_csv(r'custom_data\NASA_ALL\cleaned_features_final.csv')
X = df.iloc[:,1:-2].values
instance = 140
x_train,x_test = train_test_split(X,test_size=0.3,random_state=1)
ini_file = r'configs\nasafull.ini'
config = read_config(ini_file)
model = DeepSurv(config['network'])
model.load_state_dict(torch.load(r'logs_NASA\models\nasafull.ini.pth')['model'])
model.eval()

with torch.no_grad():
    pred = model(torch.Tensor(x_train))
    pred = pred.numpy()

h_x = pred[140][0]

y = pd.read_csv(r'custom_data\NASA_ALL\label.csv')
x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,1:-2], y,test_size=0.7,random_state=1)

y_new = y_train.iloc[:,:2]
y_new = y_new.to_numpy()
aux = [(e1,e2) for e1,e2 in y_new]
y_new = np.array(aux,dtype=[('Status','?'),('Survival_in_days','<f8')])
COX_estimator = CoxPHSurvivalAnalysis(alpha=0.5).fit(x_train,y_new)
baseline_func = COX_estimator.baseline_survival_
baseline_func.y = baseline_func.y ** (exp(h_x))

pd.DataFrame(np.column_stack((baseline_func.x,baseline_func.y))).to_csv('survival_fun_0.7.csv')



