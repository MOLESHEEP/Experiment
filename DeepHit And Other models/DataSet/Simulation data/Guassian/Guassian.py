from math import exp, log
import random
import pandas as pd

def GenerateGuassianData(DataSize = 100,savepath = 'GuassianData.csv',lambda_MAX = 5,r = 0.5,Exp_lambda = 5):
    data = []
    for i in range(DataSize):
        x0 = random.uniform(-1,1)
        x1 = random.uniform(-1,1)
        x2 = random.uniform(-1,1)
        x3 = random.uniform(-1,1)
        x4 = random.uniform(-1,1)
        x5 = random.uniform(-1,1)
        x6 = random.uniform(-1,1)
        x7 = random.uniform(-1,1)
        x8 = random.uniform(-1,1)
        x9 = random.uniform(-1,1)
        u = random.expovariate(Exp_lambda)
        h  = log(lambda_MAX) * exp(-1*(x0**2+x1**2)/(2*r**2))
        T = u/(exp(h))
        data.append([h,x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,T])
    df = pd.DataFrame(data)
    df.columns = ['hr','x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','t']
    percentile_90 = df['t'].quantile(0.9)
    df['e'] = df['t'].apply(lambda x: 0 if x > percentile_90 else 1)
    df.to_csv(savepath)


def GenerateLinearData(DataSize = 100,savepath = 'LinearData.csv',Exp_lambda = 5,shape_paramater = 2,Time_type = 1):
    '''
    DataSize: 生成数据数量
    savapath: 储存路径
    Exp_lambda: Scale Paramater
    shape_paramater: Shape paramater
    Time type:生成模拟时间数据种类： 1. Exponential distribution 2. Weibull distribution 3. Gompertz distribution
    '''
    data = []
    for i in range(DataSize):
        x0 = random.uniform(-1,1)
        x1 = random.uniform(-1,1)
        x2 = random.uniform(-1,1)
        x3 = random.uniform(-1,1)
        x4 = random.uniform(-1,1)
        x5 = random.uniform(-1,1)
        x6 = random.uniform(-1,1)
        x7 = random.uniform(-1,1)
        x8 = random.uniform(-1,1)
        x9 = random.uniform(-1,1)
        u = random.expovariate(Exp_lambda)
        h  = x0+2*x1
        #lamda = 1 v = 2
        if Time_type == 1:
            T = u/(exp(h))
        elif Time_type == 2:
            T = u/(exp(h)) ** (1/shape_paramater)
        elif Time_type == 3:
            T = (1/shape_paramater) * log(1+((shape_paramater * u)/(exp(h))))
        else:
            print("Error: Time_type should be in [1,2,3]")
            return
        data.append([h,x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,T])
    df = pd.DataFrame(data)
    df.columns = ['hr','x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','t']
    percentile_90 = df['t'].quantile(0.9)
    df['e'] = df['t'].apply(lambda x: 0 if x > percentile_90 else 1)
    df.to_csv(savepath)


GenerateLinearData(DataSize = 5000,savepath = 'GompertzLinearData.csv',Time_type=3)



