import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.distributions.empirical_distribution import ECDF
sns.set(rc = {'figure.figsize':(16,9)})

# Define columns name
column1 = ["machine_name", "cycle", "operational_setting_1", "operational_setting_2", "operational_setting_3"]
column2 = [f'sensor_measurement_{i:02}' for i in range(1,22)]
columns = column1 + column2

# Read data
turbofan_df = pd.read_csv("train_FD001.txt", header = None, sep = "\s+", names = columns)
turbofan_df.head()

# Select maximum cycle
max_cycle = turbofan_df.groupby(by = "machine_name")['cycle'].transform(max)
turbofan_df = turbofan_df[turbofan_df["cycle"] == max_cycle].set_index('machine_name')
turbofan_df.head()

# Lollipop plot for each machine name
plt.hlines(y=turbofan_df.index, xmin=1, xmax=turbofan_df['cycle'], color='skyblue')
plt.plot(turbofan_df['cycle'], turbofan_df.index, "o")
plt.plot([1 for i in range(len(turbofan_df))], turbofan_df.index, "o")

# Add titles and axis names
plt.title("Max. Cycle")
plt.xlabel('Cycle')
plt.ylabel('Machine ID')

# Show the plot
plt.show()


# Create status column
turbofan_df['status'] = turbofan_df['cycle'].apply(lambda x: False if x > 200 else True)

#判断相关性
turbofan_df.nunique()

# Change to category
category_columns = ['operational_setting_3', 'sensor_measurement_16']

turbofan_df[category_columns] = turbofan_df[category_columns].astype('category')
turbofan_df.info()

sns.heatmap(turbofan_df.corr()**2, annot = True,)
plt.show()

selected_columns = ['operational_setting_1','operational_setting_2','sensor_measurement_04','sensor_measurement_06','sensor_measurement_08','sensor_measurement_11','sensor_measurement_13','sensor_measurement_14']
cleaned_data = turbofan_df.loc[:, selected_columns + category_columns + ['status', 'cycle']]
sns.heatmap(cleaned_data.corr(), annot = True)
cleaned_data.to_csv('cleaned_data_4.csv')
plt.show()




#kaplan-meier
# One Hot Encoding for Categorical Variable
from sksurv.preprocessing import OneHotEncoder
cleaned_data = pd.read_csv('cleaned_data.csv')
cleaned_data['label'] = cleaned_data['status'].apply(lambda x:1 if x==True else 0 )
cleaned_data.to_csv('cleaned_data.csv')
data_x = OneHotEncoder().fit_transform(cleaned_data.iloc[:, :-2])
data_x.head()

data_y = list(cleaned_data.loc[:, ["status", "cycle"]].itertuples(index = None, name = None))
data_y = np.array(data_y, dtype=[('status', bool), ('cycle', float)])

from sksurv.nonparametric import kaplan_meier_estimator

time, survival_prob = kaplan_meier_estimator(data_y["status"], data_y["cycle"])
time, survival_prob = np.append(0, time), np.append(1, survival_prob)

# Plotting
#pd.concat([pd.DataFrame(time),pd.DataFrame(survival_prob)],axis=1).to_excel('KaplanMeier.xlsx')
plt.step(time, survival_prob, where = "post")
plt.xlim(left = 0, right = 220)
plt.title("Kaplan Meier Estimator")
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.show()

#Experical distribution
y = ECDF(time)
plt.step(time,y.y[1:])
plt.title("Emprical Distribution")
plt.xlabel("time $t$")
plt.show()

#summary file lifetime
cleaned_data = pd.read_csv('cleaned_data.csv')
plt.hlines(y=cleaned_data.index, xmin=1, xmax=cleaned_data['cycle'], color='skyblue')
plt.plot(cleaned_data['cycle'], cleaned_data.index, "o")
plt.plot([1 for i in range(len(cleaned_data))], cleaned_data.index, "o")

# Add titles and axis names
#plt.title("")
plt.xlabel('Time')
plt.ylabel('System ID')

# Show the plot
plt.show()
