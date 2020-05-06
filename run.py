# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:18:06 2020

@author: 28215
"""

from OuttertuneFunction import *
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF
#Load workload A
workload_A_all=pd.read_csv('./offline_workload.csv') 
tmp_flag = (workload_A_all != 0).any(axis=0)
workload_A = workload_A_all.loc[:, tmp_flag]
workload_A = workload_A.loc[:, ~(workload_A == workload_A.iloc[0]).all()]
workload_A = workload_A.loc[:, workload_A.isin([0]).mean() < .6]
knobs_list = ['k1','k2','k3','k4','k5','k6','k7','k8','s1','s2','s3','s4']

tmpframe = workload_A.T.copy()
tmpframe1 = tmpframe.tail(314-1).T
tmpframe2 = tmpframe.head(1).T
knobs_metrics_name = tmpframe1.columns
work_id_name = tmpframe2.columns

tmp_workload_knobs_metrics = (workload_A[knobs_metrics_name]).values

dataScalar = MinMaxScaler().fit(tmp_workload_knobs_metrics)

workload_knobs_metrics_Norm = dataScalar.transform(tmp_workload_knobs_metrics)
tmp_workload_id = (workload_A[work_id_name]).values

workload_knobs_metrics_Norm_id = np.hstack((tmp_workload_id,workload_knobs_metrics_Norm))

col_list = list(work_id_name)+list(knobs_metrics_name)
norm_workload_A = pd.DataFrame(data=workload_knobs_metrics_Norm_id,columns=col_list)

latencyArray = (workload_A[['latency']]).values

latencyScalar = MinMaxScaler().fit(latencyArray)

workload_A_ids = norm_workload_A["workload id"].unique()
workload_A_list = []
for num,workload_A_id in enumerate(workload_A_ids,start=1):
    tmp_workload = norm_workload_A[norm_workload_A['workload id'].isin([workload_A_id])]
    workload_A_list.append(tmp_workload)
    exec("workload_A_%s = tmp_workload"%num)
    
tmpframe = norm_workload_A.T.copy()
tmpframe = tmpframe.tail(314-1-12).T
metrics_name = tmpframe.columns
workload_A_listformetrics = []
num_workloads_A = len(workload_A_list)
for i in range(num_workloads_A):
    tmp = (workload_A_list[i])[metrics_name]
    workload_A_listformetrics.append(tmp)

print("Load workload A finish!")
#######################################################
    
#Load workload B
workload_B=pd.read_csv('./online_workload_B.csv') 


tmp_workload_B_knobs_metrics = (workload_B[knobs_metrics_name]).values


Bworkload_knobs_metrics_Norm = dataScalar.transform(tmp_workload_B_knobs_metrics)
Btmp_workload_id = (workload_B[work_id_name]).values

Bworkload_knobs_metrics_Norm_id = np.hstack((Btmp_workload_id,Bworkload_knobs_metrics_Norm))

norm_workload_B = pd.DataFrame(data=Bworkload_knobs_metrics_Norm_id,columns=col_list)

workload_B_ids = norm_workload_B["workload id"].unique()
workload_B_list = []
for num,workload_B_id in enumerate(workload_B_ids,start=1):
    tmp_workload = norm_workload_B[norm_workload_B['workload id'].isin([workload_B_id])]
    workload_B_list.append(tmp_workload)
    exec("workload_B_%s = tmp_workload"%num)
    
workload_B_mapping_data_list = []
workload_B_target_list = []

num_workloads_B = len(workload_B_list)

workload_B_mapping_data = []
workload_B_target = []

workload_B_mapping_data2 = []
workload_B_target2 = []

workload_B_mapping_data3 = []
workload_B_target3 = []

workload_B_mapping_data4 = []
workload_B_target4 = []

workload_B_mapping_data5 = []
workload_B_target5 = []

workload_B_mapping_data6 = []
workload_B_target6 = []

for i in range(num_workloads_B):
    workload_B_mapping_data.append(workload_B_list[i][:-1])
    workload_B_target.append(workload_B_list[i][5:6])

workload_B_mapping_data_list.append(workload_B_mapping_data)   
workload_B_target_list.append(workload_B_target)

for i in range(num_workloads_B):
    workload_B_mapping_data2.append(pd.concat([(workload_B_list[i][0:4]),(workload_B_list[i][5:6])]))
    workload_B_target2.append((workload_B_list[i][4:5]))

workload_B_mapping_data_list.append(workload_B_mapping_data2)   
workload_B_target_list.append(workload_B_target2)    
    
for i in range(num_workloads_B):
    workload_B_mapping_data3.append(pd.concat([(workload_B_list[i][0:3]),(workload_B_list[i][4:6])]))
    workload_B_target3.append((workload_B_list[i][3:4]))
    
workload_B_mapping_data_list.append(workload_B_mapping_data3)   
workload_B_target_list.append(workload_B_target3)

for i in range(num_workloads_B):
    workload_B_mapping_data4.append(pd.concat([(workload_B_list[i][0:2]),(workload_B_list[i][3:6])]))
    workload_B_target4.append((workload_B_list[i][2:3]))

workload_B_mapping_data_list.append(workload_B_mapping_data4)   
workload_B_target_list.append(workload_B_target4)

for i in range(num_workloads_B):
    workload_B_mapping_data5.append(pd.concat([(workload_B_list[i][0:1]),(workload_B_list[i][2:6])]))
    workload_B_target5.append((workload_B_list[i][1:2]))

workload_B_mapping_data_list.append(workload_B_mapping_data5)   
workload_B_target_list.append(workload_B_target5)

for i in range(num_workloads_B):
    workload_B_mapping_data6.append(pd.concat([(workload_B_list[i][0:0]),(workload_B_list[i][1:6])]))
    workload_B_target6.append((workload_B_list[i][0:1]))

workload_B_mapping_data_list.append(workload_B_mapping_data6)   
workload_B_target_list.append(workload_B_target6)
print("Load workload B finish!")

#######################################################################
# Load workload C
workload_C=pd.read_csv('./online_workload_C.csv') 


tmp_workload_C_knobs_metrics = (workload_C[knobs_metrics_name]).values


Cworkload_knobs_metrics_Norm = dataScalar.transform(tmp_workload_C_knobs_metrics)
Ctmp_workload_id = (workload_C[work_id_name]).values

Cworkload_knobs_metrics_Norm_id = np.hstack((Ctmp_workload_id,Cworkload_knobs_metrics_Norm))

norm_workload_C = pd.DataFrame(data=Cworkload_knobs_metrics_Norm_id,columns=col_list)

workload_C_ids = norm_workload_C["workload id"].unique()
workload_C_list = []
for num,workload_C_id in enumerate(workload_C_ids,start=1):
    tmp_workload = norm_workload_C[norm_workload_C['workload id'].isin([workload_C_id])]
    workload_C_list.append(tmp_workload)
    exec("workload_C_%s = tmp_workload"%num)
    
workload_C_target=pd.read_csv('./test.csv')

tmp_workload_C_knobs_metrics_t = (workload_C_target[knobs_list]).values
tmp_fill_value = np.zeros((100,313-12))

tmp_workload_C_knobs_metrics_t_withfill = np.hstack((tmp_workload_C_knobs_metrics_t,tmp_fill_value))

Cworkload_knobs_metrics_Norm_t = dataScalar.transform(tmp_workload_C_knobs_metrics_t_withfill)
Ctmp_workload_id_t = (workload_C_target[work_id_name]).values

Cworkload_knobs_metrics_Norm_id_t = np.hstack((Ctmp_workload_id_t,Cworkload_knobs_metrics_Norm_t))
norm_workload_C_t = pd.DataFrame(data=Cworkload_knobs_metrics_Norm_id_t,columns=col_list)

workload_C_ids_t = norm_workload_C_t["workload id"].unique()
workload_C_list_t = []
for num,workload_C_id in enumerate(workload_C_ids_t,start=1):
    tmp_workload = norm_workload_C_t[norm_workload_C_t['workload id'].isin([workload_C_id])]
    workload_C_list_t.append(tmp_workload)
    exec("workload_C_%s = tmp_workload"%num)

print("Load workload C finish!")

# Load File finish
# Trainning and validation
print("Trainning and validation test begin")
kernel_1 = 4.0 * RBF(length_scale=1.8) + WhiteKernel()
a1 = FAforAllworkloads(170,tmpframe)
nk,b1 = KmeanForallworkloads(2,a1,metrics_name)
d1 = BuildGPRForall(kernel_1,workload_A_list,b1)
e1 = FindMostSimliarIndexForTarget(workload_A_list,workload_B_mapping_data_list[0],d1,b1,kernel_1,nk)
f1 = Concatenate(workload_A_list,workload_B_mapping_data_list[0],b1,kernel_1,e1)
print("Validation test with workloads B: ")
h1 = PredictMAPE(workload_B_target_list[0],f1,dataScalar,b1,e1,latencyScalar)
print("Metrics retained after pruning: ", b1)
workload_B_target_list_pair = []
for i in range(100):
    workload_B_target_list_pair.append(workload_A_ids[e1[i]])
    
workload_B_target_list_array = np.array(workload_B_target_list_pair)
workload_B_nearset_pair = np.vstack((workload_B_ids,workload_B_target_list_array))
workload_B_nearset_pair_re = workload_B_nearset_pair.T
workload_B_nearset_pair_re_pd = pd.DataFrame(data=workload_B_nearset_pair_re,columns=['workload_B workload id','offline workload id'])
workload_B_nearset_pair_re_pd.to_csv('./workload_b_nearest_pair.csv')
print("The result for Nearest neighbor for each workload in workload B is in the file workload_b_nearest_pair.csv at current directory")

#Predict with workload C and the test file
print("Predict with workload C and the test file begin")
a = FAforAllworkloads(170,tmpframe)
nk,b = KmeanForallworkloads(2,a,metrics_name)
d = BuildGPRForall(kernel_1,workload_A_list,b)
e = FindMostSimliarIndexForTarget(workload_A_list,workload_C_list,d,b,kernel_1,nk)
f = Concatenate(workload_A_list,workload_C_list,b,kernel_1,e)
h = PredictMAPEwithoutGroundTruth(workload_C_list_t,f,dataScalar,b,e,latencyScalar)
workload_c_target_list = []
for i in range(100):
    workload_c_target_list.append(workload_A_ids[e[i]])
    
workload_c_target_list_array = np.array(workload_c_target_list)
workload_c_nearset_pair = np.vstack((workload_C_ids,workload_c_target_list_array))
workload_c_nearset_pair_re = workload_c_nearset_pair.T
workload_c_nearset_pair_re_pd = pd.DataFrame(data=workload_c_nearset_pair_re,columns=['workload_C workload id','offline workload id'])
workload_c_nearset_pair_re_pd.to_csv('./workload_c_nearest_pair.csv')
print("The result for Nearest neighbor for each workload in workload C is in the file workload_c_nearest_pair.csv at current directory")

predcit_list_col = ['workload id','k1','k2','k3','k4','k5','k6','k7','k8','s1','s2','s3','s4','latency prediction']
workloadC_predict_array = np.array(h).reshape(100,1)
workload_C_withpredict = np.hstack((tmp_workload_C_knobs_metrics_t,workloadC_predict_array))
workload_C_withpredict_id = np.hstack((Ctmp_workload_id_t,workload_C_withpredict))
pd_workload_C_withpredict_id = pd.DataFrame(data=workload_C_withpredict_id,columns=predcit_list_col)
pd_workload_C_withpredict_id.to_csv('./Test_with_latency.csv')
print("The result for latency prediction for Test file is in the file Test_with_latency.csv at current directory")