# -*- coding: utf-8 -*-
"""
Created on Wed May  6 18:08:10 2020

@author: 28215
"""
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler


knobs_list = ['k1','k2','k3','k4','k5','k6','k7','k8','s1','s2','s3','s4']


def FAforAllworkloads(n_c,frame):
    all_metrics_data = frame.values

    all_metrics_data_Trans = all_metrics_data.T
    tmp_all_transformer = FactorAnalysis(n_components=n_c, random_state=0)
    tmp_workload_A_transformed = tmp_all_transformer.fit_transform(all_metrics_data_Trans)
    
    return tmp_workload_A_transformed

def KmeanForallworkloads(n_k,faworkloads,m_name):
    workload_A_list_array_Kmeans = KMeans(n_clusters=n_k, random_state=0).fit(faworkloads)
    cluster_centers = workload_A_list_array_Kmeans.cluster_centers_
    tmp_centers_index = []
    tmp_centers_metrics_name = []
    
    for i in range(n_k):
        tmp_center = cluster_centers[i]
        tmp_center_diff = tmp_center - faworkloads
        tmp_center_distance = np.sum((tmp_center_diff * tmp_center_diff),axis=1)
        tmp_center_min_idx = np.argmin(tmp_center_distance)
        tmp_centers_index.append(tmp_center_min_idx)
    
    for j in range(n_k):
        tmp_mname = m_name[tmp_centers_index[j]]
        tmp_centers_metrics_name.append(tmp_mname)

    tmp_centers_metrics_name = ['latency'] + tmp_centers_metrics_name
    #print(tmp_centers_metrics_name)
    num_M = 1 + n_k
    workload_list_centers_metrics_name = []
    for n in range(58):
        workload_list_centers_metrics_name.append(tmp_centers_metrics_name)
        
    return num_M,tmp_centers_metrics_name

def BuildGPRForall(kernel_t,workload_A,centers_metrics_name):
    tmp_kernel = kernel_t
    
    workload_A_GPRmodels_lists = []
    workload_A_scalar_lists = []
    workload_A_normalized_m = []
    
    for i in range(58):
        tmp_GPRmodels_lists = []
        tmp_scalar_lists = []
        tmp_normalized_m = []
    
    
        tmp_Pruning_workload_A_X = ((workload_A[i])[knobs_list]).values
    
        tmp_num_centers = len(centers_metrics_name)
        
        for j in range(tmp_num_centers):
        
            tmp_Pruning_workload_A_y_m = ((workload_A[i])[centers_metrics_name[j]]).values
            tmp_workload_A_gpr_m = GaussianProcessRegressor(kernel=tmp_kernel).fit(tmp_Pruning_workload_A_X, tmp_Pruning_workload_A_y_m)
            tmp_GPRmodels_lists.append(tmp_workload_A_gpr_m)
        
        workload_A_GPRmodels_lists.append(tmp_GPRmodels_lists)
    return workload_A_GPRmodels_lists

def FindMostSimliarIndexForTarget(workload_A,workload_mapping,GPRmodelList,center_metric_name,t_kernel,n_m):
    all_distance_list = []
    for i in range(100):
        tmp_test_workload_B_mapping = workload_mapping[i]
        tmp_test_workload_B_mapping_cfigs = ((tmp_test_workload_B_mapping)[knobs_list]).values
    
        tmp_distance_list = []
    
        for j in range(58):
            tmp_test_workload_A = workload_A[j]
            tmp_test_workload_A_gpr_model = GPRmodelList[j]   
            tmp_test_workload_A_cfigs = ((tmp_test_workload_A)[knobs_list]).values
        
            tmp_corr_metric_values = []
        
            for k in range(5):
                tmp_data_point = tmp_test_workload_B_mapping_cfigs[k]
                tmp_check = np.where((tmp_data_point==tmp_test_workload_A_cfigs[:,None]).all(-1))
            
                if len(tmp_check[0]) == 0:
                    tmp_gpr_predic = []
                    for v in range(n_m):
                        tmp_gpr_predic.append(tmp_test_workload_A_gpr_model[v].predict([tmp_data_point])[0])
                    tmp_corr_metric_values.append(tmp_gpr_predic)
            
                else:
                    tmp_gpr_predic = []
                    tmp_row_idx = tmp_check[0][0]
                    for n in range(n_m):
                        tmp_gpr_predic.append((((tmp_test_workload_A)[[center_metric_name[n]]]).values)[tmp_row_idx])
                    tmp_corr_metric_values.append(tmp_gpr_predic)
        
            tmp_test_workload_B_mapping_metrics_norm_value_list = []
            for m in range(n_m):
                tmp_test_workload_B_mapping_metrics_value = tmp_test_workload_B_mapping[[center_metric_name[m]]].values

                tmp_test_workload_B_mapping_metrics_norm_value_list.append(tmp_test_workload_B_mapping_metrics_value)


            tmp_test_workload_B_mapping_metrics_norm_value_arry = np.asarray(tmp_test_workload_B_mapping_metrics_norm_value_list).reshape((n_m,5))       
            tmp_corr_metric_values_arry = np.asarray(tmp_corr_metric_values).reshape((5,n_m)).T

            up_min_list = np.min(tmp_test_workload_B_mapping_metrics_norm_value_arry,axis=0)
            down_min_list = np.min(tmp_corr_metric_values_arry,axis=0)
            tot_min_list = np.min(np.vstack((up_min_list,down_min_list)),axis=0)
        
            up_max_list = np.max(tmp_test_workload_B_mapping_metrics_norm_value_arry,axis=0)
            down_max_list = np.max(tmp_corr_metric_values_arry,axis=0)
            tot_max_list = np.max(np.vstack((up_max_list,down_max_list)),axis=0)

            tmp_test_workload_B_mapping_metrics_norm_value_arry_norm = np.divide((tmp_test_workload_B_mapping_metrics_norm_value_arry - tot_min_list),(tot_max_list - tot_min_list))
            tmp_corr_metric_values_arry_norm = np.divide((tmp_corr_metric_values_arry - tot_min_list),(tot_max_list - tot_min_list))

            tmp_distance = np.sum((tmp_test_workload_B_mapping_metrics_norm_value_arry_norm-tmp_corr_metric_values_arry_norm)*(tmp_test_workload_B_mapping_metrics_norm_value_arry_norm-tmp_corr_metric_values_arry_norm))
            tmp_distance_list.append(tmp_distance)
     
        all_distance_list.append(tmp_distance_list)


    
    all_min_idx_list = []
    for i in range(100):
        tmp_idx = all_distance_list[i].index(min(all_distance_list[i]))
        all_min_idx_list.append(tmp_idx)
    return all_min_idx_list

def Concatenate(workload_A,workload_mapping,center_metric_name,t_kernel,min_idx_list):
    predictGPRmodel_list = []



    for i in range(100):
        
        tmp_similar_idx = min_idx_list[i]
        tmp_similar_workload = workload_A[tmp_similar_idx]
        tmp_similar_workload_configs = ((tmp_similar_workload)[knobs_list]).values
        tmp_similar_workload_metrics = ((tmp_similar_workload)[[center_metric_name[0]]]).values
    
        tmp_Augmented_pruned_target_configs = np.array(tmp_similar_workload_configs)
        tmp_Augmented_pruned_target_metrics = np.array(tmp_similar_workload_metrics)
    
        tmp_workload_B_configs = workload_mapping[i][knobs_list].values
        tmp_workload_B_metrics = workload_mapping[i][[center_metric_name[0]]].values
    
        for j in range(5):
            tmp_test_workload_B_mapping_cfigs = tmp_workload_B_configs[j]
            tmp_test_workload_B_mapping_metrics_with_similar = tmp_workload_B_metrics[j]
            tmp_check = np.where((tmp_test_workload_B_mapping_cfigs==tmp_similar_workload_configs[:,None]).all(-1))
        
            if len(tmp_check[0]) == 0:
            
                tmp_Augmented_pruned_target_configs = np.append(tmp_Augmented_pruned_target_configs,[tmp_test_workload_B_mapping_cfigs],axis=0)
                tmp_Augmented_pruned_target_metrics = np.append(tmp_Augmented_pruned_target_metrics,[tmp_test_workload_B_mapping_metrics_with_similar],axis=0)
            else:
        
                tmp_row_idx = tmp_check[0][0]
                tmp_Augmented_pruned_target_metrics[tmp_row_idx] = tmp_test_workload_B_mapping_metrics_with_similar
    
        test_workload_B_gpr_o = GaussianProcessRegressor(kernel=t_kernel).fit(tmp_Augmented_pruned_target_configs, tmp_Augmented_pruned_target_metrics)
        predictGPRmodel_list.append(test_workload_B_gpr_o)
        
    return predictGPRmodel_list

def PredictMAPE(workload_target,predGPRmodelList,dscalar,center_metric_name,min_idx_list,latencyScalar):
    rightSet = {}

    totPercent = 0
    right_count = 0
    for i in range(100):
        tmp_target_config = workload_target[i][knobs_list].values
        tmp_max = dscalar.data_max_[12]
        tmp_min = dscalar.data_min_[12]
        tmp_predict = predGPRmodelList[i].predict(tmp_target_config)
        

        tmp_predict_re = latencyScalar.inverse_transform(tmp_predict)
        tmp_ground_Truth = latencyScalar.inverse_transform(workload_target[i][[center_metric_name[0]]].values)
        
        tmp_diff = np.abs(tmp_predict_re - tmp_ground_Truth)

        totPercent += tmp_diff/tmp_ground_Truth
        if (tmp_diff/tmp_ground_Truth) < 0.3:
            right_count += 1
            rightSet[i] = min_idx_list[i]

    print('MAPE: ',totPercent/100)
    print('Right Predict with absolute percentage error less than 0.3: ', right_count)
    return totPercent,rightSet

def PredictMAPEwithoutGroundTruth(workload_target,predGPRmodelList,dscalar,center_metric_name,min_idx_list,latencyScalar):
    predictList = []
    for i in range(100):
        tmp_target_config = workload_target[i][knobs_list].values
        tmp_max = dscalar.data_max_[12]
        tmp_min = dscalar.data_min_[12]
        tmp_predict = predGPRmodelList[i].predict(tmp_target_config)
        tmp_predict_re = latencyScalar.inverse_transform(tmp_predict)
        if tmp_predict_re < tmp_min:
            tmp_predict_re = np.array([[tmp_min]])
        predictList.append(tmp_predict_re)
        
    return predictList