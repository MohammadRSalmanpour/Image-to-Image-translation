import shutil
import matplotlib.pyplot as plt
import glob
import os
import shutil
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.stats.mstats import spearmanr
from sklearn.metrics import mean_absolute_percentage_error as mape
import pingouin as pg
import nibabel as nib
import random

algorithms = os.listdir('/home/mohammad/Desktop/GAN/Algorithms/RadioMix_Data/Radiomics_Feature_Results')

df_mri = pd.read_excel('/home/mohammad/Desktop/GAN/Algorithms/RadioMix_Data/MRI/All_extracted_features_01-08-2024_034324.xlsx')

df_pearson_test = pd.DataFrame()
df_spearman_test = pd.DataFrame()

df_pearson_val = pd.DataFrame()
df_spearman_val = pd.DataFrame()

df_icc_test = pd.DataFrame()
df_pval_test = pd.DataFrame()

df_icc_val = pd.DataFrame()
df_pval_val = pd.DataFrame()

df_pvalue_spearman_test = pd.DataFrame()
df_pvalue_spearman_val = pd.DataFrame()

for algorithm in algorithms:

    df_test = pd.read_excel('/home/mohammad/Desktop/GAN/Algorithms/RadioMix_Data/Radiomics_Feature_Results/{}/test/features.xlsx'.format(algorithm))
    df_val = pd.read_excel('/home/mohammad/Desktop/GAN/Algorithms/RadioMix_Data/Radiomics_Feature_Results/{}/val/features.xlsx'.format(algorithm))

    key = lambda row: (row.PatientID)
    df_mri_test = df_mri[df_mri.apply(key, axis=1).isin(df_test.apply(key, axis=1))]
    df_mri_val = df_mri[df_mri.apply(key, axis=1).isin(df_val.apply(key, axis=1))]

    df_selected_test = df_test[df_test.apply(key, axis=1).isin(df_mri_test.apply(key, axis=1))]
    df_selected_val = df_val[df_val.apply(key, axis=1).isin(df_mri_val.apply(key, axis=1))]
    
    data_pearson_test = []
    data_spearman_test = []
    data_icc_test = []
    data_pval_test = []
    data_pvalue_spearman_test = []
    df_mape_test = pd.DataFrame()
    
    for column in df_mri_test.columns[3:]:
        corr_pearson, p_value_pearson = pearsonr(df_mri_test[column], df_selected_test[column])
        corr_spearman, p_value_spearman = spearmanr(df_mri_test[column], df_selected_test[column])
        data_pvalue_spearman_test.append(p_value_spearman)
        data_pearson_test.append(corr_pearson)
        data_spearman_test.append(corr_spearman)
        df_mape_test[column] = abs(df_mri_test[column].values - df_selected_test[column].values) / df_mri_test[column].values
        
        df_icc_1_ = pd.DataFrame()
        df_icc_1_['PatientID'] = np.arange(1, len(df_mri_test['PatientID']) + 1)
        df_icc_1_['MP'] = ['M' for i in range(len(df_icc_1_['PatientID']))]
        df_icc_2_ = pd.DataFrame({'PatientID': np.arange(1, len(df_mri_test['PatientID']) + 1), 'MP':['P' for i in range(len(df_icc_1_['PatientID']))]})
        df_icc_1_ = df_icc_1_._append(df_icc_2_, ignore_index = True)
        df_icc_1_['Feature_Values'] = list(df_mri_test[column].values) + list(df_selected_test[column].values)
        results = pg.intraclass_corr(data=df_icc_1_, targets='PatientID', raters='MP', ratings='Feature_Values')
        results = results.set_index('Description')
        icc = results.loc['Average random raters', 'ICC']
        pval = results.loc['Average random raters', 'pval']
        data_icc_test.append(icc)
        data_pval_test.append(pval)
        
    df_pvalue_spearman_test[algorithm] = data_pvalue_spearman_test
    
    df_pearson_test[algorithm] = data_pearson_test
    df_spearman_test[algorithm] = data_spearman_test
    
    df_icc_test[algorithm] = data_icc_test
    df_pval_test[algorithm] = data_pval_test
    
    df_mape_test.insert(0, 'PatientID', df_mri_test['PatientID'].values)
    df_mape_test.to_excel('/home/mohammad/Desktop/GAN/Algorithms/RadioMix_Data/Result New/MAPE_test_{}.xlsx'.format(algorithm), index=False)

    #df_corr = pd.DataFrame(data, columns=['Feature', algorithm])
    
    data_pearson_val = []
    data_spearman_val = []
    data_icc_val = []
    data_pval_val = []
    data_pvalue_spearman_val = []
    df_mape_val = pd.DataFrame()
    
    for column in df_mri_val.columns[3:]:
        corr_pearson, p_value_pearson = pearsonr(df_mri_val[column], df_selected_val[column])
        corr_spearman, p_value_spearman = spearmanr(df_mri_val[column], df_selected_val[column])
        data_pvalue_spearman_val.append(p_value_spearman)
        data_pearson_val.append(corr_pearson)
        data_spearman_val.append(corr_spearman)
        df_mape_val[column] = abs(df_mri_val[column].values - df_selected_val[column].values) / df_mri_val[column].values
        
        df_icc_1_ = pd.DataFrame()
        df_icc_1_['PatientID'] = np.arange(1, len(df_mri_val['PatientID']) + 1)
        df_icc_1_['MP'] = ['M' for i in range(len(df_icc_1_['PatientID']))]
        df_icc_2_ = pd.DataFrame({'PatientID': np.arange(1, len(df_mri_val['PatientID']) + 1), 'MP':['P' for i in range(len(df_icc_1_['PatientID']))]})
        df_icc_1_ = df_icc_1_._append(df_icc_2_, ignore_index = True)
        df_icc_1_['Feature_Values'] = list(df_mri_val[column].values) + list(df_selected_val[column].values)
        results = pg.intraclass_corr(data=df_icc_1_, targets='PatientID', raters='MP', ratings='Feature_Values')
        results = results.set_index('Description')
        icc = results.loc['Average random raters', 'ICC']
        pval = results.loc['Average random raters', 'pval']
        data_icc_val.append(icc)
        data_pval_val.append(pval)
    
    df_pvalue_spearman_val[algorithm] = data_pvalue_spearman_val
    
    df_pearson_val[algorithm] = data_pearson_val
    df_spearman_val[algorithm] = data_spearman_val
    
    df_icc_val[algorithm] = data_icc_val
    df_pval_val[algorithm] = data_pval_val
    
    df_mape_val.insert(0, 'PatientID', df_mri_val['PatientID'].values)
    df_mape_val.to_excel('/home/mohammad/Desktop/GAN/Algorithms/RadioMix_Data/Result New/MAPE_val_{}.xlsx'.format(algorithm), index=False)
    #df_corr = pd.DataFrame(data, columns=['Feature', algorithm])

df_pearson_test.insert(0, 'FeatureNames', df_mri_test.columns[3:])    
df_spearman_test.insert(0, 'FeatureNames', df_mri_test.columns[3:])
df_pearson_test.to_excel('/home/mohammad/Desktop/GAN/Algorithms/RadioMix_Data/Result New/test_Pearson_corr.xlsx', index=False)
df_spearman_test.to_excel('/home/mohammad/Desktop/GAN/Algorithms/RadioMix_Data/Result New/test_Spearman_corr.xlsx', index=False)

df_pearson_val.insert(0, 'FeatureNames', df_mri_val.columns[3:])    
df_spearman_val.insert(0, 'FeatureNames', df_mri_val.columns[3:])
df_pearson_val.to_excel('/home/mohammad/Desktop/GAN/Algorithms/RadioMix_Data/Result New/val_Pearson_corr.xlsx', index=False)
df_spearman_val.to_excel('/home/mohammad/Desktop/GAN/Algorithms/RadioMix_Data/Result New/val_Spearman_corr.xlsx', index=False)

df_icc_test.insert(0, 'FeatureNames', df_mri_test.columns[3:])    
df_pval_test.insert(0, 'FeatureNames', df_mri_test.columns[3:])
df_icc_test.to_excel('/home/mohammad/Desktop/GAN/Algorithms/RadioMix_Data/Result New/test_ICC.xlsx', index=False)
df_pval_test.to_excel('/home/mohammad/Desktop/GAN/Algorithms/RadioMix_Data/Result New/test_Pvalue.xlsx', index=False)

df_icc_val.insert(0, 'FeatureNames', df_mri_val.columns[3:])    
df_pval_val.insert(0, 'FeatureNames', df_mri_val.columns[3:])
df_icc_val.to_excel('/home/mohammad/Desktop/GAN/Algorithms/RadioMix_Data/Result New/val_ICC.xlsx', index=False)
df_pval_val.to_excel('/home/mohammad/Desktop/GAN/Algorithms/RadioMix_Data/Result New/val_Pvalue.xlsx', index=False)

df_pvalue_spearman_test.insert(0, 'FeatureNames', df_mri_test.columns[3:])    
df_pvalue_spearman_val.insert(0, 'FeatureNames', df_mri_val.columns[3:])
df_pvalue_spearman_test.to_excel('/home/mohammad/Desktop/GAN/Algorithms/RadioMix_Data/Result New/test_Spearman_Pvalue.xlsx', index=False)
df_pvalue_spearman_val.to_excel('/home/mohammad/Desktop/GAN/Algorithms/RadioMix_Data/Result New/val_Spearman_Pvalue.xlsx', index=False)
