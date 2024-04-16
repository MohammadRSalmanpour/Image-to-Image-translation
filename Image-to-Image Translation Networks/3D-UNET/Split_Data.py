#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os 
import pandas as pd
import numpy as np
import shutil

base_project_dir = os.path.abspath(os.path.dirname(__file__))  
parent_dir = os.path.join(base_project_dir, 'Data', 'Preprocessed')
dst_dir = os.path.join(base_project_dir, 'Data')

train_path = os.path.join(base_project_dir, 'Data', 'Patient_Order', 'train.csv')
val_path = os.path.join(base_project_dir, 'Data', 'Patient_Order', 'val.csv')
test_path = os.path.join(base_project_dir, 'Data', 'Patient_Order', 'test.csv')

data_dir_path = os.path.join(base_project_dir, 'Data', 'Split_Data')
train_split_path = os.path.join(data_dir_path, 'train')
val_split_path = os.path.join(data_dir_path, 'val')
test_split_path = os.path.join(data_dir_path, 'test')

if not os.path.exists(data_dir_path):
    os.mkdir(data_dir_path)

if not os.path.exists(train_split_path):
    os.mkdir(train_split_path)
    
if not os.path.exists(val_split_path):
    os.mkdir(val_split_path)
    
if not os.path.exists(test_split_path):
    os.mkdir(test_split_path)


df_train = pd.read_csv(train_path)
df_val = pd.read_csv(val_path)
df_test = pd.read_csv(test_path)


for train_patient in df_train['Patient ID']:
    shutil.copytree(os.path.join(parent_dir, train_patient), os.path.join(train_split_path, train_patient))
    
for val_patient in df_val['Patient ID']:
    shutil.copytree(os.path.join(parent_dir, val_patient), os.path.join(val_split_path, val_patient))
    
for test_patient in df_test['Patient ID']:
    shutil.copytree(os.path.join(parent_dir, test_patient), os.path.join(test_split_path, test_patient))






