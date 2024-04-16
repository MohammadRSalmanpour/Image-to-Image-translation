# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:16:52 2024

@author: Amin
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 21:33:16 2024

@author: Amin
"""

import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.utils import to_categorical
from resnet50 import ResNet50
from resnet50_AB import ResNet50_AB
from resnet18 import ResNet18
from resnet18_AB import ResNet18_AB
from vgg16 import VGG16
from simple_model import SimpleModel
from book_model import base_model
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
import keras_tuner as kt

tf.keras.utils.set_random_seed(42)

algorithm = 'DualGAN'
labeling = 'Labeled_NPZ_Data_UCLA'
project_dir_synthetic = os.path.join(os.getcwd(), f'{labeling}', algorithm)
project_dir_real = os.path.join(os.getcwd(), f'{labeling}', 'US_MRI')

print('project_dir_synthetic: ', project_dir_synthetic)
print('project_dir_real: ', project_dir_real)

category_lists = ['synthetic', 'real']
train_category = 'synthetic'
#val_category = 'real'
test_category = 'synthetic'

us_type = False
train_combination = False

# train category path
if train_category == 'synthetic':
    project_dir_train = project_dir_synthetic
else:
    project_dir_train = project_dir_real

# test category path    
if test_category == 'synthetic':
    project_dir_test = project_dir_synthetic
else:
    project_dir_test = project_dir_real

if train_combination == False:
    train_patients = glob.glob(project_dir_train + '/train/*/*.npz')
else:
    combination_path_train = os.path.join(os.getcwd(), f'{labeling}', 'US_MRI')
    train_patients_synthetic = glob.glob(project_dir_train + '/train/*/*.npz')
    train_patients_real = glob.glob(combination_path_train + '/train/*/*.npz')
    train_patients = train_patients_synthetic + train_patients_real
    
#val_patients = glob.glob(project_dir_val + '/val/*/*.npz')
test_patients = glob.glob(project_dir_test + '/test/*/*.npz')

print(train_patients[0])
#print(val_patients[0])
print(test_patients[0])

def load_data(files_path, category='real', us_image=False):
    images_list = []
    labels_list = []
    for file_path in files_path:
        label_name = str(file_path).split('/')[-2]

        if label_name == 'A':
            label = 0  
        else:
            label = 1
        
        if category == 'real' and us_image == False:
            type_image = 'mri'
        elif category == 'real' and us_image == True:
            type_image = 'us'
        elif category == 'synthetic':
            type_image = 'pred'
        else:
            type_image = 'mri'
        
        data = np.load(file_path, allow_pickle=True)  # convert Tensor to numpy
        keywords = list(data.keys())
        images_list.append(data[type_image])
        labels_list.append(label)
        
    dataset_images= np.array(images_list)
    dataset_labels = np.array(labels_list)

    return dataset_images, dataset_labels
    
train_images, train_labels = load_data(train_patients, category=train_category, us_image=us_type)
print('The Number of Train: ', len(train_images))
print('Images Train Shape: ', train_images.shape)
print('Labels Train Shape: ', train_labels.shape)

test_images, test_labels = load_data(test_patients, category=test_category, us_image=us_type)
print('The Number of Test: ', len(test_images))
print('Images Test Shape: ', test_images.shape)
print('Labels Test Shape: ', test_labels.shape)

model = ResNet50_AB(input_size=(128, 128, 64), classes=1)
model.summary()

loss_fn = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]


model.compile(loss=loss_fn,
              optimizer=optimizer,
              metrics=metrics)

batch_size = 1
epochs = 12
model.fit(train_images, train_labels,
          batch_size=batch_size,
          epochs=epochs
          )

loss_train, accuracy_train, auc_train = model.evaluate(train_images, train_labels)
print('Train Accuracy: {:.2f}'.format(accuracy_train))
print('Train AUC: {:.2f}'.format(auc_train))
                    
loss_test, accuracy_test, auc_test = model.evaluate(test_images, test_labels)
print('Test Accuracy: {:.2f}'.format(accuracy_test))
print('Test AUC: {:.2f}'.format(auc_test))

