# -*- coding: utf-8 -*-
"""
This script uses the Deep ESN for the classification of audio signals recorded 
in construction sites.

The Deep ESN implementation is based on the work proposed in:
- Gallicchio,  C.,  Micheli,  A.,  Pedrelli,  L.: Deep  reservoir  computing:  
A  critical  experimental  analysis.    Neurocomputing268,  87–99  (2017).    
https://doi.org/10.1016/j.neucom.2016.12.08924. 

whose source code is available at: https://github.com/gallicch/DeepRC-TF

Created on Thu Jun  1 18:01:53 2023

@author: Michele Scarpiniti -- DIET Dpt., Sapienza University of Rome
"""

# General imports
import numpy as np
import scipy.io
# import pandas as pd
# import time
import DeepRC as DRC

# Scikit-learn imports
# from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, accuracy_score
# from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Custom imports
import matplotlib.pyplot as plt
import yaml


# ============ Loading the configuration file ==============================
config_file = 'CS.yaml'
# with open(config_file, 'r') as f:
#     config = yaml.load(f, Loader=yaml.Loader)

# print(config)


# ============ RC model configuration and hyperparameter values ============

# If you wish not using the YAML configuration file, you can set here your 
# model configuration and all the used hyperparameters. 
config = {}

# General settings
config['dataset_name'] = 'cantiere'
config['result_path'] = 'E:/Data/DeepESN/code/Results/CS/'

config['seed'] = 1

# Hyperarameters of the reservoir
config['layers'] = 5                    # number of layers of the reservoir
config['units'] = 200                   # size of the reservoir
config['spectral_radius'] = 0.9         # largest eigenvalue of the reservoir
config['leaky'] = 0.1                   # amount of leakage in the reservoir state update (None or 1.0 --> no leakage)
config['input_scaling'] = 0.8           # scaling of the input weights
config['inter_scaling'] = 0.8           # scaling of the input weights
config['connectivity_recurrent'] = 10   # percentage of nonzero connections in the reservoir
config['connectivity_input'] = 10       # percentage of nonzero connections in the input matrix
config['connectivity_inter'] = 10       # percentage of nonzero connections in the inter-layer matrix
config['concat'] = True                 # if True, concatenates all states
config['return_sequences'] = False      # if True, return the time state sequence

# Dimensionality reduction hyperparameters
config['dimred_method'] = 'pca'         # options: {None (no dimensionality reduction), 'pca'}
config['n_dim'] = 80                    # number of resulting dimensions after the dimensionality reduction procedure


# Type of readout
config['readout_type'] = 'lin'          # readout used for classification: {'lin', 'knn', 'mlp', 'svm', 'rf'}

# Linear readout hyperparameters
config['alpha'] = 0.5                   # regularization of the ridge regression readout

# kNN readout hyperparameters
config['n_neighbors'] = 5               # number of neighbors

# SVM readout hyperparameters
config['svm_kernel'] = 'rbf'            # kernel of the SVM
config['svm_gamma'] = 0.005             # bandwith of the RBF kernel
config['svm_C'] = 10                    # regularization for SVM hyperplane

# MLP readout hyperparameters
config['mlp_layout'] = (80,20)          # neurons in each MLP layer
config['num_epochs'] = 200              # number of epochs
config['w_l2'] = 0.001                  # weight of the L2 regularization
config['nonlinearity'] = 'relu'         # type of activation function {'relu', 'tanh', 'logistic', 'identity'}

# Random Forests hyperparameters
config['n_trees'] = 100                 # number of tree estimators in the forest

print(config)


# ============ Load dataset ================================================
data = scipy.io.loadmat(config['dataset_name'] + '.mat')
np.random.seed(config['seed'])

Xtr = data['X']  # shape is [N, T, V]
if len(Xtr.shape) < 3:
    Xtr = np.atleast_3d(Xtr)
Ytr = data['Y']  # shape is [N, K]

Xte = data['Xte']
if len(Xte.shape) < 3:
    Xte = np.atleast_3d(Xte)
Yte = data['Yte']

print("Xtr shape", Xtr.shape)
print("Ytr shape", Ytr.shape)
print('Loaded '+config['dataset_name']+' - Tr: '+ str(Xtr.shape)+', Te: '+str(Xte.shape))

Ytr = Ytr.ravel()
Yte = Yte.ravel()


# One-hot encoding for labels
# onehot_encoder = OneHotEncoder(sparse=False)
# Ytr = onehot_encoder.fit_transform(Ytr)
# Yte = onehot_encoder.transform(Yte)



# %% ============ Test a specific ESN configuration ========================

# DeepESN model
esn = DRC.SimpleDeepESNStates(
             units = config['units'],
             layers = config['layers'],
             concat = config['concat'],
             spectral_radius = config['spectral_radius'],
             leaky = config['leaky'],
             input_scaling = config['input_scaling'],
             inter_scaling = config['inter_scaling'],
             connectivity_recurrent = config['connectivity_recurrent'],
             connectivity_input = config['connectivity_input'],
             connectivity_inter = config['connectivity_inter'],
             return_sequences = config['return_sequences']
)

# Compute states
states_tr = esn(Xtr)
states_te = esn(Xte)


# Dimensionality reduction
if ((config['dimred_method'] == 'pca') and (config['units']*config['layers'] <= config['n_dim'])):
    N_pca = int(0.8*config['units']*config['layers'])
else:
    N_pca = config['n_dim']

if config['dimred_method'] is not None:
    pca = PCA(n_components=N_pca)
    states_tr = pca.fit_transform(states_tr)
    states_te = pca.transform(states_te)


# Select the readout classifier
if config['readout_type'] == 'lin':
    clf = RidgeClassifier(alpha=config['alpha'])
elif config['readout_type'] == 'knn':
    clf = KNeighborsClassifier(n_neighbors=config['n_neighbors'])
elif config['readout_type'] == 'mlp':
    clf = MLPClassifier(hidden_layer_sizes=config['mlp_layout'])
elif config['readout_type'] == 'svm':
    clf = SVC(kernel=config['svm_kernel'], C=config['svm_C'])
elif config['readout_type'] == 'rf':
    clf = RandomForestClassifier(n_estimators=config['n_trees'])
else:
    print('Readout classifier not set. Using linear by default.')
    clf = RidgeClassifier(alpha=config['alpha'])
    

# Train and test the selected readout classifier
clf.fit(states_tr, Ytr)
Ypred = clf.predict(states_te)

acc  = accuracy_score(Yte, Ypred)
# prec = precision_score(Yte, y_pred)
# rec  = recall_score(Yte, y_pred)
f1 = f1_score(Yte, Ypred, average='weighted')
        

# Print metrics
print("Overall accuracy: {}%".format(round(100*acc,2)))
# print("Precision: {}%".format(round(100*prec,2)))
# print("Recall: {}%".format(round(100*rec,2)))
print("F1-score: {}%".format(round(100*f1,2)))
        
        
# Confusion Matrix (CM)
class_names = ['JD50D', 'IRCOM', 'ConcreteMixer', 'CAT320E', 'Hitachi50U']
title_cm = "Accuracy: {}%".format(round(100*acc,2))
cm = confusion_matrix(Yte, Ypred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues')
plt.title(title_cm)
plt.show()
