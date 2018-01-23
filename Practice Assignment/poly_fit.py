# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 01:13:21 2017

@author: ECE521
"""

import numpy as np
import matplotlib.pyplot as plt

poly_degree = 9
lambda_reg = 0.1

def load_data():
    data = np.load('./datapoints.npy')
    train_data = data[:,0]
    train_target = data[:,1]
    return train_data, train_target


def poly_map(input_data, poly_degree):
    input_data = np.expand_dims(input_data, 1)
    poly_data = input_data

    for i in range(2, poly_degree + 1):
        poly_data = np.append(poly_data, input_data**i, 1)
    
    return poly_data 
    
def opt_weights(train_data, train_target, lambda_reg, poly_degree):
    Np = np.shape(train_data)[0]    
    poly_data = np.append(np.ones([Np, 1]), poly_map(train_data, poly_degree), 1)
    Id_matrix = np.eye(poly_degree + 1)
    Id_matrix[0,0] = 0
    xx_inv = np.linalg.inv(np.dot(np.transpose(poly_data), poly_data) + \
                                                   lambda_reg*Id_matrix)
    W_opt = np.dot(np.dot(xx_inv, np.transpose(poly_data)), train_target)
    return W_opt
    

train_data, train_target = load_data()
W_opt = opt_weights(train_data, train_target, lambda_reg, poly_degree)
print('Optimum Weights are:')
print(W_opt)
plt.close('all')
plt.scatter(train_data, train_target)
plt.xlabel('input (x)')
plt.ylabel('target (t)')

X_test = np.linspace(-2, 2, 100)
yhat = np.dot(poly_map(X_test, poly_degree), W_opt[1:poly_degree+1]) +  W_opt[0]
plt.plot(X_test, yhat, 'r')