import numpy as np
import tensorflow as tf 
import math
import matplotlib.pyplot as plt

def load_data():
    rawData =  [ [data] for data in np.load('data2D.npy')]
    mid = math.floor(len(rawData)/3)
    validData = rawData[0:mid]
    trainData = rawData[mid:]
    return validData, trainData
    
def build_graph(epochs, K, N, learning_rate):

    
    #parameters
    x = tf.placeholder(tf.float32,[None,1,2],name="x")
    mu_T = tf.Variable(tf.random_normal([1,K,2], mean = 0, stddev = 1), [1,K,2],name="mu_T")
    pairwise_distance_matrix = tf.sqrt(tf.reduce_sum(tf.square(x - mu_T),2)) # N by K
    argmins = tf.argmin(pairwise_distance_matrix, 1)
    total_loss = tf.reduce_mean(tf.reduce_min(pairwise_distance_matrix, 1))
    # Adam Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,  beta1=0.9, beta2=0.99, epsilon=1e-5)
    train = optimizer.minimize(loss=total_loss)  
    return total_loss, pairwise_distance_matrix,x,mu_T, train, argmins


if __name__ == '__main__':
    validData, trainData = load_data()
    #Hyperparameter
    epochs = 1000
    K = 5
    N = len(trainData)
    learning_rate = 8E-3
    
    total_loss, pairwise_distance_matrix,x,mu_T, train, argmins = build_graph(epochs, K, N, learning_rate)
    sess =  tf.InteractiveSession() 
    init = tf.global_variables_initializer()
    sess.run(init)  
    
    plot1_x = []
    plot1_y = []
    plot2_x = []
    plot2_y = [] 
    
    for i in range(epochs):
        _, train_loss = sess.run([train, total_loss], feed_dict = {x: trainData})
        valid_loss = sess.run(total_loss, feed_dict = {x: validData})
        plot1_x.append(i)
        plot1_y.append(train_loss)
        plot2_x.append(i)
        plot2_y.append(valid_loss)        
        
    #plot1
    plt.figure(1) 
    plt.plot(plot1_x,plot1_y, label="traing_loss")
    plt.plot(plot2_x,plot2_y, label="validation_loss")
    plt.legend()
    print("Final training loss:" + str(train_loss))
    print("Final vlidation loss:" + str(valid_loss))
    
   



'''
[    [ [0],  [0]  ]  , [ [1],[1] ]       ]  2*2*1
[    [ [0,0],[1,1]]  ]           ]          1*2*2 
'''

