import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt
import math

def load_data():
    rawData =  [ [data] for data in np.load('data100D.npy')]
    mid = math.floor(len(rawData)/3)
    validData = rawData[0:mid]
    trainData = rawData[mid:]
    return validData, trainData
    
def build_graph(epochs, K, D, learning_rate):

    
    #parameters
    x = tf.placeholder(tf.float32,[None,1,D],name="x")
    mu_T = tf.Variable(tf.random_normal([1,K,D], mean = 0.5, stddev = 0.0001), [1,K,D],name="mu_T")
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
    epochs = 75
    K = 15
    D = 100 #dim of data points
    learning_rate = 8E-2
    plot_1_x = []
    plot_1_y = []
    plot_2_x = []
    plot_2_y = []
    for K in range(1,10,1):
        total_loss, pairwise_distance_matrix,x,mu_T, train, argmins = build_graph(epochs, K, D, learning_rate)
        sess =  tf.InteractiveSession() 
        init = tf.global_variables_initializer()
        sess.run(init)    
        for i in range(epochs):
            _, = sess.run([train], feed_dict = {x: trainData})
        train_loss = sess.run(total_loss, feed_dict = {x: trainData})    
        valid_loss = sess.run(total_loss, feed_dict = {x: validData})
        plot_1_x.append(K)
        plot_1_y.append(train_loss)
        plot_2_x.append(K)
        plot_2_y.append(valid_loss)        
        
        
        
    #total_loss, pairwise_distance_matrix,x,mu_T, train, argmins = build_graph(epochs, K, D, learning_rate)
    #sess =  tf.InteractiveSession() 
    #init = tf.global_variables_initializer()
    #sess.run(init)    
    #for i in range(epochs):
        #_, train_loss = sess.run([train, total_loss], feed_dict = {x: trainData})
        #valid_loss = sess.run(total_loss, feed_dict = {x: validData})
        #plot_1_x.append(i)
        #plot_1_y.append(train_loss)
        #plot_2_x.append(i)
        #plot_2_y.append(valid_loss)            
                
    plt.figure(1) 
    plt.plot(plot_1_x,plot_1_y, label="traing_loss")
    plt.plot(plot_2_x,plot_2_y, label="validation_loss")
    plt.legend()   



