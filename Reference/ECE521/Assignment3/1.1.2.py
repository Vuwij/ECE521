import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt

def load_data():
    return  [ [data] for data in np.load('data2D.npy')]
    
    
def build_graph(epochs, K, N, learning_rate):

    
    #parameters
    x = tf.placeholder(tf.float32,[N,1,2],name="x")
    mu_T = tf.Variable(tf.random_normal([1,K,2], mean = 0, stddev = 1), [1,K,2],name="mu_T")
    pairwise_distance_matrix = tf.sqrt(tf.reduce_sum(tf.square(x - mu_T),2)) # N by K
    argmins = tf.argmin(pairwise_distance_matrix, 1)
    total_loss = tf.reduce_mean(tf.reduce_min(pairwise_distance_matrix, 1))
    # Adam Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,  beta1=0.9, beta2=0.99, epsilon=1e-5)
    train = optimizer.minimize(loss=total_loss)  
    return total_loss, pairwise_distance_matrix,x,mu_T, train, argmins


if __name__ == '__main__':
    data = load_data()
    #Hyperparameter
    epochs = 1000
    K = 1
    N = len(data)
    learning_rate = 8E-3
    
    total_loss, pairwise_distance_matrix,x,mu_T, train, argmins = build_graph(epochs, K, N, learning_rate)
    sess =  tf.InteractiveSession() 
    init = tf.global_variables_initializer()
    sess.run(init)    
    plot1_x = []
    plot1_y = []
    for i in range(epochs):
        _, loss = sess.run([train, total_loss], feed_dict = {x: data})
        plot1_x.append(i)
        plot1_y.append(loss)
        
    #plot1
    plt.figure(1) 
    plt.plot(plot1_x,plot1_y, label="loss")
    plt.legend()
    
    #finish training;query datapoints
    plot2_x = []
    plot2_y = []
    for i in range(K):
        plot2_x.append([])
        plot2_y.append([])

    mins = sess.run(argmins, feed_dict = {x:data})
    for i in range(len(data)):
        plot2_x[mins[i]].append(data[i][0][0])
        plot2_y[mins[i]].append(data[i][0][1])
         
    #plot2
    plt.figure(2) 
    for i in range(K):
        label_str = "mu" + str(i+1)
        plt.plot(plot2_x[i],plot2_y[i], 'o', label=label_str)
    plt.legend()
    
    #report percentage of points to each clusters
    mins.sort()
    temp = 0
    count = 0
    for i in range(len(data)):
        if mins[i] > temp:
            print("data points belonging to cluster"+ str(temp+1) + " = " + str(count) + "\n") 
            print("percetage = " + str(count/N) + "\n")
            temp += 1
            count = 1 
        else:
            count += 1       
    print("data points belonging to cluster"+ str(temp+1) + " = " + str(count) + "\n") 
    print("percetage = " + str(count/N) + "\n")



'''
[    [ [0],  [0]  ]  , [ [1],[1] ]       ]  2*2*1
[    [ [0,0],[1,1]]  ]           ]          1*2*2 
'''

