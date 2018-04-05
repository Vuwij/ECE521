import tensorflow as tf
import numpy as np
from utils import reduce_logsumexp, logsoftmax
import matplotlib.pyplot as plt
import math

#Q2.1.2
#log_prob_x_given_z of size [B,K] has the probability of xn given normal distribution mu_k and var_k 
def compute_log_prob_x_given_z(x, mu_T, var,D):
    # dim(x) = [B,1,2], dim(mu_T) = [1,K,2]
    distance_matrix = tf.reduce_sum(tf.square(x - mu_T),2)# B by K
    # [B,K]                     [B,K] <--- ([B,K]   [1,K])              [1,K]                  [1,1]
    log_prob_x_given_z = -0.5*tf.div(distance_matrix, var) - 0.5*D*tf.log(var) - 0.5*D*tf.log(2*math.pi)
    return log_prob_x_given_z

#Q2.1.3
def compute_log_prob_z_given_x(log_prob_x_given_z,log_pi):
    #[B,K]     [1,K]         [B,K]                     [B,1]                     
    return log_pi + log_prob_x_given_z - reduce_logsumexp(tf.add(log_prob_x_given_z, log_pi),keep_dims=True)
   

def build_graph(B, K, learning_rate, D):
    
    #Parameters
    x = tf.placeholder(tf.float32,[None,1,D],name="x") # input data
    mu_T = tf.Variable(tf.random_normal([1,K,D], mean = 0, stddev = 0.001), [1,K,D],name="mu_T") # mu tranpose 
    phi = tf.Variable(tf.random_normal([1,K], mean=1, stddev=0), [1,K], name="phi")
    psi = tf.Variable(tf.random_normal([1,K], mean=1, stddev=0), [1,K], name="psi")
    var = tf.exp(phi) #variance [0, inf)
    log_pi = logsoftmax(psi) # sum(log_pi) = 1
    
    #Q2.1.2
    log_prob_x_given_z= compute_log_prob_x_given_z(x, mu_T, var,D)
    #Q2.1.3
    log_prob_z_given_x = compute_log_prob_z_given_x(log_prob_x_given_z, log_pi)
    
    argmaxs = tf.argmax(log_prob_z_given_x, 1)
    mu = tf.reduce_sum(mu_T,0)
    #loss
    #[1,1] <------------------- [B,1]
    total_loss = -tf.reduce_mean(reduce_logsumexp(tf.add(log_prob_x_given_z, log_pi),keep_dims=True))
    
    # Adam Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,  beta1=0.9, beta2=0.99, epsilon=1e-5)
    train = optimizer.minimize(loss=total_loss)    
    
    pi = tf.exp(log_pi)
    return x, mu_T, train, mu, total_loss, argmaxs, log_prob_x_given_z, pi, var


def load_data():
    rawData =  [ [data] for data in np.load('data100D.npy')]
    mid = math.floor(len(rawData)/3)
    validData = rawData[0:mid]
    trainData = rawData[mid:]
    return validData, trainData    

    
if __name__ == '__main__':
    #Hyperparameters
    epochs = 100
    B = 10000
    K = 12
    D = 100
    learning_rate = 0.15
    
    
    validData, trainData = load_data() 
    
    plot1_x = []
    plot1_y = []  
    plot2_x = []
    plot2_y = []        
    for K in range(1,9,1):
        x, mu_T, train, mu, total_loss, argmaxs, log_prob_x_given_z, pi, var = build_graph(B, K, learning_rate, D)
        sess =  tf.InteractiveSession() 
        init = tf.global_variables_initializer()
        sess.run(init)          
        for i in range(epochs):
            _, = sess.run([train], feed_dict = {x:trainData})
        train_loss = sess.run([total_loss], feed_dict = {x:trainData})     
        valid_loss = sess.run([total_loss], feed_dict = {x:validData})
        plot1_x.append(K)
        plot1_y.append(train_loss)
        plot2_x.append(K)
        plot2_y.append(valid_loss)        
        #plot1
    plt.figure(1) 
    plt.plot(plot1_x,plot1_y, label="train loss")
    plt.plot(plot2_x,plot2_y, label="valid loss")
    plt.legend()
    
    #K = 5 
    #x, mu_T, train, mu, total_loss, argmaxs, log_prob_x_given_z, pi, var = build_graph(B, K, learning_rate, D)
    #sess =  tf.InteractiveSession() 
    #init = tf.global_variables_initializer()
    #sess.run(init)          
    #for i in range(epochs):
        #_, = sess.run([train], feed_dict = {x:trainData})
        #train_loss = sess.run([total_loss], feed_dict = {x:trainData})     
        #valid_loss = sess.run([total_loss], feed_dict = {x:validData})
        #plot1_x.append(i)
        #plot1_y.append(train_loss)
        #plot2_x.append(i)
        #plot2_y.append(valid_loss)        
    ##plot1
    #plt.figure(1) 
    #plt.plot(plot1_x,plot1_y, label="train loss")
    #plt.plot(plot2_x,plot2_y, label="valid loss")
    #plt.legend()
      
    

    
    
  
