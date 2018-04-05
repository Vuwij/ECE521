import tensorflow as tf 
import numpy as np 
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def make_image(visualization_weights):
    plt.figure(figsize = (2,2))
    gs1 = gridspec.GridSpec(2, 2)
    gs1.update(wspace=0.1, hspace=0.1) # set the spacing between axes. 
    
    for i in range(4):
       # i = i + 1 # grid spec indexes from 0
        ax1 = plt.subplot(gs1[i])
        plt.imshow(visualization_weights[i], cmap="gray")
        plt.axis('off')    
    
def build_graph(K,D):
    learning_rate = 8E-2
    #paramaters 
    x = tf.placeholder(tf.float32, [None,D], name ="input_x") #train: [B,D]
    W = tf.Variable(tf.random_normal([D,K], mean=2, stddev=0.5), [D,K], name="weights")
    mu = tf.Variable(tf.random_normal([1,D], mean=10, stddev=0.5), [1,D], name="mu")
    Psi_vector = tf.Variable(tf.random_normal([D,], mean=10, stddev=0.5),[D,],name="psi_1d") 
    Psi_matrix = tf.diag(tf.exp(Psi_vector))
    covariance_matrix = Psi_matrix + tf.matmul(W, tf.matrix_transpose(W))
    #log determinant of covariance matrix 
    log_det_covar = 2.0 * tf.reduce_sum(tf.log(tf.diag_part(tf.cholesky(covariance_matrix))))
    #last term in liklihood function; size = B by 1 
    
    distance = tf.reduce_sum(tf.multiply(tf.matmul(x-mu, tf.matrix_inverse(covariance_matrix)),x-mu),1)#Alternative way to implement this: distance = tf.diag_part(tf.matmul(tf.matmul(x-mu, tf.matrix_inverse(covariance_matrix)),tf.matrix_transpose(x-mu)))
    
    #neg log likilhood
    neg_log_liklihood = -tf.reduce_mean(-0.5*D*tf.log(2*math.pi) - 0.5*log_det_covar - distance)
    #Adam Optimizer; minimize
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train = optimizer.minimize(loss=neg_log_liklihood)     
    #visualization 
    weights_3d = tf.reshape(tf.matrix_transpose(W), (K,8,8)) #resize to 4*28*28  
    return x, train, neg_log_liklihood, weights_3d, Psi_matrix, covariance_matrix
    

def load_data():
    # Loading my data       
    with np.load ("tinymnist.npz") as data :
        trainData, trainTarget = data ["x"], data["y"]
        validData, validTarget = data ["x_valid"], data ["y_valid"]
        testData, testTarget = data ["x_test"], data ["y_test"]    
    return trainData, trainTarget,validData, validTarget,testData, testTarget

if __name__ == '__main__':
    
    trainData, trainTarget, validData, validTarget, testData, testTarget = load_data() 
    #Hyperparameter 
    K = 4 # number of latent causes
    D = 64 # number of observed dimensions 
    epochs = 1000    
    
    x, train, neg_log_liklihood, weights_3d, Psi_matrix, covariance_matrix  = build_graph(K,D)
    init = tf.global_variables_initializer()
    sess =  tf.InteractiveSession() 
    sess.run(init)         
    
    
    #plots
    plot1_x = [] #train
    plot1_y = []  
    plot2_x = [] #valid
    plot2_y = []  
    plot3_x = [] #test    
    plot3_y = []  
    
    # training
    for i in range(epochs): 
        _, train_loss = sess.run([train,neg_log_liklihood], feed_dict = {x:trainData})
        plot1_x.append(i)
        plot1_y.append(train_loss)     
        valid_loss = sess.run(neg_log_liklihood, feed_dict = {x:validData})
        plot2_x.append(i)
        plot2_y.append(valid_loss)  
        test_loss = sess.run(neg_log_liklihood, feed_dict = {x:testData})
        plot3_x.append(i)
        plot3_y.append(test_loss)          
        
    #plot1
    plt.figure(2) 
    plt.plot(plot1_x,plot1_y, label="training_loss")
    plt.plot(plot2_x,plot2_y, label="valid_loss")
    plt.plot(plot3_x,plot3_y, label="test_loss")
    plt.legend()    
    
    print("train_loss")
    print(train_loss)
    print("valid_loss")
    print(valid_loss)    
    print("test_loss")
    print(test_loss)    
    
    #plot2
    visualization_weights = sess.run(weights_3d)
    make_image(visualization_weights)