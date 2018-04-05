import tensorflow as tf 
import numpy as np
import math
import matplotlib.pyplot as plt

def load_data():
   s = np.random.multivariate_normal(np.zeros(3), np.identity(3), 10000) 
   W =np.array([[1, 1, 0],
                [0, 0.001, 0],
                [0, 0, 10]])
   return np.matmul(s,W)


def build_FA():
   learning_rate = 3E-3
   D = 3 # input dimensions
   K = 1 # latent causes
   #paramaters 
   x = tf.placeholder(tf.float32, [None,D], name ="input_x") #train: [B,D]
   W = tf.Variable(tf.random_normal([D,K], mean=2, stddev=0.1), [D,K], name="weights")
   mu = tf.Variable(tf.random_normal([1,D], mean=0, stddev=0.1), [1,D], name="mu")
   Psi_vector = tf.Variable(tf.random_normal([D,], mean=1, stddev=1),[D,],name="psi_1d") 
   Psi_matrix = tf.diag(tf.exp(Psi_vector))
   covariance_matrix = Psi_matrix + tf.matmul(W, tf.matrix_transpose(W))
   #log determinant of covariance matrix 
   log_det_covar = 2.0 * tf.reduce_sum(tf.log(tf.diag_part(tf.cholesky(covariance_matrix))))
   #last term in liklihood function; size = B by 1 
   distance = tf.reduce_sum(tf.multiply(tf.matmul(x-mu, tf.matrix_inverse(covariance_matrix)),x-mu),1)
   print(distance)
   #neg log likilhood
   neg_log_liklihood = -tf.reduce_mean(-0.5*D*tf.log(2*math.pi) - 0.5*log_det_covar - distance)
   #Adam Optimizer; minimize
   train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss=neg_log_liklihood)    
   #new W projection = Sigma * W_T * psi^-1
   W_T = tf.transpose(W)
   Psi_inverse = tf.matrix_inverse(Psi_matrix)
   Sigma = tf.matrix_inverse(1 + tf.matmul(tf.matmul(W_T,Psi_inverse),W))
   print(Sigma)
   W_proj = tf.matmul(tf.matmul(Sigma,W_T),Psi_inverse)
   return x, train, neg_log_liklihood, Psi_matrix, covariance_matrix, W_proj,W, mu

if __name__ == '__main__':
   epochs = 10000
   data = load_data()
   x, train, neg_log_liklihood, Psi_matrix, covariance_matrix, W_proj,W,mu = build_FA()
   init = tf.global_variables_initializer()
   sess =  tf.InteractiveSession() 
   sess.run(init)  
   #plots
   plot1_x = [] #train
   plot1_y = []     
   for i in range(epochs):
      _, train_loss = sess.run([train,neg_log_liklihood], feed_dict = {x:data})
      plot1_x.append(i)
      plot1_y.append(train_loss)  
   
   #plot1
   plt.figure(1) 
   plt.plot(plot1_x,plot1_y, label="training_loss")
   plt.legend()     
   projection, weight_mat, mean, psi = sess.run([W_proj,W,mu,Psi_matrix] , feed_dict = {x:load_data()})
   print("mean")
   print(mean)
   print("psi")
   print(psi)
   print("weight matrix W")
   print(weight_mat)
   print("W_proj")
   print(projection)
