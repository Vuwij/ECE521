import tensorflow as tf 
import numpy as np


def load_data():
   s = np.random.multivariate_normal(np.zeros(3), np.identity(3), 10000) 
   W =np.array([[1, 1, 0],
                [0, 0.001, 0],
                [0, 0, 10]])
   return np.matmul(s,W)
   
#PCA with maximum variance formulation     
def build_PCA():
   #Hyperparameter
   D = 3
   #parameter
   x_PCA = tf.placeholder(tf.float32, [None, D], name="x_PCA")
   #mean
   x_mean_PCA = tf.reduce_mean(x_PCA,0)
   #variance
   S_PCA = compute_outer_product(x_PCA - x_mean_PCA) #compute outer product 
   eigen_val,eigen_vectors = tf.self_adjoint_eig(S_PCA, name="eigenvect")
   #u_PCA
   max_index = tf.argmax(eigen_val,0)
   u_PCA = tf.gather(eigen_vectors,max_index)
   return x_PCA, S_PCA, eigen_val, eigen_vectors, u_PCA

def compute_outer_product(diff): 
   arg1 = tf.expand_dims(diff,2) #[N,D,1]
   arg2 = tf.expand_dims(diff,1) #[N,1,D]
   outer_product = tf.reduce_mean(tf.multiply(arg1, arg2),0)
   return outer_product   

if __name__ == '__main__':
   #PCA
   data = load_data()
   x_PCA, S_PCA, eigen_val, eigen_vectors, u_PCA =  build_PCA()
   init = tf.global_variables_initializer()
   sess =  tf.InteractiveSession() 
   sess.run(init)    
   eigenvalues,eigenvectors, principal_component = sess.run([eigen_val, eigen_vectors, u_PCA], feed_dict = {x_PCA:data})

   #Report
   print("eigenvalues:")
   print(eigenvalues)
   print("\neigenvectors:")
   print(eigenvectors)
   print("\nprincipal component:")
   print(principal_component)