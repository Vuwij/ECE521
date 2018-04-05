import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#np.set_printoptions(threshold=np.nan)

def make_image(visualization_weights):
    plt.figure(figsize = (10,10))
    gs1 = gridspec.GridSpec(10, 10)
    gs1.update(wspace=0.03, hspace=0.03) # set the spacing between axes. 
    
    for i in range(100):
       # i = i + 1 # grid spec indexes from 0
        ax1 = plt.subplot(gs1[i])
        plt.imshow(visualization_weights[i], cmap="gray")
        plt.axis('off')    
    
    


def load_data():
    with np.load("notMNIST.npz") as data:
        Data, Target = data ["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255.
        Data = Data.reshape(-1,784)
        Target = Target[randIndx]
        
        
        #convert Target to one-hot encoding 
        one_hot_target = np.zeros((len(Target), 10))
        one_hot_target[np.arange(len(Target)),Target]  = 1
       
        
        #split data into train, valid and test
        trainData, trainTarget = Data[:15000], one_hot_target[:15000]
        validData, validTarget = Data[15000:16000], one_hot_target[15000:16000]
        testData, testTarget = Data[16000:], one_hot_target[16000:]    
    return trainData, trainTarget, validData, validTarget, testData, testTarget

def compute_weighted_sum(previous_layer, n_units):
    
    #xavier initialization
    dev = tf.sqrt(3/(n_units[0] + n_units[1]))
    initW = tf.truncated_normal(shape=[n_units[0],n_units[1]], stddev = dev )
   
    #weight matrix
    weight_matrix = tf.Variable(initW, [n_units[0], n_units[1]], name = "weight_matrix")
    
    #bias (broadcasted)
    bias = tf.Variable(tf.truncated_normal(shape=[1,n_units[1]],stddev=dev), [1,n_units[1]], name="bias")
    
    #weighted sum
    Z = tf.matmul(previous_layer,weight_matrix) + bias
    
    return Z, weight_matrix
  
def build_neural_network():
 
    # Network Hyperparameters
    n_hidden_1 = 1000 # 1st layer number of features
    n_input = 784 # MNIST data input (img shape: 28*28)
    n_classes = 10 # MNIST total classes (0-9 digits)
    regularization = 3E-4
    learning_rate = 8E-4
    keep_prob = 0.5
    
    #weight matrices list 
    weight_matrices = [ [[]] for i in range(2) ]
    
    # weighted sum 
    Z =  [ [[]] for i in range(2) ]
    
    #inputs and targets
    X = tf.placeholder(tf.float32, [None,784], name = "reg" )
    target = tf.placeholder(tf.float32, [None,10], name = "target") 
    
    #hidden_layer_1
    Z[0], weight_matrices[0]= compute_weighted_sum(X, (n_input, n_hidden_1) )
    hidden_layer = tf.nn.relu(Z[0]) 
    
    #comment out this line for no dropout 
    hidden_layer = tf.nn.dropout(hidden_layer, keep_prob)
    
    #output_layer
    Z[1], weight_matrices[1] = compute_weighted_sum(hidden_layer, (n_hidden_1, n_classes) )
    Y_predicted =  tf.nn.softmax(Z[1]) 
    
    # cross entropy loss
    loss_CE = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z[1], labels=target))
    
    # deadweight (regularization) loss
    loss_w = 0
    for i in range(len(weight_matrices)):
        loss_w += tf.reduce_sum(tf.square(weight_matrices[i]))*regularization/2
        
    # sum loss    
    total_loss = loss_CE + loss_w
    
    # Adam Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train = optimizer.minimize(loss=total_loss)     
    
   
    # predictions 
    argmax = tf.reshape(tf.argmax(Y_predicted, 1), [-1,1])
    result = tf.one_hot( indices=argmax, depth=10, on_value=1., off_value=0.)
    result = tf.reshape(result, [-1,10])


    # Classification accuracy
    correction_predictions = tf.equal(tf.argmax(Y_predicted, 1), tf.argmax(target,1))
    accuracy = tf.reduce_mean(tf.cast(correction_predictions,tf.float32))
    
    
    #visualization
    layer_1_weights_transpose = tf.matrix_transpose(weight_matrices[0])
    weights_3d = tf.reshape(layer_1_weights_transpose, (n_hidden_1,28,28))*1E3 #resize to 1000*28*28    
    
    return train, total_loss, X, result, target, accuracy, loss_CE, weights_3d
    


if __name__ == '__main__':
   
    
    #Hyperparameter 
    batch_size = 500   
    
    #number of epochs +1 
    epoch = 23
    
    #display step
    display_step = 1
    
    
    trainData, trainTarget, validData, validTarget, testData, testTarget = load_data() 
    train, total_loss, X, result, target, accuracy, loss_CE, weights_3d = build_neural_network()
    
    init = tf.global_variables_initializer()
    randIdx = np.arange(15000)
    sess =  tf.InteractiveSession() 
    sess.run(init)         
  

   
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()      
    #saver.restore(sess, "/Users/winst/Google Drive/3S/ECE521/Assignment2/checkpoints/model.ckpt")
    #print("Model restored.")    
    
   
   
    
    # record accuracy and loss for plotting
    training_acc_list = []
    valid_acc_list = []
    test_acc_list = []
    
    train_cost_list = [] 
    valid_cost_list = []
    test_cost_lost = []
    
    input_x = trainData[randIdx[0 : batch_size]]
    input_y = trainTarget[randIdx[0 :batch_size]]       
    
    before_train_acc, train_cost, t_predicted = sess.run([accuracy, loss_CE, result], feed_dict= {X: input_x, target: input_y})  
    
    
    print("before training: train acc: %s "%(before_train_acc))
    
    len_of_x_plot = 0

    #fig = plt.figure()
     ##train neural net
    for i in range(epoch+1):
        np.random.shuffle(randIdx) 
        for j in range(int(len(trainData)/batch_size)):
            
            input_x = trainData[randIdx[j*batch_size : (j+1)*batch_size]]
            input_y = trainTarget[randIdx[j*batch_size : (j+1)*batch_size]]   
            
            #adam descent on training data
            _, train_acc, train_cost, t_predicted = sess.run([train, accuracy, loss_CE, result], feed_dict= {X: input_x, target: input_y})    
            
        if(i%display_step==0):    
            
            len_of_x_plot += 1
            
            #query validation 
            valid_acc, valid_cost = sess.run([accuracy, loss_CE], feed_dict = {X:validData, target:validTarget})
                
            #query test
            test_acc, test_cost = sess.run([accuracy, loss_CE], feed_dict = {X:testData, target:testTarget})
                
    
            print("i: %s "%(i))
            print("training accuracy: %s"%(train_acc))
            print("valid accuracy: %s"%(valid_acc))
            print("test accuracy: %s"%(test_acc))            
    
    
            # record for plots 
            training_acc_list.append(train_acc)  
            valid_acc_list.append(valid_acc)
            test_acc_list.append(test_acc)
            
            train_cost_list.append(train_cost) 
            valid_cost_list.append(valid_cost) 
            test_cost_lost.append(test_cost) 
        
        
       
        if(i == int(epoch*0.25) or i == int(epoch*0.5) or i == int(epoch*0.75) or i == epoch):
            visualization_weights = sess.run(weights_3d)
            make_image(visualization_weights)
              
         
 
   
    


  
  
  
