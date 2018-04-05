import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math


#LESSSON LEARN: CHECK UR INPUTS WHEN CALL FUNCTION 

def random_search_hyperparameter():
    np.random.seed(1001614853) # last 6 digits of student ID 1001 614 853 minus 1
    n_layers = np.random.random_integers(1,5)
    n_hidden_units = np.random.random_integers(100,500, size = (n_layers))
    with_dropout = np.random.random_integers(0,1) # with dropout = 1
    log_learning_rate = np.random.ranf() *3 - 7.5
    log_regularization = np.random.ranf()*3 - 9
    return n_layers, n_hidden_units, with_dropout, log_learning_rate, log_regularization

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
    dev = math.sqrt(3/(n_units[0] + n_units[1]))
    initW = tf.truncated_normal(shape=[n_units[0],n_units[1]], stddev = dev )
   
    #weight matrix
    weight_matrix = tf.Variable(initW, [n_units[0], n_units[1]], name = "weight_matrix")
    
    #bias (broadcasted)
    bias = tf.Variable(tf.truncated_normal(shape=[1,n_units[1]],stddev=dev), [1,n_units[1]], name="bias")
    
    #weighted sum
    Z = tf.matmul(previous_layer,weight_matrix) + bias
    
    return Z, weight_matrix
  
  
def build_neural_network():
    
 
    #inputs and targets
    X = tf.placeholder(tf.float32, [None,784], name = "reg" )
    target = tf.placeholder(tf.float32, [None,10], name = "target") 
    
    
    # Network Hyperparameters
    n_input = 784 # MNIST data input (img shape: 28*28)
    n_classes = 10 # MNIST total classes (0-9 digits)
    n_layers, n_hidden_units, with_dropout, log_learning_rate, log_regularization = random_search_hyperparameter()
    learning_rate = math.exp(log_learning_rate)
    regularization = math.exp(log_regularization)
    keep_prob = 0.5
    
    #seed 
    tf.set_random_seed(1001614853)
    
    #print hyperparameters 
    print("Number of Layers: %s \n"%(n_layers))
    print("hidden units in each layer: %s \n"%(n_hidden_units))
    print("learning_rate: %s \n"%(learning_rate))
    print("weight decay/regularization coefficient: %s \n"%(regularization))
    if with_dropout == 1: 
        print("With Dropout \n")
    else:
        print("No Dropout \n")
    
    
    
    
    #weight matrices list 
    weight_matrices = [ [[]] for i in range(n_layers+1) ]
    
    # weighted sum 
    Z =  [ [[]] for i in range(n_layers+1) ]    
    
    #hidden layer 
    hidden_layer= [ [[]] for i in range(n_layers)] #has one less element than z and weight
    
    #hidden_layer_1
    Z[0], weight_matrices[0]= compute_weighted_sum(X, (n_input, n_hidden_units[0]) )
    hidden_layer[0] = tf.nn.relu(Z[0]) 
    if with_dropout == 1: #with dropout
        hidden_layer[0] = tf.nn.dropout(hidden_layer[0], keep_prob)


    # generate hidden layers from second layer onwards 
    for i in range(1, n_layers):
        Z[i], weight_matrices[i] = compute_weighted_sum(hidden_layer[i-1],(n_hidden_units[i-1], n_hidden_units[i]))
        hidden_layer[i] = tf.nn.relu(Z[i]) 
        if with_dropout == 1: #with dropout
            hidden_layer[i] = tf.nn.dropout(hidden_layer[i], keep_prob)
        
    
    #output_layer
    Z[n_layers], weight_matrices[n_layers] = compute_weighted_sum(hidden_layer[-1], (n_hidden_units[-1], n_classes) )
    Y_predicted =  tf.nn.softmax(Z[n_layers]) 
    
    
    
    # cross entropy loss
    loss_CE = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z[n_layers], labels=target))
    
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
    
    
    return train, total_loss, loss_CE, X, result, target, accuracy
    


if __name__ == '__main__':
   
    
    #Hyperparameter 
    batch_size = 500   
    
    #number of epochs
    epoch = 50
    
    #display step
    display_step = 1
    
    #load data
    trainData, trainTarget, validData, validTarget, testData, testTarget = load_data() 
    
    #build network
    train, total_loss, loss_CE, X, result, target, accuracy = build_neural_network()
    
    #initialization
    init = tf.initialize_all_variables()  
    randIdx = np.arange(15000)
    sess =  tf.InteractiveSession() 
    sess.run(init)         
  
    
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()    
    
    # record accuracy and loss for plotting
    training_acc_list = []
    valid_acc_list = []
    test_acc_list = []
    train_cost_list = [] 
    valid_cost_list = []
    test_cost_lost = []
    
    #Before training inspection
    input_x = trainData[randIdx[0 : batch_size]]
    input_y = trainTarget[randIdx[0 :batch_size]]       
    before_train_acc, train_cost, t_predicted = sess.run([accuracy, total_loss, result], feed_dict= {X: input_x, target: input_y})  
    print("before training: train acc: %s "%(before_train_acc))

    
    len_of_x_plot = 0    
     ##train neural net
    for i in range(epoch):
        np.random.shuffle(randIdx) 
        for j in range(int(len(trainData)/batch_size)):
            
            input_x = trainData[randIdx[j*batch_size : (j+1)*batch_size]]
            input_y = trainTarget[randIdx[j*batch_size : (j+1)*batch_size]]   
            
            #adam descent on training data
            _, train_acc, train_cost, t_predicted = sess.run([train, accuracy, loss_CE, result], feed_dict= {X: input_x, target: input_y})    
            
            
        #query validation 
        valid_acc, valid_cost = sess.run([accuracy, loss_CE], feed_dict = {X:validData, target:validTarget})
        
        #query test
        test_acc, test_cost = sess.run([accuracy, loss_CE], feed_dict = {X:testData, target:testTarget})
        
        if (i%display_step == 0):
            len_of_x_plot+= 1
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
    
   
    
    #create plots
    x_plot = [i for i in range(len_of_x_plot)]
    plt.figure(1) 
    plt.plot(x_plot,training_acc_list, label="training acc")
    plt.plot(x_plot, valid_acc_list, label = "validationg acc")
    plt.plot(x_plot, test_acc_list, label = "test acc")
    plt.legend()
    
    #second figure for loss
    plt.figure(2) 
    plt.plot(x_plot,train_cost_list, label = "training loss")
    plt.plot(x_plot, valid_cost_list, label = "validationg loss")
    plt.plot(x_plot, test_cost_lost, label = "test loss")
    plt.legend()
 
  
  
  
  
  
  