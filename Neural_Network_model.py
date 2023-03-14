# Name: Ismail Al Abdali
# Course: CS460G
# Instructor: Dr.Brent Harrison

# ## imports



import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


### defining our neural network fucntions that are used for 
def sigmoid(x):
    return (1.0 / (1.0+np.exp(-x)))

def sigmoid_der(x):
    return (sigmoid(x) * (1.0 - sigmoid(x)))

def error(y , g_in):
    return y - g_in


# ### initialize weights for our neural network

def initialize_neural_network(hidden_layer_shape, hidden_to_out_connection_shape,output_layer_shape,bias):
    low_r,high_r = -1, 1
    in_weights = np.random.uniform(low_r,high_r,size = hidden_layer_shape)
    hid_weights = np.random.uniform(low_r,high_r,size = hidden_to_out_connection_shape)
    hid_bias = np.zeros(hidden_to_out_connection_shape) + bias
    out_bias = np.zeros(output_layer_shape) + bias
    
    # returns a dict of weights of input , hidden layer and biases.
    return {"in_w": in_weights,
           "hid_b": hid_bias,
           "hid_w": hid_weights,
           "out_b": out_bias}


# #### defining forward propagation 

# In[ ]:


## now let's code forward propagation for all x_train data.
def forward_propagation(net_parameters,X):
    """
    net_parameters: Neural network inilized paramters or resulted paramters
    x: training/testing.
    """
    # get neural_network paramets
    in_w,hid_b,hid_w,out_b = net_parameters["in_w"],net_parameters["hid_b"],net_parameters["hid_w"],net_parameters["out_b"]
    input_feat = np.array(X) # make sure to put X as np array.
    # get weighted sum between input-hidden
    weighted_sum_1 = np.dot(input_feat.T,in_w)
    # add bias with the resulted weighted sum 1
    weighted_sum_1_added_bias = np.add(weighted_sum_1,hid_b)
    # apply sigmoid funtion to each node of the hidden layer.
    sig_applied_h1 = np.array(list(map(sigmoid,weighted_sum_1_added_bias)))
    # get weighted sum between hidden-output layer
    weighted_sum_2 = np.dot(sig_applied_h1,hid_w.T)
    # add bias with the resulted wighted sum 2
    weighted_sum_2_bias_add = np.add(weighted_sum_2,out_b)
    # activate the output neurons
    in_o = sigmoid((weighted_sum_2_bias_add).flatten()[0])
    # return output of the layer, and activations of hidden layer 
    return {"out": in_o,
           "hid_act": sig_applied_h1}


# ### define Back Propagation

# In[ ]:


def backPropagate_and_getUpdateWeights(net_parameters,forward_dict,X,y_true,alpha):
    # get the weights
    in_w,hid_w = net_parameters["in_w"],net_parameters["hid_w"]
    # get biases
    hid_b,out_b = net_parameters["hid_b"],net_parameters["out_b"]
           
    in_out ,act_in_h1 = forward_dict["out"], forward_dict["hid_act"]
    # calculate error and find delta output
    err = error(y_true,in_out)
    delta_out = err * sigmoid_der(in_out) # calculcate output delta
    
    
    # now calculate hidden layer deltas
    hidden_layer_deltas = np.multiply(sigmoid_der(act_in_h1),hid_w) * delta_out
    # updates the weight of the hidden layer
    hid_w = np.add(hid_w,act_in_h1*delta_out * alpha)
    # update the weights of the input layer
    outer = np.outer(X,hidden_layer_deltas) # get the outer product with hidden layer deltas and input_feat
    in_w = np.multiply(np.add(in_w,outer),alpha)
    # update bias for output layer 
    out_b = out_b + alpha * delta_out
    # update bias for hidden layer
    hid_b = hid_b + alpha * hidden_layer_deltas

    # return the weights to update the nerual network.
    return {"up_input_w" :in_w,
           "up_hidden_w" :hid_w,
           "up_hidden_b" : hid_b,
           "up_output_b" : out_b}


# ### train/fit define Neural network

def train_neural_network(net_shapes,x_train,y_train,epoc = 1,alpha = 1,bias = 1):
    # defining shapes of neural network
    input_layer_shape = net_shapes["input_layer_shape"]
    hidden_layer_shape = net_shapes["hidden_layer_shape"]
    hidden_to_out_shape = net_shapes["hidden_to_out_shape"]
    output_layer_shape = net_shapes["output_layer_shape"]
    # initlize neural network
    net_params = initialize_neural_network(hidden_layer_shape,hidden_to_out_shape,output_layer_shape,bias = bias)
    for i in range(0,epoc):
        for i in range(len(x_train)):
            X = np.array(x_train.iloc[i])
            y_true =  np.array(y_train.iloc[i])
            # forward prapagate the network
            forward_result_dict = forward_propagation(net_params,X)
            # backward prapagate the nework and update weights.
            updated_weights = backPropagate_and_getUpdateWeights(net_params,forward_result_dict,X,y_true,alpha)
            # update weights
            net_params["in_w"],net_params["hid_w"] = updated_weights["up_input_w"],updated_weights["up_hidden_w"]
            # update bias terms
            net_params["hid_b"],net_params["out_b"] = updated_weights["up_hidden_b"],updated_weights["up_output_b"]
        # return the paramaters of the network
        return net_params


# ### define predict function

# In[ ]:
def model_predict(net_params,x_test):
    y_pred = []
    for i in range(len(x_test)):
        X = np.array(x_test.iloc[i])
        y_pred.append(round(forward_propagation(net_params,X)["out"]))
    return y_pred


# ##### define accuracy function to find our models accuracy 

# In[ ]:
def accuracy_score(y_true,y_pred):
    return np.sum(y_pred==y_true) / len(y_true)

