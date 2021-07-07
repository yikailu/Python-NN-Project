import numpy as np
from scipy.optimize import minimize
from math import sqrt
import pickle
'''
You need to modify the functions except for initializeWeights() 
'''

def initializeWeights(n_in, n_out):
    '''
    initializeWeights return the random weights for Neural Network given the
    number of node in the input layer and output layer
    Input:
    n_in: number of nodes of the input layer
    n_out: number of nodes of the output layer
    Output:
    W: matrix of random initial weights with size (n_out x (n_in + 1))
    '''
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    '''
    Notice that z can be a scalar, a vector or a matrix
    return the sigmoid of input z (same dimensions as z)
    '''
    # remove the next line and replace it with your code
    return 1 / (1 + np.exp(-z))

def nnObjFunction(params, *args):
    '''
    % nnObjFunction computes the value of objective function (cross-entropy
    % with regularization) given the weights and the training data and lambda
    % - regularization hyper-parameter.
    % Input:
    % params: vector of weights of 2 matrices W1 (weights of connections from
    %     input layer to hidden layer) and W2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of nodes in input layer (not including the bias node)
    % n_hidden: number of nodes in hidden layer (not including the bias node)
    % n_class: number of nodes in output layer (number of classes in
    %     classification problem
    % train_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of the corresponding instance 
    % train_label: the vector of true labels of training instances. Each entry
    %     in the vector represents the truee label of its corresponding training instance.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector (not a matrix) of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.
    '''
    # do not remove the next 5 lines
    n_input, n_hidden, n_class, train_data, train_label, lambdaval = args
    # First reshape 'params' vector into 2 matrices of weights W1 and W2

    W1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    W2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    ########## CODE START HERE ##########
    # 1.1: Feedforward Propagation
    # define constants
    n_sample = train_data.shape[0]
    # add bias term to train_data
    train_data = np.c_[train_data, np.ones(n_sample)]
    # compute hidden layer
    hidden_net = np.dot(train_data, W1.T)
    hidden_out = sigmoid(hidden_net)
    # add bias term to hidden layer
    hidden_data = np.c_[hidden_out, np.ones(n_sample)]
    # compute output layer
    output_net = np.dot(hidden_data, W2.T)
    output_out = sigmoid(output_net)

    # 1.2: Error Function
    # train label matrix
    label_mat = np.zeros((n_sample, n_class))
    for i in range(n_sample):
        label_mat[i][int(train_label[i])] = 1
    # compute error without regularization
    obj_err = -np.sum(label_mat * np.log(output_out) + (1 - label_mat) * np.log(1 - output_out)) / n_sample
    # compute regularization error
    obj_reg = (np.sum(np.square(W1)) + np.sum(np.square(W2))) * lambdaval / (2 * n_sample)
    # final error
    obj_val = obj_err + obj_reg

    # 1.3: Backpropagation
    # difference between output and train label
    output_delta = output_out - label_mat
    # define constants 
    len_W1 = n_hidden * (n_input + 1)
    len_W2 = n_class * (n_hidden + 1)
    # derivative of error function (for weights of input to hidden)
    grad_hidden = (np.dot(((1 - hidden_out) * hidden_out * np.dot(output_delta, W2[:, :-1])).T, train_data) / n_sample).reshape(len_W1)
    # derivative of error function (for weights of hidden to output)
    grad_output = (np.dot(hidden_data.T, output_delta) / n_sample).T.reshape(len_W2)
    # the partial derivative of regularization term (for weights of input to hidden)
    grad_hidden_reg = (lambdaval * W1 / n_sample).reshape(len_W1)
    # the partial derivative of regularization term (for weights of hidden to output)
    grad_output_reg = (lambdaval * W2 / n_sample).reshape(len_W2)
    # final gradients
    obj_grad = np.concatenate((grad_hidden + grad_hidden_reg, grad_output + grad_output_reg))

    return (obj_val,obj_grad)

def nnPredict(W1, W2, data):
    '''
    % nnPredict predicts the label of data given the parameter W1, W2 of Neural
    % Network.
    % Input:
    % W1: matrix of weights for hidden layer units
    % W2: matrix of weights for output layer units
    % data: matrix of data. Each row of this matrix represents the feature vector for the corresponding data instance
    % Output:
    % label: a column vector of predicted labels
    '''
    ########## CODE START HERE ##########
    # Predict using the Feedforward Propagation
    # add bias term to train_data
    train_data = np.c_[data, np.ones(data.shape[0])]
    # compute hidden layer using W1
    hidden_net = np.dot(train_data, W1.T)
    hidden_out = sigmoid(hidden_net)
    # add bias term to hidden layer
    hidden_data = np.c_[hidden_out, np.ones(data.shape[0])]
    # compute output layer using W2
    output_net = np.dot(hidden_data, W2.T)
    output_out = sigmoid(output_net)
    # mark the highest vote as label
    labels = np.argmax(output_out, axis=1)
    return labels
