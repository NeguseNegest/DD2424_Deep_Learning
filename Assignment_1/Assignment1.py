import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
import tarfile
import urllib.request
import os
import shutil
import torch




def LoadBatch(filename):
    with open(filename, "rb") as fo:
        batch = pickle.load(fo, encoding="bytes")

    X = batch[b"data"].astype(np.float64).T / 255.0   # 3072 x 10000  divide by 255.0 so to keep everything small 
    y = np.array(batch[b"labels"])                    # length 10000, since we have 1000 images

    K = 10
    n = X.shape[1]
    Y = np.zeros((K, n), dtype=X.dtype)
    Y[y, np.arange(n)] = 1 # each column represent the class representation of one image 

    return X, Y, y


def normalize(X, mean_X, std_X):
    return (X - mean_X) / std_X

def softmax(s):
    s = s - np.max(s, axis=0, keepdims=True)   # stability
    exp_s = np.exp(s)
    return exp_s / np.sum(exp_s, axis=0, keepdims=True)



def init_network(X_train):

    K = 10
    d = X_train.shape[0]

    rng = np.random.default_rng()
    BitGen = type(rng.bit_generator)
    seed = 42
    rng.bit_generator.state = BitGen(seed).state

    init_net = {
        'W': 0.01 * rng.standard_normal((K, d)),
        'b': np.zeros((K, 1))
    }
    return init_net

def ApplyNetwork(X, network):
    W = network['W']
    b = network['b']

    s = W @ X + b
    P = softmax(s)
    return P


def ComputeLoss(P, y):
    n = P.shape[1]
    correct_class_probs = P[y, np.arange(n)]
    L = -np.mean(np.log(correct_class_probs))
    return L

def BackwardPass(X, Y, P, network, lam):
    W = network['W']
    n = X.shape[1]

    G = P - Y

    grads = {}
    grads['W'] = (G @ X.T) / n + 2 * lam * W
    grads['b'] = np.sum(G, axis=1, keepdims=True) / n

    return grads



if __name__ == "__main__":

    FILEPATH="Datasets/cifar-10-batches-py/" # we add the directory path first 

    X_train,Y_train,y_train=LoadBatch(FILEPATH+"data_batch_1")

    X_validation,Y_validation,y_validation=LoadBatch(FILEPATH+"data_batch_2")

    X_test,Y_test,y_test=LoadBatch(FILEPATH+"test_batch")

    d = X_train.shape[0]

    mean_X = np.mean(X_train, axis=1).reshape(d, 1)
    std_X  = np.std(X_train, axis=1).reshape(d, 1)

    X_train = normalize(X_train, mean_X, std_X)
    X_validation = normalize(X_validation, mean_X, std_X)
    X_test = normalize(X_test, mean_X, std_X)




