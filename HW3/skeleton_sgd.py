#################################
# Your name:Odai Agbaria, Id:212609440
#################################

import matplotlib.pyplot as plt
import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels



def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """
    np.random.seed(0) 

    n=data.shape[0]
    d=data.shape[1]
    w=np.zeros(d)

    for t in range(1,T+1):
        eta_t=eta_0/t

        i=np.random.randint(0,n)
        x=data[i]
        y=labels[i]

        if y*np.dot(w,x)<1:
            w=(1-eta_t)*w+eta_t*C*y*x
        else:
            w=(1-eta_t)*w
    
    return w



def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    """
    np.random.seed(0) 

    n=data.shape[0]
    d=data.shape[1]
    w=np.zeros(d)

    for t in range(1,T+1):
        eta_t=eta_0/t

        i=np.random.randint(0,n)
        x=data[i]
        y=labels[i]

        dot_prod=np.dot(w, x)
        w=w+eta_t*((y*x*np.exp(-y*dot_prod))/(1+np.exp(-y*dot_prod)))
    
    return w

#################################

def Q1a():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels=helper()
    eta_0_values=np.logspace(-5,5,num=75)
    avg_accs=[]

    for eta_0 in eta_0_values:
        temp_acc=[]
        for i in range(10):
            w=SGD_hinge(train_data,train_labels,1,eta_0,1000)
            predictions=np.sign(np.dot(validation_data, w))
            acc = np.mean(predictions == validation_labels)
            temp_acc.append(acc)
        avg_acc = np.mean(temp_acc)
        avg_accs.append(avg_acc)

    plt.semilogx(eta_0_values, avg_accs,color='black', marker='*')
    plt.xlabel('Initial Learning Rate (eta_0)')
    plt.ylabel('Average Validation Accuracy')
    plt.title('Validation Accuracy vs. Initial Learning Rate (eta_0)')
    plt.grid(True)
    plt.show()


def Q1b():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels=helper()
    opt_eta=0.723
    C_values=np.logspace(-5,5,num=75)
    avg_accs=[]

    for C in C_values:
        temp_acc=[]
        for i in range(10):
            w=SGD_hinge(train_data,train_labels,C,opt_eta,1000)
            predictions=np.sign(np.dot(validation_data, w))
            acc = np.mean(predictions == validation_labels)
            temp_acc.append(acc)
        avg_acc = np.mean(temp_acc)
        avg_accs.append(avg_acc)

    plt.semilogx(C_values, avg_accs,color='orange', marker='*')
    plt.xlabel('Regularization Parameter (C)')
    plt.ylabel('Average Validation Accuracy')
    plt.title('Validation Accuracy vs. Regularization Parameter (C)')
    plt.grid(True)
    plt.show()

def Q1c():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels=helper()
    opt_eta=0.723
    opt_C=0.000161
    w = SGD_hinge(train_data, train_labels, opt_C, opt_eta, 20000)

    w_image = w.reshape(28, 28)

    plt.imshow(w_image, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Visualization of the Weight Vector')
    plt.show()

def Q1d():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels=helper()
    opt_eta=0.723
    opt_C=0.000161
    w = SGD_hinge(train_data, train_labels, opt_C, opt_eta, 20000)
    predictions=np.sign(np.dot(test_data, w))
    acc = np.mean(predictions == test_labels)
    print("Accuracy is:",acc)

def Q2a():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels=helper()
    eta_0_values=np.logspace(-5,5,num=75)
    avg_accs=[]

    for eta_0 in eta_0_values:
        temp_acc=[]
        for i in range(10):
            w=SGD_log(train_data,train_labels,eta_0,1000)
            predictions=np.sign(np.dot(validation_data, w))
            acc = np.mean(predictions == validation_labels)
            temp_acc.append(acc)
        avg_acc = np.mean(temp_acc)
        avg_accs.append(avg_acc)

    plt.semilogx(eta_0_values, avg_accs,color='magenta', marker='*')
    plt.xlabel('Initial Learning Rate (eta_0)')
    plt.ylabel('Average Validation Accuracy')
    plt.title('Validation Accuracy vs. Initial Learning Rate (eta_0)')
    plt.grid(True)
    plt.show()

def Q2b():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels=helper()
    opt_eta=0.00001
    w = SGD_log(train_data, train_labels, opt_eta, 20000)
    predictions=np.sign(np.dot(test_data, w))
    acc = np.mean(predictions == test_labels)
    print("Accuracy is:",acc)

    w_image = w.reshape(28, 28)

    plt.imshow(w_image, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Visualization of the Weight Vector')
    plt.show()

def Q2c_help(data, labels, eta_0, T):
    np.random.seed(0) 

    n=data.shape[0]
    d=data.shape[1]
    w=np.zeros(d)
    norms=[]

    for t in range(1,T+1):
        eta_t=eta_0/t

        i=np.random.randint(0,n)
        x=data[i]
        y=labels[i]

        dot_prod=np.dot(w, x)
        w=w+eta_t*((y*x*np.exp(-y*dot_prod))/(1+np.exp(-y*dot_prod)))
        norms.append(np.linalg.norm(w))
    
    return w,norms

def Q2c():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels=helper()
    opt_eta=0.00001
    w,norms=Q2c_help(train_data,train_labels,opt_eta,20000)
    plt.plot(range(1, 20001), norms,color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Norm of weight vector w')
    plt.title('Norm of w as a function of the iteration')
    plt.grid(True)
    plt.show()
    

#################################

if __name__ == '__main__':
    Q2c()