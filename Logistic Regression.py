import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
from PIL import Image
import h5py
import scipy
from load_dataset import load_data

#---------------Loading DataSet --------------------------

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_data()

#---------------------checking if datas are imported correctly (will be commented later) -----------

'''index = 25
img = Image.fromarray(train_set_x_orig[index], 'RGB') # PIL module
img.show(img)
print(train_set_x_orig.shape)
print(train_set_y_orig.shape)
print(test_set_x_orig.shape)
print(test_set_y_orig.shape)'''

#---------------------Reshaping the Data --------------------------

'''A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b$*$c$*$d, a) is to use:

X_flatten = X.reshape(X.shape[0], -1).T'''

train_set_x_flat = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flat = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

#-------------Normalize and Standardize our data---------------------------

train_set_x = train_set_x_flat /255
test_set_x = test_set_x_flat /255

'''print(train_set_x.shape)
print(train_set_y.shape)
print(test_set_x.shape)
print(test_set_y.shape)'''
#----------------------------------sigmoid ----------------------

def sigmoid(z) :
    return 1/(1+np.exp(-z))

#-------------- initialization of w and b---------
def initialization(dim):
    w = np.zeros((dim,1))
    b = 0

    assert(w.shape == (dim,1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b

#------------costs and gradient -----------------

def calculation(X, Y, w, b) :
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -(1/m)*(np.sum((Y*np.log(A) + (1-Y)*np.log(1-A))))
    dw = (1/m)*(np.dot(X, ((A-Y).T)))
    db = (1/m)*np.sum(A-Y)
    #Dimensions matching
    assert(w.shape == dw.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = { "dw" : dw, "db" : db}

    return cost, grads

#----------------------Optimize-----------------
def optimize(X, Y, w, b, num_steps, alpha) :
    for i in range(num_steps) :
        cost, grads = calculation(X, Y, w, b)
        dw = grads["dw"]
        db = grads["db"]
        w -= alpha*dw
        b -= alpha*db

    params = { "w" : w, "b" : b}
    grads = {"dw" : dw, "db" : db}

    return params, grads

''' checkkkkkkkkkking

w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])

params, grads= optimize(X, Y,w, b, num_steps= 100, alpha = 0.009)

print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))'''

#---------------prediction------------------
def predict(X, w, b) :
    A = sigmoid(np.dot(w.T, X) + b)
    y_prediction = np.zeros((1, X.shape[1]))
    for i in range(A.shape[1]) :
        if A[0][i] > 0.5 :
            y_prediction[0][i] = 1
    assert(y_prediction.shape == (1, X.shape[1]))
    return y_prediction

'''w = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
print ("predictions = " + str(predict(X, w, b)))'''


#-----------Model------------
def model(train_set_x, train_set_y, test_set_x, test_set_y, num_steps, alpha) :
    dim = train_set_x.shape[0]
    w, b = initialization(dim)
    params, grads = optimize(train_set_x, train_set_y, w, b, num_steps, alpha)
    w = params["w"]
    b = params["b"]
    y_prediction_test = predict(test_set_x, w, b)
    y_prediction_train = predict(train_set_x, w, b)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - train_set_y)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - test_set_y)) * 100))

    
    d = {"Y_prediction_test": y_prediction_test, 
         "Y_prediction_train" : y_prediction_train, 
         "w" : w, 
         "b" : b}
    
    return d
    
    
if __name__ == "__main__" :
    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_steps = 2000, alpha = 0.005)
    index = 200
    img = Image.fromarray(train_set_x_orig[index], 'RGB')
    img.show(img)
    print(d["Y_prediction_train"][:,index])
















