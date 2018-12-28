#seperately compute w and b

import numpy as np
import pandas as pd
#from scipy.misc import derivative

df = pd.read_csv("wdbc.data", header = None)

X_id = df[[0]] 
y = np.array(df[[1]])
X_origin = np.array(df.drop([0,1], axis = 1))

X_mean = np.mean(X_origin, axis = 0)
X_X_mean = X_origin - X_mean.reshape(1,30)

X_norm = np.linalg.norm(X_X_mean,axis = 1)

X = X_X_mean / X_norm.reshape(569,1)

#X = np.hstack((np.ones((569,1)),X))
Xy = np.hstack((y,X))



lambd = 0.01
gamma = 0.3 # step size multiplier 0.000001
eta = 0.1
W1 = np.zeros((30,100))
B1 = np.zeros(100)
Err = np.zeros(100)
n_iters = np.zeros((100,1))
#step = np.zeros((500,1))

for i in range(100):
    # randomly choose trainning set and test set
    np.random.shuffle(Xy)
    Xtrain, Xtest, ytrain, ytest = Xy[:500,1:], Xy[500:,1:], Xy[:500,0], Xy[500:,0]
    cur_w = np.ones(30)*10 # The algorithm starts at w0 = 0.5
    cur_b = 10
    prev_w = cur_w
    prev_b = cur_b
    z_w = cur_w
    z_b = cur_b
    prev_w_step = cur_w - prev_w
    prev_b_step = cur_b - prev_b
    err = np.zeros(10000)
#    previous_step_size = 1  #0.01
    iters = 0 #iteration counter
    while (iters < 10000):
        prev_w = cur_w
        prev_b = cur_b
        z_w = prev_w + eta * prev_w_step
        z_b = prev_b + eta * prev_b_step
        a = 1 - (1/(1+np.exp(np.dot(Xtrain,prev_w)+np.ones(500)*prev_b)))
        der_w = ((-1)*np.sum(Xtrain*ytrain.reshape(500,1),axis=0) + 
                 np.sum(Xtrain*(np.exp(np.dot(Xtrain,z_w)+prev_b)/(1+np.exp(np.dot(Xtrain,z_w)+prev_b))).reshape(500,1),axis=0) +
                 lambd*z_w)
        
#        der_w = (np.sum(Xtrain*(-1)*(ytrain.reshape((500,1))),axis=0) + 
#                 np.sum(Xtrain*(a.reshape((500,1))),axis=0) + 
#                 lambd*prev_w)
        der_b = ((-1)*np.sum(ytrain) + 
                 np.sum((np.exp(np.dot(Xtrain,prev_w)+z_b))/(1+np.exp(np.dot(Xtrain,prev_w)+z_b))))
        
#        der_b = np.sum(ytrain*(-1)) + np.sum(a)
        f = ((-1)*np.sum(np.dot(Xtrain,prev_w) * ytrain) - np.sum(ytrain * prev_b) +
             np.sum(np.log(1+np.exp(np.dot(Xtrain,prev_w)+prev_b))) + (lambd/2)*np.dot(prev_w,prev_w))
        
#        f = (np.sum((-1)*ytrain*np.dot(Xtrain,prev_w)-ytrain*prev_b+
#                    np.log(1+np.exp(np.dot(Xtrain,prev_w)+np.ones(500)*prev_b)))+
#                    (lambd/2)*np.dot(prev_w,prev_w))
        b = ((np.linalg.norm(der_w))**2 + der_b**2)
        c = ((10**(-6)) * (1 + np.abs(f)))
        if (b <= c):
            break
        cur_w = z_w - gamma * der_w
        cur_b = z_b - gamma * der_b
        prev_w_step = cur_w - prev_w
        prev_b_step = cur_b - prev_b
#        step[iters] = np.dot(prev_w,cur_w)
#        previous_step_size = abs(cur_w - prev_w)
#        print('iters:',iters," prev_w:",cur_w, " der:", der, "previous_step_size: ",previous_step_size)             
        ppred = 1/(1+np.exp((-1)*(np.dot(Xtest,cur_w)+cur_b)))
        pmal = ppred > 0.5
        ypred = pmal * np.ones(69)
        err_n = np.count_nonzero(ypred != ytest)
        err[iters] = err_n
        iters+=1 
    n_iters[i] = iters
    Err[i] = np.mean(err)
    W1[:,i] = cur_w
    B1[i]= cur_b
    
    
avg_w = W1.mean(axis = 1)
avg_b = B1.mean()
avg_err = Err.mean()
avg_iters = n_iters.mean()
print("Average w is : ", avg_w, "\nAverage b is : ", avg_b, "\nAverage error is : ", avg_err,"\nAverage n_iters is ", avg_iters)





# initialize w0, step size multiplier, step size
# 3(c)
#lambd = 0.01
#gamma = 0.01 # step size multiplier 0.000001
#max_iters = 10000 # maximum number of iterations
#W1 = np.zeros((30,100))
#B1 = np.zeros(100)
#Err = np.zeros(100)
#
#for i in range(100):
#    # randomly choose trainning set and test set
#    np.random.shuffle(Xy)
#    Xtrain, Xtest, ytrain, ytest = Xy[:500,1:], Xy[500:,1:], Xy[:500,0], Xy[500:,0]
#    cur_w = np.ones(30)*0.5 # The algorithm starts at w0 = 0.5
#    cur_b = 1
#    prev_w = cur_w
#    prev_b = cur_b
##    previous_step_size = 1  #0.01
#    iters = 0 #iteration counter
#    while (iters < 500):
#        prev_w = cur_w
#        prev_b = cur_b
#        a = 1 - 1/(1+np.exp(np.dot(Xtrain,prev_w)+np.ones(500)*prev_b))
#        der_w = np.sum(Xtrain*(-1)*(ytrain.reshape(500,1)),axis=0) + np.sum(Xtrain*(a.reshape(500,1)),axis=0) + lambd*prev_w
#        der_b = np.sum(ytrain*(-1)) + np.sum(a)
#        cur_w -= gamma * der_w
#        cur_b -= gamma * der_b
##        previous_step_size = abs(cur_w - prev_w)
##        print('iters:',iters," prev_w:",cur_w, " der:", der, "previous_step_size: ",previous_step_size)
#        iters+=1
#    W1[:,i] = cur_w
#    B1[i]= cur_b
#    fPred = np.dot(Xtest,cur_w)
#    pPred = 1/(1+np.exp((-1)*fPred))
#    label = pPred > 0.5
#    yPred = label * np.ones(69)
#    Err[i] = np.count_nonzero(yPred != ytest)
#    
#avg_w = W1.mean(axis = 1)
#avg_err = Err.mean()
#print("Average w is : ", avg_w, "\n Average error is : ", avg_err)



