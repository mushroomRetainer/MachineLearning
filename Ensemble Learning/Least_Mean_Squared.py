
import numpy as np
from scipy.optimize import fsolve

def J(w, x, y):
    return 1/2 * np.sum(y - np.sum(w * x, axis=1) )**2

def grad_J(w,x,y):
    j_max = len(x[0])
    dJ_dw = np.zeros(j_max)
    for j in range(j_max):
        dJ_dw[j] = - np.sum( (y - np.sum(w * x, axis=1))*x[:,j] )
    return dJ_dw

def batch_gradient_decent(w_guess,r, x, y):
    # update all the weights after finding the gradient based on all the data
    return w_guess - r * grad_J(w_guess,x,y)

def stochastic_gradient_decent(w_guess, r, x, y, x_row):
    # update all the weights using only one row of data at a time
    gradient = (y[x_row] - np.sum(w_guess * x[x_row])) * x[x_row]
    w = w_guess + r * gradient
    return w, gradient

def get_numerical_data_from_csv(csv_name):
    '''
    reads data in from a csv and convert to numpy array, assumes no headers and that final column is the label
    '''
    data = []
    with open(csv_name, 'r') as f:
        for line in f:
            data.append(line.strip().split(','))
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = float(data[i][j])
    data = np.array(data)
    x = data[:,:-1]
    x = np.hstack([np.ones([len(x),1]),x])
    y = data[:,-1]
    return data, x, y

def calculate_error(x,w,y):
    return np.sum( (y - np.matmul(w,np.transpose(x)) )**2)

import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    # HW 1.7
    
    data = np.array([[1,-1,2,1],
                     [1,1,3,4],
                     [-1,1,0,-1],
                     [1,2,-4,-2],
                     [3,-1,-1,0]
                    ])
    x = data[:,:-1]
    x = np.hstack([np.ones([len(x),1]),x])
    y = data[:,-1]
    
    
    w = np.array([0, 0, 0, 0])
    
    # prints out a single gradient
    #print(grad_J(w,x,y))
    #print(w - r*grad_J(w,x,y))
    
    # gets the exact solution
    #def solve_function(w_guess):
    #    return grad_J(w_guess,x,y)
    #print(fsolve(solve_function,w))
    
    
    # batch gradient decent (locks on the correct values with r = 0.05, unstable with r = 0.1)
    r = 0.05
    error = []
    for i in range(500):
        gradient_w = grad_J(w, x, y)
        w = batch_gradient_decent(w, r, x, y)
        print('\niteration',i,'\ngrad_w =',gradient_w,'\nw =',w)
        error.append(calculate_error(x,w,y))
        
#    r = 0.1
    # stochastic gradient decent
#    print('initial\nw =',w[1:],'\nb =',w[0])
#    for i in range(1):
#    #    print('data pass number',i)
#        for row in range(len(x)):
#            w, gradient_w = stochastic_gradient_decent(w, r, x, y, row)
#            print('\ndata row',row,'\ngrad_w =',gradient_w,'\nw =',w[1:],'\nb =',w[0])
    #    x = x[np.random.choice(len(x),len(x),replace=False)] # can be used to shuffle data
    plt.figure()
    plt.plot(error, label = 'train error')
    #plt.plot(test_error, label = 'test error')
    plt.ylabel('error')
    plt.xlabel('number of training iterations')
    plt.title('Test Error')
    plt.legend()
