
import numpy as np
from scipy.optimize import minimize

class Perceptron_dual:
    '''
    Suppor Vector Machine classifier for a binary label using the dual formulation, where labels are 0 and 1 (converted to -1 and 1 internally)
    '''
    
    def __init__(self, r, train_csv_name, guassian_gamma = 0.1, test_csv_name = None, kernal = 'default'):
        
        self.train_data, self.train_x, self.train_y = self.get_numerical_data_from_csv(train_csv_name)
        
        if test_csv_name is not None:
            self.test_data, self.test_x, self.test_y = self.get_numerical_data_from_csv(test_csv_name)
        
        N = len(self.train_y)
        mistakes_0 = np.ones(N)
        x = self.train_x
        y = self.train_y.reshape(N,1)
        self.r = r
        
        # compute this once so we can use it later
        self.guassian_matrix = np.zeros([N,N])
        for i in range(N):
            for j in range(N):
                self.guassian_matrix[i,j] = np.exp( -np.linalg.norm( x[i] - x[j] , 2 )**2 / guassian_gamma )
        
#        print('ojb test:',self.objective_function(mistakes_0, x, y, self.r, self.default_kernal))
#        test = lambda alpha: np.sum(alpha * self.train_y)
        
        if kernal == 'default':
            kernal_function = self.default_kernal
        else:
            kernal_function = self.guassian_kernal
        result = minimize(self.objective_function, 
                          mistakes_0, 
                          args=(x,y,self.r,kernal_function), 
                          method='SLSQP', 
#                          jac = True,
                          bounds=[[0, None]] * len(mistakes_0), 
#                          constraints={'type': 'eq', 'fun': test} 
                          )
        
        self.mistakes = result.x
        
        self.w = np.sum(self.mistakes.reshape(N,1) * self.r * self.train_y.reshape(N,1) * self.train_x, axis = 0)
        
        self.train_error = self.get_current_error()
        
        if test_csv_name is not None:
            self.test_error = self.get_current_error(use_testing_data = True)
        
        
    def objective_function(self, mistakes, x, y, r, kernal):
        mistakes = mistakes.reshape(len(mistakes),1)
        ones_vector = np.ones([len(mistakes),1])
        y_i = np.matmul(y,np.transpose(ones_vector))
        y_j = np.transpose(y_i)
        mistakes_i = np.matmul(mistakes,np.transpose(ones_vector))
        mistakes_j = np.transpose(mistakes_i)
#        value = np.sum( np.sum( -y_i * y_j * mistakes_j * r * kernal(x), axis = 1).clip(a_min = 0) )
        value = np.sum( np.clip( np.sum( -y_i * y_j * mistakes_j * r * kernal(x), axis = 1), 0, None) )
#        gradient = 1/2 * np.sum(y_i * y_j * alpha_j * kernal(x), axis=1) - 1
        return value #, gradient
    
    def default_kernal(self, x):
        return np.matmul(x,np.transpose(x))
    
    def guassian_kernal(self,x):
        return self.guassian_matrix
         
    def all_predictions(self, x_matrix):
        '''
        returns an array of the raw numerical predictions for given weights and x_matrix
        '''
        return np.matmul(self.w.transpose(), x_matrix.transpose())
    
    def get_current_error(self, use_testing_data = False):
        if use_testing_data:
            data_x = self.test_x
            data_y = self.test_y
        else:
            data_x = self.train_x
            data_y = self.train_y
            
        return np.sum( self.all_predictions(data_x) * data_y <=0 ) / len(data_x)
    
    def get_numerical_data_from_csv(self, csv_name):
        '''
        reads data in from a csv and convert to numpy array, assumes no headers, all entries are numbers, and that final column is the label
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
        x = np.hstack([np.ones([len(x),1]),x]) # put ones in the column to allow for a bias term
        y = data[:,-1]
        
        # convert binary values to signed -1 and 1
        y = 2 * y - 1
        data[:,-1] = np.copy(y)
        
        return data, x, y

if __name__ == '__main__':
#    C_array = np.array([100, 500, 700])/873
    
#    for c in C_array:
#        svm_dual = SVM_dual(c,'bank-note/train.csv', test_csv_name = 'bank-note/test.csv')
#        print('\nc:',round(c*873),'/ 873')
#        print('w:',svm_dual.w)
#        print('b:',svm_dual.b)
#    #    print('all_biases:',svm_dual.all_biases)
#        print('train_error:',round(svm_dual.train_error,3))
#        print('test_error:',round(svm_dual.test_error,3))
    
    gamma_array = np.array([0.01,0.1,0.5,1,2,5,10,100])
    
    r = .1
    
    for gamma in gamma_array:
        perceptron_dual = Perceptron_dual(r,'bank-note/train.csv', guassian_gamma = gamma, test_csv_name = 'bank-note/test.csv', kernal = 'guassian')
        print('\ngamma:',gamma)
        print('w:',perceptron_dual.w)
        print('train_error:',round(perceptron_dual.train_error,3))
        print('test_error:',round(perceptron_dual.test_error,3))
    