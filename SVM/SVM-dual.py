
import numpy as np
from scipy.optimize import minimize

class SVM_dual:
    '''
    Suppor Vector Machine classifier for a binary label using the dual formulation, where labels are 0 and 1 (converted to -1 and 1 internally)
    '''
    
    def __init__(self, hinge_weight_C, train_csv_name, guassian_gamma = 0.1, test_csv_name = None, kernal = 'default'):
        
        self.C = hinge_weight_C
        self.train_data, self.train_x, self.train_y = self.get_numerical_data_from_csv(train_csv_name)
        
        if test_csv_name is not None:
            self.test_data, self.test_x, self.test_y = self.get_numerical_data_from_csv(test_csv_name)
        
        N = len(self.train_y)
        alpha_0 = np.zeros(N)
        x = self.train_x
        y = self.train_y.reshape(N,1)
        
        # compute this once so we can use it later
        self.guassian_matrix = np.zeros([N,N])
        for i in range(N):
            for j in range(N):
                self.guassian_matrix[i,j] = np.exp( -np.linalg.norm( x[i] - x[j] , 2 )**2 / guassian_gamma )
        
        test = lambda alpha: np.sum(alpha * self.train_y)
        
        if kernal == 'default':
            kernal_function = self.default_kernal
        else:
            kernal_function = self.guassian_kernal
        result = minimize(self.objective_function, 
                          alpha_0, 
                          args=(x,y,kernal_function), 
                          method='SLSQP', 
                          jac = True,
                          bounds=[[0, self.C]] * len(alpha_0), 
                          constraints={'type': 'eq', 'fun': test} )
        
        self.alpha = result.x
        
        self.w = np.sum(self.alpha.reshape(N,1) * self.train_y.reshape(N,1) * self.train_x, axis = 0)
        
#        self.support_filter = np.logical_and(self.alpha > 1e-8, self.alpha < self.C * .9999)
        self.support_filter = np.logical_and(self.alpha > 1e-3, self.alpha < self.C * .99)
        self.support_indexes = np.arange(0,N)[self.support_filter]
        
        self.all_biases = self.train_y.reshape(N,1)[self.support_filter] - np.transpose(np.matmul( self.w.reshape(1,len(self.w)), np.transpose(self.train_x[self.support_filter]) ) )
        
        self.b = np.average(self.all_biases)
        
        self.train_error = self.get_current_error()
        
        if test_csv_name is not None:
            self.test_error = self.get_current_error(use_testing_data = True)
        
        
    def objective_function(self, alpha, x, y, kernal):
        alpha = alpha.reshape(len(alpha),1)
        ones_vector = np.ones([len(alpha),1])
        y_i = np.matmul(y,np.transpose(ones_vector))
        y_j = np.transpose(y_i)
        alpha_i = np.matmul(alpha,np.transpose(ones_vector))
        alpha_j = np.transpose(alpha_i)
        value = 1/2 * np.sum(y_i * y_j * alpha_i * alpha_j * kernal(x)) - np.sum(alpha)
        gradient = 1/2 * np.sum(y_i * y_j * alpha_j * kernal(x), axis=1) - 1
        return value, gradient
    
    def default_kernal(self, x):
        return np.matmul(x,np.transpose(x))
    
    def guassian_kernal(self,x):
        return self.guassian_matrix
         
    def all_predictions(self, x_matrix):
        '''
        returns an array of the raw numerical predictions for given weights and x_matrix
        '''
        return np.matmul(self.w.transpose(), x_matrix.transpose()) + self.b
    
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
        DOES NOT AUGMENT WITH ONES
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
        y = data[:,-1]
        
        # convert binary values to signed -1 and 1
        y = 2 * y - 1
        data[:,-1] = np.copy(y)
        
        return data, x, y

if __name__ == '__main__':
    C_array = np.array([100, 500, 700])/873
    
#    for c in C_array:
#        svm_dual = SVM_dual(c,'bank-note/train.csv', test_csv_name = 'bank-note/test.csv')
#        print('\nc:',round(c*873),'/ 873')
#        print('w:',svm_dual.w)
#        print('b:',svm_dual.b)
#    #    print('all_biases:',svm_dual.all_biases)
#        print('train_error:',round(svm_dual.train_error,3))
#        print('test_error:',round(svm_dual.test_error,3))
    
    gamma_array = np.array([5,1,2,10,100])
    
    for gamma in gamma_array:
        for c in C_array:
            svm_dual = SVM_dual(c,'bank-note/train.csv', guassian_gamma = gamma, test_csv_name = 'bank-note/test.csv', kernal = 'guassian')
            print('\ngamma:',gamma)
            print('c:',round(c*873),'/ 873')
#            print('w:',svm_dual.w)
#            print('b:',svm_dual.b)
            print('train_error:',round(svm_dual.train_error,3))
            print('test_error:',round(svm_dual.test_error,3))
            print('number of support vectors:',len(svm_dual.support_indexes))
            if gamma==5:
                print('support vector indexes:',svm_dual.support_indexes)
    