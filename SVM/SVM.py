
import numpy as np

class SVM:
    '''
    Suppor Vector Machine classifier for a binary label, where labels are 0 and 1 (converted to -1 and 1 internally)
    '''
    
    def __init__(self, learning_rate_gamma, learning_paramater_d, hinge_weight_C, train_csv_name, test_csv_name = None):
        '''
        defaults to disable voting because this takes a lot of memory. 
        Send enable_voting=True and it will store all weight arrays to allow for perceptron voting
        '''
        self.gamma_0 = learning_rate_gamma
        self.gamma = self.gamma_0
        self.d = learning_paramater_d
        self.C = hinge_weight_C
        self.epoch_count = 1
        self.train_data, self.train_x, self.train_y = self.get_numerical_data_from_csv(train_csv_name)
        
        if test_csv_name is not None:
            self.test_data, self.test_x, self.test_y = self.get_numerical_data_from_csv(test_csv_name)
        
        # includes bias term
        self.w = np.zeros(len(self.train_x[0]))
        
        np.random.seed(12345)
    
    def add_epoch(self):
        '''
        loops through every single training example to compute the weight updates for a single epoch
        '''
        
        N = len(self.train_data)
        
        # shuffle the data
        new_order = np.random.choice(N, N, replace=False)
        self.train_y = self.train_y[new_order]
        self.train_x = self.train_x[new_order]
        
        for i in range(N):
            if self.train_y[i] * np.matmul(self.w.transpose(), self.train_x[i]) <= 1:
                self.w = (1 - self.gamma) * self.w + self.gamma * self.C * N * self.train_y[i] * self.train_x[i]
            else:
                self.w[1:] = (1 - self.gamma) * self.w[1:]
            print('sub gradient:',self.gamma * self.C * N * self.train_y[i] * self.train_x[i])
        
#        self.gamma = self.gamma_0 / (1 + self.gamma_0 / self.d * self.epoch_count)
        self.gamma *= 0.5
        self.epoch_count += 1
         
    def all_predictions(self, weights, x_matrix):
        '''
        returns an array of the raw numerical predictions for given weights and x_matrix
        '''
        return np.matmul(weights.transpose(), x_matrix.transpose())
    
    def get_current_error(self, use_testing_data = False):
        if use_testing_data:
            data_x = self.test_x
            data_y = self.test_y
        else:
            data_x = self.train_x
            data_y = self.train_y
            
        return np.sum( self.all_predictions(self.w, data_x) * data_y <=0 ) / len(data_x)
    
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
        y = 2*y - 1
        data[:,-1] = np.copy(y)
        
        return data, x, y

if __name__ == '__main__':
#    learning_rate_gamma = 5e-4
#    learning_paramater_d = learning_rate_gamma / 10
#    hinge_weight_C_array =  np.array([1 ,10, 50, 100, 300, 500, 700]) / 873 # np.array([100]) / 873 
#    test_error_array = []
#    train_error_array = []
#    w_array = []
#    num_epochs = 100
#    for hinge_weight_C in hinge_weight_C_array:
#        svm = SVM(learning_rate_gamma, learning_paramater_d, hinge_weight_C, 'bank-note/train.csv', test_csv_name = 'bank-note/test.csv')
#        for i in range(num_epochs):
#            svm.add_epoch()
#        w_array.append(svm.w)
#        print('\nUsing learning rate (gamma):',learning_rate_gamma,'parameter (d):',learning_paramater_d,'\nand hinge weight (C):',int(np.round(hinge_weight_C*873)),'/ 873, with',num_epochs,'epochs')
#        test_error_array.append(svm.get_current_error(True))
#        train_error_array.append(svm.get_current_error(False))
#        print('Training error of final SVM:   ', train_error_array[-1])
#        print('Testing error of final SVM:    ', test_error_array[-1])
#        
#    test_error_array2 = []
#    train_error_array2 = []
#    w_array2 = []
#    learning_paramater_d = learning_rate_gamma
#    for hinge_weight_C in hinge_weight_C_array:
#        svm = SVM(learning_rate_gamma, learning_paramater_d, hinge_weight_C, 'bank-note/train.csv', test_csv_name = 'bank-note/test.csv')
#        for i in range(num_epochs):
#            svm.add_epoch()
#        w_array2.append(svm.w)
#        print('\nUsing learning rate (gamma):',learning_rate_gamma,'parameter (d):',learning_paramater_d,'\nand hinge weight (C):',int(np.round(hinge_weight_C*873)),'/ 873, with',num_epochs,'epochs')
#        test_error_array2.append(svm.get_current_error(True))
#        train_error_array2.append(svm.get_current_error(False))
#        print('Training error of final SVM:   ', train_error_array2[-1])
#        print('Testing error of final SVM:    ', test_error_array2[-1])
#    
#    w_difference_array = []
#    for i in range(len(w_array)):
#        w_difference_array.append(np.linalg.norm(w_array[i] - w_array2[i],2))
#    print('\nTrain error difference:',np.round(np.abs(np.array(train_error_array) - np.array(train_error_array2)),4))
#    print('Test error difference:',np.round(np.abs(np.array(test_error_array) - np.array(test_error_array2)),4))
#    print('L2-norm of weights vectors:',np.round(w_difference_array,3))
#    
#    # for problem 2:
#    for i in [3,5,6]:
#        print('\nUsing hinge weight (C):',int(np.round(hinge_weight_C_array[i]*873)),'/ 873')
#        print('bias and weights (bias is first term):', w_array2[i])
    
    SVM(0.01, 1, 1, 'train_small.csv')
    
    svm.add_epoch()