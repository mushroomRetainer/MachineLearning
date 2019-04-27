
import numpy as np
import matplotlib.pyplot as plt

class Logistic_regression:
    '''
    Logistic Regression classifier for a binary label, where labels are 0 and 1 (converted to -1 and 1 internally)
    '''
    
    def __init__(self, variance, learning_rate_gamma = None, learning_paramater_d  = None, train_csv_name = None, test_csv_name = None, use_MAP = False, data_x = None, data_y = None, learning_rate_array = None, rng_seed= 12345, debugging = False):
        '''
        defaults to disable voting because this takes a lot of memory. 
        Send enable_voting=True and it will store all weight arrays to allow for perceptron voting
        '''
        self.use_MAP = use_MAP
        self.debugging = debugging
        self.variance = variance
        self.epoch_count = 1
        if debugging:
            self.train_x = data_x
            self.train_y = data_y
            self.learning_rate_array = learning_rate_array # should be an array
        else:
            
            self.gamma_0 = learning_rate_gamma
            self.gamma = self.gamma_0
            self.d = learning_paramater_d
            self.train_data, self.train_x, self.train_y = self.get_numerical_data_from_csv(train_csv_name)
            
            if test_csv_name is not None:
                self.test_data, self.test_x, self.test_y = self.get_numerical_data_from_csv(test_csv_name)
        
        # includes bias term
        self.w = np.zeros(len(self.train_x[0]))
        
        if rng_seed is not None:
            np.random.seed(rng_seed)
    
    def add_epoch(self):
        '''
        loops through every single training example to compute the weight updates for a single epoch
        '''
        
            
        N = len(self.train_x)
        
        # shuffle the data
        new_order = np.random.choice(N, N, replace=False)
        self.train_y = self.train_y[new_order]
        self.train_x = self.train_x[new_order]
        
        for i in range(N):
            
            sub_gradient = self.get_subgradient(self.train_x[i], self.train_y[i])
            
            if self.debugging:
                self.gamma = self.learning_rate_array[i]
                print('subgradient =',sub_gradient)
                
            self.w -= self.gamma * sub_gradient
                
#        self.gamma = self.gamma_0 / (1 + self.gamma_0 / self.d * self.epoch_count)
#        self.gamma *= 0.5
        self.epoch_count += 1
         
    def get_subgradient(self, x, y):
        exp_term = np.exp(-y * np.sum(self.w * x))
        if self.use_MAP:
            return (-y * exp_term) / (1 + exp_term) * x + 2/self.variance * self.w
        else:
            return (-y * exp_term) / (1 + exp_term) * x
            
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
    lg = Logistic_regression(1, learning_rate_array = [.1, .005, .0025], data_x = np.array([[.5,-1,.3],[1,-2,-2],[1.5,.2,2.5]]), data_y = np.array([1,-1,1]), debugging=True)
    lg.add_epoch()
    
    print('')
    
    for use_MAP in [False, True]:
        plt.figure()
        if use_MAP:
            plt.title('Results for MAP')
        else:
            plt.title('Results for ML')
        subplot_counter = 1
        for variance in [0.01,0.1,0.5,1,3,5,10,100]:
            if use_MAP:
                print('MAP Using variance =',variance)
            else:
                print('ML Using variance =',variance)
            lg = Logistic_regression(variance, learning_rate_gamma = 0.003, learning_paramater_d  = 0.01, use_MAP = use_MAP, train_csv_name='bank-note/train.csv', test_csv_name = 'bank-note/test.csv')
            
            epoch_array = []
            train_error_array = []
            test_error_array = []
            
            train_error = lg.get_current_error(use_testing_data=False)
            test_error = lg.get_current_error(use_testing_data=True)
            epoch_array.append(0)
            train_error_array.append(train_error)
            test_error_array.append(test_error)
            
            for i in range(100):
                lg.add_epoch()
                train_error = lg.get_current_error(use_testing_data=False)
                test_error = lg.get_current_error(use_testing_data=True)
                epoch_array.append(i+1)
                train_error_array.append(train_error)
                test_error_array.append(test_error)
            print('final testing and training error:', round(train_error,5), round(test_error,5))
            
            
            plt.subplot(2,4,subplot_counter)
            plt.title('Variance = '+str(variance))
            plt.plot(epoch_array, train_error_array, label='train')
            plt.plot(epoch_array, test_error_array, label='test')
            subplot_counter += 1
            
        plt.xlabel('epoch')
        plt.ylabel('error')
        plt.legend()
        plt.show()
        
            