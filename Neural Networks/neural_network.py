
#  three layer artificial neural network

import numpy as np

def sigmoid_F(x):
    return 1 / ( 1 + np.e**(-x) )

def d_sigmoid_F(x):
    sig = sigmoid_F(x)
    return sig * ( 1 - sig )

def loss_F(y, y_star):
    return 1/2 * (y - y_star)**2

def d_loss_F(y, y_star):
    return y - y_star

class Nerual_network:
    
    def __init__(self, nonbias_nodes = 2, train_csv_name = None, test_csv_name = None, gamma_0 = 1, d = 1, rng_seed = None, zero_init_w = False, num_inputs = None, edge_weights = None, debugging = False):
        
        self.nonbias_nodes = nonbias_nodes
        self.nodes = nonbias_nodes + 1
        
        self.hidden_layers = 2
        self.layers = self.hidden_layers + 1
        
        if debugging:
            self.num_inputs = num_inputs
            self.w = edge_weights
        else:
            self.train_data, self.train_x, self.train_y = self.get_numerical_data_from_csv(train_csv_name)
            
            if test_csv_name is not None:
                self.test_data, self.test_x, self.test_y = self.get_numerical_data_from_csv(test_csv_name)
                
            self.num_inputs = len(self.train_x[0])
            if rng_seed is not None:
                np.random.seed(rng_seed)
            if zero_init_w:
                self.w = np.zeros( [self.layers + 1, self.nodes, self.nodes] )
            else:
                self.w = np.random.normal( size = [self.layers + 1, self.nodes, self.nodes] )
            self.gamma_0 = gamma_0
            self.gamma = gamma_0
            self.d = d
            self.epoch_count = 0
        
        self.x = np.zeros([self.num_inputs + 1])
        self.z = np.zeros([self.layers, self.nodes])
        self.s = np.zeros([self.layers, self.nodes])
        self.y = 1
        self.L = 1
        
        # bias terms:
        self.z[:,0] = 1
        
        self.dL_dy = 1
        self.dy_dz = np.zeros([self.layers, self.nodes]) # from y to any given z
        self.dz_ds = np.zeros([self.layers, self.nodes])
        self.dL_dw = np.zeros([self.layers + 1, self.nodes, self.nodes]) # from the we all the way to the L

    def prediction(self, x):
        self.x = np.array(x)
        self.s[1,1:] = np.sum( self.w[1,0:self.num_inputs,1:] * np.repeat(self.x.reshape([self.num_inputs, 1]), self.nonbias_nodes, axis = 1 ), axis = 0 )
        self.z[1,1:] = sigmoid_F(self.s[1,1:])
        
        # could be looped if desired. replace all the '2's with a counter variable
        self.s[2,1:] = np.sum( self.w[2,:,1:] * np.repeat(self.z[2-1,:].reshape([self.nodes, 1]), self.nonbias_nodes, axis = 1 ), axis = 0 )
        self.z[2,1:] = sigmoid_F(self.s[2,1:])
        
        self.y = np.sum( self.w[3,:,1].reshape([self.nodes]) * self.z[3-1,:] )
        
        return self.y
    
    def back_propigation(self, x, y_star): # y_star is the correct label
        # returns a matrix of the derivatives of the loss function with respect to each weight
#        print('input values:',x, y_star)
        self.dL_dy = d_loss_F(self.prediction(x), y_star) # note: prediction also updates all the values
        
        # first layer
        self.dL_dw[3,:,1] = self.dL_dy * self.z[3-1,:].reshape([self.nodes])
        # right underneath the first layer
        self.dy_dz[2,:] = self.w[3,:,1]
        self.dz_ds[2,:] = d_sigmoid_F(self.s[2,:])
        self.dL_dw[2,:,1:] = self.dL_dy * np.repeat( (self.dy_dz[2,1:] * self.dz_ds[2,1:]).reshape([1, self.nonbias_nodes]), self.nodes, axis = 0) * np.repeat(self.z[2-1,:].reshape([self.nodes, 1]), self.nonbias_nodes, axis = 1 )
        
        # all other layers
        # (don't have any)
        
        # bottom layer (could be looped for the other layers if you switch out the final term x for the z term above and replace the 1 indexes with a counter)
        self.dy_dz[1,:] = np.sum( self.dy_dz[1+1,:] * self.dz_ds[1+1,:] * self.w[1+1,:,:] , axis =1)
        self.dz_ds[1,:] = d_sigmoid_F(self.s[1,:])
        
        self.dL_dw[1,0:self.num_inputs,1:] = self.dL_dy * np.repeat( (self.dy_dz[1,1:] * self.dz_ds[1,1:]).reshape([1, self.nonbias_nodes]), self.num_inputs, axis = 0) * np.repeat(self.x.reshape([self.num_inputs, 1]), self.nonbias_nodes, axis = 1 )
        
        return self.dL_dw
    
    def add_epoch(self):
        '''
        loop through every single training example to compute the weight updates based on the back propigation derivatives for a single epoch
        '''
        
        N = len(self.train_data)
        
        # shuffle the data
        new_order = np.random.choice(N, N, replace=False)
        self.train_y = self.train_y[new_order]
        self.train_x = self.train_x[new_order]
        
        for i in range(N):
            dL_dw_matrix = self.back_propigation(self.train_x[i], self.train_y[i])
            self.w -= self.gamma * dL_dw_matrix
        
        self.gamma = self.gamma_0 / (1 + self.gamma_0 / self.d * self.epoch_count)
#        self.gamma *= 0.5, for debugging prior to tuning
        self.epoch_count += 1
         
    def all_predictions(self, weights, x_matrix):
        '''
        returns an array of the raw numerical predictions for given weights and x_matrix
        '''
        results = []
        for i in range(len(x_matrix)):
            results.append(self.prediction(x_matrix[i,:])) 
        return results
    
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
        
    
if __name__ == "__main__":
    
    # problem 3.1 (test against the handwritten solution)
    
    given_Ws = np.zeros([3+1,2+1,2+1])
    
    given_Ws[1,0,1:] = [-1, 1]
    given_Ws[1,1,1:] = [-2, 2]
    given_Ws[1,2,1:] = [-3, 3]
    given_Ws[2,0,1:] = [-1, 1]
    given_Ws[2,1,1:] = [-2, 2]
    given_Ws[2,2,1:] = [-3, 3]
    given_Ws[3,:,1] = [-1, 2, -1.5]
    
    nn = Nerual_network(num_inputs = 3, nonbias_nodes = 2, edge_weights = given_Ws, debugging = True)
    
    print('prediction:', round(nn.prediction([1,1,1]),4))
    
    dL_dw = nn.back_propigation([1,1,1], 1)
    print('\nderivatives:')
    # the weight derivative matrix is 3 levels deep
    for layer in range(len(dL_dw)):
        for bottom_connection in range(len(dL_dw[0])):
            for top_connection in range(len(dL_dw[0,0])):
                value = dL_dw[layer, bottom_connection, top_connection]
                if value != 0: # many of the weights are unused and remain zero
                    print('w^'+str(layer)+'_'+str(bottom_connection)+str(top_connection)+'='+str(round(value,5)))
    
    # problem 3.2 (unleash it on real data)
    
    for zero_init_w in [False,True]:
        for nodes in [5,10,25,50,100]:
            if zero_init_w:
                s = 'zero matrix initialization for w'
            else:
                s = 'Gaussian initialization for w'
            print('\nTraining on bank dataset with',nodes,'nodes and with',s)
            
            nn = Nerual_network(nonbias_nodes = nodes, train_csv_name='bank-note/train.csv', test_csv_name = 'bank-note/test.csv', gamma_0 = 0.05, d = 0.1, rng_seed = 12345, zero_init_w = zero_init_w)
            
#            epoch_array = []
#            train_error_array = []
#            test_error_array = []
            
            epoch_min_improvement = 1e-5  # minimum for what is considered "improvement" from one epoch to the next
            epoch_convergence_count = 3   # this is how many consequitive epoch must pass without improvment (minus an improvment margin, as defined above)
            epoch_convergence_counter = 0 # this keeps a tall of how many consequtive epochs have not improved
            best_train_error = np.inf     # the training error is checked at the end of every epoch. The running best is stored here
            
            for i in range(100):
                nn.add_epoch()
                
                train_error = nn.get_current_error(use_testing_data=False)
                test_error = nn.get_current_error(use_testing_data=True)
#                epoch_array.append(i)
#                train_error_array.append(train_error)
#                test_error_array.append(test_error)
                
                if train_error < best_train_error - epoch_min_improvement:
                    epoch_convergence_counter = 0
                    best_train_error = train_error
                elif epoch_convergence_counter > epoch_convergence_count:
                    print('Convergence criterion met with',nn.epoch_count,'epochs')
                    print('Final training and testing error:',round(train_error,7),round(test_error,7))
                    break
                else:
                    epoch_convergence_counter += 1 