
import numpy as np

class Perceptron:
    '''
    perceptron classifier for a binary label, where labels are 0 and 1 (converted to -1 and 1 internally)
    '''
    
    def __init__(self, learning_rate_r, train_csv_name, test_csv_name = None, enable_voting = False):
        '''
        defaults to disable voting because this takes a lot of memory. 
        Send enable_voting=True and it will store all weight arrays to allow for perceptron voting
        '''
        self.enable_voting = enable_voting
        self.r = learning_rate_r
        self.train_data, self.train_x, self.train_y = self.get_numerical_data_from_csv(train_csv_name)
        
        if test_csv_name is not None:
            self.test_data, self.test_x, self.test_y = self.get_numerical_data_from_csv(test_csv_name)
        
        self.w = np.zeros(len(self.train_x[0]))
        self.avg_w = np.zeros(len(self.train_x[0]))
        
        if self.enable_voting:
            self.voting_weights = []
            self.voting_counts = []
        
    
    def add_epoch(self):
        '''
        loops through every single training example to compute the weight updates for a single epoch
        stores the voting information only if specified since this is a major memory hog
        '''
        for i in range(len(self.train_data)):
#            if i <25:
#                print(self.train_y[i],' * ',np.matmul(self.w.transpose(), self.train_x[i]))
            if self.train_y[i] * np.matmul(self.w.transpose(), self.train_x[i]) <= 0:
                self.w += self.r * self.train_y[i] * self.train_x[i]
                if self.enable_voting:
                    self.voting_weights.append(np.copy(self.w)) # need a copy, or every single array item will get update each iteration
                    self.voting_counts.append(1)
            elif self.enable_voting:
                self.voting_counts[-1] += 1
            self.avg_w += self.w
        
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
            
        return np.sum( self.all_predictions(self.w, data_x) * data_y <=0 )/ len(data_x)
    
    def get_average_error(self, use_testing_data = False):
        if use_testing_data:
            data_x = self.test_x
            data_y = self.test_y
        else:
            data_x = self.train_x
            data_y = self.train_y
            
        return np.sum( self.all_predictions(self.avg_w, data_x) * data_y <=0 )/ len(data_x)
    
    def get_voted_error(self, use_testing_data = False):
        if use_testing_data:
            data_x = self.test_x
            data_y = self.test_y
        else:
            data_x = self.train_x
            data_y = self.train_y
            
        predictions = np.zeros(len(data_x))
        for i in range(len(self.voting_weights)):
            current_predictions = self.all_predictions(self.voting_weights[i], data_x)
            current_predictions /= abs(current_predictions) # get rid of the magnitudes and only keep the signs
            predictions += self.voting_counts[i] * current_predictions # multiply by the vote weights
        return np.sum( predictions * data_y <=0 )/ len(data_x)
    
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
    learning_rate_r = .1
    num_epochs = 10
    perceptron = Perceptron(learning_rate_r, 'bank-note/test.csv', test_csv_name = 'bank-note/test.csv', enable_voting = True)
    for i in range(num_epochs):
        perceptron.add_epoch()
        
    print('\nUsing learning rate (r):',learning_rate_r,'with',num_epochs,'epochs')
    
    print('\nTraining error of final perceptron:   ', perceptron.get_current_error(False))
    print('Testing error of final perceptron:    ', perceptron.get_current_error(True))
    print('Final Perceptron weight (scaled to bias=1):',np.round(perceptron.w / perceptron.w[0],4))
    
    print('\nTraining error of average perceptron: ', perceptron.get_average_error(False))
    print('Testing error of average perceptron:  ', perceptron.get_average_error(True))
    print('Average Perceptron weight (scaled to bias=1):',np.round(perceptron.avg_w / perceptron.avg_w[0],4))
    
    print('\nTraining error of voted perceptron:   ', perceptron.get_voted_error(False))
    print('Testing error of voted perceptron:    ', perceptron.get_voted_error(True))
    
    print('Voted Perceptron unique weights and counts (each scaled to bias=1, if bias!=0):')
    for i in range(len(perceptron.voting_weights)):
        weights = perceptron.voting_weights[i]
        if weights[0] != 0:
            weights /= weights[0]
        print('\tCount =',perceptron.voting_counts[i],'\tand weights =',np.round(weights,4))
    
    
    