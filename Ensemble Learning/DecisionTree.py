
import numpy as np
import copy
from collections import Counter

class Tree:
    '''
    this class only stores the tree structure and leaf values for the decision tree
    it uses indexes starting at zero for attributes, so it is designed to be called 
    from within DecisionTree_ID3 where the meanings of the indexes are stored
    '''
    def __init__(self, depth):
        self.children = []
        self.depth = depth # needed for pruning, etc.
        
        # one of these must be defined later:
        self.value = None
        self.split_attribute = None
        
        self.split_rule = None
        self.split_data = None
        
        # TODO: add supoort for this:
#        self.default_label = None # this will handle any attribute values that aren't in the testing set
    
    def set_value(self, value):
        '''
        for leaf nodes, sets the value that this leaf will always return
        '''
        self.value = value
    
    def set_split_attribute(self, split_attribute, num_children):
        '''
        for branch nodes, sets the attribute that we are splitting on
        and adds num_children trees as children, each with depth that is one more than current level
        '''
        self.split_attribute = split_attribute
        
        for i in range(num_children):
            self.children.append( Tree( self.depth + 1 ) )
    
    # TODO: add supoort for this:
#    def set_default_label(self, label):
#        self.default_label = label
    
    def get_value(self, attributes):
        '''
        recursively drills down into the tree to find the value based on a list of attributes
        '''
        if self.value is not None:
#            print('found leaf value of',self.value)
            return self.value
        else:
            attribute_value = attributes[self.split_attribute]
#            print('using value of',attribute_value,'for attribute',self.split_attribute)
            return self.children[attribute_value].get_value(attributes)
    
    def get_children(self):
        return self.children
    
    def store_split_data(self, split_rule, split_data):
        self.split_rule = split_rule
        self.split_data = split_data

    def print_tree(self, index_to_attributes_list, split_attribute, attribute_value):
        output = '\t'*(self.depth)
        if self.depth == 0:
            output += 'Decision Tree Structure (attributes are zero indexed):'
        else:
#            output += 'if attribute ' + str(split_attribute) + ' is value ' + str(attribute_value) +': '
            output += 'if attribute ' + str(split_attribute) + " is value '" + str(index_to_attributes_list[split_attribute][attribute_value]) +"': "
            
        if self.value is not None: 
#            output += 'label is ' + str(self.value)
            output += "label is '" + str(index_to_attributes_list[-1][self.value]) + "'"
            
        print(output)
        if self.split_data is not None:
            output = '\t'*(self.depth + 1) + self.split_rule + ': '
            for i in range(len(self.split_data)):
                output += str(i) + ' = ' + str(round(self.split_data[i],3))
                if i < len(self.split_data) - 1:
                     output += ', '
            print(output)
        
        if self.value is None: 
            attribute_counter = 0
            for child in self.children:
                child.print_tree(index_to_attributes_list, self.split_attribute, attribute_counter)
                attribute_counter += 1


class DecisionTree_ID3:
    
    def __init__(self, split_rule, csv_name = None, data = None, data_weights = None, maxDepth = None, 
                 quantitative_columns = [], missing_data_flag = None, missing_data_method = None, full_output = False,
                 random_split_limit = None):
        '''
        slitting_rule is a string that determines how to determine which attribute best splits the data
        must provide either csv_name or data
        '''
        self.random_split_limit = random_split_limit
        
        if csv_name is not None:
            data = self.get_data_from_csv(csv_name)
        self.data = data # the original data. Keep in mind that the quantitative filter does modify this to change them into categorical data columns
        if data_weights is None:
            data_weights = np.ones(len(self.data))
        self.data_weights = np.array(data_weights)
        
        self.quantitative_columns = quantitative_columns
        self.quantitative_data = []
        self.missing_data_flag = missing_data_flag
        self.missing_data_method = missing_data_method
        self.full_output = full_output
        
        self.default_labels = [] # need this to cover our bases in case the testing data constains something that isn't in the training data
        
        self.data_numerical = self.convert_data_to_integers(self.data) # converts all categorical and quantitative data into integers starting at zero
        
        # need to remove data that is zero weighted and store the original for error testing
        self.data_weights_original = self.data_weights
        self.data_numerical_original = self.data_numerical
        self.data_numerical, self.data_weights = self.exclude_zero_weights(self.data_numerical, self.data_weights)
        
        self.decision_tree = Tree(0) # start with a tree that has zero depth
        self.ID3_algorithm(self.data_numerical, self.data_weights, self.decision_tree, maxDepth, split_rule)
        
        
    
    def exclude_zero_weights(self, data, weights):
        data_filter = weights != 0
#        filtered_data = []
#        for i in range(len(data)):
#            if data_filter[i]:
#                filtered_data.append(data[i])
#        print('data_filter:',data_filter)
#        return filtered_data, weights[data_filter]
        return data[data_filter], weights[data_filter]
    
    def fill_missing_data(self, data):
        # most common value techniques: https://stackoverflow.com/questions/1518522/find-the-most-common-element-in-a-list
        missing_data_count = 0
        np_data = np.array(data) # the data is still all strings, but I need to use numpy's slicing methods to fill in the data
        
        if self.missing_data_method == 'most common':
            for col in range(len(data[0])-1): # don't need to look through the labels
                column = list(np_data[:,col])
                replacement_value = max(set(column), key = column.count)
                data_filter = np_data[:,col] == self.missing_data_flag
                np_data[data_filter,col] = replacement_value
                missing_data_count += np.sum(data_filter)
            print('Replaced',missing_data_count,'values labeled',self.missing_data_flag,'using the method',self.missing_data_method)
            
        elif self.missing_data_method == 'most common for label':
            
            # need the data row and columns inverted to make it easier to work with
            inverted_data = []
            for col in range(len(data[0])):
                inverted_data.append([])
                for row in range(len(data)):
                    inverted_data[col].append(data[row][col])
            inverted_data = np.array(inverted_data)
            
            for attribute in range(len(inverted_data)-1):
                for instance in range(len(inverted_data[0])):
                    if inverted_data[attribute,instance] == self.missing_data_flag:
                        data_filter = inverted_data[-1,instance] == inverted_data[-1,:]
#                        replacement_value = max(set(inverted_data[attribute,data_filter]), key = inverted_data[attribute,data_filter].count)
                        replacement_value = Counter(inverted_data[attribute,data_filter]).most_common(1)[0][0]
                        inverted_data[attribute,instance] = replacement_value
                        missing_data_count += 1
            #invert the data back
            np_data = np.transpose(inverted_data)
            print('Replaced',missing_data_count,'values labeled',self.missing_data_flag,'using the method',self.missing_data_method)
        else:
#            print('Did not replace missing values')
            pass
        return np_data
        
    def get_data_from_csv(self, csv_name):
        '''
        reads data in from a csv, assumes no headers and that final column is the label
        '''
        data = []
        with open(csv_name, 'r') as f:
            for line in f:
                data.append(line.strip().split(','))
        return data
    
    def quantitative_to_categorical(self, data, initial = True):
        '''
        converts quantitative data to categorical based on the median of the training dataset
        if this data is the training data, use initial = True to calculate and store the median
        otherwise, send initial = False to use the current stored median value(s)
        '''
        
        # replace all numerical values with strings that describe whether they are at or below the cutoff value (chosen to be the median)
        for counter, quantitative_column in enumerate(self.quantitative_columns):
            numeric_values = []
            for i in range(len(data)):
                numeric_values.append( float(data[i][quantitative_column]) )
            
            # if we are initalizing the dataset, we need to calculate the median and store it
            if initial:
                cutoff = np.median(numeric_values)
                self.quantitative_data.append(cutoff)
            # otherwise, we need to use the previously calculated median
            else:
                cutoff = self.quantitative_data[counter]
            cutoff = float(cutoff)
            high_string = 'at or above ' + str(cutoff)
            low_string = 'below ' + str(cutoff)
            for i, value in enumerate(numeric_values):
                if value >= cutoff:
                    data[i][quantitative_column] = high_string
                else:
                    data[i][quantitative_column] = low_string
        return data
        
    
    def convert_data_to_integers(self, data, initial = True):
        '''
        converts data from strings to a numpy array
        data should be a list of lists, where each row is a single labled instance
        the final column is assumed to be the label
        
        for quantitative varaibles, the algorithm cuts the list in half at the median into a binary label
        '''
        
        data_categorical = self.quantitative_to_categorical(copy.copy(data), initial = initial)
        
        data_categorical = self.fill_missing_data(data_categorical)
        
        length = len(data_categorical)
        width = len(data_categorical[0])
        
        
        # dictionaries for converting between numberical values and attribute strings. Each is a list that is a length of num_attributes,
        # with a single dictionary for each attribute
        if initial:
            self.unused_attribute_indexes = np.zeros(width, dtype = 'int32') # array of counters that store the current unused value for each attribute
            self.attributes_to_index_list = []
            self.index_to_attributes_list = []
            for i in range(width):
                self.attributes_to_index_list.append({})
                self.index_to_attributes_list.append({})
        
        data_numerical = np.zeros([length, width], dtype = 'int32') # start with a blank matrix and fill it later
        
        # start filling in the dictionaries starting at zero for the first attribute encountered and incrementing by one for each unique attribute
        for row in range(length):
            for col in range(width):
                attribute = data_categorical[row][col]
                if not attribute in self.attributes_to_index_list[col]:
                    index = self.unused_attribute_indexes[col]
                    if initial or col == len(self.data[0])-1: # allow the label dataset to keep growing even if this is the testing data. 
                        self.unused_attribute_indexes[col] += 1
                        self.attributes_to_index_list[col][attribute] = index
                        self.index_to_attributes_list[col][index] = attribute
                    else:
#                        print('Attribute',col,'does not have value',attribute,'in training set. Replacing with most common value',self.default_labels[col])
                        index = self.default_labels[col] # overwrite with most common value. This is a attribute value in the testing set that wasn't in the training set
                else:
                    index = self.attributes_to_index_list[col][attribute]
                data_numerical[row][col] = index
                
        # TODO: determine the default labels in case testing data contains values that the training data did not:
        if initial:
            for col in range(width - 1):
                self.default_labels.append(np.bincount(data_numerical[:,col]).argmax())
        
        return data_numerical
    
    def indexes_to_attributes(self, indexes):
        '''
        takes a single set of indexes and converts it to a string attributes
        '''
        attributes = []
        for i in range(len(indexes)):
            index = indexes[i]
            attributes.append( self.index_to_attributes_list[i][index] )
        return attributes
    
    def ID3_algorithm(self, subset_data_numerical, weights, tree, max_depth, split_rule):
        '''
        recursively calls itself until the entire decision tree is built using the ID3 algorithm
        the splitting rule is determined by the rule provided when this class was declared
        assumes that the parent tree object is stored in this class, so doesn't return anything
        '''
        if self.full_output:
                print('\nPerforming ID3 algorithm on the following dataset:')
                print(subset_data_numerical)
                print()
        labels = subset_data_numerical[:,-1]
        
        # there are a few exit conditions:
        #   * all labels are the same, so there is no point in splitting further
        #   * maximum depth has been reached  (if applicable)
        #   * the attribute chosen only has one value so it has no way to split the data set 
        #     (this helps in the event of noise because you may not be able to ever get pure labels)
        
        # check to see if all examples have the same label
        if np.all(labels == labels[0]):
            if self.full_output:
                print('ALL LABELS ARE THE SAME. FOUND A LEAF NODE! Label is',labels[0])
                print('='*100)
            # if so, add a leaf with this label
            tree.set_value(labels[0])
            
            
        # if we have hit the maximum depth are it is not possible to further split the tree, use the most common label
        elif tree.depth == max_depth: # or not self.is_splittable(subset_data_numerical):
            if self.full_output:
                print('MAX DEPTH REACHED. FOUND A LEAF NODE! Label is',labels[0])
            tree.set_value(self.get_most_common_label(labels, weights))
            
            
        else:
            split_attribute, split_values = self.get_best_split_attribute(subset_data_numerical, weights, split_rule)
            
            # one last exit condition: end if the chosen attribute is pure (sometimes machine precision errors result in slightly nonzero purity values for a dead-end attribute):
            if np.all(subset_data_numerical[:,split_attribute] == subset_data_numerical[0,split_attribute]):
                tree.set_value(self.get_most_common_label(labels, weights))
            else:
                num_children = len(self.index_to_attributes_list[split_attribute])
                tree.set_split_attribute(split_attribute, num_children)
                tree.store_split_data(split_rule, split_values)
                # get children and repeat the algorithm on each child with a subset of the data
                attribute_index = 0
                for child in tree.get_children():
                    data_row_filter = subset_data_numerical[:,split_attribute] == attribute_index
                    # check for empty dataset, then make the child a leaf using the most common label:
                    if np.all(data_row_filter == False):
                        child.set_value(self.get_most_common_label(labels, weights))
                    # otherwise, call the algorithm again using the subset of the data
                    else:
                        self.ID3_algorithm(subset_data_numerical[data_row_filter], weights[data_row_filter], child, max_depth, split_rule)
                    attribute_index += 1
                    
    def predict_value(self, attributes):
        return self.decision_tree.get_value(attributes)
        
    def measure_error(self, dataset = None, weights = None, csv_name =None, use_training_data = False):
        '''
        returns the error of the predictions using the provided dataset (can be the test or training data)
        the error is simply the proportion of the data that is predicted incorrectly, where each data point is weighted accoring to the weights
        this also returns the full results as a 0 and 1 array, where 1 is a correct prediction and 0 is incorrect
        '''
        if use_training_data:
            dataset = self.data_numerical_original
            weights = self.data_weights_original
        else:
            if csv_name is not None:
                dataset = self.get_data_from_csv(csv_name)
            dataset = self.convert_data_to_integers(dataset, initial=False)
            if weights is None:
                weights = np.ones(len(dataset))
                
        num_instances = len(dataset)
        correct_array = np.zeros(num_instances)
        for i in range(num_instances):
            instance = dataset[i]
#            print('current instance:', instance)
            if instance[-1] == self.decision_tree.get_value(instance[:-1]):
                correct_array[i] = 1
        incorrect_array = -correct_array + 1
        return np.sum(incorrect_array * weights) / np.sum(weights), correct_array
#        return correct / len(dataset)
    
    def get_most_common_label(self, labels, weights):
        '''
        given a numpy array, returns the value of the most common entry. If there are ties, returns the lowest value based on sorting rules
        '''
        # source: https://stackoverflow.com/questions/6252280/find-the-most-frequent-number-in-a-numpy-vector
#        (values,counts) = np.unique(labels,return_counts=True)
#        ind=np.argmax(counts)
#        return values[ind]
        
        label_dictionary = {}
        for i in range(len(labels)):
            label = labels[i]
            weight = weights[i]
            if label in label_dictionary:
                label_dictionary[label] += weight
            else:
                label_dictionary[label] = weight
                
        # get key with max value from dictionary (i.e. label with highest vote), see https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        return max(label_dictionary, key=lambda key: label_dictionary[key]) 
        
    
    def get_best_split_attribute(self, data, weights, rule = None):
        '''
        calculates all the information gains for the attributes and returns the index for the highest one
        '''
        if rule == 'information gain':
            purity_function = self.get_entropy
        elif rule == 'majority error':
            purity_function = self.get_majority_error
        elif rule == 'Gini index':
            purity_function = self.get_Gini_index
        else: 
            raise Exception
                
        num_attributes = len(data[0]) - 1
        split_values = []
        current_purity = purity_function(data, weights, -1)
        if self.full_output:
            print('current',rule,':',current_purity)
            
        for attribute in range(num_attributes):
            if self.full_output:
                print('\nEvaluating',rule,'for splitting the data along attribute',attribute)
            split_values.append(current_purity - self.expected_purity(data, weights, attribute, purity_function))
        
        # TODO: add capability to randomly downsample the non-zero values
        if self.random_split_limit is not None:
            
            # for debugging:
#            split_values_duplicate = copy.deepcopy(split_values)
            
            split_values = np.array(split_values)
            nonzero_indices = np.argwhere(split_values > 0) # this might need to be 1e-10 or something slightly nonzero
            nonzero_indices = nonzero_indices.reshape(np.size(nonzero_indices)) # for some reason, argwhere returns a 2D array, so this puts it back to 1D
            if self.random_split_limit > len(nonzero_indices):
                allowable_split_indices = nonzero_indices
            else:
                allowable_split_indices = np.random.choice(nonzero_indices, self.random_split_limit, replace = False)
            
            for i in range(len(split_values)):
                if i not in allowable_split_indices:
                    split_values[i] = 0 # this will just zero out the ones that aren't options. This allows all the remaining code to function the same as before
            
            split_values = list(split_values) # need to change this back to a list
            
#            print('Only these indexes are allowed:',allowable_split_indices)
            
            
        best_index = split_values.index(max(split_values))
        if self.full_output:
            print('\nCalculated gains using',rule,'for each attribute:', split_values)
            print("'Best' attribute to split on:", best_index)
            print('='*100)
        
        # for debugging:
#        if self.random_split_limit is not None:
#            best_index_unlimitted = split_values_duplicate.index(max(split_values_duplicate))
#            if best_index != best_index_unlimitted:
#                print('Normally would have chosen',best_index,'but chose',best_index_unlimitted,'due to random limitations')
        
        
        return best_index, split_values
    
    def expected_purity(self, data, weights, column, purity_function):
        '''
        calculates the expected purity (entropy, majority error, GINI index, etc.) of the labels along a given attribute column,
        meaning the sum of the purity of the subsets of each label in the given column each multiplied by the proporiton of the dataset that has that attribute value
        purity funciton should be one of the available functions: entropy, majority error, GINI index, etc.
        the purity function should take data and column as inputs
        '''
        proportions, data_subsets, column_values, weight_subsets = self.get_proportions(data, weights, column)
        purities = []
        for i in range(len(data_subsets)):
            purities.append(purity_function(data_subsets[i], weight_subsets[i], -1))
        purities = np.array(purities)
        if self.full_output:
            print('attribute values are:',column_values)
            print('proportions of dataset that are the above attribute values:',proportions)
            print('datasubsets are:\n',data_subsets)
            print('purity function (info gain, ME, or Gini index) of each datasubset evaluates to:', purities)
            print('expected purity (weighted by proportions) of this potential split evaluates to:', np.sum(proportions * purities))
            print('-'*100)
        return np.sum(proportions * purities)
    
    def get_proportions(self, data, weights, column):
        '''
        returns a numpy array of the proportions, each of the data_subsets, and a list of the unique values in the given column
        '''
        proportions = []
        data_subsets, column_values, weight_subsets = self.split_data(data, weights, column)
#        for data_subset in data_subsets:
#            proportions.append(len(data_subset)/len(data))
        total_weight = 0
        for weight_subset in weight_subsets:
            total_weight += np.sum(weight_subset)
        for weight_subset in weight_subsets:
            proportions.append(np.sum(weight_subset)/total_weight)
        proportions = np.array(proportions)
        return proportions, data_subsets, column_values, weight_subsets
    
    def split_data(self, data, weights, column):
        '''
        splits data into subsets along a column (one subset for each unique value in the column)
        '''
        data_subsets = []
        weight_subsets = []
        column_values = set(data[:,column])
        for value in column_values:
            data_filter = data[:,column] == value
            data_subsets.append(data[data_filter])
#            print('data:',data)
#            print('weights:',weights)
#            print('data_filter:',data_filter)
            weight_subsets.append(weights[data_filter])
        return data_subsets, column_values, weight_subsets
    
    def print_tree(self):
        self.decision_tree.print_tree(self.index_to_attributes_list, None, None)
    
    ####################
    # purity functions #
    ####################
    
    def get_entropy(self, data, weights, column):
        '''
        calculates the entropy of the data along a given column
        '''
        proportions, _, _, _ = self.get_proportions(data, weights, column)
        non_zero_proportions = proportions[proportions>0] # filter out the zero proportions because they should evaluate to zero, but cause problems in the log
        return np.sum(-non_zero_proportions * np.log2(non_zero_proportions))
    
    def get_majority_error(self, data, weights, column):
        '''
        calculates the majority error of the data along a given column
        '''
        proportions, _, _, _ = self.get_proportions(data, weights, column)
        
        return 1 - np.max(proportions) # the error is the proportion of the non-majority label(s), so just subtract the max from unity
    
    def get_Gini_index(self, data, weights, column):
        '''
        calculates the Gini index of the data along a given column
        '''
        proportions, _, _, _ = self.get_proportions(data, weights, column)
        return 1 - np.sum(proportions**2) 
    
ln = np.log # I really hate the way they defined log, so this is for my sanity

class AdaBoost:
    
    def __init__(self, csv_train_name, maxDepth, csv_test_name = None, split_rule = 'information gain',
                 quantitative_columns = [], missing_data_flag = None, missing_data_method = None):
        self.training_data_strings = self.get_data_from_csv(csv_train_name)
        if csv_test_name is not None:
            self.testing_data_strings = self.get_data_from_csv(csv_test_name)
        else:
            self.testing_data_strings = None
        self.split_rule = split_rule
        self.maxDepth = maxDepth
        self.quantitative_columns = quantitative_columns
        self.missing_data_flag = missing_data_flag
        self.missing_data_method = missing_data_method
        
        data_length = len(self.training_data_strings)
        self.weights = np.ones(data_length)/data_length
        self.stump_array = []
        self.alpha_array = []
        
        self.stump_train_error_weighted_array = []
        self.stump_train_error_unweighted_array = []
        self.stump_test_error_array = []
        
        # need to get numerical data saved at this level, rather than in the tree object so we can also feed it through the trees for testing
        decisionStump_ID3 = DecisionTree_ID3(self.split_rule, data = copy.deepcopy(self.training_data_strings), data_weights = self.weights, maxDepth = self.maxDepth, 
                                             quantitative_columns = self.quantitative_columns, missing_data_flag = self.missing_data_flag, 
                                             missing_data_method = self.missing_data_method)
        
        # store the numerical attribute values (rather than string values) of the training and testing data
        self.training_data = decisionStump_ID3.convert_data_to_integers(copy.deepcopy(self.training_data_strings), initial = False)
        self.testing_data = decisionStump_ID3.convert_data_to_integers(copy.deepcopy(self.testing_data_strings), initial = False)
        
    def add_itteration(self):
        # the data is already in pure numerical categorical form, so don't need to pass additional arguements to ID3
        decisionStump_ID3 = DecisionTree_ID3(self.split_rule, data = self.training_data, data_weights = self.weights, maxDepth = self.maxDepth)
        # use this if you want to have each tree work with the string data. Useful for debugging is you want to print the tree strcuture, but slows it down.
#        decisionStump_ID3 = DecisionTree_ID3(self.split_rule, data = copy.deepcopy(self.training_data_strings), data_weights = self.weights, maxDepth = self.maxDepth, 
#                                             quantitative_columns = self.quantitative_columns, missing_data_flag = self.missing_data_flag, 
#                                             missing_data_method = self.missing_data_method)
        self.stump_array.append(decisionStump_ID3)
        
        stump_error_weighted_train, is_correct_array = decisionStump_ID3.measure_error(use_training_data=True)
        self.stump_train_error_weighted_array.append(stump_error_weighted_train)
        
        stump_error_unweighted_train, _ = decisionStump_ID3.measure_error(dataset=self.training_data)
        self.stump_train_error_unweighted_array.append(stump_error_unweighted_train)
        
        stump_error_test, _ = decisionStump_ID3.measure_error(dataset=self.testing_data)
        self.stump_test_error_array.append(stump_error_test)
        
        # convert binary is_correct_array to -1 for incorrect and 1 for correct 
        is_correct_array = is_correct_array * 2 - 1
        
        # calculate alpha, the weight for this decision stump's final vote, as well as the parameter for changing the data weights
        alpha = 1/2 * ln( (1 - stump_error_weighted_train) / stump_error_weighted_train)
        self.alpha_array.append(alpha)
        self.weights *= np.exp( - alpha * is_correct_array)
        
        # for debugging:
#        print_counter = 0
#        index = 0
#        while print_counter < 10 and index < len(is_correct_array):
#            if is_correct_array[index] == -1:
##                print('Incorrectly guessed index',index)
#                print_counter += 1
#            index += 1
#        print('alpha value:',alpha)
#        print('stump error:',error_t)
        
        # normalize
        self.weights /= np.sum(self.weights)
    
    def measure_error(self, use_training = True):
        if use_training:
            data = self.training_data
        else:
            data = self.testing_data
        incorrect = 0
        for data_instance in data:
            if self.predict_value(data_instance[:-1]) != data_instance[-1]:
                incorrect += 1
        return incorrect/len(data)
    
    def predict_value(self, data_instance):
        # acucmulate votes into dictionary
        votes = {}
        for i in range(len(self.stump_array)):
            predicted_label = self.stump_array[i].decision_tree.get_value(data_instance)
            if predicted_label in votes:
                votes[predicted_label] += self.alpha_array[i]
            else:
                votes[predicted_label] = self.alpha_array[i]
               
        # get key with max value from dictionary (i.e. label with highest vote), see https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        return max(votes, key=lambda key: votes[key]) 
    
    # I copied this method from the ID3algorithm class to avoid having to read the csv every single time we build a stump
    def get_data_from_csv(self, csv_name):
        '''
        reads data in from a csv, assumes no headers and that final column is the label
        '''
        data = []
        with open(csv_name, 'r') as f:
            for line in f:
                data.append(line.strip().split(','))
        return data

class Bagging:
    def __init__(self, csv_train_name, sample_size, maxDepth = None, csv_test_name = None, split_rule = 'information gain',
                 quantitative_columns = [], missing_data_flag = None, missing_data_method = None, random_seed = None, downsample_training_count = None,
                 random_forest_downsample_limit = None):
        # send None to simply use a sample size that is the size of the data set
        # the downsample_training_count is specifically for HW2.2c, computing the variance. This parameter is ignored if None, but if a number, randomly downsamples the training data without replacement
         
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.training_data_strings = self.get_data_from_csv(csv_train_name)
        
        if csv_test_name is not None:
            self.testing_data_strings = self.get_data_from_csv(csv_test_name)
        else:
            self.testing_data_strings = None        
            
        if sample_size is None:
            self.sample_size = len(self.testing_data_strings)
        else:
            self.sample_size = sample_size
        
        self.split_rule = split_rule
        self.maxDepth = maxDepth
        self.quantitative_columns = quantitative_columns
        self.missing_data_flag = missing_data_flag
        self.missing_data_method = missing_data_method
        self.random_forest_downsample_limit = random_forest_downsample_limit
        
        self.tree_array = []
        self.tree_train_error_array = []
        self.tree_test_error_array = []
        
        # need to get numerical data saved at this level, rather than in the tree object so we can also feed it through the trees for 
        # this stump object is only constructed in order to get the numerical arrays saved at this level to avoid processing them every time
        decisionTree_ID3 = DecisionTree_ID3(self.split_rule, data = copy.deepcopy(self.training_data_strings), maxDepth = 1, 
                                             quantitative_columns = self.quantitative_columns, missing_data_flag = self.missing_data_flag, 
                                             missing_data_method = self.missing_data_method)
        
        # store the numerical attribute values (rather than string values) of the training and testing data
        self.training_data = decisionTree_ID3.convert_data_to_integers(copy.deepcopy(self.training_data_strings), initial = False)
        self.testing_data = decisionTree_ID3.convert_data_to_integers(copy.deepcopy(self.testing_data_strings), initial = False)
        
        if downsample_training_count is not None:
            self.training_data = self.training_data[np.random.choice(len(self.training_data_strings), downsample_training_count, replace=False)]
        
        self.data_length = len(self.training_data)
        
    def add_itteration(self):
        # choose a set of random weights, effectively choosing a bootstrap sample
        random_indexes = np.random.randint(0,self.data_length,self.sample_size)
        weights = np.zeros(self.data_length)
        for i in random_indexes:
            weights[i] += 1
        
        # the data is already in pure numerical categorical form, so don't need to pass additional arguements to ID3
        decisionTree_ID3 = DecisionTree_ID3(self.split_rule, data = self.training_data, data_weights = weights, 
                                            maxDepth = self.maxDepth, random_split_limit = self.random_forest_downsample_limit)
        
        self.tree_array.append(decisionTree_ID3)
        
        tree_error_train, _ = decisionTree_ID3.measure_error(use_training_data=True)
        self.tree_train_error_array.append(tree_error_train)
        
        tree_error_test, _ = decisionTree_ID3.measure_error(dataset=self.testing_data)
        self.tree_test_error_array.append(tree_error_test)

    def measure_error(self, use_training = True):
        if use_training:
            data = self.training_data
        else:
            data = self.testing_data
        incorrect = 0
        for data_instance in data:
            if self.predict_value(data_instance[:-1]) != data_instance[-1]:
                incorrect += 1
        return incorrect/len(data)
    
    def predict_value(self, data_instance):
        # acucmulate votes into dictionary
        votes = {}
        for i in range(len(self.tree_array)):
            predicted_label = self.tree_array[i].decision_tree.get_value(data_instance)
            if predicted_label in votes:
                votes[predicted_label] += 1
            else:
                votes[predicted_label] = 1
               
        # get key with max value from dictionary (i.e. label with highest vote), see https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        return max(votes, key=lambda key: votes[key]) 
    
    # I copied this method from the ID3algorithm class to avoid having to read the csv every single time we build a stump
    def get_data_from_csv(self, csv_name):
        '''
        reads data in from a csv, assumes no headers and that final column is the label
        '''
        data = []
        with open(csv_name, 'r') as f:
            for line in f:
                data.append(line.strip().split(','))
        return data

if __name__ == '__main__':
    pass # room to test any new features without another script

    