
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
    
    def __init__(self, split_rule, csv_name = None, data = None, maxDepth = None, quantitative_columns = [], missing_data_flag = None, missing_data_method = None, full_output = False):
        '''
        slitting_rule is a string that determines how to determine which attribute best splits the data
        must provide either csv_name or data
        '''
        if csv_name is not None:
            data = self.get_data_from_csv(csv_name)
        self.data = data # the original data. Keep in mind that the quantitative filter does modify this to change them into categorical data columns
        self.quantitative_columns = quantitative_columns
        self.quantitative_data = []
        self.missing_data_flag = missing_data_flag
        self.missing_data_method = missing_data_method
        self.full_output = full_output
        
        self.data_numerical = self.convert_data_to_integers(self.data) # converts all categorical and quantitative data into integers starting at zero
        self.decision_tree = Tree(0) # start with a tree that has zero depth
        self.ID3_algorithm(self.data_numerical, self.decision_tree, maxDepth, split_rule)
        
    
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
            print('Did not replace missing values')
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
        
        unused_attribute_indexes = np.zeros(width, dtype = 'int32') # array of counters that store the current unused value for each attribute
        # dictionaries for converting between numberical values and attribute strings. Each is a list that is a length of num_attributes,
        # with a single dictionary for each attribute
        if initial:
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
                    index = unused_attribute_indexes[col]
                    if initial:
                        unused_attribute_indexes[col] += 1
                        self.attributes_to_index_list[col][attribute] = index
                        self.index_to_attributes_list[col][index] = attribute
                index = self.attributes_to_index_list[col][attribute]
                data_numerical[row][col] = index
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
    
    def ID3_algorithm(self, subset_data_numerical, tree, max_depth, split_rule):
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
            tree.set_value(self.get_most_common_label(labels))
            
            
        else:
            split_attribute, split_data = self.get_best_split_attribute(subset_data_numerical, split_rule)
            
            # one last exit condition: end if the chosen attribute is pure (sometimes machine precision errors result in slightly nonzero purity values for a dead-end attribute):
            if np.all(subset_data_numerical[:,split_attribute] == subset_data_numerical[0,split_attribute]):
                tree.set_value(self.get_most_common_label(labels))
            else:
                num_children = len(self.index_to_attributes_list[split_attribute])
                tree.set_split_attribute(split_attribute, num_children)
                tree.store_split_data(split_rule, split_data)
                # get children and repeat the algorithm on each child with a subset of the data
                attribute_index = 0
                for child in tree.get_children():
                    data_row_filter = subset_data_numerical[:,split_attribute] == attribute_index
                    # check for empty dataset, then make the child a leaf using the most common label:
                    if np.all(data_row_filter == False):
                        child.set_value(self.get_most_common_label(labels))
                    # otherwise, call the algorithm again using the subset of the data
                    else:
                        self.ID3_algorithm(subset_data_numerical[data_row_filter], child, max_depth, split_rule)
                    attribute_index += 1
                    
    def measure_error(self, dataset = None, csv_name =None):
        '''
        returns the error of the predictions using the provided dataset (can be the test or training data)
        the error is simply the proportion of the data that is predicted incorrectly
        '''
        if csv_name is not None:
            dataset = self.get_data_from_csv(csv_name)
        dataset = self.convert_data_to_integers(dataset, initial=False)
        correct = 0
        for instance in dataset:
#            print('current instance:', instance)
            if instance[-1] == self.decision_tree.get_value(instance[:-1]):
                correct += 1
        return correct / len(dataset)
    
    def get_most_common_label(self, labels):
        '''
        given a numpy array, returns the value of the most common entry. If there are ties, returns the lowest value based on sorting rules
        '''
        # source: https://stackoverflow.com/questions/6252280/find-the-most-frequent-number-in-a-numpy-vector
        (values,counts) = np.unique(labels,return_counts=True)
        ind=np.argmax(counts)
        return values[ind]
    
    def get_best_split_attribute(self, data, rule = None):
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
        current_purity = purity_function(data, -1)
        if self.full_output:
            print('current',rule,':',current_purity)
            
        for attribute in range(num_attributes):
            if self.full_output:
                print('\nEvaluating',rule,'for splitting the data along attribute',attribute)
            split_values.append(current_purity - self.expected_purity(data, attribute, purity_function))
        
            
        best_index = split_values.index(max(split_values))
        if self.full_output:
            print('\nCalculated gains using',rule,'for each attribute:', split_values)
            print("'Best' attribute to split on:", best_index)
            print('='*100)
        return best_index, split_values
    
    def expected_purity(self, data, column, purity_function):
        '''
        calculates the expected purity (entropy, majority error, GINI index, etc.) of the labels along a given attribute column,
        meaning the sum of the purity of the subsets of each label in the given column each multiplied by the proporiton of the dataset that has that attribute value
        purity funciton should be one of the available functions: entropy, majority error, GINI index, etc.
        the purity function should take data and column as inputs
        '''
        proportions, data_subsets, column_values = self.get_proportions(data, column)
        purities = []
        for data_subset in data_subsets:
            purities.append(purity_function(data_subset, -1))
        purities = np.array(purities)
        if self.full_output:
            print('attribute values are:',column_values)
            print('proportions of dataset that are the above attribute values:',proportions)
            print('datasubsets are:\n',data_subsets)
            print('purity function (info gain, ME, or Gini index) of each datasubset evaluates to:', purities)
            print('expected purity (weighted by proportions) of this potential split evaluates to:', np.sum(proportions * purities))
            print('-'*100)
        return np.sum(proportions * purities)
    
    def get_proportions(self, data, column):
        '''
        returns a numpy array of the proportions, each of the data_subsets, and a list of the unique values in the given column
        '''
        proportions = []
        data_subsets, column_values = self.split_data(data, column)
        for data_subset in data_subsets:
            proportions.append(len(data_subset)/len(data))
        proportions = np.array(proportions)
        return proportions, data_subsets, column_values
    
    def split_data(self, data, column):
        '''
        splits data into subsets along a column (one subset for each unique value in the column)
        '''
        data_subsets = []
        column_values = set(data[:,column])
        for value in column_values:
            data_filter = data[:,column] == value
            data_subsets.append(data[data_filter])
        return data_subsets, column_values
    
    def print_tree(self):
        self.decision_tree.print_tree(self.index_to_attributes_list, None, None)
    
    ####################
    # purity functions #
    ####################
    
    def get_entropy(self, data, column):
        '''
        calculates the entropy of the data along a given column
        '''
        proportions, _, _ = self.get_proportions(data, column)
        non_zero_proportions = proportions[proportions>0] # fliter out the zero proportions because they should evaluate to zero, but cause problems in the log
        return np.sum(-non_zero_proportions * np.log2(non_zero_proportions))
    
    def get_majority_error(self, data, column):
        '''
        calculates the majority error of the data along a given column
        '''
        proportions, _, _ = self.get_proportions(data, column)
        
#        with open("Output.txt", "a+") as text_file:
#            text_file.write('length of data: '+str(len(data))+ '\n')
#            text_file.write('column of interest: '+str(column)+ '\n')
#            text_file.write('proportions in column of interest:' + str(proportions)+'\n')
#            text_file.write('majority error:' + str(1 - np.max(proportions))+'\n')
#            text_file.write('-'*100+'\n')
        
        return 1 - np.max(proportions) # the error is the proportion of the non-majority label(s), so just subtract the max from unity
    
    def get_Gini_index(self, data, column):
        '''
        calculates the Gini index of the data along a given column
        '''
        proportions, _, _ = self.get_proportions(data, column)
        return 1 - np.sum(proportions**2) 
    
if __name__ == '__main__':
    
#    decisionTree_ID3 = DecisionTree_ID3('information gain', csv_name='car/train.csv', maxDepth = 3)
#    decisionTree_ID3 = DecisionTree_ID3('majority error', csv_name='car/train.csv', maxDepth = None)
#    decisionTree_ID3 = DecisionTree_ID3('Gini index', csv_name='car/train.csv', maxDepth = None)
    
    decisionTree_ID3 = DecisionTree_ID3('information gain', csv_name='boolean/train.csv', maxDepth = None, full_output = True)
#    decisionTree_ID3 = DecisionTree_ID3('majority error', csv_name='boolean/train.csv', maxDepth = None)
#    decisionTree_ID3 = DecisionTree_ID3('Gini index', csv_name='boolean/train.csv', maxDepth = None)
    
    decisionTree_ID3.print_tree() 
    
#    bank_quantitative_columns = [0,5,9,11,12,13,14]
#    decisionTree_ID3 = DecisionTree_ID3('information gain', csv_name='bank/train.csv', maxDepth = 4, quantitative_columns = bank_quantitative_columns)
#    decisionTree_ID3 = DecisionTree_ID3('majority error', csv_name='bank/train.csv', maxDepth = None, quantitative_columns = bank_quantitative_columns)
#    decisionTree_ID3 = DecisionTree_ID3('Gini index', csv_name='bank/train.csv', maxDepth = 2, quantitative_columns = bank_quantitative_columns)
#    decisionTree_ID3 = DecisionTree_ID3('information gain', csv_name='bank/train.csv', maxDepth = 4, quantitative_columns = bank_quantitative_columns, missing_data_flag = 'unknown', missing_data_method = 'most common')
#    decisionTree_ID3 = DecisionTree_ID3('information gain', csv_name='bank/train.csv', maxDepth = 4, quantitative_columns = bank_quantitative_columns, missing_data_flag = 'unknown', missing_data_method = 'most common for label')
    
#    tree = decisionTree_ID3.decision_tree
#    index_to_attributes_list = decisionTree_ID3.index_to_attributes_list
#    tree.print_tree(index_to_attributes_list, None, None)
    
#    training_error = decisionTree_ID3.measure_error(csv_name='bank/train.csv')
#    print('training_error:',training_error)
#    
#    test_error = decisionTree_ID3.measure_error(csv_name='bank/test.csv')
#    print('test_error:',test_error)
    