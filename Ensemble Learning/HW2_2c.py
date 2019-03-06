from DecisionTree import Bagging
import matplotlib.pyplot as plt
import numpy as np

# problem 2.2c

bank_quantitative_columns = [0,5,9,11,12,13,14]

num_runs = 100

trees_per_bag = 100 # if you want to wait a week for the answer, feel free to change this to 1000, but the same point can be proven with a smaller bag

tree_array = []
bag_array = []

sample_size = 500 # the TA suggested using about half the dataset. However, there doesn't seem to be a strong effect of the end results

for run in range(num_runs):
    print('Training bag',run,'of',num_runs)
    
    bagging = Bagging('bank/train.csv', sample_size, csv_test_name = 'bank/test.csv', 
                      quantitative_columns = bank_quantitative_columns, downsample_training_count = 1000)
    for i in range(trees_per_bag):
        bagging.add_itteration()
        
    bag_array.append(bagging)
    
    first_tree = bagging.tree_array[0]
    tree_array.append(first_tree)

print('\nComputing bias and variance for single tree vs bags')
test_numerical_data = bagging.testing_data # the testing data is always the same, so just grab it from the last bag object

num_test_instances = len(test_numerical_data)

bias_single_tree_array      = np.zeros([num_test_instances])
bias_bag_array      = np.zeros([num_test_instances])
variance_single_tree_array  = np.zeros([num_test_instances])
variance_bag_array  = np.zeros([num_test_instances])

for test_instance_counter in range(num_test_instances):
    tree_prediction = np.zeros(num_runs)
    bag_prediction = np.zeros(num_runs)
    
    instance_data = test_numerical_data[test_instance_counter]
    attributes = instance_data[:-1]
    correct_label = instance_data[-1]
    
    for run_counter in range(num_runs):
        tree_prediction[run_counter] = tree_array[run_counter].predict_value(attributes) == correct_label
        bag_prediction[run_counter] = bag_array[run_counter].predict_value(attributes) == correct_label
    
    instance_tree_avg_prediction = np.average(tree_prediction)
    instance_bag_avg_prediction = np.average(bag_prediction)
    
    instance_bias_tree = (1 - instance_tree_avg_prediction ) ** 2
    instance_bias_bag = (1 - instance_bag_avg_prediction ) ** 2
    
    instance_variance_tree = 1 / (num_runs - 1) * np.sum( ( instance_tree_avg_prediction - tree_prediction ) ** 2)
    instance_variance_bag = 1 / (num_runs - 1) * np.sum( ( instance_bag_avg_prediction - bag_prediction ) ** 2)
    
    bias_single_tree_array[test_instance_counter]       = instance_bias_tree
    bias_bag_array[test_instance_counter]               = instance_bias_bag
    variance_single_tree_array[test_instance_counter]   = instance_variance_tree
    variance_bag_array[test_instance_counter]           = instance_variance_bag
    
average_bias_tree       = np.average(bias_single_tree_array)
average_bias_bag        = np.average(bias_bag_array)
average_variance_tree   = np.average(variance_single_tree_array)
average_variance_bag    = np.average(variance_bag_array)

general_squared_error_tree = average_bias_tree + average_variance_tree
general_squared_error_bag = average_bias_bag + average_variance_bag

print('average_bias_tree:',average_bias_tree)
print('average_bias_bag:',average_bias_bag)
print('average_variance_tree:',average_variance_tree)
print('average_variance_bag:',average_variance_bag)
print('general_squared_error_tree:',general_squared_error_tree)
print('general_squared_error_bag:',general_squared_error_bag)