from DecisionTree import Bagging
import matplotlib.pyplot as plt
import numpy as np

# problem 2.2d

bank_quantitative_columns = [0,5,9,11,12,13,14]

sample_size = 2500 # the TA suggested using about half the dataset. However, there doesn't seem to be a strong effect of the end results

random_forest_downsample_limit_array = [2,4,6]

bagging_array = []
for random_forest_downsample_limit in random_forest_downsample_limit_array: 
    bagging_array.append(Bagging('bank/train.csv', sample_size, csv_test_name = 'bank/test.csv', 
                                 quantitative_columns = bank_quantitative_columns, random_seed = 12345,
                                 random_forest_downsample_limit = random_forest_downsample_limit))

train_error_matrix = []
test_error_matrix = []
max_itter = 1000 # assignment asks for 1000, put this to a lower number if you want a result in a timely manner

for i in range(max_itter):
    print('Itteration:',i,'of',max_itter)
    train_error_array = []
    test_errors_array = []
    for bagging in bagging_array:
        bagging.add_itteration()
        train_error_array.append(bagging.measure_error())
        test_errors_array.append(bagging.measure_error(use_training = False))
    train_error_matrix.append(train_error_array)
    test_error_matrix.append(test_errors_array)
    
train_error_matrix = np.array(train_error_matrix)
test_error_matrix = np.array(test_error_matrix)

plt.figure()
for i in range(len(random_forest_downsample_limit_array)):
    downsample_num = random_forest_downsample_limit_array[i]
    plt.plot(train_error_matrix[:,i], label = 'train error, feature subset size: ' + str(downsample_num) )
    plt.plot(test_error_matrix[:,i], label = 'test error, feature subset size: ' + str(downsample_num))
plt.ylabel('error')
plt.xlabel('number of training iterations')
plt.title('Random Forest Total Prediction Error')
plt.legend()

plt.figure()
for i in range(len(random_forest_downsample_limit_array)):
    downsample_num = random_forest_downsample_limit_array[i]
    plt.plot(bagging_array[i].tree_train_error_array, label = 'train error, feature subset size: ' + str(downsample_num) )
    plt.plot(bagging_array[i].tree_test_error_array, label = 'test error, feature subset size: ' + str(downsample_num))
plt.ylabel('error')
plt.xlabel('number of training iterations')
plt.title('Random Forest Individual Tree Error')
plt.legend()

plt.show()
