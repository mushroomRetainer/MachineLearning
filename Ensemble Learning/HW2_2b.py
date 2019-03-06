from DecisionTree import Bagging
import matplotlib.pyplot as plt

# problem 2.1b

bank_quantitative_columns = [0,5,9,11,12,13,14]

sample_size = 2500 # the TA suggested using about half the dataset. However, there doesn't seem to be a strong effect of the end results

bagging = Bagging('bank/train.csv', sample_size, csv_test_name = 'bank/test.csv', quantitative_columns = bank_quantitative_columns, random_seed = 12345)

train_error = []
test_error = []
max_itter = 2 # assignment asks for 1000, put this to a lower number if you want a result in a timely manner
for i in range(max_itter):
    print('Itteration:',i,'of',max_itter)
    bagging.add_itteration()
    train_error.append(bagging.measure_error())
    test_error.append(bagging.measure_error(use_training = False))
    
plt.figure()
plt.plot(train_error, label = 'train error')
plt.plot(test_error, label = 'test error')
plt.ylabel('error')
plt.xlabel('number of training iterations')
plt.title('Bagging Total Prediction Error')
plt.legend()

plt.figure()
plt.plot(bagging.tree_train_error_array, label = 'train error')
plt.plot(bagging.tree_test_error_array, label = 'test error')
plt.ylabel('error')
plt.xlabel('number of training iterations')
plt.title('Bagging Individual Tree Error')
plt.legend()

plt.show()
