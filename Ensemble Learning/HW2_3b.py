
from DecisionTree import DecisionTree_ID3

# problem 3, bagged trees

credit_card_quantitative_columns = list(range(23))
credit_card_quantitative_columns.remove(1)
credit_card_quantitative_columns.remove(2)
credit_card_quantitative_columns.remove(3)
credit_card_quantitative_columns.remove(5)

from DecisionTree import Bagging
import matplotlib.pyplot as plt


sample_size = 12000 # the TA suggested using about half the dataset. However, there doesn't seem to be a strong effect of the end results

bagging = Bagging('credit_card/train.csv', sample_size, csv_test_name = 'credit_card/test.csv', quantitative_columns = credit_card_quantitative_columns, random_seed = 12345)

train_error = []
test_error = []
max_itter = 250 # assignment asks for 1000, put this to a lower number if you want a result in a timely manner
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
plt.title('Credit Card Bagging Total Prediction Error')
plt.legend()

plt.figure()
plt.plot(bagging.tree_train_error_array, label = 'train error')
plt.plot(bagging.tree_test_error_array, label = 'test error')
plt.ylabel('error')
plt.xlabel('number of training iterations')
plt.title('Credit Card Bagging Individual Tree Error')
plt.legend()

plt.show()