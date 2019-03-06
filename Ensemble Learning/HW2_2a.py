from DecisionTree import AdaBoost
import matplotlib.pyplot as plt

# problem 2.1a

bank_quantitative_columns = [0,5,9,11,12,13,14]
adaBoost = AdaBoost('bank/train.csv', 1, csv_test_name = 'bank/test.csv', quantitative_columns = bank_quantitative_columns)
train_error = []
test_error = []
max_itter = 1000 # assignment asks for 1000, put this to a lower number if you want a result in a timely manner
for i in range(max_itter):
    print('Itteration:',i,'of',max_itter)
    adaBoost.add_itteration()
    train_error.append(adaBoost.measure_error())
    test_error.append(adaBoost.measure_error(use_training = False))
    
plt.figure()
plt.plot(train_error, label = 'train error')
plt.plot(test_error, label = 'test error')
plt.ylabel('error')
plt.xlabel('number of training iterations')
plt.title('AdaBoosting Total Prediction Error')
plt.legend()

plt.figure()
plt.plot(adaBoost.stump_train_error_weighted_array, label = 'train error weighted')
plt.plot(adaBoost.stump_train_error_unweighted_array, label = 'train error unweighted')
plt.plot(adaBoost.stump_test_error_array, label = 'test error')
plt.ylabel('error')
plt.xlabel('number of training iterations')
plt.title('AdaBoosting Individual Stump Error')
plt.legend()

plt.figure()
plt.plot(adaBoost.alpha_array)
plt.ylabel('alpha')
plt.xlabel('number of training iterations')
plt.title('AdaBoosting alpha')

plt.show()