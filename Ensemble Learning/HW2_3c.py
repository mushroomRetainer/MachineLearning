from DecisionTree import AdaBoost
import matplotlib.pyplot as plt

# problem 3c AdaBoosting

credit_card_quantitative_columns = list(range(23))
credit_card_quantitative_columns.remove(1)
credit_card_quantitative_columns.remove(2)
credit_card_quantitative_columns.remove(3)
credit_card_quantitative_columns.remove(5)

adaBoost = AdaBoost('credit_card/train.csv', 1, csv_test_name = 'credit_card/test.csv', quantitative_columns = credit_card_quantitative_columns)
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
plt.title('Credit Card AdaBoosting Total Prediction Error')
plt.legend()

plt.figure()
plt.plot(adaBoost.stump_train_error_weighted_array, label = 'train error weighted')
plt.plot(adaBoost.stump_train_error_unweighted_array, label = 'train error unweighted')
plt.plot(adaBoost.stump_test_error_array, label = 'test error')
plt.ylabel('error')
plt.xlabel('number of training iterations')
plt.title('Credit Card AdaBoosting Individual Stump Error')
plt.legend()

plt.figure()
plt.plot(adaBoost.alpha_array)
plt.ylabel('alpha')
plt.xlabel('number of training iterations')
plt.title('Credit Card AdaBoosting alpha')

plt.show()
