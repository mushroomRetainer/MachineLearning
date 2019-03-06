
import numpy as np
from Least_Mean_Squared import get_numerical_data_from_csv, batch_gradient_decent, stochastic_gradient_decent, calculate_error
import matplotlib.pyplot as plt

train_data, x_train, y_train = get_numerical_data_from_csv('concrete/train.csv')
test_data, x_test, y_test = get_numerical_data_from_csv('concrete/test.csv')

w_batch = np.zeros(len(x_train[0]))

r_batch = 1e-2
convergence = 1e5
i = 0
error = []
error.append(calculate_error(x_train, w_batch, y_train))
while convergence > 1e-6:
    w_previous = w_batch[:]
    w_batch = batch_gradient_decent(w_batch, r_batch, x_train, y_train)
    convergence = np.linalg.norm(w_batch - w_previous,2)
    error.append(calculate_error(x_train, w_batch, y_train))
    print('\niteration',i,'\nconvergence =',convergence)
    i += 1

plt.figure()
plt.plot(error, label = 'train error')
#plt.plot(test_error, label = 'test error')
plt.ylabel('error')
plt.xlabel('number of training iterations')
plt.title('Concrete LMS Batch Error')
plt.legend()



w_stoch = np.zeros(len(x_train[0]))

r_stoch = 1e-3
best_error = 1e5
i = 0
error = []
error.append(calculate_error(x_train, w_stoch, y_train))
np.random.seed(12345)
exit_counter = 0
while exit_counter < 1000:
    w_previous = w_stoch[:]
    data_index = np.random.randint(0,len(x_train))
    w_stoch = stochastic_gradient_decent(w_stoch, r_stoch, x_train, y_train, data_index)[0]
    error.append(calculate_error(x_train, w_stoch, y_train))
    print('\niteration',i,'\nbest_error =',best_error)
    if error[-1] <= best_error:
        best_error = error[-1]
        exit_counter = 0
    else:
        exit_counter += 1
    i += 1

plt.figure()
plt.plot(error, label = 'train error')
#plt.plot(test_error, label = 'test error')
plt.ylabel('error')
plt.xlabel('number of training iterations')
plt.title('Concrete LMS Stochastic Error')
plt.legend()



w_analytical = np.linalg.lstsq(x_train, y_train, rcond=None)[0]

batch_test_error = calculate_error(x_test, w_batch, y_test)
stochastic_test_error = calculate_error(x_test, w_stoch, y_test)
analytical_test_error = calculate_error(x_test, w_analytical, y_test)


print('\nOptimal weights batch:',w_batch)
print('Optimal weights stochastic:',w_stoch)
print('Optimal weights analytic:',w_analytical)

print('\nBatch test error:',batch_test_error)
print('Stochastic test error:',stochastic_test_error)
print('Analytic test error:',analytical_test_error)

print('\nLearning rate, r, batch:',r_batch)
print('Learning rate, r, stochastic:',r_stoch)



