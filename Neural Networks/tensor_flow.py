

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt



def get_numerical_data_from_csv(csv_name):
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
    
    return data, x, y





_, x_train, y_train = get_numerical_data_from_csv('bank-note/train.csv')
_, x_test, y_test = get_numerical_data_from_csv('bank-note/test.csv')


layers = 3 #3,5,9
width = 10 #5,10,25,50,100
use_tanh = True # True, FAlse

for layers in [3,5,9]:
    for width in [5,10,25,50,100]:
        for use_tanh in [False, True]:
            
            keras_input = []
            for i in range(layers):
                if use_tanh:
                    keras_input.append( keras.layers.Dense(width, kernel_initializer=keras.initializers.he_uniform(seed=None), activation=tf.nn.tanh ) )
                else:
                    keras_input.append( keras.layers.Dense(width, kernel_initializer=keras.initializers.he_uniform(seed=None), activation=tf.nn.relu) )
            keras_input.append( keras.layers.Dense(2, kernel_initializer=keras.initializers.he_uniform(seed=None), activation=tf.nn.softmax) )
            
            model = keras.Sequential(keras_input)
                
            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy', #mean_squared_error, sparse_categorical_crossentropy
                          metrics=['accuracy'])
            
            model.fit(x_train, y_train, epochs=50, verbose=0)
            
            train_loss, train_acc = model.evaluate(x_train, y_train)
            test_loss, test_acc = model.evaluate(x_test, y_test)
            if use_tanh:
                function = 'tanh'
            else:
                function = 'relu'
            print('Layers:',layers,'Width:',width,'Activation Function:',function)
            print('Train accuracy:', train_acc)
            print('Test accuracy:', test_acc)
