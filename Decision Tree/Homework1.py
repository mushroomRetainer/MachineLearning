
from DecisionTree import DecisionTree_ID3


# problem 1.1a (uncomment these blocks to run them)

#decisionTree_ID3 = DecisionTree_ID3('information gain', csv_name='boolean/train.csv', maxDepth = None, full_output = True)
#decisionTree_ID3.print_tree()

#decisionTree_ID3 = DecisionTree_ID3('majority error', csv_name='tennis/train.csv', maxDepth = None, full_output = True)
#decisionTree_ID3.print_tree()

#decisionTree_ID3 = DecisionTree_ID3('Gini index', csv_name='tennis/train.csv', maxDepth = None, full_output = False)
#decisionTree_ID3.print_tree()

#decisionTree_ID3 = DecisionTree_ID3('information gain', csv_name='tennis/train_missing.csv', maxDepth = None, full_output = False, missing_data_flag = 'Missing', missing_data_method = 'most common')
#decisionTree_ID3.print_tree()
#decisionTree_ID3 = DecisionTree_ID3('information gain', csv_name='tennis/train_missing.csv', maxDepth = None, full_output = False, missing_data_flag = 'Missing', missing_data_method = 'most common for label')
#decisionTree_ID3.print_tree()




gain_functions = ['information gain', 'majority error', 'Gini index']

print('-'*50)
print(' '*18,'Problem 2.2')
print('-'*50)

latex_table = ''

for maxDepth in range(1,7):
    latex_table += str(maxDepth)
    for gain_function in gain_functions:
        print('\nCurrent run:',gain_function,'with max depth of',maxDepth)
        decisionTree_ID3 = DecisionTree_ID3(gain_function, csv_name='car/train.csv', maxDepth = maxDepth)
        
        training_error = decisionTree_ID3.measure_error(csv_name='car/train.csv')
        test_error = decisionTree_ID3.measure_error(csv_name='car/test.csv')
        print('training error (testing error):\t',round(training_error,3),'('+str(round(test_error,3))+')')
        
        latex_table += ' & ' + str(round(training_error,3)) + ' (' + str(round(test_error,3)) + ')'
    latex_table += ' \\\\ \n'

print('\nLatex table:')
print(latex_table)

print('-'*50)
print(' '*18,'Problem 2.3a')
print('-'*50)

bank_quantitative_columns = [0,5,9,11,12,13,14]

latex_table = ''

for maxDepth in range(1,17):
    latex_table += str(maxDepth)
    for gain_function in gain_functions:
        print('\nCurrent run:',gain_function,'with max depth of',maxDepth)
        decisionTree_ID3 = DecisionTree_ID3(gain_function, csv_name='bank/train.csv', maxDepth = maxDepth, quantitative_columns = bank_quantitative_columns)
        
        training_error = decisionTree_ID3.measure_error(csv_name='bank/train.csv')
        test_error = decisionTree_ID3.measure_error(csv_name='bank/test.csv')
        print('training error (testing error):\t',round(training_error,3),'('+str(round(test_error,3))+')')
        
        latex_table += ' & ' + str(round(training_error,3)) + ' (' + str(round(test_error,3)) + ')'
    latex_table += ' \\\\ \n'

print('\nLatex table:')
print(latex_table)

print('-'*50)
print(' '*18,'Problem 2.3b')
print('-'*50)


latex_table = ''

for maxDepth in range(1,17):
    latex_table += str(maxDepth)
    for gain_function in gain_functions:
        print('\nCurrent run:',gain_function,'with max depth of',maxDepth)
        decisionTree_ID3 = DecisionTree_ID3(gain_function, csv_name='bank/train.csv', maxDepth = maxDepth, quantitative_columns = bank_quantitative_columns, missing_data_flag = 'unknown', missing_data_method = 'most common')
        training_error = decisionTree_ID3.measure_error(csv_name='bank/train.csv')
        test_error = decisionTree_ID3.measure_error(csv_name='bank/test.csv')
        print('training error (testing error):\t',round(training_error,3),'('+str(round(test_error,3))+')')
        
        latex_table += ' & ' + str(round(training_error,3)) + ' (' + str(round(test_error,3)) + ')'
    latex_table += ' \\\\ \n'

print('\nLatex table:')
print(latex_table)





# this is for replacing the attribute with the most common that shares the same label. It only runs up to a dept of 8 because this is very slow.

#latex_table = ''
#
#for maxDepth in range(1,9):
#    latex_table += str(maxDepth)
#    for gain_function in gain_functions:
#        print('\nCurrent run:',gain_function,'with max depth of',maxDepth)
#        decisionTree_ID3 = DecisionTree_ID3(gain_function, csv_name='bank/train.csv', maxDepth = maxDepth, quantitative_columns = bank_quantitative_columns, missing_data_flag = 'unknown', missing_data_method = 'most common for label')
#        training_error = decisionTree_ID3.measure_error(csv_name='bank/train.csv')
#        test_error = decisionTree_ID3.measure_error(csv_name='bank/test.csv')
#        print('training error (testing error):\t',round(training_error,3),'('+str(round(test_error,3))+')')
#        
#        latex_table += ' & ' + str(round(training_error,3)) + ' (' + str(round(test_error,3)) + ')'
#    latex_table += ' \\\\ \n'
#
#print('\nLatex table:')
#print(latex_table)

