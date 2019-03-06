
from DecisionTree import DecisionTree_ID3

# problem 3, fully expanded decision tree

credit_card_quantitative_columns = list(range(23))
credit_card_quantitative_columns.remove(1)
credit_card_quantitative_columns.remove(2)
credit_card_quantitative_columns.remove(3)
credit_card_quantitative_columns.remove(5)

decisionTree_ID3 = DecisionTree_ID3('information gain', csv_name='credit_card/train.csv', quantitative_columns = credit_card_quantitative_columns)
        
training_error = decisionTree_ID3.measure_error(csv_name='credit_card/train.csv')
test_error = decisionTree_ID3.measure_error(csv_name='credit_card/test.csv')

#decisionTree_ID3.print_tree()

print('training_error:',training_error[0])
print('test_error:',test_error[0])
