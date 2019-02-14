
from DecisionTree import DecisionTree_ID3

# problem 1.1a

#decisionTree_ID3 = DecisionTree_ID3('information gain', csv_name='boolean/train.csv', maxDepth = None, full_output = True)
#decisionTree_ID3.print_tree()

decisionTree_ID3 = DecisionTree_ID3('information gain', csv_name='tennis/train.csv', maxDepth = None, full_output = False)
decisionTree_ID3.print_tree()

decisionTree_ID3 = DecisionTree_ID3('majority error', csv_name='tennis/train.csv', maxDepth = None, full_output = False)
decisionTree_ID3.print_tree()

decisionTree_ID3 = DecisionTree_ID3('Gini index', csv_name='tennis/train.csv', maxDepth = None, full_output = False)
decisionTree_ID3.print_tree()


