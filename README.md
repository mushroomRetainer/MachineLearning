This is a machine learning library developed by Landen Blackburn for CS5350/6350 in University of Utah

To Use DecisionTree, simply import the class create a DecisionTree_ID3 object with the desired parameters:
split_rule						purity rule for splitting data. Options are 'information gain', 'majority error', or 'Gini index'
csv_name = None					csv file to pull data from. Should be without headers, have one instance per line, and the last column is assumed to be the label
data = None						can supply a list of lists for the data instead of a csv. The program will then skip reading from the csv
maxDepth = None					limits the max depth of the tree. Use a depth of 1 for a decision stump, leave it at none for a fully expanded tree 
quantitative_columns = [] 		a list of the columns that represent quantitative data. These will be partitioned at the median into categorical data 
missing_data_flag = None 		Value for missing attribute, for example "unknown" or "missing". Must be supplied along with the following parameter, or all value will assume to be unique inputs
missing_data_method = None		Method for dealing with missing data (defaul is to treat all labels as unique). Options are 'most common' or 'most common for label'
full_output = False 			Give the full output of each step of the decision tree. Useful for debugging or analysizing small trees.

