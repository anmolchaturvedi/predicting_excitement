import pandas as pd
from itertools import product
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import graphviz 
from sklearn.metrics import accuracy_score as accuracy


def split_data(df, outcome_var, geo_columns, test_size):
    '''
    Separate data frame into training and test subsets based on specified size 
    for model training and evaluation.

    Inputs:
        df: pandas dataframe
        outcome_var: (string) variable model will predict
        geo_columns:  (list of strings) list of column names corresponding to 
            columns with numeric geographical information (ex: zipcodes)
        test_size: (float) proportion of data to hold back from training for 
            testing
    
    Output: testing and training data sets for predictors and outcome variable
    '''
    # remove outcome variable and highly correlated variables
    all_drops = [outcome_var] + geo_columns
    X = df.drop(all_drops, axis=1)
    # isolate outcome variable in separate data frame
    Y = df[outcome_var]

    return train_test_split(X, Y, test_size = test_size)



def loop_dt(param_dict, training_predictors, testing_predictors, 
                training_outcome, testing_outcome):
    '''
    Loop over series of possible parameters for decision tree classifier to 
    train and test models, storing accuracy scores in a data frame

    Inputs: 
        param_dict: (dictionary) possible decision tree parameters 
        training_predictors: data set of predictor variables for training
        testing_predictors: data set of predictor variables for testing
        training_outcome: outcome variable for training
        testing_outcome: outcome variable for testing

    Outputs: 
        accuracy_df: (data frame) model parameters and accuracy scores for 
            each iteration of the model

    Attribution: adapted combinations of parameters from Moinuddin Quadri's 
    suggestion for looping: https://stackoverflow.com/questions/42627795/i-want-to-loop-through-all-possible-combinations-of-values-of-a-dictionary 
    and method for faster population of a data frame row-by-row from ShikharDua: 
    https://stackoverflow.com/questions/10715965/add-one-row-in-a-pandas-dataframe
    '''
    rows_list = []
    for params in list(product(*param_dict.values())):
        dec_tree = DecisionTreeClassifier(criterion = params[0], 
                                          max_depth = params[1],
                                          max_features = params[2],
                                          min_samples_split = params[3])
        dec_tree.fit(training_predictors, training_outcome)

        train_pred = dec_tree.predict(training_predictors)
        test_pred = dec_tree.predict(testing_predictors)

        # evaluate accuracy
        train_acc = accuracy(train_pred, training_outcome)
        test_acc = accuracy(test_pred, testing_outcome)

        acc_dict = {}
        acc_dict['criterion'], acc_dict['max_depth'], acc_dict['max_features'], acc_dict['min_samples_split'] = params
        acc_dict['train_acc'] = train_acc
        acc_dict['test_acc'] = test_acc
        
        rows_list.append(acc_dict)

    accuracy_df = pd.DataFrame(rows_list) 

    return accuracy_df


def create_best_tree(accuracy_df, training_predictors, training_outcome):
    '''
    Create decision tree based on highest accuracy score in model testing, to 
    view feature importance of each fitted feature

    Inputs:
        accuracy_df: (data frame) model parameters and accuracy scores for 
            each iteration of the model
        training_predictors: data set of predictor variables for training
        training_outcome: outcome variable for training

    Outputs:
        best_tree: (classifier object) decision tree made with parameters used 
            for highest-ranked model in terms of accuracy score during 
            parameters loop
    '''
    accuracy_ranked = accuracy_df.sort_values('test_acc', ascending = False)
    dec_tree = DecisionTreeClassifier(
    criterion = accuracy_ranked.loc[accuracy_ranked.iloc[0].name, 'criterion'],
    max_depth = accuracy_ranked.loc[accuracy_ranked.iloc[0].name, 'max_depth'],
    max_features = accuracy_ranked.loc[accuracy_ranked.iloc[0].name, 'max_features'], 
    min_samples_split = accuracy_ranked.loc[accuracy_ranked.iloc[0].name, 'min_samples_split'])

    dec_tree.fit(training_predictors, training_outcome)
    
    return dec_tree
    

def feature_importance_ranking(best_tree, training_predictors):
    '''
    View feature importance of each fitted feature

    Inputs:
        best_tree: (classifier object) decision tree made with parameters used 
            for highest-ranked model in terms of accuracy score during 
            parameters loop

    Outputs:
        features_df: (data frame) table of feature importance for each 
        predictor variable
    '''
    features_df = pd.DataFrame(best_tree.feature_importances_, 
                                training_predictors.columns).rename(
                                columns = {0: 'feature_importance'}).sort_values(
                                by = 'feature_importance', ascending = False)
    return features_df


def visualize_best_tree(best_tree, training_predictors):
    '''
    Visualize decision tree object with GraphWiz 
    '''
    viz = sklearn.tree.export_graphviz(best_tree, 
                    feature_names = training_predictors.columns,
                    class_names=['Financially Stable', 'Financial Distress'],
                    rounded=False, filled=True)

    with open("tree.dot") as f:
        dot_graph = f.read()
        graph = graphviz.Source(dot_graph)
    
    return graph
