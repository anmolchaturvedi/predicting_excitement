import pandas as pd
import itertools
import sklearn
from sklearn import preprocessing, svm, metrics, tree, decomposition
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import graphviz 
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import f1_score, precision_score, recall_score, f1_score, roc_auc_score

NOTEBOOK = 1

def split_data(df, outcome_var, geo_columns, test_size, seed = None):
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

    return train_test_split(X, Y, test_size = test_size, random_state = seed)


def temporal_train_test_split(df, outcome_var, exclude = []):
    skips = [outcome_var] + exclude
    Xs = df.drop(skips, axis = 1)
    Ys = df[outcome_var]
    return (Xs, Ys)


def loop_multiple_classifiers(training_predictors, testing_predictors,
         training_outcome, testing_outcome, param_dict = None, set_num = None):
    '''
    Attribution: Adapted from Rayid Ghani magicloop and simpleloop examples
    https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py
    '''
    classifier_type = {
        "Logistic Regression": LogisticRegression()
    }
        # "KNN": KNeighborsClassifier(penalty='l1', C=1e5),
        # "Decision Tree": DecisionTreeClassifier(),
        # "SVM": svm.SVC(),
        # "Naive Bayes": GaussianNB(),
        # "Random Forest": RandomForestClassifier()
        # 'AdaBoost': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        # 'Bagging': BaggingClassifier(base_estimator=LogisticRegression(penalty='l1', C=1e5))
    


    results_df =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'baseline_precision',
                                    'baseline_recall','auc-roc', 'f1', 'p_at_1', 'r_at_1',
                                    'p_at_2', 'r_at_2', 'p_at_5', 'r_at_5', 'p_at_10', 
                                    'r_at_10','p_at_20','r_at_20','p_at_30','r_at_30', 'p_at_50', 'r_at_50'))
                                    

    if param_dict is None:
    # define parameters to loop over. Thanks to the DSSG team for the recommendations!
        param_dict = {
        "Logistic Regression": { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10], 'random_state':[1008]},
        # 'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
        # "Decision Tree": {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': [None, 'sqrt','log2'],'min_samples_split': [2,5,10], 'random_state':[1008]},
        # 'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear'], 'probability':[True, False], 'random_state':[1008]},
        # "Naive Bayes": {},
        # "Random Forest": {'n_estimators': [100, 10000], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs':[-1], 'random_state':[1008]},
        # "AdaBoost": { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
        # "Bagging": {base_estimator=LogisticRegression(penalty='l1', C=1e)}
               }

    for name, classf in classifier_type.items():
        # create dictionaries for each possible tuning option specified 
        # in param_dict
        print("name", name)
        print("classf", classf)
        options = param_dict[name] 
        tuners = list(options.keys())
        list_params = list(itertools.product(*options.values()))
        all_model_params = []

        for params in list_params:
            kwargs_dict = dict(zip(tuners, params))
            all_model_params.append(kwargs_dict)
            print("all_model_params", all_model_params)
        # create all possible models using tuners in dictionaries created 
        # above
        for args in all_model_params:
            print("args", args)
            
            classf.set_params(**args)

            classf.fit(training_predictors, training_outcome)


            # retain only column associated with outcome of interest (1)
            train_pred = classf.predict_proba(training_predictors)[:,1]
            test_pred = classf.predict_proba(testing_predictors)[:,1]


            y_pred_probs_sorted, y_test_sorted = joint_sort_descending(np.array(test_pred), np.array(testing_outcome))
            results_df.loc[len(results_df)] = [name, clf, args, precision_at_k(y_test_sorted, y_pred_probs_sorted, 100.0), 
                recall_at_k(y_test_sorted, y_pred_probs_sorted, 100.0), roc_auc_score(testing_outcome, test_pred),
                precision_at_k(y_test_sorted,y_pred_probs_sorted,1.0), recall_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                f1_at_k(y_test_sorted,y_pred_probs_sorted,1.0),
                precision_at_k(y_test_sorted,y_pred_probs_sorted,2.0), recall_at_k(y_test_sorted,y_pred_probs_sorted,2.0), 
                f1_at_k(y_test_sorted,y_pred_probs_sorted,2.0),
                precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0), recall_at_k(y_test_sorted,y_pred_probs_sorted,5.0),  
                f1_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0), recall_at_k(y_test_sorted,y_pred_probs_sorted,10.0),  
                f1_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0), recall_at_k(y_test_sorted,y_pred_probs_sorted,20.0),  
                f1_at_k(y_test_sorted,y_pred_probs_sorted,30.0),
                precision_at_k(y_test_sorted,y_pred_probs_sorted,30.0), recall_at_k(y_test_sorted,y_pred_probs_sorted,30.0),  
                f1_at_k(y_test_sorted,y_pred_probs_sorted,50.0),
                precision_at_k(y_test_sorted,y_pred_probs_sorted,50.0), recall_at_k(y_test_sorted,y_pred_probs_sorted,50.0)]

            if NOTEBOOK == 1:
                plot_precision_recall_n(testing_outcome, test_pred, model_name)

    return results_df

            
            


def generate_binary_at_k(test_pred, k):
    '''
    Attribution: Rayid Ghani, https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py
    '''
    cutoff_index = int(len(test_pred) * (k / 100.0))
    predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(test_pred))]
    return predictions_binary


def joint_sort_descending(l1, l2):
    '''
    Attribution: Adapted from code by Rayid Ghani, https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py
    '''
    # l1 and l2 have to be numpy arrays
    if not isinstance(l1, (np.ndarray)):
        l1 = np.array(l1)
    if not isinstance(l2, (np.ndarray)):
        l1 = np.array(l2)

    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]


def precision_at_k(testing_outcome, test_pred, k):
    '''
    Attribution: Rayid Ghani, 
    https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py
    '''

    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(test_pred), np.array(testing_outcome))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    #precision = precision[1]  # only interested in precision for label 1
    precision = precision_score(y_true_sorted, preds_at_k)
    return precision

def f1_at_k(testing_outcome, test_pred, k):
    '''
    Attribution: Adapted from prediction and recoll score calculation by Rayid Ghani,
    https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py
    '''

    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(test_pred), np.array(testing_outcome))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    #precision = precision[1]  # only interested in precision for label 1
    f1 = f1_score(y_true_sorted, preds_at_k)
    return f1

def recall_at_k(testing_outcome, test_pred, k):
    '''
    Attribution: Rayid Ghani, 
    https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(test_pred), np.array(testing_outcome))
    preds_at_k = generate_binary_at_k(y_scores_sorted, k)
    #precision = precision[1]  # only interested in precision for label 1
    recall = recall_score(y_true_sorted, preds_at_k)
    return recall



def plot_precision_recall_n(testing_outcome, test_pred, model_name):
    '''
    Attribution: Rayid Ghani, 
    https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py
    '''
    from sklearn.metrics import precision_recall_curve
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(testing_outcome, test_pred)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(test_pred)
    for value in pr_thresholds:
        num_above_thresh = len(test_pred[test_pred>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()

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
    for clf_type, classifier in classifier_type.items():
    
        for params in list(itertools.product(*param_dict.values())):
            classifier(params)
            dec_tree.fit(training_predictors, training_outcome)


    rows_list = []
    for params in list(itertools.product(*param_dict.values())):
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
                                columns = {0: 'feature_importance'}, inplace = True)
    features_df.sort_values(by = 'feature_importance', ascending = False)
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
