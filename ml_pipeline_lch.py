import pandas as pd
import numpy as np
import re
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta

def retrieve_data(filename, headers = False, set_ind = None):
    '''
    Read in data from CSV to a pandas dataframe

    Inputs:
        filename: (string) filename of CSV
        headers: (boolean) whether or not CSV includes headers
        ind: (integer) CSV column number of values to be used as indices in 
            data frame

    Output: pandas data frame
    '''
    if headers and isinstance(set_ind, int):
        data_df = pd.read_csv(filename, header = 0, index_col = set_ind)
    elif headers and not set_ind:
        data_df = pd.read_csv(filename, header = 0)
    else:
        data_df = pd.read_csv(filename)
    return data_df



def print_null_freq(df, blanks_only = False):
    '''
    For all columns in a given dataframe, calculate and print number of null and non-null values

    Attribution: Adapted from https://github.com/yhat/DataGotham2013/blob/master/analysis/main.py
    '''
    df_lng = pd.melt(df)
    null_variables = df_lng.value.isnull()
    all_rows = pd.crosstab(df_lng.variable, null_variables)
        
    if blanks_only:
        return all_rows[all_rows[True] > 0]
    else: 
        return all_rows

def still_blank(train_test_tuples):
    '''
    Check for remaining null values after dummy variable creation is complete.
    '''
    to_impute = []
    for train, test in train_test_tuples:
        with_blanks = print_null_freq(train, blanks_only = True)
        print(with_blanks)
        print()
        to_impute.append(list(with_blanks.index))
    return to_impute


def create_col_ref(df):
    '''
    Develop quick check of column position via dictionary
    '''
    col_list = df.columns
    col_dict = {}
    for list_position, col_name in enumerate(col_list):
        col_dict[col_name] = list_position
    return col_dict


def abs_diff(col, factor, col_median, MAD):
    '''
    Calculate modified z-score of value in pandas data frame column, using 
    sys.float_info.min to avoid dividing by zero

    Inputs:
        col: column name in pandas data frame
        factor: factor for calculating modified z-score (0.6745)
        col_median: median value of pandas data frame column
        MAD: mean absolute difference calculated from pandas dataframe column
    
    Output: (float) absolute difference between column value and column meaan 
        absolute difference

    Attribution: workaround for MAD = 0 adapted from https://stats.stackexchange.com/questions/339932/iglewicz-and-hoaglin-outlier-test-with-modified-z-scores-what-should-i-do-if-t
    '''
    if MAD == 0:
        MAD = 2.2250738585072014e-308 
    return (x - y)/ MAD



def outliers_modified_z_score(df, col):
    '''
    Identify outliers (values falling outside 3.5 times modified z-score of 
    median) in a column of a given data frame

    Output: (pandas series) outlier values in designated column

    Attribution: Modified z-score method for identifying outliers adapted from 
    http://colingorrie.github.io/outlier-detection.html
    '''
    threshold = 3.5
    zscore_factor = 0.6745
    col_median = df[col].astype(float).median()
    median_absolute_deviation = abs(df[col] - col_median).mean()
    
    modified_zscore = df[col].apply(lambda x: abs_diff(x, zscore_factor, 
                                    col_median, median_absolute_deviation))
    return modified_zscore[modified_zscore > threshold]


def convert_dates(date_series):
    '''
    Faster approach to datetime parsing for large datasets leveraging repated dates.

    Attribution: https://github.com/sanand0/benchmarks/commit/0baf65b290b10016e6c5118f6c4055b0c45be2b0
    '''
    dates = {date:pd.to_datetime(date) for date in date_series.unique()}
    return date_series.map(dates)


def view_max_mins(df, max = True):
    '''
    View top and bottom 10% of values in each column of a given data frame

    Inputs: 
        df: pandas dataframe
        max: (boolean) indicator of whether to return to or bottom values

    Output: (dataframe) values at each 100th of a percentile for top or bottom 
        values dataframe column
    '''
    if max:
        return df.quantile(q=np.arange(0.99, 1.001, 0.001))
    else: 
        return df.quantile(q=np.arange(0.0, 0.011, 0.001))



def view_likely_outliers(df, max = True):
    '''
    View percent change between percentiles in top or bottom 10% of values in  
    each column of a given data frame 

    Inputs: 
        df: pandas dataframe
        max: (boolean) indicator of whether to return to or bottom values

    Output: (dataframe) percent changes between values at each 100th of a 
        percentile for top or bottom values in given dataframe column
    '''
    if max:
        return df.quantile(q=np.arange(0.9, 1.001, 0.001)).pct_change()
    else: 
        return df.quantile(q=np.arange(0.0, 0.011, 0.001)).pct_change()



def remove_over_under_threshold(df, col, min_val = False, max_val = False, lwr_threshold = None, upr_threshold = False):
    '''
    Remove values over given percentile or value in a column of a given data 
    frame
    '''
    if max_val:
        df.loc[df[col] > max_val, col] = None
    if min_val:
        df.loc[df[col] < min_val, col] = None
    if upr_threshold:
        maxes = view_max_mins(df, max = True)
        df.loc[df[col] > maxes.loc[upr_threshold, col], col] = None
    if lwr_threshold:
        mins = view_max_mins(df, max = False)
        df.loc[df[col] < mins.loc[lwr_threshold, col], col] = None
    

def remove_dramatic_outliers(df, col, threshold, max = True):
    '''
    Remove values over certain level of percent change in a column of a given 
    data frame
    '''
    if max:
        maxes = view_max_mins(df, max = True)
        likely_outliers_upper = view_likely_outliers(df, max = True)
        outlier_values = list(maxes.loc[likely_outliers_upper[likely_outliers_upper[col] > threshold][col].index, col])
    else: 
        mins = view_max_mins(df, max = False)
        likely_outliers_lower = view_likely_outliers(df, max = False)
        outlier_values = list(mins.loc[likely_outliers_lower[likely_outliers_lower[col] > threshold][col].index, col])
    
    df = df[~df[col].isin(outlier_values)]



def basic_fill_vals(df, col_name, test_df = None, method = None, replace_with = None):
    '''
    For columns with more easily predicatable null values, fill with mean, median, or zero

    Inputs:
        df: pandas data frame
        col_name: (string) column of interest
        method: (string) desired method for filling null values in data frame. 
            Inputs can be "zeros", "median", or "mean"
    '''
    if method == "zeros":
        df[col_name] = df[col_name].fillna(0)
    elif method == "replace":
        replacement_val = replace_with
        df[col_name] = df[col_name].fillna(replacement_val)
    elif method == "median":
        replacement_val = df[col_name].median()
        df[col_name] = df[col_name].fillna(replacement_val)
    elif method == "mean":
        replacement_val = df[col_name].mean()
        df[col_name] = df[col_name].fillna(replacement_val)

    # if imputing train-test set, fill test data frame with same values
    if test_df:
        test_df[col_name] = test_df[col_name].fillna(replacement_val)



def check_col_types(df):
    return pd.DataFrame(df.dtypes, df.columns).rename({0: 'data_type'}, axis = 1)


def view_cols(df):
    '''
    View unique values across columns in given data frame.
    '''
    for col in df.columns:
        print(col)
        print(df[col].unique())
        print()


def is_category(col_name, flag = None, geos = True):
    '''
    Utility function to determine whether a given column name includes key words or
    phrases indicating it is categorical.

    Inputs:
        col_name: (string) name of a column
        geos: (boolean) whether or not to include geographical words or phrases
            in column name search
    '''
    search_for = ["_bin","_was_null"]

    if flag:
        search_for += [flag]

    if geos:
        search_for += ["city", "state", "county", "country", "zip", "zipcode", "latitude", "longitude"]

    search_for = "|".join(search_for)

    return re.search(search_for, col_name)



def replace_dummies(df, cols_to_dummy):
    return pd.get_dummies(df, columns = cols_to_dummy, dummy_na=True)



def isolate_categoricals(df, categoricals_fcn, ret_categoricals = False, keyword = None, geos_indicator = True):
    '''
    Retrieve list of cateogrical or non-categorical columns from a given dataframe

    Inputs:
        df: pandas dataframe
        categoricals_fcn: (function) Function to parse column name and return boolean
            indicating whether or not column is categorical
        ret_categoricals: (boolean) True when output should be list of  
            categorical colmn names, False when output should be list of 
            non-categorical column names

    Outputs: list of column names from data frame
    '''
    categorical = [col for col in df.columns if categoricals_fcn(col, flag = keyword, geos = geos_indicator)]
    non_categorical = [col for col in df.columns if not categoricals_fcn(col, flag = keyword, geos = geos_indicator)]
    
    if ret_categoricals:
        return categorical
    else:
        return non_categorical



def change_col_name(df, current_name, new_name):
    '''
    Change name of a single column in a given data frame
    '''
    df.columns = [new_name if col == current_name else col for col in df.columns]


def record_nulls(df):
    for col in list(df.columns):
        title = col + "_was_null"
        df[title] = df[col].isnull().astype(int)
    df = df.loc[:, (df != 0).any(axis=0)]


def drop_unwanted(df, drop_list):
    df.drop(drop_list, axis = 1, inplace = True)


def organize_variables(df, col_names, indicator, var_dict = None):
    if var_dict is None:
        var_dict = {'binary': [], 'tops': [], 'drop': [], 'ids': [], 'geo': [], 'multi': [], 'numeric': []}
    
    if indicator == 'binary':
        var_dict[indicator] += col_names
    elif indicator == 'multi':
        var_dict[indicator] += col_names
    elif indicator == 'numeric':
        var_dict[indicator] += col_names
    elif indicator == 'geo':
        var_dict[indicator] += col_names
    elif indicator == 'ids':
        var_dict[indicator] += col_names
    elif indicator == 'tops':
        var_dict[indicator] += col_names
    elif indicator == 'drop':
        var_dict[indicator] += col_names
    
    return var_dict




def time_series_split(df, date_col, train_size, test_size, increment = 'month', specify_start = None):
    
    if specify_start:
        min_date = datetime.strptime(specify_start, '%Y-%m-%d')
    else:
        min_date = df[date_col].min()

        if min_date.day > 25:
            min_date += datetime.timedelta(days = 7)
            min_date = min_date.replace(day=1, hour=0, minute=0, second=0)

        else:
            min_date = min_date.replace(day=1, hour=0, minute=0, second=0)
    
    if increment == 'month':
        train_max = min_date + relativedelta(months = train_size)
        test_min = train_max + timedelta(days = 1)
        test_max = min(test_min + relativedelta(months = test_size), df[date_col].max())
        
    if increment == 'day':
        train_max = min_date + relativedelta(days = train_size)
        test_min = train_max + timedelta(days = 1)
        test_max = min((test_min + relativedelta(days = test_size)), df[date_col].max())
    
    if increment == 'year':
        train_max = timedelta(months = train_size)
        test_min = train_max + relativedelta(years = train_size)
        test_max = min(test_min + relativedelta(years = test_size), df[date_col].max())
    
    new_df = df[df.columns]
    train_df = new_df[(new_df[date_col] >= min_date) & (new_df[date_col] <= train_max)]
    test_df = new_df[(new_df[date_col] >= test_min) & (new_df[date_col] <= test_max)]
    
    return [train_df, test_df]



def create_expanding_splits(df, total_periods, dates, train_period_base, test_period_size, period = 'month', defined_start = None):
    num_months = total_periods / test_period_size
    months_used = train_period_base
    
    tt_sets = []
    
    while months_used < total_periods:
        
        print("original train period lenth: {}".format(train_period_base))
        train, test = time_series_split(df, date_col = dates, train_size = train_period_base, test_size = test_period_size, increment = period, specify_start = defined_start)
        print("train: {}, test: {}".format(train.shape, test.shape))
        tt_sets.append((train, test))
        train_period_base += test_period_size
        months_used += test_period_size
    
    return tt_sets



def determine_top_dummies(train_test_tuples, var_dict, threshold, max_options = 10):
    set_distro_dummies = []
    counter = 1
    for train, test in train_test_tuples:
        print("starting set {}...".format(counter))
        dummies_dict = {}
        for col in train[var_dict['tops']]:
            print("col: ", col)
            col_sum = train[col].value_counts().sum()
            top = train[col].value_counts().nlargest(max_options)
            
            top_value = 0
            num_dummies = 0

            while ((top_value / col_sum) < threshold) & (num_dummies < max_options):
                top_value += top[num_dummies]
                num_dummies += 1
            print("Keeping top {} values.".format(num_dummies, (top_value / col_sum)))
            print()
            keep_dummies = list(top.index)[:num_dummies]
            dummies_dict[col] = keep_dummies
            
        counter += 1
        set_distro_dummies.append(dummies_dict)

    return set_distro_dummies



def lower_vals_to_other(set_specific_dummies, train_test_tuples):
    counter = 0
    for i, set_dict in enumerate(set_specific_dummies):
        print("starting set {}...".format(counter))
        counter += 1
        for col, vals in set_dict.items():
            train, test = train_test_tuples[i]
            train.loc[~train[col].isin(vals), col] = 'Other'
            test.loc[~test[col].isin(vals), col] = 'Other'





def replace_set_specific_dummies(train_test_tuples, to_dummies):
    augmented_sets = []
    for i, (train, test) in enumerate(train_test_tuples):
        print("Starting set {}...".format(i))
        print(train.shape)
        print(test.shape)
        print("new shapes")
        train = replace_dummies(train, to_dummies)
        print(train.shape)
        test = replace_dummies(test, to_dummies)
        print(test.shape)
        print()
        augmented_sets.append((train, test))
    return augmented_sets
        

def convert_geos(train_test_tuples, geo_cols):
    for train, test in train_test_tuples:
        for col in geo_cols:
            train[col] = train[col].astype('category')
            test[col] = test[col].astype('category')




def dummies_tt_timeporal(train_test_tuples, replace):
    updates = []
    for train, test in train_test_tuples:
        cats_train = ml.isolate_categoricals(train, ml.is_category, ret_categoricals = True, geos_indicator = False)
        train = pd.get_dummies(train, columns = cats_train, dummy_na = True)
        
        cats_test = ml.isolate_categoricals(train, ml.is_category, ret_categoricals = True, geos_indicator = False)
        test = pd.get_dummies(train, columns = cats_test, dummy_na = True)
        updates.append((train, test))
        
    return updates



