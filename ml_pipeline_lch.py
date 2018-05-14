import pandas as pd
import numpy as np
import re


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



def print_null_freq(df):
    '''
    For all columns in a given dataframe, calculate and print number of null and non-null values

    Attribution: https://github.com/yhat/DataGotham2013/blob/master/analysis/main.py
    '''
    df_lng = pd.melt(df)
    null_variables = df_lng.value.isnull()
    print(pd.crosstab(df_lng.variable, null_variables))



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



def basic_fill_vals(df, col_name, method = None):
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
    elif method == "median":
        replacement_val = df[col_name].median()
        df[col_name] = df[col_name].fillna(replacement_val)
    elif method == "mean":
        replacement_val = df[col_name].mean()
        df[col_name] = df[col_name].fillna(replacement_val)


# def isolate_noncategoricals(df, ret_categoricals = False, geo_cols = None):
#     '''
#     Retrieve list of cateogrical or non-categorical columns from a given dataframe

#     Inputs:
#         df: pandas dataframe
#         ret_categoricals: (boolean) True when output should be list of  
#             categorical colmn names, False when output should be list of 
#             non-categorical column names

#     Outputs: list of column names from data frame
#     '''
#     if ret_categoricals:
#         categorical = [col for col in df.columns if re.search("_bin|_was_null", col)]
#         return categorical + geo_cols
#     else:
#         non_categorical = [col for col in df.columns if not \
#         re.search("_bin", col) and col not in geo_cols]
#         return non_categorical


def is_category(col_name, geos = True):
    '''
    Utility function to determine whether a given column name includes key words or
    phrases indicating it is categorical.

    Inputs:
        col_name: (string) name of a column
        geos: (boolean) whether or not to include geographical words or phrases
            in column name search
    '''
    if geos:
        return re.search("_bin|_was_null|city|state|county|country|zip|zipcode|latitude|longitude", col_name)
    else:
        return re.search("_bin|_was_null", col_name)


def isolate_categoricals(df, categoricals_fcn, ret_categoricals = False, geos_indicator = True):
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
    categorical = [col for col in df.columns if categoricals_fcn(col, geos = geos_indicator)]
    non_categorical = [col for col in df.columns if not categoricals_fcn(col, geos = geos_indicator)]
    
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

