import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
import glob
import pickle

def load_data():
    """
    Load data from pickle files in the 'profiles' directory.
    
    Returns:
        out_flattened (dict): Flattened dictionary containing all profiles.
        out_sorted (dict): Dictionary containing profiles sorted by file.
    """
    files = glob.glob('profiles/*.pkl')

    out_flattened = {}
    out_sorted = {}
    for file in files:
        with open(file, 'rb') as f:
            tmp = pickle.load(f)
            out_flattened = {**out_flattened, **tmp}
            out_sorted[file] = tmp

    return out_flattened, out_sorted


def rolling_windows(X, nwindow):
    """
    Takes as input a DataFrame and a window size with which to compute various statistics.
    
    Args:
        X (DataFrame): The profile data as a DataFrame.
        nwindow (int): The window size of the moving window
    
    Returns:
        df (DataFrame): Rolling features used for the model
    """
        
    rolling = X.rolling(nwindow)

    mean = rolling.mean()
    mean.columns = mean.columns + f'_mean_rolling_{nwindow}'

    std = rolling.std()
    std.columns = std.columns + f'_std_rolling_{nwindow}'

    skew = rolling.skew()
    skew.columns = skew.columns + f'_skew_rolling_{nwindow}'

    kurt = rolling.kurt()
    kurt.columns = kurt.columns + f'_kurt_rolling_{nwindow}'

    df = pd.concat([mean, std, skew, kurt], axis=1)
    df = df.fillna(0)   

    return df

def process_df_for_ensemble(df, nwindow):
    """
    Takes as input a DataFrame and a window size with which to compute various statistics.
    The window size can be an int of a list of ints.

    Args:
        df (DataFrame): The profile data as a DataFrame.
        nwindow (int or list of ints): The window sizes of the moving window
    
    Returns:
        X (DataFrame): Rolling features used for the model.
    """

    X = pd.DataFrame()

    if isinstance(nwindow, int):
        nwindow = [nwindow]
    
    for n in nwindow:
        roll = rolling_windows(df.copy(), n)
        X = pd.concat([X, roll], axis=1)

    cumsumX = df.cumsum()
    cumsumX = cumsumX.fillna(0)
    cumsumX.columns = cumsumX.columns + '_cumsum'
    
    X = pd.concat([X, cumsumX], axis=1)

    return X

def profile_to_df(profile):
    """
    Takes as input a DataFrame containing the profile information. 
    Normalizes every feature between 0 and 1 and drops the depth.
    Returns two dataframes, one containing the features and the other the target variable,
    i.e. the layer category.

    Args:
        profile (DataFrame): The profile data 
    
    Returns:
        X (DataFrame): The normalized data
        y (DataFrame or maybe Series idk): The target variable.
    """

    X = profile.iloc[:, 1:-1].copy()
    X = X.fillna(0)
    X = (X - X.min())/ (X.max() - X.min())

    y = profile['layer'].fillna(method='ffill')
    y = y.apply(lambda x: 'S' if 'S' in x else x)

    return X, y

def pickle_to_data(survey_dict, nwindow=100):
    """
    Takes as input a dictionary of DataFrames containing the profile information. 
    Computes, for different window sizes, rolling statistics for the data. 

    Args:
        survey_dict (dict): The various profiles 
    
    Returns:
        X (DataFrame): The normalized data
        y (DataFrame or maybe Series idk): The target variable.
    """

    X = pd.DataFrame()
    y = pd.DataFrame()

    for key in survey_dict.keys():
        profile = survey_dict[key]

        if 'layer' not in profile.columns:
            continue 
        
        Xi, yi = profile_to_df(profile)
        Xi = process_df_for_ensemble(Xi, nwindow=nwindow)

        X = pd.concat([X, Xi], ignore_index=True)
        y = pd.concat([y, yi], ignore_index=True)
        
    return X, y