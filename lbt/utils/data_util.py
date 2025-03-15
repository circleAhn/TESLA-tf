import numpy as np
import pandas as pd
import numbers

def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)

def train_test_split(X, y = None, valid_ratio = 0.1, test_ratio = 0.1):
    if isinstance(X, pd.DataFrame):
        X_len = X.shape[0]
        test_split_num = X_len - int(test_ratio * X_len)
        valid_split_num = test_split_num - int(valid_ratio * X_len)
        train_df, valid_df, test_df = X.iloc[:valid_split_num, :], X.iloc[valid_split_num:test_split_num, :], X.iloc[test_split_num:, :] 
        return train_df, valid_df, test_df
    else:
        X_len = X.shape[0]
        test_split_num = X_len - int(test_ratio * X_len)
        valid_split_num = test_split_num - int(valid_ratio * X_len)
        X_train, X_valid, X_test = X[:valid_split_num, ...], X[valid_split_num:test_split_num, ...], X[test_split_num:, ...]

        if y is not None:
            y_train, y_valid, y_test = y[:valid_split_num, ...], y[valid_split_num:test_split_num, ...], y[test_split_num:, ...]
            return X_train, X_valid, X_test, y_train, y_valid, y_test
    return X_train, X_valid, X_test


def label_smoothing(df, y_col, roll):
    return df[y_col].rolling(roll, center=True, min_periods=1).mean()
    
def get_window_dataset(df, 
                       X_col, 
                       y_col=None, 
                       window_length=20, 
                       target_window_length=1, 
                       accept_window_interval=None, 
                       temporal_info=None,
                       label_smooth=0):
    
    if accept_window_interval is None:
        accept_window_interval = int(window_length * 1.2) * 15 * 1000

    df['time_interval'] = df['date'].diff().dt.total_seconds() / 60
    df.loc[:, 'time_interval'] = df.loc[:, 'time_interval'].rolling(window_length).sum().fillna(-1)
    valid_indices = df['time_interval'].between(window_length, accept_window_interval, inclusive='both')

    
    X, X_col_values = None, None
    if temporal_info is not None:
        assert isinstance(temporal_info, list)
        X = -np.ones([len(df), window_length, len(temporal_info) + 1]) * 100
        X_col_values = df[[X_col] + temporal_info].values
    else:
        X = -np.ones([len(df), window_length, 1])* 100
        X_col_values = np.expand_dims(df[X_col].values, axis=-1)
        
    for i in np.where(valid_indices)[0]:
        if i >= window_length:
            X[i, ...] = X_col_values[i - window_length:i, ...]

    X = np.float32(X[~np.all(X[..., 0] < -90, axis=-1)])

    
    if not y_col:
         return X

    y = -np.ones([len(df), target_window_length])* 100
    
    if label_smooth > 0:
        y_col_values = label_smoothing(df, y_col, roll=label_smooth).values
    else:
        y_col_values = df[y_col].values
    #y_col_values = label_smoothing(df, y_col, roll=10).values
    for i in np.where(valid_indices)[0]:
        if i >= target_window_length:
            y[i] = y_col_values[i - target_window_length:i]

    y = np.float32(y[~np.all(y < -90, axis=-1)])
    
    if target_window_length > 1:
        y = np.expand_dims(y, axis=-1)


    return X, y

