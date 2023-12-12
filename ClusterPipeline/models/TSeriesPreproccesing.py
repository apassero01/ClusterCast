import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
import yfinance as yf
from .SequencePreprocessing import StockSequenceSet,ScalingMethod
# from ClusterProcessing import ClusterGroupParams, StockClusterGroupParams
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    LabelBinarizer,
    RobustScaler,
)
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSet:
    """
    Class that encapsulates a set of features. A set of features is a subset of the columns in a Dateset object. 
    This is helpful for keeping track of different requirements for different groups of features
    """
    def __init__(self, scaling_method, name ='', scale_range = (-1,1)):
        self.name = name
        self.cols = []
        self.scaling_method = scaling_method

        self.scaler = None

        self.range = scale_range



class DataSet:
    """
    Class to encapsuate a dataset. The dataset is essentially a list of dataframes of the
    same shape and size with the same features.
    """
    def __init__(self,group_params):
        self.df = pd.DataFrame()
        self.group_params = group_params

        self.initialize_group_params() 

        self.training_df = pd.DataFrame()
        self.test_df = pd.DataFrame()

        self.created_dataset = False
        self.created_features = False
        self.created_y_targets = False

        self.scaling_dict = group_params.get_scaling_dict()

        self.X_feature_sets = []
        self.y_feature_sets = []
    
    def initialize_group_params(self):
        self.group_params.X_cols = set() 
        self.group_params.y_cols = set()
    
    def create_dataset(self):
        """
        Create the dataset from the tickers and start/end dates and interval
        """
        pass
    def create_features(self):
        """
        Create additonal featues for the dataset
        """
        pass

    def create_y_targets(self, cols_to_create_targets):
        """
        Create target y values for every column in cols_to_create_targets.
        """
        pass

    def preprocess_pipeline(self):
        """
        Preprocess the dataset
        """
        pass 

class StockDataSet(DataSet):

    """
    Class to encapsuate a dataset. The dataset is essentially a list of dataframes of the
    same shape and size with the same features.

    In the case of stocks this is 1 or more stock dataframes that are combined into one dataset
    """

    def __init__(self, group_params, ticker):
        super().__init__(group_params)
        self.ticker = ticker
    
    def preprocess_pipeline(self):
        """
        Preprocess the dataset
        """
        if not self.created_dataset:
            self.create_dataset()
            if len(self.df) < 1 :
                raise ValueError("No dataframes created")
        if not self.created_features:
            self.create_features()
            if len(self.group_params.X_cols) < 0:
                raise ValueError("No features created")
        
        if not self.created_y_targets:
            self.create_y_targets(self.group_params.target_cols)

        self.train_test_split()

        if len(self.training_df) < 1: 
            raise ValueError("training_set is None")
        if len(self.test_df) < 1:
            raise ValueError("test_set is None")

        quant_min_max_feature_sets = [feature_set for feature_set in self.X_feature_sets if feature_set.scaling_method.value == ScalingMethod.QUANT_MINMAX.value or feature_set.scaling_method.value == ScalingMethod.QUANT_MINMAX_G.value] 
        quant_min_max_feature_sets += [feature_set for feature_set in self.y_feature_sets if feature_set.scaling_method.value == ScalingMethod.QUANT_MINMAX.value or feature_set.scaling_method.value == ScalingMethod.QUANT_MINMAX_G.value]

        if len(quant_min_max_feature_sets) > 0:
            self.training_df, self.test_df = self.scale_quant_min_max(quant_min_max_feature_sets, self.training_df, self.test_df)
        
        
        if len(self.group_params.X_cols) < 1:
            raise ValueError("No features created")
        if len(self.group_params.y_cols) < 1:
            raise ValueError("No target features created")
        if len(self.X_feature_sets) < 1:
            raise ValueError("No X_feature_sets created")
        if len(self.y_feature_sets) < 1:
            raise ValueError("No y_feature_sets created")
        


    def create_dataset(self):
        """
        Create the dataset from the tickers and start/end dates and interval
        """
        if self.created_dataset:
            return


        self.df, cols = get_stock_data(self.ticker, self.group_params.start_date, self.group_params.end_date, self.group_params.interval)

        self.group_params.X_cols.update(cols)

        self.created_dataset = True

    def create_features(self):
        """
        Create additonal featues for the dataset
        """
        if self.created_features:
            return
        
        X_feature_sets = []
        X_cols = set() 

        # Create price features
        self.df, feature_set = create_price_vars(self.df,scaling_method=self.scaling_dict['price_vars'])

        self.X_feature_sets.append(feature_set)
        X_cols.update(feature_set.cols)

        # Create trend features
        self.df, feature_set = create_trend_vars(self.df,scaling_method=self.scaling_dict['trend_vars'])

        self.X_feature_sets.append(feature_set)
        X_cols.update(feature_set.cols)

        # Create percent change variables 
        self.df, feature_set = create_pctChg_vars(self.df,scaling_method=self.scaling_dict['pctChg_vars'])
        self.X_feature_sets.append(feature_set)
        X_cols.update(feature_set.cols)

        # Create rolling sum variables
        pctChg_cols = [col for col in X_cols if "pctChg" in col]
        
        for col in pctChg_cols: 
            self.df, feature_set = create_rolling_sum_vars(self.df, col, scaling_method=self.scaling_dict['rolling_vars'])
            self.X_feature_sets.append(feature_set)
            X_cols.update(feature_set.cols)
        
        # print(df.tail())

        # Update the group params with the new  columns
        self.group_params.X_cols.update(X_cols)

        self.created_features = True
    
    def train_test_split(self,feature_list = None, training_percentage = 0.8):
    
        if not feature_list:
            feature_list = list(self.group_params.X_cols.union(self.group_params.y_cols))

            self.training_df, self.test_df = df_train_test_split(self.df, feature_list, training_percentage)
            # test_df = test_df.iloc[self.n_steps:,:] # Remove the first n_steps rows to prevent data leakage
            #TODO mininmal data leakage needs to be addressed, when refactored, this class does not know the steps

    def scale_quant_min_max(self,feature_sets, training_df, test_df):
        """
        Scales the features in the feature sets in dataframe form using custom MinMaxPercentileScaler
        """


        training_df = training_df.copy()
        test_df = test_df.copy()

        
        #Scale the features in the feature sets
        for feature_set in feature_sets: 
            if feature_set.scaling_method.value == ScalingMethod.QUANT_MINMAX.value:
                scaler = MinMaxPercentileScaler()
            if feature_set.scaling_method.value == ScalingMethod.QUANT_MINMAX_G.value:
                scaler = MinMaxPercentileScaler(scaling_mode = 'global')
            scaler.fit(training_df[feature_set.cols])

            # After fitting the scaler to the combined training dataframe, transform the individual dataframes
                
            training_df[feature_set.cols] = scaler.transform(training_df[feature_set.cols])
            test_df[feature_set.cols] = scaler.transform(test_df[feature_set.cols])

            feature_set.scaler = scaler
        
        
        return training_df, test_df

    def create_y_targets(self, cols_to_create_targets):
        """
        Create target y values for every column in cols_to_create_targets.

        NOTE - This is a specific implementation for stock data. The target values are gathered
        in a special way read @create_forward_rolling_sums for more info.
        """
        if self.created_y_targets:
            return
        
        self.df, feature_set = add_forward_rolling_sums(self.df, cols_to_create_targets, scaling_method=self.scaling_dict['target_vars'])

        self.y_feature_sets.append(feature_set)
        self.group_params.y_cols.update(feature_set.cols)
        self.created_y_targets = True 

    def create_sequence_set(self):
        seq = StockSequenceSet(self.group_params, self.training_dfs,self.test_dfs)
        
        return seq


def get_stock_data(
    ticker: str, start_date: datetime, end_date: datetime, interval="1d"
) -> pd.DataFrame:
    """
    Use the yfinance package and read the requested ticker from start_date to end_date. The following additional
    variables are created:

    additional variables: binarized weekday, month, binarized q1-q4

    All scaled variables are

    :param ticker:
    :param start_date:
    :param end_date:
    :param interval: 1d, 1wk, 1mo etc consistent with yfinance api
    :return:
    """

    df = pd.DataFrame()
    if end_date:
        df = yf.download([ticker], start=start_date, end=end_date, interval=interval)
        # vix = yf.download(["^VIX"], start=start_date, end=end_date, interval=interval)
    else:
        df = yf.download([ticker], start=start_date, interval=interval)
        # vix = yf.download(["^VIX"], start=start_date, interval=interval)

    df = df.drop(columns="Adj Close")
    # df['Vix'] = vix['High']
    # Standard column names needed for pandas-ta
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    return df, df.columns.tolist()

def create_price_vars(df: pd.DataFrame,moving_average_vals = [5,10,20,30,50,100], scaling_method = ScalingMethod.SBSG) -> (pd.DataFrame, FeatureSet):
    """
    Create price features from the OHLC data.
    """

    feature_set = FeatureSet(scaling_method, 'price_vars',(0,1))

    feature_set.cols +=["open", "high", "low", "close"]

    # Create price features
    for length in moving_average_vals:
        df["sma" + str(length)] = ta.sma(df.close, length=length)
        df["ema" + str(length)] = ta.ema(df.close, length=length)
        feature_set.cols.append("sma" + str(length))
        feature_set.cols.append("ema" + str(length))
        # df["sma" + str(length)].fillna(method='bfill', inplace=True)
        # df["ema" + str(length)].fillna(method='bfill', inplace=True)

    # lag_df = create_lag_vars(df, feature_set.cols, start_lag, end_lag)
    # df = pd.concat([df, lag_df], axis=1)
    # feature_set.cols += lag_df.columns.tolist()
    
    for col in feature_set.cols:
        df[col] = df[col].fillna(method='bfill')

    return df, feature_set

def create_trend_vars(df: pd.DataFrame,scaling_method = ScalingMethod.SBS) -> (pd.DataFrame, FeatureSet):
    """
    Create trend features which are made up of features that are not price features but are not percent change features.
    """
    feature_set = FeatureSet(scaling_method,'trend_vars', (0,1))

    feature_set.cols += ["volume"]

    # lag_df = create_lag_vars(df, feature_set.cols, start_lag, end_lag)
    # df =  pd.concat([df, lag_df], axis=1)
    # feature_set.cols += lag_df.columns.tolist()

    for col in feature_set.cols:
        df[col] = df[col].fillna(df[col].mean())

    return df, feature_set
       

def create_pctChg_vars(
    df: pd.DataFrame, scaling_method = ScalingMethod.QUANT_MINMAX, start_lag = 1, end_lag = 1
) -> (pd.DataFrame, FeatureSet):
    """
    Create key target variables from the OHLC processed data.

    New variables created:

    The following variables are created based on the OHLC data:
    opHi, opLo, opCl, loCl, hiLo, hiCl.

    The following variables are created based on volume:
    minMaxVol, stdVolume, volChange (change from yesterday to today),

    pctChgClOp - percent change from yesterday's close to today's open
    pctChgClLo - percent change from yesterday's close to today's low
    pctChgClHi - percent change from yesterday's close to today's high
    pctChgClCl - percent change from yesterday's close to today's close

    :param df: A data frame containing OHLC data
    :param rolling_sum_windows: A list of summed window sizes for pctChgClCl to create, or None if not creating
    summed windows
    """
    df = df.copy()

    feature_set = FeatureSet(scaling_method,"pctChg_vars")

    for column in df.columns:
        df['pctChg' + column] = df[column].pct_change() * 100.0
        feature_set.cols.append('pctChg' + column)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # % jump from open to high
    df['opHi'] = (df.high - df.open) / df.open * 100.0
    feature_set.cols.append('opHi')

    # % drop from open to low
    df['opLo'] = (df.low - df.open) / df.open * 100.0
    feature_set.cols.append('opLo')

    # % drop from high to close
    df['hiCl'] = (df.close - df.high) / df.high * 100.0
    feature_set.cols.append('hiCl')

    # % raise from low to close
    df['loCl'] = (df.close - df.low) / df.low * 100.0
    feature_set.cols.append('loCl')

    # % spread from low to high
    df['hiLo'] = (df.high - df.low) / df.low * 100.0
    feature_set.cols.append('hiLo')

    # % spread from open to close
    df['opCl'] = (df.close - df.open) / df.open * 100.0
    feature_set.cols.append('opCl')

    # Calculations for the percentage changes
    df["pctChgClOp"] = np.insert(np.divide(df.open.values[1:], df.close.values[0:-1]) * 100.0 - 100.0, 0, np.nan)
    feature_set.cols.append('pctChgClOp')

    df["pctChgClLo"] = np.insert(np.divide(df.low.values[1:], df.close.values[0:-1]) * 100.0 - 100.0, 0, np.nan)
    feature_set.cols.append('pctChgClLo')

    df["pctChgClHi"] = np.insert(np.divide(df.high.values[1:], df.close.values[0:-1]) * 100.0 - 100.0, 0, np.nan)
    feature_set.cols.append('pctChgClHi')

    for col in feature_set.cols:
        df[col] = df[col].fillna(df[col].mean())

    return df, feature_set

def create_rolling_sum_vars(df: pd.DataFrame, col_name : str, rolling_sum_windows=(1, 2, 3, 4, 5, 6), scaling_method = ScalingMethod.QUANT_MINMAX_G ) -> (pd.DataFrame, FeatureSet):
    """
    Create rolling sum variables for the specified column.

    :param df: A data frame containing all features including the column to create rolling sums for
    :param col_name: The column name to create rolling sums for
    :param rolling_sum_windows: A list of summed window sizes to create

    :return: The dataframe with the new columns added
    """

    if "pctChg" not in col_name:
        raise ValueError("Column name must contain pctChg")
    if col_name not in df.columns:
        raise ValueError("Column name not in dataframe")
    
    df = df.copy()

    feature_set = FeatureSet(scaling_method,col_name + "_rolling")

    for roll in rolling_sum_windows:
        new_col_name = "sum" + col_name + "_" + str(roll)
        df[new_col_name] = df[col_name].rolling(roll).sum()
        feature_set.cols.append(new_col_name)

    for col in feature_set.cols:
        df[col] = df[col].fillna(df[col].mean())

    return df, feature_set 

def create_weekday_month_cols(df):
    """
    Create binarized weekday and and continuous month columns
    """

    encoder = LabelBinarizer()
    wds = pd.DataFrame(
        data=encoder.fit_transform(df.index.weekday),
        index=df.index,
        columns=["Mon", "Tue", "Wed", "Thu", "Fri"],
    )
    cols_to_add = [col for col in wds.columns if col not in df.columns]

    # Concatenate only the unique columns from df_lags to df
    df = pd.concat([df, wds[cols_to_add]], axis=1)
    del wds

    df["month"] = pd.Series(data=df.index.month, index=df.index)

    return df


def create_quarter_cols(df):
    """
    Create a binarized quarter vector. For shorter datasets, some quarters may
    be missing, thus the extra check here
    """
    df_quarters = pd.get_dummies(df.index.quarter, prefix="q", prefix_sep="")
    df_quarters.index = df.index
    for q_missing in filter(
        lambda x: x not in df_quarters, ["q" + str(q) for q in range(1, 5)]
    ):
        df_quarters.loc[:, q_missing] = 0
    cols_to_add = [col for col in df_quarters.columns if col not in df.columns]

    # Concatenate only the unique columns from df_lags to df
    df = pd.concat([df, df_quarters[cols_to_add]], axis=1)
    del df_quarters

    return df


def add_forward_rolling_sums(df: pd.DataFrame, columns: list, scaling_method = ScalingMethod.QUANT_MINMAX) -> (pd.DataFrame, FeatureSet):
    """
    Add the y val for sumPctChgClCl_X for the next X periods. For example sumPctChgClCl_1
    is the percent change from today's close to tomorrow's close, sumPctChgClCl_2 is the percent
    change from today's close to the close 2 days from now, etc. If we want to predict returns
    two days from now the new column sumPctChgClCl+2 would be the training target value.

    :param df: DataFrame to use
    :param columns: a list of column names
    :return: the DataFrame with the new columns added
    """
    feature_set = FeatureSet(scaling_method)

    max_shift = -1
    # print(df.tail())

    for col in columns:
        # Extract the number X from the column name
        num_rows_ahead = int(col.split("_")[-1])

        # Create a new column name based on X
        new_col_name = "sumPctChgclose+" + str(num_rows_ahead)
        feature_set.cols.append(new_col_name)
        # Shift the column values by -X to fetch the value X rows ahead
    
        shifted_col = df[col].shift(-num_rows_ahead)



        df[new_col_name] = shifted_col

        max_shift = max(max_shift, num_rows_ahead)


    print(df.tail())


    return df, feature_set


def create_lag_vars(
    df: pd.DataFrame, cols_to_create_lags: list, start_lag, end_lag
) -> list:
    """
    Create a DataFrame of lag variables

    :param df: DataFrame to use
    :param cols_to_create_lags: a list of column names to create lags for
    :param start_lag: start lag (default = 1)
    :param end_lag: end lag (inclusive, default = 1)
    :return: a list of the new lag variable column names
    """

    new_cols = {}

    for lag in range(start_lag, end_lag + 1):
        for var in cols_to_create_lags:
            if lag >= 1:
                col = var + "-" + str(lag)
                new_cols[col] = df[var].shift(lag)
            elif lag <= -1:
                col = var + "+" + str(-lag)
                new_cols[col] = df[var].shift(lag)

    # Convert new columns to a new DataFrame
    new_df = pd.DataFrame(new_cols)

    return new_df

def df_train_test_split(dataset, feature_list, train_percentage = 0.8):
    '''
    Split the dataset into train and test sets
    '''
    total_rows = len(dataset)
    train_rows = int(total_rows * train_percentage)

    train = dataset.iloc[:train_rows][feature_list]
    test = dataset.iloc[train_rows:][feature_list]

    return train, test

# class MinMaxPercentileScaler(BaseEstimator, TransformerMixin):
#     """
#     Custom transformer that clips data to the defined percentiles and scales it between -1 and 1. This is important as zero is maintained before and after scaling. This ensures
#     the same number of values <, = and > zero are maintained.
#     """
#     def __init__(self,percentile=[1,99]):
#         self.max_abs_trimmed_ = None
#         self.percentile = percentile
    
#     def fit(self, X, y=None):
#         # Ensure we're working with a copy
#         X_copy = X.copy() if isinstance(X, pd.DataFrame) else np.copy(X)
        
#         # Assuming X is a DataFrame or numpy array
#         low, high = np.percentile(X_copy, self.percentile, axis=0)  # axis=0 computes percentiles column-wise
#         self.max_abs_trimmed_ = np.maximum(np.abs(low), np.abs(high))
#         return self

#     def transform(self, X):
#         # Ensure we're working with a copy
#         X_copy = X.copy() if isinstance(X, pd.DataFrame) else np.copy(X)
        
#         # Clip data column-wise
#         X_copy = np.clip(X_copy, -self.max_abs_trimmed_, self.max_abs_trimmed_)

#         # Scale values between -1 and 1 for each column, maintaining zero
#         pos_mask = X_copy > 0
#         neg_mask = X_copy < 0
        
#         X_copy[pos_mask] = X_copy[pos_mask] / self.max_abs_trimmed_
#         X_copy[neg_mask] = X_copy[neg_mask] / self.max_abs_trimmed_

#         return X_copy
    
#     def inverse_transform(self, X):
#         # Ensure we're working with a copy
#         X_copy = X.copy() if isinstance(X, pd.DataFrame) else np.copy(X)
        
#         # If it's a DataFrame, get the numpy array
#         if isinstance(X_copy, pd.DataFrame):
#             X_copy = X_copy.values  # Convert to NumPy array

#         X_copy = X_copy * self.max_abs_trimmed_

#         return X_copy

class MinMaxPercentileScaler(BaseEstimator, TransformerMixin):
    def __init__(self, percentile=[5, 95], scaling_mode='column'):
        """
        Custom transformer that clips data to the defined percentiles and scales it between -1 and 1. 
        This ensures the same number of values <, = and > zero are maintained.
        scaling_mode can be 'column' or 'global'. 
        """
        self.max_abs_trimmed_ = None
        self.percentile = percentile
        self.scaling_mode = scaling_mode
    
    def fit(self, X, y=None):
        X_copy = X.copy() if isinstance(X, pd.DataFrame) else np.copy(X)
        
        if self.scaling_mode == 'column':
            low, high = np.percentile(X_copy, self.percentile, axis=0)
            # print(low, high)
            self.max_abs_trimmed_ = np.maximum(np.abs(low), np.abs(high))
        elif self.scaling_mode == 'global':
            low, high = np.percentile(X_copy, self.percentile)
            self.max_abs_trimmed_ = np.maximum(np.abs(low), np.abs(high))
            if self.max_abs_trimmed_ == 0:
                print("Warning: Global max absolute trimmed value is zero.")
        else:
            raise ValueError("Invalid scaling_mode. Choose either 'column' or 'global'.")
        return self

    def transform(self, X):
        X_copy = X.copy() if isinstance(X, pd.DataFrame) else np.copy(X)
        
        if self.scaling_mode == 'column':
            # Clip data column-wise
            X_copy = np.clip(X_copy, -self.max_abs_trimmed_, self.max_abs_trimmed_)
            pos_mask = X_copy > 0
            neg_mask = X_copy < 0
            X_copy[pos_mask] = X_copy[pos_mask] / self.max_abs_trimmed_
            X_copy[neg_mask] = X_copy[neg_mask] / self.max_abs_trimmed_
        elif self.scaling_mode == 'global':
            # Clip data globally
            X_copy = np.clip(X_copy, -self.max_abs_trimmed_, self.max_abs_trimmed_)
            X_copy = X_copy / self.max_abs_trimmed_
        else:
            raise ValueError("Invalid scaling_mode. Choose either 'column' or 'global'.")
            
        return X_copy
    
    def inverse_transform(self, X):
        X_copy = X.copy() if isinstance(X, pd.DataFrame) else np.copy(X)

        if self.scaling_mode == 'column':
            X_copy = X_copy * self.max_abs_trimmed_
        elif self.scaling_mode == 'global':
            X_copy = X_copy * self.max_abs_trimmed_
        else:
            raise ValueError("Invalid scaling_mode. Choose either 'column' or 'global'.")
        
        return X_copy
