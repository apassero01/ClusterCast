import numpy as np
import pandas as pd
import pandas_ta as ta

import ta as technical_analysis
from datetime import datetime, timedelta
import yfinance as yf
from .SequencePreprocessing import StockSequenceSet, ScalingMethod
from .StockPatterns import OHLCVFactory, MovingAverageFactory, BandFactory, MomentumFactory

# from ClusterProcessing import ClusterGroupParams, StockClusterGroupParams
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    LabelBinarizer,
    RobustScaler,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor


class FeatureSet:
    """
    Class that encapsulates a set of features. A set of features is a subset of the columns in a Dateset object.
    This is helpful for keeping track of different requirements for different groups of features
    """

    def __init__(
        self,
        scaling_method,
        name="",
        scale_range=(-1, 1),
        ticker=None,
        percentiles=[5, 95],
    ):
        self.name = name
        self.cols = []
        self.scaling_method = scaling_method

        self.scaler = None

        self.range = scale_range
        self.ticker = ticker
        self.percentiles = percentiles

        self.sub_categories = {}

    def __repr__(self) -> str:
        return self.name


class DataSet:
    """
    Class to encapsuate a dataset. The dataset is essentially a list of dataframes of the
    same shape and size with the same features.
    """

    def __init__(self, group_params):
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

    def preprocess_pipeline(self, to_train=True):
        """
        Preprocess the dataset
        """
        print("Creating and Processing Dataset")
        if not self.created_dataset:
            self.create_dataset()
            if len(self.df) < 1:
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

        quant_min_max_feature_sets = [
            feature_set
            for feature_set in self.X_feature_sets
            if feature_set.scaling_method.value == ScalingMethod.QUANT_MINMAX.value
            or feature_set.scaling_method.value == ScalingMethod.QUANT_MINMAX_G.value
        ]
        quant_min_max_feature_sets += [
            feature_set
            for feature_set in self.y_feature_sets
            if feature_set.scaling_method.value == ScalingMethod.QUANT_MINMAX.value
            or feature_set.scaling_method.value == ScalingMethod.QUANT_MINMAX_G.value
        ]

        print("Scaling Quant Min Max Features")
        if len(quant_min_max_feature_sets) > 0:
            self.training_df, self.test_df = self.scale_quant_min_max(
                quant_min_max_feature_sets, self.training_df, self.test_df
            )
        print("Quant Min Max Features Scaled")

        standard_feature_sets = [
            feature_set
            for feature_set in self.X_feature_sets
            if feature_set.scaling_method.value == ScalingMethod.STANDARD.value
        ]
        if len(standard_feature_sets) > 0:
            self.scale_standard(standard_feature_sets, self.training_df, self.test_df)

        if to_train:
            # print("Running RandomForest Regressor to find strong predictors")
            # if not hasattr(self.group_params, 'strong_predictors'):
            #     strong_predictors = self.strong_predictors_rf()
            #     self.group_params.strong_predictors = strong_predictors

            self.group_params.strong_predictors = (
                []
            )  # TODO remove this when strong predictors are implemented

            print("RandomForest Compete")

        if len(self.group_params.X_cols) < 1:
            raise ValueError("No features created")
        if len(self.group_params.y_cols) < 1:
            raise ValueError("No target features created")
        if len(self.X_feature_sets) < 1:
            raise ValueError("No X_feature_sets created")
        if len(self.y_feature_sets) < 1:
            raise ValueError("No y_feature_sets created")

        self.group_params.X_feature_sets += self.X_feature_sets
        self.group_params.y_feature_sets += self.y_feature_sets

        if self.training_df.isna().any().any():
             # Identifying columns with NaN values
            columns_with_nans = self.training_df.columns[self.training_df.isna().any()].tolist()

            # Printing the names of columns with NaN values
            print("Columns with NaN values:", columns_with_nans)
            raise ValueError("Training set has NaN values")

        print("Dataset Preprocessing Complete")

    def create_dataset(self):
        """
        Create the dataset from the tickers and start/end dates and interval
        """
        if self.created_dataset:
            return

        self.df, cols = get_stock_data(
            self.ticker,
            self.group_params.start_date,
            self.group_params.end_date,
            self.group_params.interval,
        )

        self.group_params.X_cols.update(cols)

        self.created_dataset = True

    def create_features(self):
        """
        Create additonal featues for the dataset
        """
        if self.created_features:
            return
        
        self.factories = {
            'ohlcv_factory' : OHLCVFactory(self.df),
            'moving_average_factory' : MovingAverageFactory(self.df),
            'band_factory' : BandFactory(self.df),
            'momentum_factory' : MomentumFactory(self.df)
        }

        X_cols = set()

        # Create price features
        self.df, feature_sets = create_price_vars(
            self.df,
            self.factories,
            scaling_method=self.scaling_dict["price_vars"],
            cluster_features=self.group_params.cluster_features,
            ticker=self.ticker,
        )
        for feature_set in feature_sets:
            self.X_feature_sets.append(feature_set)
            X_cols.update(feature_set.cols)


        # Create trend features
        self.df, feature_set = create_trend_vars(
            self.df, self.factories, scaling_method=self.scaling_dict["trend_vars"], ticker=self.ticker
        )

        self.X_feature_sets.append(feature_set)
        X_cols.update(feature_set.cols)

        # Create momentum features
        self.df, feature_set = create_momentum_vars(self.df, self.factories, scaling_method=self.scaling_dict["momentum_vars"])
        self.X_feature_sets.append(feature_set)
        X_cols.update(feature_set.cols)

        # Create percent change variables
        self.df, feature_set = create_pctChg_vars(
            self.df, self.X_feature_sets, self.factories, scaling_method=self.scaling_dict["pctChg_vars"], ticker=self.ticker
        )
        self.X_feature_sets.append(feature_set)
        X_cols.update(feature_set.cols)

        pctChgFeatureSet = [feature_set for feature_set in self.X_feature_sets if feature_set.name == 'pctChg_vars'][0]
        self.df, feature_set = create_lag_vars_features(self.df, pctChgFeatureSet, lags = [3,6], ticker = self.ticker, scaling_method = self.scaling_dict['lag_feature_vars'])
        self.X_feature_sets.append(feature_set)
        X_cols.update(feature_set.cols)

        # Create rolling sum variables
        pctChgCols = pctChgFeatureSet.cols

        for col in pctChgCols:
            self.df, feature_set = create_rolling_sum_vars(
                self.df,
                col,
                rolling_sum_windows=[50],
                scaling_method=self.scaling_dict["rolling_vars"],
                ticker=self.ticker,
            )
            self.X_feature_sets.append(feature_set)
            X_cols.update(feature_set.cols)
        
        self.df, feature_set = create_rolling_sum_vars(self.df, 'pctChgclose', scaling_method=self.scaling_dict['rolling_vars'],ticker = self.ticker,rolling_sum_windows=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
        self.X_feature_sets.append(feature_set)
        X_cols.update(feature_set.cols)
        

        # Update the group params with the new  columns
        self.group_params.X_cols.update(X_cols)

        self.created_features = True

        print(self.df.columns.tolist())

    def train_test_split(self, feature_list=None, training_percentage=0.8):

        if not feature_list:
            feature_list = list(
                self.group_params.X_cols.union(self.group_params.y_cols)
            )

            self.training_df, self.test_df = df_train_test_split(
                self.df, feature_list, training_percentage
            )
            # test_df = test_df.iloc[self.n_steps:,:] # Remove the first n_steps rows to prevent data leakage
            # TODO mininmal data leakage needs to be addressed, when refactored, this class does not know the steps

    def scale_quant_min_max(self, feature_sets, training_df, test_df):
        """
        Scales the features in the feature sets in dataframe form using custom MinMaxPercentileScaler
        """

        training_df = training_df.copy()
        test_df = test_df.copy()

        # Scale the features in the feature sets
        for feature_set in feature_sets:
            if feature_set.scaling_method.value == ScalingMethod.QUANT_MINMAX.value:
                scaler = MinMaxPercentileScaler(percentile=feature_set.percentiles)
            if feature_set.scaling_method.value == ScalingMethod.QUANT_MINMAX_G.value:
                scaler = MinMaxPercentileScaler(
                    scaling_mode="global", percentile=feature_set.percentiles
                )
            scaler.fit(training_df[feature_set.cols])

            # After fitting the scaler to the combined training dataframe, transform the individual dataframes

            training_df[feature_set.cols] = scaler.transform(
                training_df[feature_set.cols]
            )
            test_df[feature_set.cols] = scaler.transform(test_df[feature_set.cols])

            feature_set.scaler = scaler

        return training_df, test_df
    
    def scale_standard(self, feature_sets, training_df, test_df):
        """
        Scales the features in the feature sets in dataframe form using custom MinMaxPercentileScaler
        """
        print("Scaling Standard")
        training_df = training_df.copy()
        test_df = test_df.copy()

        for feature_set in feature_sets:
            print(f"Name: {feature_set.name} len {len(feature_set.cols)}")
            scaler = StandardScaler()
            scaler.fit(training_df[feature_set.cols])

            # After fitting the scaler to the combined training dataframe, transform the individual dataframes

            training_df[feature_set.cols] = scaler.transform(
                training_df[feature_set.cols]
            )
            test_df[feature_set.cols] = scaler.transform(test_df[feature_set.cols])

            feature_set.scaler = scaler

    def scale_transform(self, df, feature_sets):
        """
        Scales the features in the feature sets in dataframe form using custom MinMaxPercentileScaler
        """
        df = df.copy()
        # Scale the features in the feature sets
        for feature_set in feature_sets:
            df[feature_set.cols] = feature_set.scaler.transform(df[feature_set.cols])

        return df

    def create_y_targets(self, cols_to_create_targets):
        """
        Create target y values for every column in cols_to_create_targets.

        NOTE - This is a specific implementation for stock data. The target values are gathered
        in a special way read @create_forward_rolling_sums for more info.
        """
        if self.created_y_targets:
            return
        
        rolling_features = ['sumpctChgclose_1','sumpctChgclose_2','sumpctChgclose_3','sumpctChgclose_4','sumpctChgclose_5','sumpctChgclose_6','sumpctChgclose_7','sumpctChgclose_8','sumpctChgclose_9','sumpctChgclose_10','sumpctChgclose_11','sumpctChgclose_12','sumpctChgclose_13','sumpctChgclose_14','sumpctChgclose_15']

        
        self.df, feature_set = add_forward_rolling_sums(self.df, rolling_features, scaling_method=self.scaling_dict['target_vars'],ticker=self.ticker)

        self.y_feature_sets.append(feature_set)
        self.group_params.y_cols.update(feature_set.cols)

        # print(self.df.columns.tolist())

        pctChg_target_f_set = FeatureSet(scaling_method = self.scaling_dict['target_vars'], name = "lag_target_vars", ticker = self.ticker, percentiles = [10,90])

        self.df, feature_set = create_lag_vars_target(self.df, ['pctChgclose'], start_lag = -6, end_lag = -1, ticker = self.ticker)
        pctChg_target_f_set.cols += feature_set.cols

        self.df, feature_set = create_lag_vars_target(self.df, ['pctChgclose'], start_lag = 0, end_lag = 14, ticker = self.ticker)
        pctChg_target_f_set.cols += feature_set.cols

        self.y_feature_sets.append(pctChg_target_f_set)
        self.group_params.y_cols.update(pctChg_target_f_set.cols)
        print("TARGETS IN THE FEATURE SETS \n")
        print(pctChg_target_f_set.cols)


        self.created_y_targets = True 


    def strong_predictors_rf(self, num_features = 10): 
        """
        Use a random forest regressor to determine the most important features
        """
        y_feautures = list(self.group_params.y_cols)

        X_features = self.group_params.training_features

        X_strong_features = []
        for i in range(len(y_feautures)):
            rf = RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=4)
            rf.fit(self.training_df[X_features], self.training_df[y_feautures[i]])

            # Get numerical feature importances
            importances = pd.Series(
                data=rf.feature_importances_, index=X_features
            ).sort_values(ascending=False)

            X_strong_features.append(importances[:num_features].index.tolist())

        # combine the strong features for all y and remove duplicates
        X_strong_features = list(
            set([item for sublist in X_strong_features for item in sublist])
        )
        return X_strong_features


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
        vix = yf.download(["^VIX"], start=start_date, end=end_date, interval=interval)
    else:
        df = yf.download([ticker], start=start_date, interval=interval)
        vix = yf.download(["^VIX"], start=start_date, interval=interval)

    df = df.drop(columns="Adj Close")
    df["Vix"] = vix["Close"]
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


def create_price_vars(
    df: pd.DataFrame,
    factories: dict,
    scaling_method=ScalingMethod.SBSG,
    cluster_features=None,
    ticker=None,
) -> (pd.DataFrame, FeatureSet):
    """
    Create price features from the OHLC data.
    """
    feature_sets = []
    price_feature_set = FeatureSet(scaling_method, "price_vars", (0, 1), ticker=ticker)

    ohlc_cols = ["open", "high", "low", "close"]
    price_feature_set.cols += ohlc_cols
    price_feature_set.sub_categories['ohlc'] = ohlc_cols

    ma_factory = factories['moving_average_factory']
    ma_cols = []

    sma_df = ma_factory.createSMA()   
    df = pd.concat([df, sma_df], axis=1)
    ma_cols += sma_df.columns.tolist()

    ema_df = ma_factory.createEMA()
    df = pd.concat([df, ema_df], axis=1)
    ma_cols += ema_df.columns.tolist()


    price_feature_set.sub_categories['ma'] = ma_cols
    price_feature_set.cols += ma_cols

    bb_factory = factories['band_factory'] 
    bb_cols = []

    bb_df = bb_factory.createBB()
    df = pd.concat([df, bb_df], axis=1)
    bb_cols += bb_df.columns.tolist()

    price_feature_set.cols += bb_cols
    price_feature_set.sub_categories['bb'] = bb_cols

    feature_sets.append(price_feature_set)
    
    if cluster_features:
        # We want to scale all of the price cluster features on the same scale.
        # If cluster_feaure[0] is a price_var, the rest will also be
        if cluster_features[0] in price_feature_set.cols:
            cluster_feature_set = FeatureSet(
                ScalingMethod.SBSG, "cluster_vars", (0, 1), ticker=ticker
            )
            for feature in cluster_features:
                cluster_feature_set.cols.append(feature)
                price_feature_set.cols.remove(feature)

            feature_sets.append(cluster_feature_set)

    all_cols = []
    for feature_set in feature_sets:
        all_cols += feature_set.cols

    feature_sets.append(price_feature_set)

    return df, feature_sets


def create_trend_vars(
    df: pd.DataFrame, factories: dict, scaling_method=ScalingMethod.SBS, ticker=None
) -> (pd.DataFrame, FeatureSet):
    """
    Create trend features which are made up of features that are not price features but are not percent change features.
    """
    feature_set = FeatureSet(scaling_method, "trend_vars", (0, 1), ticker=ticker)

    feature_set.cols += ["volume"]


    for col in feature_set.cols:
        df[col] = df[col].fillna(method = "bfill")

    ma_factory = factories['moving_average_factory']
    ma_cols = []

    vol_sma_df = ma_factory.createSMAVolume() 
    df = pd.concat([df, vol_sma_df], axis=1)
    ma_cols += vol_sma_df.columns.tolist()

    feature_set.cols += ma_cols
    feature_set.sub_categories['volMa'] = ma_cols

    all_cols = feature_set.cols

    return df, feature_set


def create_pctChg_vars(
    df: pd.DataFrame, X_feature_sets, factories, scaling_method = ScalingMethod.QUANT_MINMAX, start_lag = 1, end_lag = 1, ticker = None
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

    feature_set = FeatureSet(scaling_method, "pctChg_vars", ticker=ticker)

    ohlcv_factory = factories['ohlcv_factory']
    ohlcv_pct_chg = ohlcv_factory.createPctChg()
    df = pd.concat([df, ohlcv_pct_chg], axis=1)

    feature_set.cols += ohlcv_pct_chg.columns.tolist()
    feature_set.sub_categories['ohlcv'] = ohlcv_pct_chg.columns.tolist()

    # % difference price moving averages and close 

    ma_diff_cols = [] 
    ma_factory = factories['moving_average_factory']
    smaPctDiff_df = ma_factory.createSMAPctDiff()
    df = pd.concat([df, smaPctDiff_df], axis=1)
    ma_diff_cols += smaPctDiff_df.columns.tolist()

    emaPctDiff_df = ma_factory.createEMAPctDiff()
    df = pd.concat([df, emaPctDiff_df], axis=1)
    ma_diff_cols += emaPctDiff_df.columns.tolist()

    smaDeriv_df = ma_factory.createSMADerivative()
    df = pd.concat([df, smaDeriv_df], axis=1)
    ma_diff_cols += smaDeriv_df.columns.tolist()

    emaDeriv_df = ma_factory.createEMADerivative()
    df = pd.concat([df, emaDeriv_df], axis=1)
    ma_diff_cols += emaDeriv_df.columns.tolist()


    feature_set.sub_categories['ma_diff'] = ma_diff_cols
    feature_set.cols += ma_diff_cols

    percent_diff_vol_cols = []

    pct_diff_vol_df = ma_factory.createSMAPctDiffVol()
    df = pd.concat([df, pct_diff_vol_df], axis=1)
    percent_diff_vol_cols += pct_diff_vol_df.columns.tolist()

    smaDerivVol_df = ma_factory.createSMADerivativeVol()
    df = pd.concat([df, smaDerivVol_df], axis=1)
    percent_diff_vol_cols += smaDerivVol_df.columns.tolist()
    
    feature_set.sub_categories['volMa_diff'] = percent_diff_vol_cols
    feature_set.cols += percent_diff_vol_cols

    bb_diff_cols = []

    band_factory = factories['band_factory']
    bbPctDiff_df = band_factory.createBBPctDiff()
    df = pd.concat([df, bbPctDiff_df], axis=1)
    bb_diff_cols += bbPctDiff_df.columns.tolist()
        
    feature_set.sub_categories['bb_diff'] = bb_diff_cols
    feature_set.cols += bb_diff_cols

    intraday_df = ohlcv_factory.createIntraDay()
    df = pd.concat([df, intraday_df], axis=1)
    feature_set.sub_categories['intra_day'] = intraday_df.columns.tolist()
    feature_set.cols += intraday_df.columns.tolist()
    
    return df, feature_set

def create_momentum_vars(df, factory, scaling_method = ScalingMethod.STANDARD):
    """
    Create momentum variables
    """

    df = df.copy()
    feature_set = FeatureSet(scaling_method, "momentum_vars")

    momentum_factory = factory['momentum_factory']
    momentum_cols = []

    rsi_df = momentum_factory.createRSI()
    df = pd.concat([df, rsi_df], axis=1)
    momentum_cols += rsi_df.columns.tolist()

    macd_df = momentum_factory.createMACD()
    df = pd.concat([df, macd_df], axis=1)
    momentum_cols += macd_df.columns.tolist()

    stoch_df = momentum_factory.createStoch()

    print(f"first date stoch: {stoch_df.index[0]}")
    print(f"last date stoch: {stoch_df.index[-1]}")
    print(f"first date df: {df.index[0]}")
    print(f"last date df: {df.index[-1]}")
    df = pd.concat([df, stoch_df], axis=1)
    momentum_cols += stoch_df.columns.tolist()

    feature_set.cols += momentum_cols

    return df, feature_set



def create_rolling_sum_vars(
    df: pd.DataFrame,
    col_name: str,
    rolling_sum_windows=(1, 2, 3, 4, 5, 6),
    scaling_method=ScalingMethod.QUANT_MINMAX_G,
    ticker=None,
) -> (pd.DataFrame, FeatureSet):
    """
    Create rolling sum variables for the specified column.

    :param df: A data frame containing all features including the column to create rolling sums for
    :param col_name: The column name to create rolling sums for
    :param rolling_sum_windows: A list of summed window sizes to create

    :return: The dataframe with the new columns added
    """

    if col_name not in df.columns:
        raise ValueError("Column name not in dataframe")

    df = df.copy()

    feature_set = FeatureSet(scaling_method, col_name + "_rolling", ticker=ticker)

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


def add_forward_rolling_sums(
    df: pd.DataFrame,
    columns: list,
    scaling_method=ScalingMethod.QUANT_MINMAX,
    ticker=None,
) -> (pd.DataFrame, FeatureSet):
    """
    Add the y val for sumPctChgClCl_X for the next X periods. For example sumPctChgClCl_1
    is the percent change from today's close to tomorrow's close, sumPctChgClCl_2 is the percent
    change from today's close to the close 2 days from now, etc. If we want to predict returns
    two days from now the new column sumPctChgClCl+2 would be the training target value.

    :param df: DataFrame to use
    :param columns: a list of column names
    :return: the DataFrame with the new columns added
    """
    feature_set = FeatureSet(scaling_method, "target_vars_cumulative", ticker=ticker)

    max_shift = -1

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

    return df, feature_set


def create_lag_vars_target(
    df: pd.DataFrame, cols_to_create_lags: list, start_lag, end_lag, ticker = None, scaling_method = ScalingMethod.QUANT_MINMAX
) -> list:
    """
    Create a DataFrame of lag variables

    :param df: DataFrame to use
    :param cols_to_create_lags: a list of column names to create lags for
    :param start_lag: start lag (default = 1)
    :param end_lag: end lag (inclusive, default = 1)
    :return: a list of the new lag variable column names
    """
    df = df.copy()

    feature_set = FeatureSet(scaling_method=scaling_method, name = "lag_target_vars", ticker = ticker)

    df_lags = pd.DataFrame(index=df.index)
    for lag in range(start_lag, end_lag + 1):
        for var in cols_to_create_lags:
            if lag >= 0:
                df_lags[var + "-" + str(lag) + "_target"] = pd.Series(df[var].shift(lag),
                                                        index=df.index[(lag):])
                # replace nans with mean in this col
                df_lags[var + "-" + str(lag)+ "_target"] = df_lags[var + "-" + str(lag)+ "_target"].fillna(df_lags[var + "-" + str(lag)+ "_target"].mean())
            elif lag <= -1:
                df_lags[var + "+" + str(lag * -1)+ "_target"] = pd.Series(df[var].shift(lag),
                                                            index=df.index[:(lag)])

    # Reverse the columns if shifting ahead
    if start_lag < 0:
        x = list(df_lags.columns)
        x.reverse()
        df_lags = df_lags[x]

    # concat the new DataFrame to the original DataFrame
    df = pd.concat([df, df_lags], axis=1)

    feature_set.cols = df_lags.columns.tolist()

    return df, feature_set


def create_lag_vars_features(
    df: pd.DataFrame, feature_set: list, lags, ticker = None, scaling_method = ScalingMethod.QUANT_MINMAX
) -> list:
    """
    Create a DataFrame of lag variables

    :param df: DataFrame to use
    :param cols_to_create_lags: a list of column names to create lags for
    :param start_lag: start lag (default = 1)
    :param end_lag: end lag (inclusive, default = 1)
    :return: a list of the new lag variable column names
    """
    df = df.copy()

    lag_feature_set = FeatureSet(scaling_method=scaling_method, name = "lag_features_vars", ticker = ticker)

    df_lags = pd.DataFrame(index=df.index)

    for lag in lags:

        for sub_category in feature_set.sub_categories:
            cols_to_create_lags = feature_set.sub_categories[sub_category]
            new_lag_cols = {}

            for var in cols_to_create_lags:
                if lag >= 0:
                    new_col_name = var + "-" + str(lag)
                    new_lag_cols[new_col_name] = pd.Series(df[var].shift(lag),
                                                            index=df.index[(lag):])
                    # replace nans with mean in this col
                    new_lag_cols[new_col_name] = new_lag_cols[new_col_name]

            lag_feature_set.sub_categories[sub_category] = list(new_lag_cols.keys())
            lag_feature_set.cols += list(new_lag_cols.keys())

            df_lags = pd.concat([df_lags, pd.DataFrame(new_lag_cols)], axis=1)
            df_lags = df_lags.fillna(df_lags.mean())

    # concat the new DataFrame to the original DataFrame
    df = pd.concat([df, df_lags], axis=1)

    return df, lag_feature_set

def df_train_test_split(dataset, feature_list, train_percentage = 0.8):
    '''
    Split the dataset into train and test sets
    '''
    total_rows = len(dataset)
    train_rows = int(total_rows * train_percentage)

    train = dataset.iloc[:train_rows][feature_list]
    test = dataset.iloc[train_rows:][feature_list]

    return train, test



class MinMaxPercentileScaler(BaseEstimator, TransformerMixin):
    def __init__(self, percentile=[5, 95], scaling_mode="column"):
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

        if self.scaling_mode == "column":
            low, high = np.percentile(X_copy, self.percentile, axis=0)
            self.max_abs_trimmed_ = np.maximum(np.abs(low), np.abs(high))
        elif self.scaling_mode == "global":
            low, high = np.percentile(X_copy, self.percentile)
            self.max_abs_trimmed_ = np.maximum(np.abs(low), np.abs(high))
            if self.max_abs_trimmed_ == 0:
                print("Warning: Global max absolute trimmed value is zero.")
        else:
            raise ValueError(
                "Invalid scaling_mode. Choose either 'column' or 'global'."
            )
        return self

    def transform(self, X):
        X_copy = X.copy() if isinstance(X, pd.DataFrame) else np.copy(X)

        if self.scaling_mode == "column":
            # Clip data column-wise
            X_copy = np.clip(X_copy, -self.max_abs_trimmed_, self.max_abs_trimmed_)
            pos_mask = X_copy > 0
            neg_mask = X_copy < 0
            X_copy[pos_mask] = X_copy[pos_mask] / self.max_abs_trimmed_
            X_copy[neg_mask] = X_copy[neg_mask] / self.max_abs_trimmed_
        elif self.scaling_mode == "global":
            # Clip data globally
            X_copy = np.clip(X_copy, -self.max_abs_trimmed_, self.max_abs_trimmed_)
            X_copy = X_copy / self.max_abs_trimmed_
        else:
            raise ValueError(
                "Invalid scaling_mode. Choose either 'column' or 'global'."
            )

        return X_copy

    def inverse_transform(self, X):
        X_copy = X.copy() if isinstance(X, pd.DataFrame) else np.copy(X)

        if self.scaling_mode == "column":
            X_copy = X_copy * self.max_abs_trimmed_
        elif self.scaling_mode == "global":
            X_copy = X_copy * self.max_abs_trimmed_
        else:
            raise ValueError(
                "Invalid scaling_mode. Choose either 'column' or 'global'."
            )

        return X_copy
