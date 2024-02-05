from django.test import TestCase
from ClusterPipeline.models import SequencePreprocessing as SP
from ClusterPipeline.models import ClusterProcessing as CP
from ClusterPipeline.models import TSeriesPreproccesing as TSPP
import pandas as pd
import random


class TestStockDataSet(TestCase):

    def setUp(self):
        """
        This method is called before each test
        """
        self.tickers = "AAPL"  # one ticker, string
        self.start_date = "2010-01-01"
        self.target_cols = [
            "sumPctChgclose_1",
            "sumPctChgclose_2",
            "sumPctChgclose_3",
            "sumPctChgclose_4",
            "sumPctChgclose_5",
            "sumPctChgclose_6",
        ]
        self.n_steps = 20
        self.interval = "1d"
        scaling_dict = {
            "price_vars": SP.ScalingMethod.UNSCALED,
            "trend_vars": SP.ScalingMethod.UNSCALED,
            "pctChg_vars": SP.ScalingMethod.QUANT_MINMAX,
            "rolling_vars": SP.ScalingMethod.QUANT_MINMAX,
            "target_vars": SP.ScalingMethod.QUANT_MINMAX,
        }
        group_params = CP.StockClusterGroupParams(
            start_date=self.start_date,
            tickers=self.tickers,
            interval=self.interval,
            target_cols=self.target_cols,
            n_steps=self.n_steps,
        )
        group_params.initialize()
        group_params.set_scaling_dict(scaling_dict)
        self.stockDataSet = TSPP.StockDataSet(group_params, self.tickers)

        # Initialize self.df
        self.df = self.stockDataSet.df

    ## Non Class Methods
    def test_create_price_vars(self):
        """
        Test that the non class function create_price_vars works as expected
        This function is outside of the class and takes a dataframe as an input and moving averages values with default values of 5, 10, 20, 50, 100]

        Tests the following:
            - The dataset contains the following columns:
                - 'open', 'high', 'low', 'close' sma5, sma10, ... ema5, ema10....
                - There are no nan values in the dataset
                - A non empty feature set is returned and the cols in the feature set are all in the dataframe

        """
        self.stockDataSet.create_dataset()
        df = self.stockDataSet.df  # .df , update each
        df, feature_set = TSPP.create_price_vars(
            df
        )  # Calling the create_price_vars function that takes a df and returns the updated df with the new feature set

        # Test that X_feature_dict contains the correct key-value pairs
        expected_keys = [
            "open",
            "high",
            "low",
            "close",
            "sma5",
            "sma10",
            "sma20",
            "sma50",
            "sma100",
            "ema5",
            "ema10",
            "ema20",
            "ema50",
            "ema100",
        ]

        # check to see if expected_keys is in df
        for key in expected_keys:
            self.assertIn(key, df.columns)
            self.assertIn(key, feature_set[0].cols)

    def test_create_trend_vars(self):
        """
        Test that the non class function create_trend_vars works as expected
        This function is outside of the class and takes a dataframe as an input

        Tests the following:
            - The dataset contains the volume as a new column
            - There are no nan values in the dataset
            - A non empty feature set is returned and the cols in the feature set are all in the dataframe

        """
        self.stockDataSet.create_dataset()
        df = self.stockDataSet.df
        df, feature_set = TSPP.create_trend_vars(df)

        # Test that new columns are added
        expected_columns = ["volume"]
        for col in expected_columns:
            self.assertIn(col, df.columns)

        # Test that there are no nan values in the dataset
        self.assertFalse(df.isnull().values.any())

        # Test that the feature set is not empty and contains the correct columns
        self.assertFalse(
            len(feature_set.cols) == 0
        )  # Feature set is a class we have defined so it does not have an empty attribute.
        self.assertEqual(feature_set.cols, expected_columns)

        # check to see expected columns "volume" is in feature_set.cols

    def test_create_pctChg_vars(self):
        """
        Test that the non class function create_pctChg_vars works as expected
        This function is outside of the class and takes a dataframe as an input

        Tests the following:
            - The dataset contains new columns for all new percent change variables (don't have to test for each specific column but make sure that the columns are there)
            - Test a few randomly selected columns ie. pctChgclose etc and make sure that the values are correct by extracting the actual values from the dataframe
                - take a random day and calculate the pctChgclose manually and compare it to the value in the dataframe. close on 5/10/21 - close on 5/9/21 / close on 5/9/21 * 100 == pctChgclose on 5/10/21
                - Do this for a few random days and columns including the single day pct chg vars such as opHi (pct change from open to high) etc
            - More Difficult: Confirm rolling sum variables are correct. This function will return sumPctChgclose_1 -> sumPctChgclose_6 which consists of the sum of pctChgClose from today to t-1, t-2, t-3, t-4, t-5, t-6
            respectively. sumPctChgclose_1 is the same as pctChgClose and sumPctChgclose_6 is the sum of pctChgClose from today to 6 days ago.

            - There are no nan values in the dataset
            - A non empty feature set is returned and the cols in the feature set are all in the dataframe
        """
        self.stockDataSet.create_dataset()
        df = self.stockDataSet.df
        df, feature_set = TSPP.create_price_vars(df)
        df, feature_set = TSPP.create_pctChg_vars(df)

        # Test that new columns are added
        expected_columns = [
            "pctChgclose",
            "pctChgopen",
            "pctChghigh",
            "pctChglow",
            "pctChgvolume",
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns)

        # Test that there are no nan values in the dataset
        self.assertFalse(df[expected_columns].isnull().values.any())
        for col in expected_columns:
            self.assertIn(col, feature_set.cols)

        # Test that the pctChgclose values are correct, select random days and calculate the pctChgclose manually and compare it to the value in the dataframe
        test_dates = ["2010-01-04", "2010-01-05", "2010-01-06", "2010-01-07"]
        if all(date in df.index for date in test_dates):
            for date in test_dates:
                self.assertAlmostEqual(
                    df.loc[date, "pctChgclose"],
                    (
                        df.loc[date, "close"]
                        - df.loc[date - pd.Timedelta(days=1), "close"]
                    )
                    / df.loc[date - pd.Timedelta(days=1), "close"]
                    * 100,
                )

        calculated_pctChgclose = (
            (df.loc["2021-05-10", "close"] - df.loc["2021-05-09", "close"])
            / df.loc["2021-05-09", "close"]
            * 100
        )
        self.assertAlmostEqual(
            df.loc["2021-05-10", "pctChgclose"], calculated_pctChgclose
        )

        # TODO right idea need to debug this the tests do not run without errors. The date you provide is a string and you try to subtract a timedelta from it.
        # the date is a string and cannot do pd.Timedelta(days=1) on it. Also look up panda dataframe operations to do this in a much easier way

    def test_add_forward_rolling_sums(self):
        """
        Test that the non class function add_forward_rolling_sums works as expected
        This function is outside of the class and takes a dataframe as an input

        Tests the following:
            - The dataset contains new columns for all new forward rolling sum variables
            - This function takes the sumPctChgClose_X variables and creates columns with the future sum of pctChgClose from t+1 to t+X days. (used as a target variable)
            - Difficult : Confirm these values are correct by manually calculating sumPctChgClose+X for a few random days and comparing them to the values in the dataframe.
                - if today is t, sumPctChgClose+1 is sumPctChgClose_1 for t+1, sumPctChgClose+2 is sumPctChgClose_2 for t+2, etc. (diagnol of the dataframe)
            - There are no nan values in the dataset
            - A non empty feature set is returned and the cols in the feature set are all in the dataframe
        """
        expected_columns = [
            "sumpctChgclose_1",
            "sumpctChgclose_2",
            "sumpctChgclose_3",
            "sumpctChgclose_4",
            "sumpctChgclose_5",
            "sumpctChgclose_6",
        ]

        self.stockDataSet.create_dataset()
        self.stockDataSet.create_features()

        df = self.stockDataSet.df
        df, feature_set = TSPP.add_forward_rolling_sums(df, expected_columns)
        print(df.columns)

        print(
            df[
                [
                    "pctChgclose", "sumpctChgclose_1",
                    "sumpctChgclose_2",
                    "sumpctChgclose_3",
                    "sumpctChgclose_4",
                    "sumpctChgclose_5",
                    "sumpctChgclose_6",
                ]
            ].head(20)
        )

        # Test that new columns are added
        for col in expected_columns:
            self.assertIn(col, df.columns)

        # Test for nan values
        self.assertFalse(df[expected_columns].isnull().values.any())

        # Test that feature set is not empty and contains the correct columns
        self.assertFalse(feature_set.empty)
        self.assertEqual(feature_set, expected_columns)

        # Test that the sumPctChgclose+X values are correct, select random days and calculate the sumPctChgclose+X manually and compare it to the value in the dataframe
        test_dates = ["2010-01-04", "2010-01-05", "2010-01-06", "2010-01-07"]
        if all(date in df.index for date in test_dates):
            for date in test_dates:
                self.assertAlmostEqual(
                    df.loc[date, "sumPctChgclose_1"],
                    (
                        df.loc[date + pd.Timedelta(days=1), "pctChgclose"]
                        + df.loc[date + pd.Timedelta(days=2), "pctChgclose"]
                        + df.loc[date + pd.Timedelta(days=3), "pctChgclose"]
                        + df.loc[date + pd.Timedelta(days=4), "pctChgclose"]
                        + df.loc[date + pd.Timedelta(days=5), "pctChgclose"]
                        + df.loc[date + pd.Timedelta(days=6), "pctChgclose"]
                    ),
                )
                self.assertAlmostEqual(
                    df.loc[date, "sumPctChgclose_2"],
                    (
                        df.loc[date + pd.Timedelta(days=2), "pctChgclose"]
                        + df.loc[date + pd.Timedelta(days=3), "pctChgclose"]
                        + df.loc[date + pd.Timedelta(days=4), "pctChgclose"]
                        + df.loc[date + pd.Timedelta(days=5), "pctChgclose"]
                        + df.loc[date + pd.Timedelta(days=6), "pctChgclose"]
                    ),
                )
                self.assertAlmostEqual(
                    df.loc[date, "sumPctChgclose_3"],
                    (
                        df.loc[date + pd.Timedelta(days=3), "pctChgclose"]
                        + df.loc[date + pd.Timedelta(days=4), "pctChgclose"]
                        + df.loc[date + pd.Timedelta(days=5), "pctChgclose"]
                        + df.loc[date + pd.Timedelta(days=6), "pctChgclose"]
                    ),
                )
                self.assertAlmostEqual(
                    df.loc[date, "sumPctChgclose_4"],
                    (
                        df.loc[date + pd.Timedelta(days=4), "pctChgclose"]
                        + df.loc[date + pd.Timedelta(days=5), "pctChgclose"]
                        + df.loc[date + pd.Timedelta(days=6), "pctChgclose"]
                    ),
                )
                self.assertAlmostEqual(
                    df.loc[date, "sumPctChgclose_5"],
                    (
                        df.loc[date + pd.Timedelta(days=5), "pctChgclose"]
                        + df.loc[date + pd.Timedelta(days=6), "pctChgclose"]
                    ),
                )

    # TODO same problem as the previous test. The date you provide is a string and you try to subtract a timedelta from it. Also look up panda dataframe
    # operations to do this in a much easier way
    # Same deal with date string to. pandas has a built in function to do this.

    ## Class Methods
    def test_create_stock_dataset(self):
        """
        Test that the stock dataset is created using stockDataSet.create_dataset() method
        Tests the following:
            - The dataset is not None
            - The length of stockDataSet.tickers is equal to the length of self.tickers
            - The length of stockDataSet.dfs = len(self.tickers)
            - The length of stockDataSet.dfs[i] = len(self.tickers[i]) for all i in range(len(self.tickers))
            - stockDataSet.cols = ['open', 'high', 'low', 'close', 'volume']
        """
        # create dataset
        self.stockDataSet.create_dataset()

        # Test that the dataset is not None
        self.assertTrue(self.stockDataSet.dfs)

        # Test that the length of stockDataSet.tickers is equal to the length of self.tickers
        self.assertEqual(len(self.stockDataSet.group_params.tickers), len(self.tickers))

        # Test each datafram is not empty and contains the correct columns
        expected_columns = ["open", "high", "low", "close", "volume"]
        for df in self.stockDataSet.dfs:
            self.assertFalse(df.empty)
            for col in expected_columns:
                self.assertIn(col, df.columns)

    def test_create_features(self):
        """
        Test that the feature set is created using stockDataSet.create_features() method.
        This method calls all of the previously tested non class methods. Run tests to confirm it is done correctly

        Tests the following:
            - The dataset is not None
            - This method calls create_price_vars, create_trend_vars, create_pctChg_vars functions ensure that the individual outputs from these functions are reflected in the dataset object after this method is called
                - ie. stockDataSet.dfs[0] contains all the new columns created by these functions
                - ie. stockDataSet.X_feature_sets contains all feature sets we expected to see earlier returned from non class functions
                - if we are running tests with multiple tickers, is df[0] the same as df[1] etc. (has all same columns, dates, num_rows, for each different stock)
            - X_cols contains all of the column names that we expected from individual functions
            - There are no nan values in the dataframe
        """
        self.assertIsNotNone(self.stockDataSet)
        self.stockDataSet.create_dataset()

        # Pull out one of the data frames as a copy to test passing into non class functions is the same as calling create_features method
        df_test = self.stockDataSet.df.copy()

        self.stockDataSet.create_features()

        # Test that the individual functions are called and the outputs are reflected in the dataset object
        df_test, price_feature_set = TSPP.create_price_vars(df_test)
        df_test, trend_feature_set = TSPP.create_trend_vars(df_test)
        df_test, pctChg_feature_set = TSPP.create_pctChg_vars(df_test)

        # Test that the feature sets are the same as the ones returned from the individual functions
        self.assertEqual(
            self.stockDataSet.X_feature_sets[0],
            price_feature_set + trend_feature_set + pctChg_feature_set,
        )

        # TODO
        # 1. Cant add feature sets together that is an object we made not a list. to test the feature sets are correct
        # I would iterate over all of the cols in each feature set and make sure that they are in the first data frame
        # 2. Check that the columns in each dataframe are the same.
        # 3 I have since added a another method where we create "rolling_sum" features in the create_features() method so we need to test for those

    def test_train_test_split(self):
        """
        Tests that the train test split function works as expected

        Tests the following:
            - train and test sets are not empty
            - The last date in the training set happens before the first day in the test set
        """
        # create dataset and features
        self.stockDataSet.create_dataset()
        self.stockDataSet.create_features()

        # split data
        self.stockDataSet.train_test_split()
        training_percent = 0.8

        # Test that the train and test sets are not empty and are split correctly
        self.assertFalse(self.stockDataSet.training_dfs[0].empty)
        self.assertFalse(self.stockDataSet.test_dfs[0].empty)
        self.assertEqual(
            self.stockDataSet.training_dfs[0].index[-1] + pd.Timedelta(days=1),
            self.stockDataSet.test_dfs[0].index[0],
        )
        self.assertEqual(
            len(self.stockDataSet.training_dfs[0]),
            int(len(self.stockDataSet.dfs[0]) * training_percent),
        )
        self.assertEqual(
            len(self.stockDataSet.test_dfs[0]),
            len(self.stockDataSet.dfs[0])
            - int(len(self.stockDataSet.dfs[0]) * training_percent),
        )

        # TODO  Dont need to check to see if the next date after training is the first date in test because it might not be (weekends) Just make sure the first date in test is after the last date in train
        # Also run one more test to make sure the length of the training and test is the same for all 3 tickers

    def scale_quant_min_max(self):
        """
        Tests that the scale_quant_min_max function works as expected
        This function takes a training set df and a test set df. Save a df
        before calling this function and after calling this function and compare the values

        Tests the following:
            - All features with featureSet.scalingMethod == QUANT_MINMAX are scaled between -1 and 1
            - For every feature, the number of instances > 0 and < 0 are the same before and after calling the method
            - The scaler is saved in feature_set.scaler and inverse transforming with the scaler returns original values (close to)
                - test this for a few random rows in the dataframe
        """
        # create dataset and features and split data
        self.stockDataSet.create_dataset()
        self.stockDataSet.create_features()
        self.stockDataSet.train_test_split()

        # Identify features that will be scaled
        quant_minmax_features = []
        for feature in self.stockDataSet.X_feature_sets[
            0
        ]:  # TODO look at how to access cols in the feature set
            if feature.scalingMethod == "QUANT_MINMAX":
                quant_minmax_features.append(feature)

        # Save a copy of the training and test sets before scaling
        training_df = self.stockDataSet.training_dfs[0].copy()
        test_df = self.stockDataSet.test_dfs[0].copy()

        # Scale the data
        self.stockDataSet.scale_quant_min_max()

        # Test scaling and inverse scaling
        for feature in quant_minmax_features:
            # Test that all values are scaled between -1 and 1
            self.assertTrue(self.stockDataSet.training_dfs[0][feature].max() <= 1)
            self.assertTrue(self.stockDataSet.training_dfs[0][feature].min() >= -1)
            self.assertTrue(self.stockDataSet.test_dfs[0][feature].max() <= 1)
            self.assertTrue(self.stockDataSet.test_dfs[0][feature].min() >= -1)

            # Test that the number of instances > 0 and < 0 are the same before and after scaling
            self.assertEqual(
                len(training_df[training_df[feature] > 0]),
                len(
                    self.stockDataSet.training_dfs[0][
                        self.stockDataSet.training_dfs[0][feature] > 0
                    ]
                ),
            )
            self.assertEqual(
                len(training_df[training_df[feature] < 0]),
                len(
                    self.stockDataSet.training_dfs[0][
                        self.stockDataSet.training_dfs[0][feature] < 0
                    ]
                ),
            )
            self.assertEqual(
                len(test_df[test_df[feature] > 0]),
                len(
                    self.stockDataSet.test_dfs[0][
                        self.stockDataSet.test_dfs[0][feature] > 0
                    ]
                ),
            )
            self.assertEqual(
                len(test_df[test_df[feature] < 0]),
                len(
                    self.stockDataSet.test_dfs[0][
                        self.stockDataSet.test_dfs[0][feature] < 0
                    ]
                ),
            )

            # Test that the scaler is saved in feature_set.scaler and inverse transforming with the scaler returns original values (close to)
            # test this for a few random rows in the dataframe
            for i in range(5):
                # TODO have to actually inverse transform. Unscale the data with the inverse transform and compare it to the original values
                random_row = random.randint(0, len(training_df))
                self.assertAlmostEqual(
                    training_df.loc[random_row, feature],
                    self.stockDataSet.training_dfs[0].loc[random_row, feature],
                )
                random_row = random.randint(0, len(test_df))
                self.assertAlmostEqual(
                    test_df.loc[random_row, feature],
                    self.stockDataSet.test_dfs[0].loc[random_row, feature],
                )

    def test_create_target(self):
        """
        Test that the target set is created using stockDataSet.create_target() method. This method calls the add_forward_rolling_sum function
        tested before. Run similar tests to confirm it is done correctly and the dataframe contains the target columns we expect

        Tests the following:
            - The dataset is not None
            - This method calls add_forward_rolling_sum function ensure that the individual outputs from this function are reflected in the dataset object after this method is called
                - ie. stockDataSet.dfs[0] contains all the new columns created by this function
                - ie. stockDataSet.y_feature_sets contains all feature sets we expected to see earlier returned from non class functions
                - if we are running tests with multiple tickers, is df[0] the same as df[1] etc. (has all same columns, dates, num_rows, for each different stock)
            - y_cols contains all of the column names that we expected from individual functions
            - There are no nan values in the dataframe
            - stockDataSet.y_cols contains the columns we expect. ie. sumPctChgClose+1, sumPctChgClose+2, etc.
        """

        self.assertIsNotNone(self.stockDataSet)

        self.stockDataSet.create_dataset()
        self.stockDataSet.create_features()

        # Pull out one of the data frames as a copy to test passing into non class functions is the same as calling create_features method
        df_test = self.stockDataSet.dfs[0].copy()
        df_test = TSPP.add_forward_rolling_sums(df_test)

        # mentually generate the columns we will be using to generate target columns
        # TODO this needs to be changed target cols end up being added somewhere else we can extract
        target_cols = [
            col
            for col in self.stockDataSet.df.columns
            if "sumPctChgclose_" in col and "-" not in col
        ]

        self.stockDataSet.create_y_targets(target_cols)
        # TODO then we need to confirm that the target cols are correct

    def test_preprocess_pipeline(self):
        """
        Test that the preprocess pipeline works as expected. This function is responsible for
        running through all of the steps required to create our dataset by calling the methods
        we tested above.


        Tests the following:
            - training_dfs and test_dfs are not empty
            - group_param variable contains the correct X_col, y_col values
            - X_feature_sets and y_feature_sets contain the correct values (again in group_param variable)
            - training_dfs and test_dfs all contain the correct columns as seen in X_col, y_col
        """
        # run the preprocess pipeline
        self.stockDataSet.preprocess_pipeline()

        # check that the training and test sets are not empty
        for training_df in self.stockDataSet.training_dfs:
            self.assertFalse(training_df.empty)
        for test_df in self.stockDataSet.test_dfs:
            self.assertFalse(test_df.empty)

        # Check that the group_param variable contains the correct X_col, y_col values

        # TODO cols in the feature set is an object we made not a list.
        self.assertEqual(
            self.stockDataSet.group_params.X_cols, self.stockDataSet.X_feature_sets
        )
        self.assertEqual(
            self.stockDataSet.group_params.y_cols, self.stockDataSet.y_feature_sets
        )

        # Check that the training and test sets all contain the correct columns as seen in X_col, y_col
        for training_df in self.stockDataSet.training_dfs:
            for col in self.stockDataSet.X_feature_sets:
                self.assertIn(col, training_df.columns)
            for col in self.stockDataSet.y_feature_sets:
                self.assertIn(col, training_df.columns)
