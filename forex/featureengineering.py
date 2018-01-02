import pandas as pd


class FeatureEngineering:

    def __init__(self, dataframe):
        self._dataframe = dataframe

    def run(self):
        """
        execute the data prep and feature engineering.
        """
        self._prep_data()
        self._engineer_features()
        self._fill_na()

    def _prep_data(self):
        """
        clean up data, and preparing the data for feature engineering, set the dataframe.
        """
        # combine the date and time together and format it into a usable format
        self._dataframe["datetime"] = pd.to_datetime(self._dataframe["date"] + " " + self._dataframe["time"],
                                                     format="%Y.%m.%d %H:%M")
        # drop redundant columns
        self._dataframe = self._dataframe.drop(["date", "time", "volume"], axis=1)
        # sort the data by datetime in ascending order
        self._dataframe = self._dataframe.sort_values(by=["datetime"])

    def _engineer_features(self):
        """
        create all the feature for the model in a central place.
        """
        # shift label by one to predict the future price
        self._dataframe["close"] = self._dataframe["close"].shift(-1)
        # percentage changes
        self._dataframe["percentage_change"] = self._dataframe["close"].pct_change()
        # high low spread
        self._dataframe["spread"] = self._dataframe["high"] - self._dataframe["low"]
        # rolling avg. 1 hour
        self._dataframe["rolling_mean_1h"] = self._dataframe["close"].rolling(window=60).mean()
        # rolling avg. 24 hours
        self._dataframe["rolling_mean_24h"] = self._dataframe["close"].rolling(window=60 * 24).mean()
        # rolling avg. 7 days
        self._dataframe["rolling_mean_7d"] = self._dataframe["close"].rolling(window=60 * 24 * 7).mean()

    def _fill_na(self):
        """
        fill the NaN / NA backwards, then forward
        """
        self._dataframe = self._dataframe.fillna(method='backfill')
        self._dataframe = self._dataframe.fillna(method='ffill')

    def get_dataframe(self):
        """
        returns the pandas dataframe object.

        :return: pandas dataframe
        """
        return self._dataframe
