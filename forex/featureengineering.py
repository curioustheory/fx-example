import pandas as pd


class FeatureEngineering:

    def __init__(self, dataframe):
        self._dataframe = dataframe

    def run(self):
        self._prep_data()
        self._engineer_features()

    def _prep_data(self):
        """
        clean up data
        :return:
        """
        self._dataframe["datetime"] = pd.to_datetime(self._dataframe["date"] + " " + self._dataframe["time"],
                                                     format="%Y.%m.%d %H:%M")
        self._dataframe = self._dataframe.drop(["date", "time", "volume"], axis=1)
        # fill na

    def _engineer_features(self):
        """

        :return:
        """
        # percentage changes
        self._dataframe["percentage_change"] = self._dataframe["close"].pct_change()
        # high low spread
        self._dataframe["spread"] = self._dataframe["high"] - self._dataframe["low"]
        # rolling avg. 1 hour
        self._dataframe["rolling_mean_1h"] = self._dataframe["close"].rolling(window=60).mean()
        # rolling avg. 24 hour
        self._dataframe["rolling_mean_24h"] = self._dataframe["close"].rolling(window=60 * 24).mean()
        # rolling avg. 7 day
        self._dataframe["rolling_mean_7d"] = self._dataframe["close"].rolling(window=60 * 24 * 7).mean()

    def get_dataframe(self):
        return self._dataframe
