import pandas as pd


class FeatureEngineering:

    def __init__(self, dataframe):
        self._dataframe = dataframe

    def run(self):
        self._prep_data()
        self._engineer_features()
        self._clean()

    def _prep_data(self):
        """
        clean up data
        :return:
        """
        self._dataframe["datetime"] = pd.to_datetime(self._dataframe["date"] + " " + self._dataframe["time"],
                                                     format="%Y.%m.%d %H:%M")
        self._dataframe = self._dataframe.drop(["date", "time", "volume"], axis=1)

    def _clean(self):
        # remove invariant

        # outlier

        # fill na
        pass

    def _engineer_features(self):
        # percentage changes
        self._dataframe["percentage_change"] = self._dataframe["close"].pct_change()
        # high low spread
        self._dataframe["spread"] = self._dataframe["high"] - self._dataframe["low"]

        # avg. 1 hour

        # avg. 4 hour

        # avg. 12 hour

    def get_dataframe(self):
        return self._dataframe
