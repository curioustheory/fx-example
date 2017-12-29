import pandas as pd


class FeatureEngineering:

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def run(self):
        self._prep_data()
        self._clean()
        self._engineer_features()

    def _prep_data(self):
        """
        clean up data
        :return:
        """
        self.dataframe["datetime"] = pd.to_datetime(self.dataframe["date"] + " " + self.dataframe["time"],
                                                    format="%Y.%m.%d %H:%M")
        self.dataframe = self.dataframe.drop(["date", "time", "volume"], axis=1)

    def _clean(self):
        # remove invariant

        # outlier

        # fill na
        pass

    def _engineer_features(self):
        # percentage changes
        self.dataframe["percentage_change"] = self.dataframe["close"].pct_change()
        # high low spread
        self.dataframe["spread"] = self.dataframe["high"] - self.dataframe["low"]

        # avg. 1 hour

        # avg. 4 hour

        # avg. 12 hour

        # flippening

        pass

    def cache_data(self):
        pass

    def get_dataframe(self):
        return self.dataframe
