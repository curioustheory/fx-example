import pandas as pd


class FeatureEngineering:

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def prep_data(self):
        self.dataframe["datetime"] = pd.to_datetime(self.dataframe["date"] + " " + self.dataframe["time"],
                                                    format="%Y.%m.%d %H:%M")
        self.dataframe = self.dataframe.drop(["date", "time", "volume"], axis=1)

    def fill_na(self):
        # remove invariant

        # outlier

        # fill na
        pass

    def engineer_features(self):
        # high low spread

        # avg. 1 hour

        # avg. 4 hour

        # avg. 12 hour

        # flippening

        pass

    def cache_data(self):
        pass

    def get_dataframe(self):
        return self.dataframe
