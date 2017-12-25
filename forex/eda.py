import pandas as pd

from forex.chartutil import plot_time_series


class ExploratoryDataAnalysis:

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def run(self):
        # shape
        print("no. of rows:", self.dataframe.shape[0])
        print("no. of columns:", self.dataframe.shape[1])
        print()

        # describe data
        data_summary = self.dataframe.describe()
        print(data_summary)

        # preprocess the date and time
        self.dataframe["datetime"] = pd.to_datetime(self.dataframe["date"] + " " + self.dataframe["time"],
                                                    format="%Y.%m.%d %H:%M")
        self.dataframe = self.dataframe.drop(["date", "time", "volume"], axis=1)
        print(self.dataframe.head())
        print()

        # gaussian check
        # TODO:
        self._is_gaussian()
        print("Normal Distribution: ")

        # plot time series
        x = self.dataframe["datetime"]
        y = self.dataframe.drop(["datetime"], axis=1)
        plot_time_series(x, y, "Time Series Plot")

        # plot distribution from summary

        # feature correlation

    def _is_gaussian(self, column_name):
        pass
