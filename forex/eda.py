from forex.chartutil import plot_time_series, plot_histogram


class ExploratoryDataAnalysis:

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def run(self):
        self._summarise_data()
        self._plot_chart()

    def _summarise_data(self):
        # shape
        print("no. of rows:", self.dataframe.shape[0])
        print("no. of columns:", self.dataframe.shape[1])
        print()

        # describe data
        data_summary = self.dataframe.describe()
        print(data_summary)

    def _plot_chart(self):
        # plot time series
        x = self.dataframe["datetime"]
        y = self.dataframe.drop(["datetime"], axis=1)
        plot_time_series(x, y, "Time Series Plot", "Time", "Price")

        # plot distribution from summary
        x = self.dataframe["close"]
        plot_histogram(x, "Histogram of Closing Price", "", "")
