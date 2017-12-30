from forex.chartutil import plot_time_series, plot_histogram
import matplotlib.pyplot as plt


class ExploratoryDataAnalysis:

    def __init__(self, dataframe):
        self._dataframe = dataframe
        self._data_summary = None

    def run(self):
        self._summarise_data()
        self._plot_chart()
        self._check_feature_correlation()

    def _summarise_data(self):
        # shape
        print("no. of rows:", self._dataframe.shape[0])
        print("no. of columns:", self._dataframe.shape[1])
        print()

        # describe data
        self._data_summary = self._dataframe.describe()
        print(self._data_summary)
        print()

    def _plot_chart(self):
        # plot distribution from summary
        x = self._dataframe["close"]
        plot_histogram(x=x,
                       x_label="Price",
                       y_label="Count",
                       bins=50,
                       heading="Histogram of Closing Price",
                       mean=self._data_summary["close"]["mean"],
                       std=self._data_summary["close"]["std"])
        plt.show()

        # plot time series
        x = self._dataframe["datetime"]
        # select closing price
        y = self._dataframe["close"]
        plt.subplot(3, 1, 1)
        plot_time_series(x=x,
                         y=y,
                         y_label="Price",
                         heading="Closing Price",
                         mean=self._data_summary["close"]["mean"])

        # plot spread between high and low / volatility
        plt.subplot(3, 1, 2)
        y = self._dataframe["spread"]
        plot_time_series(x=x,
                         y=y,
                         y_label="Volatility",
                         mean=self._data_summary["spread"]["mean"])

        # plot spread between changes
        plt.subplot(3, 1, 3)
        y = self._dataframe["percentage_change"]
        plot_time_series(x=x,
                         y=y,
                         x_label="Datetime",
                         y_label="Percentage",
                         mean=self._data_summary["percentage_change"]["mean"])

        plt.subplots_adjust(left=0.2, wspace=0.8, top=0.8)
        plt.show()

    def _check_feature_correlation(self):
        corr = self._dataframe.corr()
        print("feature correlation...")
        print(corr)
        print()

        plt.matshow(corr)
        plt.xticks(range(len(corr.columns)), corr.columns)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.show()
