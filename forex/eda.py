from forex.chartutil import plot_time_series, plot_histogram
import matplotlib.pyplot as plt


class ExploratoryDataAnalysis:

    def __init__(self, dataframe):
        self._dataframe = dataframe
        self._data_summary = None
        self._feature_correlation = None

    def run(self, plot_chart=True):
        self._summarise_data()
        self._check_feature_correlation()
        if plot_chart:
            self._plot_chart()

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

        plt.matshow(self._feature_correlation)
        plt.xticks(range(len(self._feature_correlation.columns)), self._feature_correlation.columns)
        plt.yticks(range(len(self._feature_correlation.columns)), self._feature_correlation.columns)
        plt.show()

    def _summarise_data(self):
        self._data_summary = self._dataframe.describe()

    def _check_feature_correlation(self):
        self._feature_correlation = self._dataframe.corr()

    def get_summary(self):
        return self._data_summary

    def get_feature_correlation(self):
        return self._feature_correlation
