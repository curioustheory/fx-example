import matplotlib.pyplot as plt
import numpy as np


def _plot_histogram(x, x_label="", y_label="", bins=20, heading=""):
    plt.hist(x, bins)
    plt.title(heading)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axvline(x=np.median(x), color='r', linestyle='--', label="Median")
    plt.legend()


def _plot_time_series(y, legend, x=None, x_label="", y_label="", heading="", plot_median=True):
    if x is None:
        plt.plot(y)
    else:
        plt.plot(x, y)

    plt.title(heading)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    if plot_median:
        plt.axhline(y=np.median(y), color='r', linestyle='--', label="Median")
    plt.legend(legend)


def _plot_result_comparision(y1, y2, legend, y_label="", heading=""):
    plt.title(heading)
    plt.plot(y1)
    plt.plot(y2, "--")
    plt.ylabel(y_label)
    plt.legend(legend)


def _plot_feature_correlation(feature_correlation, heading):
    plt.matshow(feature_correlation)
    plt.title(heading)
    plt.xticks(range(len(feature_correlation.columns)), feature_correlation.columns)
    plt.yticks(range(len(feature_correlation.columns)), feature_correlation.columns)


def plot_eda_chart(dataframe, feature_correlation):
    x = dataframe["close"]
    _plot_histogram(x=x, x_label="Price", y_label="Count", heading="Histogram of Closing Price")
    plt.show()

    x = dataframe["datetime"]
    y = dataframe["close"]
    plt.subplot(3, 1, 1)
    _plot_time_series(x=x, y=y, y_label="Price", heading="Closing Price", legend=["Close", "Median"])

    # plot spread between high and low / volatility
    plt.subplot(3, 1, 2)
    y = dataframe["spread"]
    _plot_time_series(x=x, y=y, y_label="Spread", heading="Price Volatility", legend=["Spread", "Median"])

    # plot spread between changes
    plt.subplot(3, 1, 3)
    y = dataframe["percentage_change"]
    _plot_time_series(x=x, y=y, x_label="Datetime", y_label="Percentage", heading="Percentage Change",
                      legend=["% Change", "Median"])

    plt.subplots_adjust(left=0.2, wspace=0.8, top=0.8)
    plt.tight_layout()
    plt.show()

    _plot_feature_correlation(feature_correlation, "Feature Correlation")
    plt.show()


def plot_modelling_result(y_train, y_hat_train, y_test, y_hat_test, loss_curve, loss_train, loss_test):
    plt.subplot(4, 1, 1)
    _plot_result_comparision(y1=y_train, y2=y_hat_train, y_label="Price", heading="Training Set",
                             legend=["Actual", "Prediction"])

    plt.subplot(4, 1, 2)
    _plot_time_series(y=loss_train, y_label="Loss", heading="Training set loss (Lower the better)",
                      legend=["Loss", "Median"])

    plt.subplot(4, 1, 3)
    _plot_result_comparision(y1=y_test, y2=y_hat_test, y_label="Price", heading="Test Set",
                             legend=["Actual", "Prediction"])

    plt.subplot(4, 1, 4)
    _plot_time_series(y=loss_test, y_label="Loss", heading="Test set loss (Lower the better)",
                      legend=["Loss", "Median"])

    plt.subplots_adjust(left=0.2, wspace=0.8, top=0.8)
    plt.tight_layout()
    plt.show()

    _plot_time_series(loss_curve, y_label="Loss", x_label="Iteration", heading="Loss Curve over Iteration",
                      legend=["Loss"], plot_median=False)
    plt.show()
