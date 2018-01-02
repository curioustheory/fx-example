import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np


def _plot_histogram(x, x_label="", y_label="", bins=20, heading="", mean=0.0, std=0.0):
    normal_distribution = stats.norm.pdf(x, mean, std)
    plt.plot(x, normal_distribution, '-')
    plt.hist(x, bins)
    plt.title(heading)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axvline(x=mean, color='r', linestyle='--', label="Mean")
    plt.legend()


def _plot_time_series(x, y, x_label="", y_label="", heading="", mean=0.0):
    plt.plot(x, y)
    plt.title(heading)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axhline(y=mean, color='r', linestyle='--', label="Mean")
    plt.legend()


def _plot_feature_correlation(feature_correlation, heading):
    plt.matshow(feature_correlation)
    plt.title(heading)
    plt.xticks(range(len(feature_correlation.columns)), feature_correlation.columns)
    plt.yticks(range(len(feature_correlation.columns)), feature_correlation.columns)


def plot_eda_chart(dataframe, data_summary, feature_correlation):
    # plot distribution from summary
    x = dataframe["close"]
    _plot_histogram(x=x,
                    x_label="Price",
                    y_label="Count",
                    heading="Histogram of Closing Price",
                    mean=data_summary["close"]["mean"],
                    std=data_summary["close"]["std"])
    plt.show()

    # plot time series
    x = dataframe["datetime"]
    # select closing price
    y = dataframe["close"]
    plt.subplot(3, 1, 1)
    _plot_time_series(x=x,
                      y=y,
                      y_label="Price",
                      heading="Closing Price",
                      mean=data_summary["close"]["mean"])

    # plot spread between high and low / volatility
    plt.subplot(3, 1, 2)
    y = dataframe["spread"]
    _plot_time_series(x=x,
                      y=y,
                      y_label="Volatility",
                      mean=data_summary["spread"]["mean"])

    # plot spread between changes
    plt.subplot(3, 1, 3)
    y = dataframe["percentage_change"]
    _plot_time_series(x=x,
                      y=y,
                      x_label="Datetime",
                      y_label="Percentage",
                      mean=data_summary["percentage_change"]["mean"])

    plt.subplots_adjust(left=0.2, wspace=0.8, top=0.8)
    plt.tight_layout()
    plt.show()

    _plot_feature_correlation(feature_correlation, "Feature Correlation")
    plt.show()


def plot_modelling_result(y_train, y_hat_train, y_test, y_hat_test, loss_curve, loss_train, loss_test):
    plt.subplot(4, 1, 1)
    plt.title("Training Set")
    plt.plot(y_train)
    plt.plot(y_hat_train, "--")
    plt.legend(["Actual", "Prediction"])

    plt.subplot(4, 1, 2)
    plt.title("Training set loss (Lower the better)")
    plt.plot(loss_train)
    plt.axhline(y=np.mean(loss_train), color='r', linestyle='--', label="Mean")
    plt.legend(["loss", "Mean"])

    plt.subplot(4, 1, 3)
    plt.title("Test Set")
    plt.plot(y_test)
    plt.plot(y_hat_test, "--")
    plt.legend(["Actual", "Prediction"])

    plt.subplot(4, 1, 4)
    plt.title("Test set loss (Lower the better)")
    plt.plot(loss_test)
    plt.axhline(y=np.mean(loss_test), color='r', linestyle='--', label="Mean")
    plt.legend(["loss", "Mean"])

    plt.subplots_adjust(left=0.2, wspace=0.8, top=0.8)
    plt.tight_layout()
    plt.show()

    plt.title("Loss Curve over Iteration")
    plt.plot(loss_curve)
    plt.show()
