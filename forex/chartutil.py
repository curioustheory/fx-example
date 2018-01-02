import matplotlib.pyplot as plt
import scipy.stats as stats


def plot_histogram(x, x_label="", y_label="", bins=20, heading="", mean=0.0, std=0.0):
    normal_distribution = stats.norm.pdf(x, mean, std)
    plt.plot(x, normal_distribution, '-')
    plt.hist(x, bins)
    plt.title(heading)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axvline(x=mean, color='r', linestyle='--', label="Mean")
    plt.legend()


def plot_time_series(x, y, x_label="", y_label="", heading="", mean=0.0):
    plt.plot(x, y)
    plt.title(heading)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axhline(y=mean, color='r', linestyle='--', label="Mean")
    plt.legend()


def plot_feature_correlation(feature_correlation, heading):
    plt.matshow(feature_correlation)
    plt.title(heading)
    plt.xticks(range(len(feature_correlation.columns)), feature_correlation.columns)
    plt.yticks(range(len(feature_correlation.columns)), feature_correlation.columns)
