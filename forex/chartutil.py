import matplotlib.pyplot as plt


def plot_histogram(x, x_label="", y_label="", bins=20, heading="", mean=0.0):
    plt.hist(x, bins)
    plt.title(heading)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axvline(x=mean, color='r', linestyle='--', label="Median")
    plt.legend()


def plot_time_series(x, y, x_label="", y_label="", heading="", mean=0.0):
    plt.plot(x, y)
    plt.title(heading)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axhline(y=mean, color='r', linestyle='--', label="Median")
    plt.legend()
