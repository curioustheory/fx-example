import matplotlib.pyplot as plt


def plot_histogram(x, heading, x_label, y_label):
    plt.hist(x)
    plt.title(heading)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def plot_time_series(x, y, heading, x_label, y_label):
    plt.plot(x, y)
    plt.title(heading)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
