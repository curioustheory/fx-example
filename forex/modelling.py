import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

from forex.util import drop_highly_correlated_features


class Modelling:
    def __init__(self, dataframe, eda_summary, train_size, correlation_limit=0.95):
        self._dataframe = dataframe
        self._eda_summary = eda_summary
        self._correlation_limit = correlation_limit
        self._train_size = train_size
        self._model = None
        self._learning_curve = None

    def run(self):
        self._prep()
        self._train_model()

    def _prep(self):
        self._remove_outlier()
        self._fill_na()
        self._split_train_test_data()

    def _fill_na(self):
        self._dataframe = self._dataframe.fillna(method='backfill')

    def _remove_outlier(self):
        outlier = self._eda_summary["spread"]["75%"]
        print("remove high volatile trades / outliers where spread > {}".format(outlier))
        self._dataframe = self._dataframe[self._dataframe["spread"] <= outlier]

    def _scale_feature(self):
        pass

    def _split_train_test_data(self):
        y = self._dataframe["close"]
        # remove the label column and store in local variable i.e. we are going to predict the closing price
        dataframe = self._dataframe.drop(["close", "datetime"], axis=1)
        X = drop_highly_correlated_features(dataframe, self._correlation_limit)

        print("splitting train / test set:")
        train_size = int(len(X) * self._train_size)
        self._X_train, self._X_test = X[0:train_size], X[train_size:len(X)]
        self._y_train, self._y_test = y[0:train_size], y[train_size:len(y)]

        # reset index for plotting
        self._X_train = self._X_train.reset_index(drop=True)
        self._X_test = self._X_test.reset_index(drop=True)
        self._y_train = self._y_train.reset_index(drop=True)
        self._y_test = self._y_test.reset_index(drop=True)

        print("Total rows: {}".format(len(X)))
        print("Training rows: {}".format(len(self._X_train)))
        print("Testing rows: {}".format(len(self._X_test)))
        print()
        return self._X_train, self._y_train, self._X_test, self._y_test

    def _train_model(self):
        print("Training MLPRegressor:")
        X_train, y_train, X_test, y_test = self._split_train_test_data()
        mlp_regressor = MLPRegressor(alpha=0.001,
                                     hidden_layer_sizes=(10, 20, 10),
                                     max_iter=5000,
                                     shuffle=False,
                                     activation='logistic',
                                     verbose='True',
                                     learning_rate='adaptive')
        model = mlp_regressor.fit(X_train, y_train)
        self._model = model
        self._learning_curve = model.loss_curve_

        y_hat_train = model.predict(X_train)
        y_hat_test = model.predict(X_test)
        loss_curve = mlp_regressor.loss_curve_

        self._plot_result(y_train, y_hat_train, y_test, y_hat_test, loss_curve)

    def _plot_result(self, y_train, y_hat_train, y_test, y_hat_test, loss_curve):
        plt.subplot(3, 1, 1)
        plt.title("Training Set")
        plt.plot(y_train)
        plt.plot(y_hat_train, "--")
        plt.legend(["actual", "prediction"])

        plt.subplot(3, 1, 2)
        plt.title("Test Set")
        plt.plot(y_test)
        plt.plot(y_hat_test, "--")
        plt.legend(["actual", "prediction"])

        plt.subplot(3, 1, 3)
        plt.title("Loss Curve over Iteration")
        plt.plot(loss_curve)

        plt.subplots_adjust(left=0.2, wspace=0.8, top=0.8)
        plt.tight_layout()

        plt.show()

    def get_model(self):
        return self._model
