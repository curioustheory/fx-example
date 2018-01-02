import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

from forex.util import drop_highly_correlated_features
import numpy as np


class Modelling:
    def __init__(self, dataframe, eda_summary, train_size, correlation_limit=0.95):
        self._dataframe = dataframe
        self._eda_summary = eda_summary
        self._correlation_limit = correlation_limit
        self._train_size = train_size
        self._model = None
        self._learning_curve = None

    def run(self,
            alpha,
            hidden_layer_sizes,
            max_iter,
            shuffle,
            activation,
            verbose,
            learning_rate):
        self._prep()
        self._train_model(alpha,
                          hidden_layer_sizes,
                          max_iter,
                          shuffle,
                          activation,
                          verbose,
                          learning_rate)

    def _prep(self):
        self._remove_outlier()
        self._fill_na()
        self._split_train_test_data()

    def _fill_na(self):
        self._dataframe = self._dataframe.fillna(method='backfill')
        self._dataframe = self._dataframe.fillna(method='ffill')

    def _remove_outlier(self):
        outlier = self._eda_summary["spread"]["75%"]
        print("remove high volatile trades / outliers where spread > {}".format(outlier))
        self._dataframe = self._dataframe[self._dataframe["spread"] <= outlier]

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

    def _train_model(self,
                     alpha=0.01,
                     hidden_layer_sizes=(10,),
                     max_iter=5000,
                     shuffle=False,
                     activation='logistic',
                     verbose='True',
                     learning_rate='adaptive'):
        print("Training MLPRegressor:")
        X_train, y_train, X_test, y_test = self._split_train_test_data()
        mlp_regressor = MLPRegressor(alpha=alpha,
                                     hidden_layer_sizes=hidden_layer_sizes,
                                     max_iter=max_iter,
                                     shuffle=shuffle,
                                     activation=activation,
                                     verbose=verbose,
                                     learning_rate=learning_rate)
        model = mlp_regressor.fit(X_train, y_train)
        print(model)
        self._model = model
        self._learning_curve = model.loss_curve_

        y_hat_train = model.predict(X_train)
        y_hat_test = model.predict(X_test)
        residual_train = abs(y_hat_train - y_train)
        residual_test = abs(y_hat_test - y_test)
        loss_curve = mlp_regressor.loss_curve_

        self._plot_result(y_train, y_hat_train, y_test, y_hat_test, loss_curve, residual_train, residual_test)

    def _plot_result(self, y_train, y_hat_train, y_test, y_hat_test, loss_curve, residual_train, residual_test):
        plt.subplot(4, 1, 1)
        plt.title("Training Set")
        plt.plot(y_train)
        plt.plot(y_hat_train, "--")
        plt.legend(["Actual", "Prediction"])

        plt.subplot(4, 1, 2)
        plt.title("Training Set Residual (Lower the better)")
        plt.plot(residual_train)
        plt.axhline(y=np.mean(residual_train), color='r', linestyle='--', label="Mean")
        plt.legend(["Residual", "Mean"])

        plt.subplot(4, 1, 3)
        plt.title("Test Set")
        plt.plot(y_test)
        plt.plot(y_hat_test, "--")
        plt.legend(["Actual", "Prediction"])

        plt.subplot(4, 1, 4)
        plt.title("Test Set Residual (Lower the better)")
        plt.plot(residual_test)
        plt.axhline(y=np.mean(residual_test), color='r', linestyle='--', label="Mean")
        plt.legend(["Residual", "Mean"])

        plt.subplots_adjust(left=0.2, wspace=0.8, top=0.8)
        plt.tight_layout()
        plt.show()

        plt.title("Loss Curve over Iteration")
        plt.plot(loss_curve)
        plt.show()

    def get_model(self):
        return self._model
