import numpy as np
from sklearn.neural_network import MLPRegressor

from forex import chartutil
from forex.util import drop_highly_correlated_features


class Modelling:
    def __init__(self, dataframe, eda_summary, train_size, correlation_limit=0.95):
        self._dataframe = dataframe
        self._eda_summary = eda_summary
        self._correlation_limit = correlation_limit
        self._train_size = train_size
        self._model = None
        self._learning_curve = None

    def run(self, alpha, hidden_layer_sizes, max_iter, activation, verbose, learning_rate):
        """
        execute the preparation for moddelling then trains the model based on the configuration.

        :param alpha: learning rate i.e. 0.001
        :param hidden_layer_sizes: defines the layers and nodes i.e. 10, 25, 10
        :param max_iter: number of max iteration to train
        :param activation: the activation function i.e. tanh, logistic, identuty, relu
        :param verbose: display training data
        :param learning_rate: types of learning i.e. adaptive
        """
        self._prep()
        self._train_model(alpha, hidden_layer_sizes, max_iter, activation, verbose, learning_rate)

    def _prep(self):
        """
        preparing the data for modelling by removing outliers and splitting the dataset into training and test.
        """
        self._remove_outlier()
        self._split_train_test_data()

    def _remove_outlier(self):
        """
        remove volatile trades that is not representative of the trend.
        """
        outlier = self._eda_summary["spread"]["75%"]
        print("remove high volatile trades / outliers where spread > {}".format(outlier))
        self._dataframe = self._dataframe[self._dataframe["spread"] <= outlier]

    def _split_train_test_data(self):
        """
        splits the data into train and test set

        :return: train and test set
        """
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

    def _train_model(self, alpha, hidden_layer_sizes, max_iter, activation, verbose, learning_rate):
        """
        train a MLPRegressor and validating it against a test set.

        :param alpha: learning rate i.e. 0.001
        :param hidden_layer_sizes: defines the layers and nodes i.e. 10, 25, 10
        :param max_iter: number of max iteration to train
        :param activation: the activation function i.e. tanh, logistic, identuty, relu
        :param verbose: display training data
        :param learning_rate: types of learning i.e. adaptive
        """
        print("Training MLPRegressor:")
        X_train, y_train, X_test, y_test = self._split_train_test_data()
        mlp_regressor = MLPRegressor(alpha=alpha,
                                     hidden_layer_sizes=hidden_layer_sizes,
                                     max_iter=max_iter,
                                     shuffle=False,
                                     activation=activation,
                                     verbose=verbose,
                                     learning_rate=learning_rate)
        model = mlp_regressor.fit(X_train, y_train)

        # TODO: refactor
        self._model = model
        self._learning_curve = model.loss_curve_

        y_hat_train = model.predict(X_train)
        y_hat_test = model.predict(X_test)
        loss_train = abs(y_hat_train - y_train)
        loss_test = abs(y_hat_test - y_test)
        loss_curve = mlp_regressor.loss_curve_

        print("Avg. train set loss: {}".format(np.mean(loss_train)))
        print("Avg. test set loss: {}".format(np.mean(loss_test)))

        chartutil.plot_modelling_result(y_train, y_hat_train, y_test, y_hat_test, loss_curve, loss_train, loss_test)

    def get_model(self):
        """
        get the trained model.

        :return: MLPRegressor model
        """
        return self._model
