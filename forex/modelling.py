import numpy as np
from sklearn.neural_network import MLPRegressor

from forex import chartutil
from forex.util import split_train_test_data


class Modelling:
    def __init__(self, X, y, train_size):
        self._X = X
        self._y = y
        self._train_size = train_size
        self._model = None

    def run(self, alpha, hidden_layer_sizes, max_iter, activation, verbose, learning_rate):
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
        X_train, y_train, X_test, y_test = split_train_test_data(self._X, self._y, self._train_size)
        mlp_regressor = MLPRegressor(alpha=alpha,
                                     hidden_layer_sizes=hidden_layer_sizes,
                                     max_iter=max_iter,
                                     shuffle=False,
                                     activation=activation,
                                     verbose=verbose,
                                     learning_rate=learning_rate)
        self._model = mlp_regressor.fit(X_train, y_train)
        self._validate_model(X_train, y_train, X_test, y_test, self._model)

    def _validate_model(self, X_train, y_train, X_test, y_test, model):
        """
        validate the training and the test set

        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        :param model:
        """
        y_hat_train = model.predict(X_train)
        y_hat_test = model.predict(X_test)
        loss_train = abs(y_hat_train - y_train)
        loss_test = abs(y_hat_test - y_test)
        loss_curve = model.loss_curve_

        print()
        print("Avg. train set loss: {}".format(np.mean(loss_train)))
        print("Avg. test set loss: {}".format(np.mean(loss_test)))

        # plot the result
        chartutil.plot_modelling_result(y_train, y_hat_train, y_test, y_hat_test, loss_curve, loss_train, loss_test)

    def get_model(self):
        """
        get the trained model.

        :return: MLPRegressor model
        """
        return self._model
