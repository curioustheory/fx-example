import numpy as np


class Modelling:
    def __init__(self, dataframe, correlation_limit=0.95):
        self._dataframe = dataframe
        self._x_train = None
        self._y_train = None
        self._x_test = None
        self._y_test = None
        self._optimum_model = None
        self._correlation_limit = correlation_limit

    def run(self):
        self._prep()
        self._find_optimum_model()

    def _prep(self):
        self._drop_highly_correlated_features()
        self._fill_na()
        self._split_train_test_data()
        # TODO define x, y dataframe

    def _drop_highly_correlated_features(self):
        # Create correlation matrix
        corr_matrix = self._dataframe.corr().abs()

        # Select upper triangle of correlation matrix
        upper_corr_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Find index of feature columns with correlation greater than the limit
        columns_to_drop = [column for column in upper_corr_triangle.columns if
                           any(upper_corr_triangle[column] > self._correlation_limit)]

        print("highly correlated columns (corr > ): ", str(columns_to_drop))
        self._dataframe = self._dataframe.drop(columns_to_drop, axis=1)

    def _fill_na(self):
        # outlier

        # fill na
        pass

    def _find_optimum_model(self):
        # ensemble

        # neural network

        pass

    def _split_train_test_data(self):
        pass

    def _cross_validation(self):
        pass

    def get_optimum_model(self):
        return self._optimum_model
