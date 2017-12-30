import numpy as np


class Modelling:
    def __init__(self, dataframe):
        self._dataframe = dataframe
        self._optimum_model = None

    def run(self):
        self._prep()
        self._find_optimum_model()

    def _prep(self):
        self._drop_highly_correlated_features()

    def _drop_highly_correlated_features(self):
        pass
        # Create correlation matrix
        # corr_matrix = self._dataframe.corr().abs()

        # Select upper triangle of correlation matrix
        # upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Find index of feature columns with correlation greater than 0.95
        # to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        # self._dataframe.drop(self._dataframe.columns[to_drop], axis=1)

    def _feature_selection(self):
        pass

    def _find_optimum_model(self):
        pass

    def get_optimum_model(self):
        return self._optimum_model
