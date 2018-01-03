from forex import chartutil


class ExploratoryDataAnalysis:

    def __init__(self, dataframe):
        self._dataframe = dataframe
        self._data_summary = None
        self._feature_correlation = None

    def run(self, plot_eda_chart=True):
        """
        execute the data summary and visualising the charts.

        :param plot_eda_chart: enable the visualisation of the eda chart
        """
        self._summarise_data()
        self._check_feature_correlation()
        if plot_eda_chart:
            chartutil.plot_eda_chart(self._dataframe, self._feature_correlation)

    def _summarise_data(self):
        """
        describes the data
        """
        self._data_summary = self._dataframe.describe()

    def _check_feature_correlation(self):
        """
        calculates the feature correlation
        """
        self._feature_correlation = self._dataframe.corr()

    def get_summary(self):
        """
        returns the data summary
        """
        return self._data_summary

    def get_feature_correlation(self):
        """
        returns the feature correlation matrix
        """
        return self._feature_correlation
