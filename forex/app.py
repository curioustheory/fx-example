from forex.eda import ExploratoryDataAnalysis
from forex.featureengineering import FeatureEngineering
from forex.modelling import Modelling
from forex.util import load_config, load_data


def run(config_file_path):
    """
    runs the entire data pipeline consisting of:
        - loading config
        - loading data
        - data engineering
        - exploratory data analysis
        - modelling

    :param config_file_path: the path of the configuration file
    :return:
    """
    print("reading configuration file... ")
    config = load_config(config_file_path=config_file_path)
    input_file_path = config["input_file_path"]
    input_file_header = config["input_file_header"]
    plot_eda_chart = config["plot_eda_chart"]
    correlation_limit = config["correlation_limit"]
    train_size = config["train_size"]
    nn_alpha = config["nn_alpha"]
    nn_hidden_layer_sizes = config["nn_hidden_layer_sizes"]
    nn_max_iter = config["nn_max_iter"]
    nn_activation = config["nn_activation"]
    nn_verbose = config["nn_verbose"]
    nn_learning_rate = config["nn_learning_rate"]

    print("loading dataframe...")
    print("---------------------------------------------------------------------------------------------")
    dataframe = load_data(input_file_path=input_file_path,
                          input_file_header=input_file_header)
    print(dataframe.head())
    print()

    print("data prep / engineering...")
    print("---------------------------------------------------------------------------------------------")
    engineering = FeatureEngineering(dataframe)
    engineering.run()
    dataframe = engineering.get_dataframe()
    print(dataframe.head())
    print()

    print("loading exploratory analysis...")
    print("---------------------------------------------------------------------------------------------")
    eda = ExploratoryDataAnalysis(dataframe)
    eda.run(plot_eda_chart)
    print("no. of rows:", dataframe.shape[0])
    print("no. of columns:", dataframe.shape[1])
    print()
    print("data summary:")
    print(eda.get_summary())
    print()
    print("feature correlation:")
    print(eda.get_feature_correlation())
    print()

    print("modelling...")
    print("---------------------------------------------------------------------------------------------")
    modelling = Modelling(dataframe, eda.get_summary(), train_size, correlation_limit)
    modelling.run(nn_alpha,
                  nn_hidden_layer_sizes,
                  nn_max_iter,
                  nn_activation,
                  nn_verbose,
                  nn_learning_rate)


if __name__ == "__main__":
    run(config_file_path="../config.json")
