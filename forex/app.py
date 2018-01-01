from forex.eda import ExploratoryDataAnalysis
from forex.featureengineering import FeatureEngineering
from forex.modelling import Modelling
from forex.util import load_config, load_data


def run(config_file_path):
    print("reading configuration file... ")
    config = load_config(config_file_path=config_file_path)
    train_input_file_path = config["train_input_file_path"]
    test_input_file_path = config["test_input_file_path"]
    input_file_header = config["input_file_header"]
    process_eda = config["process_eda"].lower() == "true"
    plot_chart = config["plot_chart"].lower() == "true"
    correlation_limit = float(config["correlation_limit"])

    print("loading dataframe...")
    print("---------------------------------------------------------------------------------------------")
    dataframe = load_data(input_file_path=train_input_file_path,
                          input_file_header=input_file_header)
    print(dataframe.head())
    print()

    print("data prep / engineering...")
    print("\t sort out datetime format and removing volume")
    print("---------------------------------------------------------------------------------------------")
    engineering = FeatureEngineering(dataframe)
    engineering.run()
    dataframe = engineering.get_dataframe()
    print(dataframe.head())
    print()

    if process_eda:
        print("loading exploratory analysis...")
        print("---------------------------------------------------------------------------------------------")
        eda = ExploratoryDataAnalysis(dataframe)
        eda.run(plot_chart)
        print("no. of rows:", dataframe.shape[0])
        print("no. of columns:", dataframe.shape[1])
        print()
        print(eda.get_summary())
        print()
        print("feature correlation...")
        print(eda.get_feature_correlation())
        print()

    print("modelling...")
    print("---------------------------------------------------------------------------------------------")
    modelling = Modelling(dataframe, correlation_limit)
    modelling.run()
    model = modelling.get_optimum_model()
    print()

    print("making a prediction...")
    print("---------------------------------------------------------------------------------------------")
    test_dataframe = load_data(input_file_path=test_input_file_path,
                               input_file_header=input_file_header)
    print(test_dataframe.head())
    print()


if __name__ == "__main__":
    run(config_file_path="../config.json")
