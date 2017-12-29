import pandas as pd

from forex.eda import ExploratoryDataAnalysis
from forex.util import load_config, load_data
from forex.featureengineering import FeatureEngineering


def run(config_file_path):
    # try:
        print("reading configuration file... ")
        config = load_config(config_file_path=config_file_path)
        print("loading dataframe...")
        print("---------------------------------------------------------------------------------------------")
        dataframe = load_data(input_file_path=config["input_file_path"],
                              input_file_header=config["input_file_header"])
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

        if config["process_eda"].lower() == "true":
            print("loading exploratory analysis...")
            print("---------------------------------------------------------------------------------------------")
            eda = ExploratoryDataAnalysis(dataframe)
            eda.run()
            print()

        print("modelling...")
        print("---------------------------------------------------------------------------------------------")
        if config["cache_data"].lower() == "true":
            pass
        print()

        print("validation...")
        print("---------------------------------------------------------------------------------------------")


    # except Exception as e:
    #     print(e)


"""
        # check feature correlation
        # TODO:

        print("running models...")
        print("---------------------------------------------------------------------------------------------")
        trainer = ModelTrainer(dataframe)
        trainer.run()
        model = trainer.get_best_model()
        print()

        print("validating models on unseen data...")
        print("---------------------------------------------------------------------------------------------")
        validator = ModelValidator()
        validator.run()
        print()
"""

if __name__ == "__main__":
    run(config_file_path="../config.json")
