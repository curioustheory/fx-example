import pandas as pd

from forex.eda import ExploratoryDataAnalysis
from forex.util import load_config, load_data


def run(config_file_path):
    try:
        print("reading configuration file... ")
        config = load_config(config_file_path=config_file_path)
        print("loading dataframe...")
        print("---------------------------------------------------------------------------------------------")
        dataframe = load_data(input_file_path=config["input_file_path"], input_file_header=config["input_file_header"])
        print(dataframe.head())
        print()

        if config["process_eda"].lower() == "true":
            print("loading exploratory analysis...")
            print("---------------------------------------------------------------------------------------------")
            eda = ExploratoryDataAnalysis(dataframe)
            eda.run()
            print()

    except Exception as e:
        print(e)


"""

        print("feature engineering...")
        print("---------------------------------------------------------------------------------------------")
        engineering = FeatureEngineering(dataframe)
        engineering.run()
        dataframe = engineering.get_dataframe()
        print()

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
