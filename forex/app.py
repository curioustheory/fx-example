from forex.util import load_config, load_data


def run(config_file_path):
    print("reading configuration file... ")
    config = load_config(config_file_path=config_file_path)
    print("loading dataframe...")
    print("---------------------------------------------------------------------------------------------")
    dataframe = load_data(input_file_path=config["input_file_path"], input_file_header=config["input_file_header"])
    print(dataframe.head())
    print()

    print("data summary...")
    print("---------------------------------------------------------------------------------------------")
    print("Data Shape: ")
    print(dataframe.shape)
    print("Summary Stats: ")
    print(dataframe.describe())

    """



    """

    print()
    # what data am i looking at

    print("some chart")
    print("---------------------------------------------------------------------------------------------")
    print()

    print("sampling")
    print("---------------------------------------------------------------------------------------------")
    print()

    print("preprocessing...")
    print("---------------------------------------------------------------------------------------------")
    print()
    # clean / fill na preprocess

    # chart

    # feature engineering

    # modelling

    # evaluation i.e. predictive modelling

    # outcome


if __name__ == "__main__":
    run(config_file_path="../config.json")
