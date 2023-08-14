from src.ml_projects_utils.load_config_utils import load_config
from src.training_and_evaluation_pipeline.model_pipeline import ml_pipeline


# EXECUTION FUNCTION #

def main():
    # -------------------------------- RAID Tutorials Pipeline -------------------------------- #

    # define model names/types to predict with
    models_name = ["raid_tutorials"]
    model_types = ["clf", "reg"]  # model_types = ["clf", "reg"]
    # load config files
    config_objects = load_config(model_names=models_name, model_types=model_types)

    # initialize raid_tutorials pipeline
    ml_pipeline(model_types, config_objects)

    # -------------------------------- RAID Deposits Pipeline -------------------------------- #

    # define model names/types to predict with
    models_name = ["raid_deposits"]
    model_types = ["clf"]  # model_types = ["clf", "reg"]

    # load config files
    config_objects = load_config(model_names=models_name, model_types=model_types)

    # initialize raid_deposits pipeline
    ml_pipeline(model_types, config_objects)


# RUN #

if __name__ == "__main__":
    # initialize main function
    main()
