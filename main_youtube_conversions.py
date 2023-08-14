from src.ml_projects_utils.load_config_utils import load_config
from src.training_and_evaluation_pipeline.model_pipeline import ml_pipeline


# EXECUTION FUNCTION #

def main():
    # -------------------------------- YouTube Conversions Pipeline -------------------------------- #

    # define model names/types to predict with
    models_name = ["youtube_conversions"]
    model_types = ["clf"]

    # load config file
    config_objects = load_config(model_names=models_name, model_types=model_types)

    # initialize YouTube conversions pipeline
    ml_pipeline(model_types, config_objects)


# RUN #

if __name__ == "__main__":
    # initialize main function
    main()
