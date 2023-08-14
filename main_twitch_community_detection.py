from src.ml_projects_utils.load_config_utils import load_config
from src.projects.twitch_community_detection.pipeline import twitch_community_detection_pipeline


def main(schema_interim: str = "dev_yehuda",
         is_filter_highly_connected_channels: bool = False,
         add_usernames: bool = False):
    # -------------------------------- Twitch Community Detection -------------------------------- #

    # define model names/types to predict with
    models_name = ["twitch_sna"]
    model_types = ["community_detection"]

    # load config files
    config_objects = load_config(model_names=models_name, model_types=model_types)

    # initialize community_detection pipeline
    twitch_community_detection_pipeline(config_objects=config_objects,
                                        schema_interim=schema_interim,
                                        is_filter_highly_connected_channels=is_filter_highly_connected_channels,
                                        is_add_usernames=add_usernames)


if __name__ == "__main__":
    # initialize main function
    main()
