import time
from src.ml_projects_utils.load_config_utils import load_config
from src.projects.twitch_fake_channels_detection.pipeline import twitch_fake_channels_detection_pipeline


def main(schema_interim: str = "ds_db"):
    # -------------------------------- Twitch Fake Channels Detection -------------------------------- #
    start_time = time.time()

    # Define model names/types to predict with
    models_name = ["twitch"]
    model_types = ["fake_channels_detection"]

    # Load config files
    config_objects = load_config(model_names=models_name, model_types=model_types)

    # Initialize face_detection pipeline
    twitch_fake_channels_detection_pipeline(config_objects=config_objects,
                                            schema_interim=schema_interim)

    end_time = time.time()
    print(end_time - start_time)


if __name__ == "__main__":
    main()
