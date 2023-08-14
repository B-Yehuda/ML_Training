import ast

DATA_PROCESSING = "Data_Processing"
PIPELINE = "Pipeline"
MODEL_PARAMETERS = "Model_Parameters"
LOCATIONS = "Locations"
AWS = "AWS"
GCS = "GCS"


class TwitchFCommunityDetectionConfig:
    def __init__(self, config):
        self.pipeline = config[PIPELINE].get("pipeline")

        self.pipeline_name = config[PIPELINE].get("pipeline_name")

        self.location_for_writing_communities_data = config[LOCATIONS].get(
            "location_for_writing_communities_data")

        self.predictions_to_bigquery_bucket = config[GCS].get("predictions_to_bigquery_bucket")

        self.predictions_to_aws_bucket = config[AWS].get("predictions_to_aws_bucket")

        self.features_data_dicts = ast.literal_eval(config[DATA_PROCESSING].get("features_data_dicts"))

        self.n_neighbors_per_channel = int(config[DATA_PROCESSING].get("n_neighbors_per_channel"))