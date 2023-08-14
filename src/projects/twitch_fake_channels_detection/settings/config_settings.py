import ast

PIPELINE = "Pipeline"
MODEL_PARAMETERS = "Model_Parameters"
LOCATIONS = "Locations"
AWS = "AWS"
GCS = "GCS"


class TwitchFakeChannelsDetectionConfig:
    def __init__(self, config):
        self.pipeline = config[PIPELINE].get("pipeline")

        self.pipeline_name = config[PIPELINE].get("pipeline_name")

        self.location_for_writing_face_detection_data = config[LOCATIONS].get(
            "location_for_writing_face_detection_data")

        self.video_max_process_time_hrs = int(config[MODEL_PARAMETERS].get("video_max_process_time_hrs"))

        self.skip_frames_by_seconds = ast.literal_eval(config[MODEL_PARAMETERS].get("skip_frames_by_seconds"))

        self.skip_frames_by_rate = ast.literal_eval(config[MODEL_PARAMETERS].get("skip_frames_by_rate"))

        self.videos_to_scrape_per_channel = int(config[MODEL_PARAMETERS].get("videos_to_scrape_per_channel"))

        self.video_quality = config[MODEL_PARAMETERS].get("video_quality")

        self.predictions_to_bigquery_bucket = config[GCS].get("predictions_to_bigquery_bucket")

        self.predictions_to_aws_bucket = config[AWS].get("predictions_to_aws_bucket")
