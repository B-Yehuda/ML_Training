from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import numpy as np
import pandas as pd
from threading import Thread
import cv2
from queue import Queue
import time
from datetime import datetime
from src.ml_projects_utils.writing_output_utils import write_output, copy_from_aws_to_redshift_table
from src.vault import get_vault_secrets
from src.load_and_process_data import connect_redshift, load_data, load_local_credentials
from src.projects.twitch_fake_channels_detection.services.scrape_twitch_videos_urls import \
    scrape_videos_urls_for_twitch_channel, ScrapeVideosOfChannelConf
from src.projects.twitch_fake_channels_detection.settings.config_settings import TwitchFakeChannelsDetectionConfig
import matplotlib.pyplot as plt
from src.projects.twitch_fake_channels_detection.services.convert_videos_urls_to_stream_links import \
    VideoUrlToStreamLinkConf, video_url_to_stream_link
from src.settings.twitch_credentials_settings import TwitchCredentials


@dataclass
class VideoUrlToStreamLinkParallelConf:
    channel_data: dict
    video_quality: str


@dataclass
class VideoFaceDetectionConf:
    s_c_d: dict
    skip_frames_by_seconds: int or bool
    skip_frames_by_rate: int or bool
    video_max_process_time_hrs: int


class FileVideoStream:
    def __init__(self,
                 video_url,
                 video_max_process_time_hrs,
                 skip_frames_by_seconds,
                 skip_frames_by_rate,
                 queue_size=0):
        # Initialize video stream capture
        self.video_cap = cv2.VideoCapture(video_url)

        # Set frames to skip
        self.video_frames_cnt = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Set fps
        fetched_fps = int(self.video_cap.get(cv2.CAP_PROP_FPS))
        self.fps = fetched_fps if fetched_fps < 120 else 120

        # Set max frames to process
        self.video_max_process_time_hrs = video_max_process_time_hrs
        self.max_frames_to_process = int(self.fps * video_max_process_time_hrs * 3600)

        if skip_frames_by_rate:
            self.frames_to_skip = int(self.video_frames_cnt * skip_frames_by_rate)

        elif skip_frames_by_seconds:
            self.frames_to_skip = int(self.fps * skip_frames_by_seconds)

        # Initialize boolean used to indicate if the thread should be stopped/transform
        self.fetched_all_frames = False
        self.stopped = False
        self.transform = False

        # Initialize the queue used to store frames read from the video file
        self.Q = Queue(maxsize=queue_size)

        # Initialize thread
        self.thread = Thread(target=self.read_frames_from_video, args=())
        self.thread.daemon = True

    def start(self):
        # Start a thread to read frames from the file video stream
        self.thread.start()
        return self

    def read_frames_from_video(self):
        # Set 1st frame index
        frame_index = 1

        # Keep looping infinitely
        while self.video_cap.isOpened():
            # If the thread indicator variable is set, stop the thread
            if self.fetched_all_frames or self.stopped:
                break

            # Otherwise, ensure the queue has room in it
            if not self.Q.full():
                # Grab the next frame from the file
                grabbed = self.video_cap.grab()

                # If the `grabbed` boolean is `False`, then we have reached the end of the video file
                if not grabbed:
                    self.fetched_all_frames = True
                    break

                # Rolling through all the frames, but only writing the N-th frame to the queue
                if frame_index == 1 or frame_index % self.frames_to_skip == 0:
                    # Retrieve (decode) frame
                    _, frame = self.video_cap.retrieve()
                    # Skip empty frames
                    if frame is None:
                        continue
                    else:
                        # Convert frame to grayscale for face detection
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        # add the frame to the queue
                        self.Q.put(frame)

                # Set frames counter
                frame_index = frame_index + 1

                # Stop when reach the limit of max frames to process
                if frame_index > self.max_frames_to_process:
                    print(
                        f"Max duration of {self.video_max_process_time_hrs} hrs process reached. Stopping fetching frames.")
                    self.fetched_all_frames = True
                    break

            else:
                # Rest for 10ms, we have a full queue
                time.sleep(0.1)

        self.video_cap.release()

    def read_next_frame_from_queue(self):
        # return next frame in the queue
        return self.Q.get()

    def more(self):
        # If consumer faster than producer - hold
        tries = 0
        while self.Q.qsize() == 0 and not self.fetched_all_frames and tries < 100:
            time.sleep(0.1)
            tries += 1

        # Continue if there are still frames in the queue else stop since producer has reached end of file stream
        is_continue = False if (self.stopped or (self.Q.qsize() == 0 and self.fetched_all_frames)) else True
        return is_continue

    def stop(self):
        # Indicate that the thread should be stopped
        self.stopped = True

        # Wait until stream resources are released (producer thread might be still grabbing frame)
        self.thread.join()

        # Release the video capture
        self.video_cap.release()
        cv2.destroyAllWindows()


def frame_face_detection(face_classifier, frame) -> bool:
    faces = face_classifier.detectMultiScale(frame,
                                             scaleFactor=1.1,
                                             minNeighbors=30,
                                             minSize=(50, 50))

    if len(faces) > 0:
        save_frames_with_face = False
        if save_frames_with_face:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(20, 10))
                plt.imshow(img_rgb)
                plt.axis('off')
                plt.savefig(f"{x}.png")
        return True

    else:
        return False


def video_face_detection(conf: VideoFaceDetectionConf) -> dict:
    print(
        f"- Processing video {conf.s_c_d['twitch_url']} of channel {conf.s_c_d['channel_id']} - started at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")

    # TODO: Is it possible to use ProcessPoolExecutor() for the face detection part?

    # Load the pre-trained face detection model
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize thread process
    fvs = FileVideoStream(video_url=conf.s_c_d['streamlink_url'],
                          video_max_process_time_hrs=conf.video_max_process_time_hrs,
                          skip_frames_by_seconds=conf.skip_frames_by_seconds,
                          skip_frames_by_rate=conf.skip_frames_by_rate).start()

    detected = False

    while fvs.more():
        # Read next frame
        frame = fvs.read_next_frame_from_queue()

        display_frames_during_process = False
        if display_frames_during_process:
            # Display frame image
            cv2.imshow('frame', frame)
            # Display each frame for 1 ms
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Run face detection process
        detected = frame_face_detection(face_classifier=face_classifier, frame=frame)
        if detected:
            fvs.stop()
            print(
                f"- Processing video {conf.s_c_d['twitch_url']} (face detection = \033[1mTRUE\033[0m) of channel {conf.s_c_d['channel_id']} - finished at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")
            conf.s_c_d['face_detected'] = True
            return conf.s_c_d

    if not detected:
        fvs.stop()
        print(
            f"- Processing video {conf.s_c_d['twitch_url']} (face detection = \033[1mFALSE\033[0m) of channel {conf.s_c_d['channel_id']} - finished at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")
        conf.s_c_d['face_detected'] = False
        return conf.s_c_d


def twitch_fake_channels_detection_pipeline(config_objects: dict,
                                            schema_interim: str):
    """
    Description:
        A function that:
         1. Load from Redshift data of suspicious fake Twitch channels.
         2. Scrape for each channel - its L30D VODs URLs.
         3. Run face detection algorithm on each VOD.
         4. Aggregate the results and load them to GCS.

    Background:
        By using this function, we assume that in each video:
         1. If there is a face in - they will appear in at least one of the frames that appear every 5 seconds.
         2. If there is no face in the first 3 hours - they will not appear later.

    Parameters:
        1. config_objects: Configuration object
        2. schema_interim: The schema to write the output to.
    """

    # PART 1 - LOAD SUSPICIOUS CHANNELS #
    """ Load pre tagged suspicious channels from Redshift"""

    for config in config_objects.values():
        config = config[0]

    # Retrieve dats from config file
    twitch_fake_channels_detection_config = TwitchFakeChannelsDetectionConfig(config)

    print("--------------------------------------------------------------------------------------------------------")
    print(
        f"Pipeline {twitch_fake_channels_detection_config.pipeline_name} using {twitch_fake_channels_detection_config.pipeline} - started at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")
    print("--------------------------------------------------------------------------------------------------------")

    # Retrieve redshift credentials
    if twitch_fake_channels_detection_config.location_for_writing_face_detection_data in {"GCS", "AWS"}:
        redshift_credentials = get_vault_secrets()
        # TODO: Add twitch_credentials to Vault

        twitch_credentials = get_vault_secrets()
        twitch_credentials = TwitchCredentials(twitch_credentials)
    elif twitch_fake_channels_detection_config.location_for_writing_face_detection_data in {"LOCAL_CSV", "LOCAL_JSON",
                                                                                            "REDSHIFT",
                                                                                            "GCS_VIA_LOCAL",
                                                                                            "AWS_VIA_LOCAL"}:
        redshift_credentials = load_local_credentials(file_name="redshift_config.json")
        twitch_credentials = load_local_credentials(file_name="twitch_config.json")
        twitch_credentials = TwitchCredentials(twitch_credentials)
    else:
        raise ValueError("No location_for_writing_model was specified in the config file")

    # Connect redshift
    engine, conn, cur = connect_redshift(credentials=redshift_credentials)

    # Load channels data from redshift
    df_suspicious_channels = load_data(cur=cur, config=config)

    with ThreadPoolExecutor() as executor:
        # PART 2 - SCRAPE TWITCH URLs #
        """ Scrape for each suspicious channel its VODs URLs """

        print(
            f"\nScraping URLs from Twitch API in parallel - started at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")

        suspicious_channels_data = [s_c_d for s_c_d in
                                    executor.map(scrape_videos_urls_for_twitch_channel,
                                                 [ScrapeVideosOfChannelConf(
                                                     channel_id=channel_id,
                                                     videos_to_scrape_per_channel=twitch_fake_channels_detection_config.videos_to_scrape_per_channel,
                                                     client_id=twitch_credentials.TWITCH_CLIENT_ID,
                                                     authorization=twitch_credentials.TWITCH_CLIENT_SECRET)
                                                     for channel_id in df_suspicious_channels.provider_id]
                                                 )]
        suspicious_channels_data = [d for d in suspicious_channels_data if d]

        print(f"- Scraped videos data of - {len(suspicious_channels_data)} Twitch channels")
        print(
            f"Scraping URLs from Twitch API in parallel - finished at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")

        if not suspicious_channels_data:
            raise ValueError("No Twitch URLs were found for any suspicious channel")

        # Refactor suspicious_channels_data array
        output_list = []
        for d in suspicious_channels_data:
            channel_id = d["channel_id"]
            twitch_urls = d["twitch_urls"]
            for twitch_url in twitch_urls:
                output_list.append({"channel_id": channel_id, "twitch_url": twitch_url})
        suspicious_channels_data = output_list

        # PART 3 - PROCESS TWITCH URLs TO STREAMLINK URLs #
        """ Process each Twitch URL so that it is possible to access it """

        print(
            f"\nConverting Twitch URLs to StreamLink URLs in parallel - started at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")

        suspicious_channels_data = [updated_s_c_d for updated_s_c_d in
                                    executor.map(video_url_to_stream_link,
                                                 [VideoUrlToStreamLinkConf(
                                                     s_c_d=s_c_d,
                                                     video_quality=twitch_fake_channels_detection_config.video_quality)
                                                     for s_c_d in suspicious_channels_data]
                                                 )]
        suspicious_channels_data = [d for d in suspicious_channels_data if d]

        channels_cnt = [d.get("channel_id") for d in suspicious_channels_data if d.get("channel_id") is not None]
        unique_channels_cnt = set(channels_cnt)

        print(f"- Converted URLs data of - {len(unique_channels_cnt)} Twitch channels")
        print(
            f"Converting Twitch URLs to StreamLink URLs in parallel - finished at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")

        if not suspicious_channels_data:
            raise ValueError("No StreamLink URLs were found for any suspicious channel")

        # PART 4 - RUN FACE DETECTION MODEL #
        """ Run face detection algorithm on each video URL for each channel """

        print(
            f"\nFace detection parallel process - started at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m\n")


        suspicious_channels_data = [{"channel_id": "82550365",
                                     "twitch_url": "blabla",
                                     "streamlink_url": "https://xxdgeft87wbj63p.cloudfront.net/60ced98521314deafc28_imandruill_40975463800_1691612669/480p30/index-muted-J4V9YIV319.m3u8"
                                     }]


        suspicious_channels_data = [updated_s_c_d for updated_s_c_d in
                                    executor.map(video_face_detection,
                                                 [VideoFaceDetectionConf(
                                                     s_c_d=s_c_d,
                                                     skip_frames_by_seconds=twitch_fake_channels_detection_config.skip_frames_by_seconds,
                                                     skip_frames_by_rate=twitch_fake_channels_detection_config.skip_frames_by_rate,
                                                     video_max_process_time_hrs=twitch_fake_channels_detection_config.video_max_process_time_hrs)
                                                     for s_c_d in suspicious_channels_data]
                                                 )]

        print(
            f"\nFace detection parallel process - finished at: \033[1m{datetime.now().isoformat(' ', 'seconds')}\033[0m")

        # PART 5 - AGGREGATE RESULTS IN DF #
        """ Combine results into single df """

        # Write results in df
        df_face_detection_granular = pd.DataFrame(suspicious_channels_data)
        df_face_detection = df_face_detection_granular.groupby('channel_id').agg({
            'streamlink_url': 'count',
            'face_detected': lambda x: (x == True).sum()
        }).reset_index()

        # Rename the columns
        df_face_detection.columns = ['channel_id', 'cnt_videos', 'cnt_videos_w_face_detected']

        # Create face detection flag based on algorithm results
        df_face_detection['no_face_flag'] = df_face_detection.apply(
            lambda row: (True if row['cnt_videos_w_face_detected'] / row['cnt_videos'] <= 0.1 else False), axis=1)
        df_face_detection = df_face_detection[
            ['channel_id', 'cnt_videos', 'cnt_videos_w_face_detected', 'no_face_flag']]

        # Join flags
        df_results = df_suspicious_channels.merge(df_face_detection[['channel_id', 'no_face_flag']],
                                                  how='left',
                                                  left_on='provider_id',
                                                  right_on='channel_id').drop(columns=['channel_id'])
        df_results['no_face_flag'] = np.where(df_results['no_face_flag'].isnull(), False, df_results['no_face_flag'])

        # Calculate fake_channel_score based on all flags
        df_results['fake_channel_score'] = df_results.apply(
            lambda row: (row['viewers_trend_flag'] * 0.3 +
                         row['chatters_to_watchtime_flag'] * 0.2 +
                         row['no_face_flag'] * 0.15 +
                         row['creator_chat_inactive_flag'] * 0.15 +
                         row['social_links_missing_flag'] * 0.1 +
                         row['description_missing_flag'] * 0.1), axis=1)

        # PART 6 - LOAD FACE DETECTION RESULTS TO REDSHIFT #
        """ Load the face detection results to AWS then write from there into Redshift """
        # TODO: Remove this part since its unnecessary + vault is not defined for AWS credentials yet

        # Write df_results into AWS bucket
        object_name = "face_detection"
        aws_file_path = write_output(location_for_writing_output="AWS_VIA_LOCAL",
                                     engine=engine,
                                     conn=conn,
                                     cur=cur,
                                     schema_name=schema_interim,
                                     df=df_face_detection,
                                     object_name=object_name,
                                     database_name=None,
                                     skip_if_exists=False,
                                     drop_if_exists=True,
                                     create_table_from_sql_or_df="df",
                                     buckets=[twitch_fake_channels_detection_config.predictions_to_aws_bucket],
                                     config=config)

        # Write df_results into Redshift table
        copy_from_aws_to_redshift_table(conn=conn,
                                        cur=cur,
                                        schema=schema_interim,
                                        table_name=object_name,
                                        columns_and_types_dict={'channel_id': 'VARCHAR(255)',
                                                                'cnt_videos': 'INTEGER',
                                                                'cnt_videos_w_face_detected': 'INTEGER',
                                                                'no_face_flag': 'BOOLEAN'},
                                        skip_if_exists=False,
                                        drop_if_exists=True,
                                        bucket_name=twitch_fake_channels_detection_config.predictions_to_aws_bucket,
                                        aws_file_path=aws_file_path)

        # PART 7 - LOAD AGGREGATED RESULTS TO GCS #
        """ Load the results to GCS """

        # Write df_results into GCS bucket
        object_name = "fake_channel_detection"
        write_output(
            location_for_writing_output=twitch_fake_channels_detection_config.location_for_writing_face_detection_data,
            engine=engine,
            conn=conn,
            cur=cur,
            schema_name=schema_interim,
            df=df_results,
            object_name=object_name,
            database_name=object_name,
            skip_if_exists=False,
            drop_if_exists=False,
            create_table_from_sql_or_df=None,
            buckets=[twitch_fake_channels_detection_config.predictions_to_bigquery_bucket],
            config=config)
