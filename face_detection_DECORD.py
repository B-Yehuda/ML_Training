import streamlink
import cv2
import time
import matplotlib.pyplot as plt
from decord import VideoReader, cpu, gpu


def stream_to_url(url, quality):
    """ Get URL, and return streamlink URL """
    session = streamlink.Streamlink()
    streams = session.streams(url)

    if streams:
        return streams[quality].to_url()
    else:
        raise ValueError('Could not locate your stream.')


def face_detection(frame):
    # Load the pre-trained face detection model
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_classifier.detectMultiScale(gray_frame,
                                             scaleFactor=1.1,
                                             minNeighbors=30,
                                             minSize=(50, 50))

    if len(faces) > 0:
        # # Display the detected face
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        #     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     plt.figure(figsize=(20, 10))
        #     plt.imshow(img_rgb)
        #     plt.axis('off')
        #     plt.savefig(f"{video_url}_{x}.png")

        # Release the video capture
        cv2.destroyAllWindows()

        # Face detected in the current frame
        return True


def process_using_decord(video_url):
    # Open the video file
    vr = VideoReader(video_url, ctx=cpu(0), num_threads=1)
    print('video frames:', len(vr))

    # Define frames to skip
    video_frames_cnt = int(len(vr))
    frames_skip_rate = 0.1
    frames_to_skip = int(video_frames_cnt * frames_skip_rate)

    for frame_idx in range(1, video_frames_cnt, frames_to_skip):
        # Read the current frame
        frame = vr[frame_idx].asnumpy()

        # Display frame image
        cv2.imshow('frame', frame)

        # Display each frame for 1 ms
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Run face detection process
        if face_detection(frame):
            return True

    # Release the video capture
    cv2.destroyAllWindows()

    # No face detected in the video
    return False


def main():
    start = time.time()

    # Provide the link to the Twitch video
    urls = ["https://www.twitch.tv/videos/1858308141", "https://www.twitch.tv/videos/1802532738",
            "https://www.twitch.tv/videos/658341831", "https://www.twitch.tv/videos/1738959153"]

    for video_url in urls:
        # Convert stream to url
        live_video_url = stream_to_url(url=video_url, quality='best')

        # Call the function to detect faces in the video
        face_detected = process_using_decord(live_video_url)

        # Print the result
        print(f"For {video_url} - face_detected is: {face_detected}")

    end = time.time()
    print(end - start)


if __name__ == "__main__":
    # initialize main function
    main()
