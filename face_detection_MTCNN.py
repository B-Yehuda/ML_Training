import cv2
import time
import streamlink
from facenet_pytorch.models.mtcnn import MTCNN
import torch
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)


def stream_to_url(url, quality):
    """ Get URL, and return streamlink URL """
    session = streamlink.Streamlink()
    streams = session.streams(url)

    if streams:
        return streams[quality].to_url()
    else:
        raise ValueError('Could not locate your stream.')


def face_detection(video_url):
    # Open the video file
    video_cap = cv2.VideoCapture(video_url)
    video_len = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while video_cap.isOpened():
        # Read the current frame
        ret, frame = video_cap.read()

        if not ret:
            # Video has ended
            break

        # Display frame image
        cv2.imshow('frame', frame)

        # Display each frame for 1 ms
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        # mtcnn_detect(frame)
        detector = MTCNN(select_largest=False)
        faces = detector(frame)

        try:
            if len(faces) > 0:
                return True
        except Exception as e:
            continue

    # Release the video capture
    video_cap.release()
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
        face_detected = face_detection(live_video_url)

        # Print the result
        print(f"For {video_url} - face_detected is: {face_detected}")

    end = time.time()
    print(end - start)


if __name__ == "__main__":
    # initialize main function
    main()
