from dataclasses import dataclass
import streamlink


@dataclass
class VideoUrlToStreamLinkConf:
    s_c_d: dict
    video_quality: str


def video_url_to_stream_link(conf: VideoUrlToStreamLinkConf) -> dict:
    session = streamlink.Streamlink()

    try:
        streams = session.streams(conf.s_c_d['twitch_url'])
        if streams:
            # Check if the given quality stream is available, else find the next closest quality
            if conf.video_quality in streams:
                conf.s_c_d['streamlink_url'] = streams[conf.video_quality].to_url()
                return conf.s_c_d
            else:
                qualities = ['360p', '360p30', '360p60', '480p', '480p30', '480p60',
                             '540p', '540p30', '540p60', '720p', '720p30', '720p60', 'best']
                for video_quality in qualities:
                    if video_quality in streams:
                        conf.s_c_d['streamlink_url'] = streams[conf.video_quality].to_url()
                        return conf.s_c_d

                raise ValueError("No medium or higher quality streams found for the provided URL.")
        else:
            raise ValueError('Could not locate your stream.')

    except Exception:
        pass
