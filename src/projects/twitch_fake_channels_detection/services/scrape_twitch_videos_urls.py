import requests
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class ScrapeVideosOfChannelConf:
    channel_id: str
    videos_to_scrape_per_channel: int
    client_id: str
    authorization: str


def scrape_videos_urls_for_twitch_channel(conf: ScrapeVideosOfChannelConf) -> dict:
    # Construct the API URL to fetch videos for the given channel ID
    base_url = "https://api.twitch.tv/helix/videos"

    # Set up the query parameters
    params = {
        'user_id': conf.channel_id,
        'first': conf.videos_to_scrape_per_channel,
        'period': 'all',
        'type': 'archive',
    }

    # Set up headers with the Client-ID for authentication
    headers = {
        'Client-ID': conf.client_id,
        'Authorization': 'Bearer ' + conf.authorization,
    }

    # Initialize dictionary to store scraped data
    channel_data = {}

    try:
        # Fetch videos data from Twitch API
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
        videos_data = data['data']
        twitch_urls = []

        # Save only last 30 days videos data
        for video in videos_data:
            if datetime.strptime(video['created_at'], '%Y-%m-%dT%H:%M:%SZ') >= (datetime.utcnow() - timedelta(days=30)):
                video_url = video['url']
                if video_url:
                    twitch_urls.append(video_url)

        twitch_urls = [item for item in twitch_urls if item]

        # Return only if videos were fetched
        if len(twitch_urls) > 0:
            channel_data['channel_id'] = conf.channel_id
            channel_data['twitch_urls'] = twitch_urls
            return channel_data

    except requests.exceptions.RequestException as e:
        print("Error occurred while fetching data:", e)
