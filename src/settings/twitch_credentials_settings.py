import ast

TWITCH_CLIENT_ID = "TWITCH_CLIENT_ID"
TWITCH_CLIENT_SECRET = "TWITCH_CLIENT_SECRET"


class TwitchCredentials:
    def __init__(self, credentials):
        self.TWITCH_CLIENT_ID = credentials[TWITCH_CLIENT_ID]

        self.TWITCH_CLIENT_SECRET = credentials[TWITCH_CLIENT_SECRET]
