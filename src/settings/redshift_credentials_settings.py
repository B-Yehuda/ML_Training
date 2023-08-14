import ast

USER = "user"
PASSWORD = "password"
HOST = "host"
PORT = "port"
DBNAME = "dbname"


class RedshiftCredentials:
    def __init__(self, credentials):
        self.USER = credentials[USER]

        self.PASSWORD = credentials[PASSWORD]

        self.HOST = credentials[HOST]

        self.PORT = credentials[PORT]

        self.DBNAME = credentials[DBNAME]
