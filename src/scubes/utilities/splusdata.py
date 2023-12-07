from splusdata import Core
from splusdata.core import AuthenticationError

def connect_splus_cloud(username=None, password=None):
    n_tries = 0
    conn = None
    while (n_tries < 3) and (conn is None):
        try:
            conn = Core(username=username, password=password)
        except AuthenticationError:
            n_tries += 1
    return conn