import os

# Always get the absolute path to the project root (where requirements.txt is)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'ml-1m')
MOVIES_PATH = os.path.join(DATA_DIR, 'movies.dat')
RATINGS_PATH = os.path.join(DATA_DIR, 'ratings.dat')
USERS_PATH = os.path.join(DATA_DIR, 'users.dat') 