# Imports
import pickle
import numpy as np


# Constants
DATABASE_FP = "data/song_features.txt"

# Variables
database = {}


def retrieve_database(filepath=DATABASE_FP):
    """
    Retrieves database dictionary from filepath.
    """
    with open(filepath, "rb") as f:
        db = pickle.load(f)
    return db


def write_database(filepath=DATABASE_FP):
    """
    Writes database dictionary to filepath.
    """
    with open(filepath, "wb") as f:
        pickle.dump(database, f)


def add_song(features, name, artist):
    """
    Adds song to database.
    :param features: List of feature tuples.
    :param name: String
    :param artist: String
    """
    song = (name, artist)
    for feat in features:
        if feat in database:
            database[feat].append(song)
        else:
            database[feat] = [song]


def get_songs_from_db():
    """
    Retrieves song set from database.
    """
    songs = set()
    for feature in database.keys():
        for song in database[feature]:
            if song not in songs:
                songs.add(song)
    return songs


def display_songs():
    """
    Displays songs and artists from database.
    """
    for song in list(get_songs_from_db()):
        print(song[0], "by", song[1])


def remove_song(song):
    """
    Removes a song from the database and deletes empty features.
    :param song: tuple (name, artist)
    """
    for feature in database.keys():
        songs = database[feature]
        if song in songs:
            songs.remove(song)
    i = database.copy()
    for k, v in i.items():
        if v == []:
           del database[k]


def clear_database(password):
    """
    Clears database with two confirms.
    """
    if password.lower() == "yes i am sure":
        if input("Are you very sure?").lower() == "y":
            global database
            database = {}


def initialize():
    """Automatically retrieves database."""
    global database
    database = retrieve_database()

initialize()

