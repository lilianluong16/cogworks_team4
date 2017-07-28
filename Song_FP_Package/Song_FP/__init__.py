import itertools
import collections
import pickle
import librosa
from os import path, listdir
from os.path import isfile, join
from pathlib import Path
from microphone import record_audio
import matplotlib.mlab as mlab
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.morphology import iterate_structure

_path = Path(path.dirname(path.abspath(__file__)))
DATABASE_FP = "data/song_features.txt"

__all__ = ["Song", 'record' , 'spectrogram', 'get_recording', 'find_peaks', 'find_fingerprint', 'get_matches', 'best_match', 
'train_single', 'train', 'identify', 'retrieve_database', 'write_database', 'add_song', 'get_songs_from_db', 
'retrieve_song_features', 'display_songs', 'get_song_object', 'remove_song', 'clear_database', 'initialize']

class Song:
    """This Song class is useful for storing songs in our database as objects and allows for quick retrieval of the song's name,
    artist and path


    NOTA BENE: When creating an instance of the Song class, an 'r' must be placed in front of the song path, which is the third parameter
    in the initialization function
    """
    def __init__(self,nm,artst,sng_pth=None):
        self.name = nm
        self.artist = artst
        self.song_path = sng_pth
    
    def __repr__(self):
        return self.name
    
    def show_Artist(self):
        return self.artist
    
    def show_Song_Path(self):
        return self.song_path


def record(seconds):
    """
    Records a period of sound from the microphone and returns a list of samples.
    :param seconds: Int, seconds to record for
    :return: 1-D numpy array of samples
    """
    byte_encoded_signal, sampling_rate = record_audio(seconds)
    sample = np.hstack([np.fromstring(i, dtype=np.int16) for i in byte_encoded_signal])
    return sample

def spectrogram(sample):
    """
    Creates spectrogram of a sample.
    :param sample: 1-D numpy array of samples
    :return: tuple (2-D numpy array of spectrogram indexes, 1-D array of real freq values, 1-D real time values)
    """

    S, freqs, times = mlab.specgram(sample, NFFT=4096, Fs=44100,
                                     window=mlab.window_hanning,
                                     noverlap=(4096 // 2))
    return S, freqs, times

def get_recording(seconds):
    """
    Gets a recording and its spectrogram.
    :param seconds: Int, seconds to record for
    :return: tuple (2-D numpy array of spectrogram indexes, 1-D array of real freq values, 1-D real time values)
    """
    ss = record(seconds)
    return spectrogram(ss)

def find_peaks(song, freqs):
    """
    Find the peaks in the two-dimensional array that describes a song
    Parameters:
    ----------
    song: numpy.ndarray (MxN)
        the two dimensional array of Fourier-constants describing the song
        song[i,j] is the magnitude of the Fourier-constant for frequency i at time j
    Returns:
    --------
    peaks: binary array (MxN)
        the binaray "mask" that identifies the locations of peaks
        peaks[i,j] is True if there is a local peak for frequency i at time j 
    """
    #generates proper neighborhood
    struct = generate_binary_structure(2, 1)
    neighborhood = iterate_structure(struct, 25)  # this incorporates roughly 20 nearest neighbors
    #finds foreground
    ys, xs = np.histogram(song.flatten(), bins=len(freqs)//2, normed=True)
    dx = xs[-1] - xs[-2]
    cdf = np.cumsum(ys)*dx  # this gives you the cumulative distribution of amplitudes
    cutoff = xs[np.searchsorted(cdf, 0.77)]
    foreground = (song >= cutoff)
    #generates boolean array of peaks that are both peaks and in the foreground
    peaks = np.logical_and((song == maximum_filter(song, footprint=neighborhood)), foreground)
    return peaks

def find_fingerprint(peaks, freqs, times):
    """
    Find the features (which are each a tuple of two peaks and the distance between them) of a song based on its peaks
    Parameters:
    ----------
    peaks: binary array (MxN)
        the binary "mask" that identifies the locations of peaks
        peaks[i,j] is True if there is a local peak for frequency i at time j
    freqs: float array (MxN)
        the array in which freqs[k] is the real value of the frequency value in bin k
    times: float array (MxN)
        the array in which time[k] is the real value of the time value in bin k
    Returns:
    --------
    song_fp: list of tuples (arbitrary length, all peaks in the song)
        the list of of tuples tuples of length three, each containing with two peaks and the distance between the two peaks
        of the form ((f1,f2,delta t), t1)
    """
    song_fp_t = []
    indices = np.argwhere(peaks == True)[::-1]
    comparisons = itertools.combinations(indices, 2)
    threshold = 30
    filtered = itertools.filterfalse(lambda x: abs(x[1][1] - x[0][1]) > threshold, comparisons)
    for (f1, t1), (f2, t2) in filtered:
        song_fp_t.append(tuple([tuple([int(freqs[f1]), int(freqs[f2]), int(abs(t2 - t1))]),
                                int(t1)]))
    return song_fp_t


def get_matches(sample_fp_t, db):
    """
    Find the features (which are each a tuple of two peaks and the distance between them) of a song based on its peaks
    Parameters:
    ----------
    sample_fp: list of tuples (arbitrary length, all peaks in the sample)
        the list of tuples of length three, each containing with two peaks and the distance between the two peaks 
    db: dictionary
        the dictionary with features as keys and song names as values
    Returns:
    --------
    matches: list of tuples of song ids and time differences 
        the list of song ids in the database that share features with the supplied sample
        and the amount of time between the feature occuring in the sample and in the 
    """
    matches = []
    for feature, time in sample_fp_t:
        if feature in db: #feat[0] is the actual finger print of the form (f1,f2,delta t)
            match = db.get(feature)
            matches += [(match[x][0], int(match[x][1] - time)) for x in np.arange(len(match)) if int(match[x][1] - time) >= 0] #feat[1] is the time at which the feature occurs
    return matches


def best_match(matches, displayc=False):
    """
    Find the features (which are each a tuple of two peaks and the distance between them) of a song based on its peaks
    Parameters:
    ----------
    matches: list of song names
        the list of song names in the database that share features with the supplied sample    Returns:
    --------
    best_match: song name
        the song name that occurs the most frequently in the list
    """
    if len(matches) < 1:
        return None
    c = collections.Counter([x for x in matches])
    if displayc:
        print(c)
        print(c.most_common(1))
    threshold = 15
    if c.get(c.most_common(1)[0][0]) < threshold:
        return None
    return c.most_common(1)[0][0][0]

def mp3_to_samparr(songpath):    # WORKS
    
    """
    ACCEPTS: a single string that represents the song's file path (songpath).
    
    This function samples the song data, records the sampling rate (fs), determines the total length of the audio (T) and
    splits up the total duration into equal intervals (time_arr)
    
    NOTA BENE: This function assumes the songpath string is in the following format:
    
    'C:\\Users\\Bunch of Directory Stuff\\Name_Artist.mp3'
    
    
    RETURN:
        
    """

    samples, fs = librosa.load(songpath, sr=44100, mono=True)
    N = len(samples) # N represents the number of samples
    T = N/fs   # N divided by the sampling rate (fs) gives you the total duration in seconds
    time_arr = np.linspace(0,T,N) # time represents a numpy array of equally spaced time values
    all_list = [samples,N,T,time_arr]
    
    return samples


def convert_files_to_songpaths(directory_name):          # WORKS
    """
    ACCEPTS: a single string that represents the directory which contains ALL of your songs.

    SIDE NOTE: The songs in this directory are in the following format:
        Name_Artist.mp3
    
    RETURNS: A list of strings that represent the file paths of each song
    """
    onlyfiles = [f for f in listdir(directory_name) if isfile(join(directory_name, f))]
    str_of_songpaths = []

    for i in range(len(onlyfiles)):
        str_of_songpaths.append(directory_name + '\\' +onlyfiles[i])

    return str_of_songpaths
        
def songpath_to_name(songpath):         # WORKS
    """ACCEPTS: a single string that is a songpath and manipulates it 
    
    NOTE: This function assumes the songpath string is in the following format:
    
    'C:\\Users\\Bunch of Directory Stuff\\Name_Artist.mp3'
    
    RETURNS: The name of the song
    
    """


    the_first = songpath.rfind('\\')
    the_second = songpath.rfind('_')
    
    name = songpath[the_first+1:the_second]
    return name


def songpath_to_artist(songpath):         # WORKS
    """ACCEPTS: a single string that is a songpath and manipulates it 
    
    NOTE: This function assumes the songpath string is in the following format:
    
    'C:\\Users\\Bunch of Directory Stuff\\Name_Artist.mp3'
    
    RETURNS: The artist of the song
    
    """

    the_first = songpath.rfind('_')
    the_second = songpath.rfind('.')

    artist = songpath[the_first+1:the_second]
    return artist

def train_single(filepath):
    s_i = mp3_to_samparr(filepath)
    name = songpath_to_name(filepath)
    artist = songpath_to_artist(filepath)
    spectro_i = spectrogram(s_i)
    peaks_i = find_peaks(spectro_i[0], spectro_i[1])
    features = find_fingerprint(peaks_i, spectro_i[1], spectro_i[2])
    return name, artist, features


def train(folder="audio"):
    filepaths = convert_files_to_songpaths(folder)
    # load songs
    for i in filepaths:
        name, artist, features = train_single(i)
        print("Adding:", name, "by", artist)
        add_song(features, name, artist)
        print("Added:", name, "by", artist)
        write_database()


def identify():
    song = record(7)
    spectro_i = spectrogram(song)
    peaks_i = find_peaks(spectro_i[0], spectro_i[1])
    features = find_fingerprint(peaks_i, spectro_i[1], spectro_i[2])
    matches = get_matches(features, database)
    good_match = best_match(matches)
    if good_match is not None:
         return " ".join([good_match.name, "by", good_match.artist])
    return None

def retrieve_database(filepath=DATABASE_FP):
    """
    Retrieves database dictionary from filepath.
    :param filepath: filepath of database
    :return: dictionary of database
    """
    with open(filepath, "rb") as f:
        db = pickle.load(f)
    return db


def write_database(filepath=DATABASE_FP):
    """
    Writes database dictionary to filepath.
    :param filepath: filepath of database
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
    song = song_class.Song(name, artist)
    for feat in features:
        if feat[0] in database:
            database[feat[0]].append((song, feat[1]))
        else:
            database[feat[0]] = [(song, feat[1])]


def get_songs_from_db():
    """
    Retrieves song set from database.
    :return: List of songs
    """
    songs = set()
    for feature in database.keys():
        for song in database[feature]:
            if song[0] not in songs:
                songs.add(song[0])
    return songs


def retrieve_song_features(song_name):
    feats = set()
    for feature in database.keys():
        for song in database[feature]:
            if song[0].name == song_name:
                if feature not in feats:
                    feats.add(feature)
                break
    return feats


def display_songs():
    """
    Displays songs and artists from database.
    """
    for song in list(get_songs_from_db()):
        print(song.name, "by", song.artist)


def get_song_object(song_name):
    song_feature = list(retrieve_song_features(song_name))[0]
    print(song_feature)
    return [i[0]
            for i in database[song_feature]
            if i[0].name == song_name][0]


def remove_song(song):
    for feature in database.keys():
        song_time_tup = database[feature]

        zipped = zip(*song_time_tup)
        combined = list(zipped)
        songs = list(combined[0])
        times = list(combined[1])

        while song in songs:
            ind = songs.index(song)
            songs.remove(song)
            times.remove(times[ind])

        updated = list(zip(songs, times))
        database[feature] = updated

    i = database.copy()
    for k, v in i.items():
        if v == []:
            del database[k]


def clear_database(password):
    """
    Clears database with two confirms. Enter 'y' to confirm when prompted.
    Remember to write to database afterwards.
    :param password: String ("yes i am sure")
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