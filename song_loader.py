import librosa
import numpy as np
from os import listdir
from os.path import isfile, join

def mp3_to_samparr(songpath):    # WORKS
    
    """
    ACCEPTS: a single string that represents the song's file path (songpath).
    
    This function samples the song data, records the sampling rate (fs), determines the total length of the audio (T) and
    splits up the total duration into equal intervals (time_arr)
    
    NOTE: This function assumes the songpath string is in the following format:
    
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


def train_database(songpaths_arr):
    """Accept the array of all song paths and register then in the database
    
        New Song = Song(songpath_to_name(songpaths_arr[0]),songpath_to_artist(songpaths_arr[0]),songpaths_arr[0])
    """
    pass
        

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


