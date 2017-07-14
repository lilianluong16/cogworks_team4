# This file contains functions for:
# song fingerprinting (finding peaks)
# fingerprint comparision (comparing peaks to those looked up in the database_
# determining the best match for a song sample
# determining whether the best match is sufficient to identify the song

import numpy as np
import itertools
# the imports below could be removed if you didn't wanna visualize things!
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.morphology import iterate_structure

def find_peaks(song, neighborhood=iterate_structure(generate_binary_structure(rank=2,connectivity=1), 30)):
    """
    Find the peaks in the two-dimensional array that describes a song
    Parameters:
    ----------
    song: numpy.ndarray (MxN)
        the two dimensional array of Fourier-constants describing the song
        song[i,j] is the magnitude of the Fourier-constant for frequency i at time j
    neighborhood: boolean array
        the footprint being used to define the scope in which a point must be the maximum to be considered a peak
        (has default)
    Returns:
    --------
    peaks: binary array (MxN)
        the binaray "mask" that identifies the locations of peaks
        peaks[i,j] is True if there is a local peak for frequency i at time j 
    """
    fp = generate_binary_structure(rank=2,connectivity=1)
    fp = iterate_structure(fp, 10)
    peaks = np.logical_and((song == maximum_filter(song, footprint=neighborhood)), song > 0)
    return peaks


def compare_features(a_frq, a_time, b_frq, b_time):
    """
    Compares two peaks for storage as part of the song's fingerprint
    Parameters:
    ----------
    a_frq: float
        the frequency at which peak a occurs    
    a_time: float
        the time at which peak a occurs
    b_frq: float
        the frequency at which peak b occurs    
    b_time: float
        the time at which peak b occurs
    Returns:
    --------
    r: tuple (of length 3)
        a tuple containing the frequency of a, the frequncy of b, and the absolute value of difference between their respective times
    """
    return (a_frq, b_frq, abs(a_time - b_time))


def find_fingerprint(peaks):
    """
    Find the features (which are each a tuple of two peaks and the distance between them) of a song based on its peaks
    Parameters:
    ----------
    peaks: binary array (MxN)
        the binary "mask" that identifies the locations of peaks
        peaks[i,j] is True if there is a local peak for frequency i at time j     
    Returns:
    --------
    song_fp: list of tuples (arbitrary length, all peaks in the song)
        the array of tuples of length three, each containing with two peaks and the distance between the two peaks
    """
    song_fp = []
    indices = np.argwhere(peaks == True)
    comparisons = itertools.combinations(indices, 2)
    threshold = 7
    filtered = itertools.filterfalse(lambda x: abs(x[1][1] - x[1][0]) > threshold, comparisons)
    for x in filtered:
        song_fp.append((x[0][0], x[0][1], abs(x[1][1] - x[1][0])))
    return song_fp[::2]