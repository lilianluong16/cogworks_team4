# This file contains functions for:
# song fingerprinting (finding peaks)
# fingerprint comparision (comparing peaks to those looked up in the database_
# determining the best match for a song sample
# determining whether the best match is sufficient to identify the song

import numpy as np
import itertools
import collections
# the imports below could be removed if you didn't wanna visualize things!
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.morphology import iterate_structure


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


# In[82]:

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
    threshold = 15
    filtered = itertools.filterfalse(lambda x: abs(x[1][1] - x[1][0]) > threshold, comparisons)
    for (f1, t1), (f2, t2) in filtered:
        song_fp_t.append(tuple([tuple([round(freqs[f1], 2), round(freqs[f2], 2), round(abs(times[t2] - times[t1]), 2)]),
                                round(times[t1], 2)]))
    print(len(song_fp_t))
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
            matches += tuple(match[0], round(match[1] - time)) #feat[1] is the time at which the feature occurs
    return matches


def best_match(matches):
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
    c = Counter(x[0] for x in matches)
    best_matches = c.most_common(2)
    threshold = 20
    if c.get(best_matches[0]) - c.get(best_matches[1]) < threshold:
        return "Not found"
    return best_matches[0]

