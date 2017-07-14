
# coding: utf-8

# In[24]:

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


# In[25]:

def find_peaks(song):
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
    neighborhood = iterate_structure(struct, 20)  # this incorporates roughly 20 nearest neighbors
    #finds foreground
    ys, xs = np.histogram(S.flatten(), bins=len(freqs)//2, normed=True)
    dx = xs[-1] - xs[-2]
    cdf = np.cumsum(ys)*dx  # this gives you the cumulative distribution of amplitudes
    cutoff = xs[np.searchsorted(cdf, 0.77)]
    foreground =  (S >= cutoff)
    #generates boolean array of peaks that are both peaks and in the foreground
    peaks = np.logical_and((song == maximum_filter(song, footprint=neighborhood)), song == foreground)
    return peaks


# In[21]:

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
    indices = np.argwhere(peaks == True)
    comparisons = itertools.combinations(indices, 2)
    threshold = 15
    filtered = itertools.filterfalse(lambda x: abs(times[x[1][1]]- times[x[1][0]]) > threshold, comparisons)
    for x in filtered:
        song_fp_t.append((freqs[x[0][0]], freq[x[0][1]], abs(times[x[1][1]] - times[x[1][0]])), times[x[1][0]])
    return song_fp_t


# In[23]:

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
    for feat in sample_fp_t:
        if feat[0] in db: #feat[0] is the actual finger print of the form (f1,f2,delta t)
            match = db.get(feat[0])
            matches += tuple(match[0], round(match[1] - feat[1])) #feat[1] is the time at which the feature occurs
    return matches


# In[26]:

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


# In[20]:

#testing 
v = find_fingerprint(find_peaks(S))
v


# In[10]:

fp = generate_binary_structure(rank=2,connectivity=1)
fp = iterate_structure(fp, 10)


# In[11]:

fig, ax = plt.subplots()
ax.imshow(find_peaks(S, fp))


# In[18]:

len(v)

