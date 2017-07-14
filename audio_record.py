from microphone import record_audio

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import numpy as np

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.morphology import iterate_structure


def record(seconds):
    byte_encoded_signal, sampling_rate = record_audio(seconds)
    sample = np.fromstring(byte_encoded_signal, dtype=np.int16)
    return sample


def local_peaks(data):
    fp = generate_binary_structure(rank=2, connectivity=1)
    fp = iterate_structure(fp, 3)
    local_maxima = maximum_filter(data, footprint=fp)
    peaks = local_maxima == data
    return np.logical_and(peaks, local_maxima > 0)


def plot_compare(data, peak_finding_function):
    fig, ax = plt.subplots()
    peaks = peak_finding_function(data)
    ax.imshow(peaks)
    return peaks, data, fig, ax


def spectrogram(sample):
    fig, ax = plt.subplots()
    S, freqs, times, im = ax.specgram(sample, NFFT=4096, Fs=44100,
                                     window=mlab.window_hanning,
                                     noverlap=(4096 // 2))
    return S


def get_recording(seconds):
    ss = record(seconds)
    return spectrogram(ss)