# Imports
from microphone import record_audio

import matplotlib.mlab as mlab
<<<<<<< HEAD
=======
import matplotlib.pyplot as plt
>>>>>>> 11aed466ba4d544526f86b70d50bc6df1faa4f54

import numpy as np


def record(seconds):
    """
    Records a period of sound from the microphone and returns a list of samples.
    :param seconds: Int, seconds to record for
    :return: 1-D numpy array of samples
    """
    byte_encoded_signal, sampling_rate = record_audio(seconds)
    sample = np.hstack(np.fromstring(byte_encoded_signal, dtype=np.int16))
    return sample


def spectrogram(sample):
    """
    Creates spectrogram of a sample.
    :param sample: 1-D numpy array of samples
    :return: tuple (2-D numpy array of spectrogram indexes, 1-D array of real freq values, 1-D real time values)
    """
<<<<<<< HEAD

    S, freqs, times = mlab.specgram(sample, NFFT=4096, Fs=44100,
                                     window=mlab.window_hanning,
                                     noverlap=(4096 // 2))
=======
    fig, ax = plt.subplots()
    S, freqs, times, im = ax.specgram(sample, NFFT=4096, Fs=44100,
                                     window=mlab.window_hanning,
                                     noverlap=(4096 // 2))
    plt.close()
>>>>>>> 11aed466ba4d544526f86b70d50bc6df1faa4f54
    return S, freqs, times


def get_recording(seconds):
    """
    Gets a recording and its spectrogram.
    :param seconds: Int, seconds to record for
    :return: tuple (2-D numpy array of spectrogram indexes, 1-D array of real freq values, 1-D real time values)
    """
    ss = record(seconds)
    return spectrogram(ss)