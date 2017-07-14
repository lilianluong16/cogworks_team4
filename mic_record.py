
import pyaudio
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion 
from scipy.ndimage.morphology import iterate_structure

# Constants
FORMAT = pyaudio.paInt16
SF = 44100
FRAME_SIZE = 1024
NO_FRAMES = 220


def record(secs, format=FORMAT, sampling_frequency=SF, frame_size=FRAME_SIZE):
    nframes = secs / 5 * 220
    p = pyaudio.PyAudio()
    print('Recording for', secs, 'seconds')

    stream = p.open(format=format,
                    channels=1,
                    rate=sampling_frequency,
                    input=True,
                    frames_per_buffer=frame_size) # frames_per_buffer refers to a number of sample frames

    data = stream.read(nframes * frame_size)
    print('Done recording.')
    stream.stop_stream()
    stream.close()
    p.terminate()

    return np.fromstring(data, 'Int16')


def spectrogram(decoded):
    fig, ax = plt.subplots()
    S, freqs, times, im = ax.specgram(decoded, NFFT=4096, Fs=44100,
                                     window=mlab.window_hanning,
                                     noverlap=(4096 // 2))
    return S


def get_recording(seconds):
    samples = record(seconds)
    return spectrogram(samples), samples