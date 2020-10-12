from pylsl import StreamInlet, resolve_stream
import joblib
import scipy.signal
import numpy as np
import time
import mne
import serial
import time

from features.features import create_features_matrix
from features.utils import SAMPLING_FREQ

DURATION = 3
CYTON_SRATE = 250
CHANNELS = ['Cz', 'FC2', 'CP2', 'C4', 'FC6', 'CP6', 'T8']
T_MIN=0.
T_MAX=3.

def init_stream_inlet():
    # first resolve an EEG stream on the lab network
    print("looking for an EEG control stream...")
    streams = resolve_stream('type', 'EEG')
    return StreamInlet(streams[0])

def pull_epoch(inlet):
    epoch = []

    for _ in range(DURATION * CYTON_SRATE):
        sample, _ = inlet.pull_sample()
        epoch.append(sample)
    return epoch

def create_mne_epochs(epoch: np.array):
    epoch = np.array(scipy.signal.resample(epoch, DURATION * SAMPLING_FREQ)).transpose()
    epoch = np.delete(epoch, (-1), axis=0)
    info = mne.create_info(ch_names=CHANNELS, sfreq=SAMPLING_FREQ, ch_types=['eeg'] * len(CHANNELS))
    raw = mne.io.RawArray(epoch, info, verbose=False)
    return mne.make_fixed_length_epochs(raw, duration=DURATION, preload=True, verbose=False)

def main():
    pipeline = joblib.load('../saved_models/svm.joblib')
    stream_inlet = init_stream_inlet()
    ser = serial.Serial('COM4', 9600)

    while True: # Wait and process epoch per epoch
        epoch = pull_epoch(stream_inlet)
        epochs = create_mne_epochs(epoch)
        X = create_features_matrix(epochs)
        Y = pipeline.predict(X)
        ser.write(b'H') if Y else ser.write(b'L')

main()
