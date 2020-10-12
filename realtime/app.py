from pylsl import StreamInlet, resolve_stream
from features.features import create_features_matrix
from features.utils import SAMPLING_FREQ
import joblib
import numpy as np
import serial
import time

DURATION = 3

def init_stream_inlet():
    # first resolve an EEG stream on the lab network
    print("looking for an EEG control stream...")
    streams = resolve_stream('type', 'EEG')
    return StreamInlet(streams[0])

def pull_epoch(inlet):
    epoch = []

    for _ in range(DURATION * SAMPLING_FREQ):
        sample, _ = inlet.pull_sample()
        epoch.append(sample)
    return epoch

def main():
    stream_inlet = init_stream_inlet()

    pipeline = joblib.load('../saved_models/svm.joblib')    
    epochs = joblib.load('../saved_models/epoch.joblib')
    while True: # Wait and process epoch per epoch

        epoch = pull_epoch(stream_inlet)

        print(epoch)
        
        # TODO: create a mne.Epochs object that contains only one epoch with the data received by LSL (with same sampling freq)
        s = time.time()
        X = create_features_matrix(epochs)

        s = time.time()
        Y = pipeline.predict(X)
        print(f'{time.time()-s}')
        print(Y)
        # TODO: Move motor with Y's result

main()
