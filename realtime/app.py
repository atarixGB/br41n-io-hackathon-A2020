import joblib
import numpy as np

from features.features import create_features_matrix


def main():    
    pipeline = joblib.load('../saved_models/svm.joblib')    
    epochs = joblib.load('../saved_models/epoch.joblib')
    #while True: # Wait and process epoch per epoch
    # TODO: receive LSL data and process one epoch
    # TODO: create a mne.Epochs object that contains only one epoch with the data received by LSL (with same sampling freq)
    import time
    s = time.time()
    X = create_features_matrix(epochs)

    s = time.time()
    Y = pipeline.predict(X)
    print(f'{time.time()-s}')
    print(Y)
    # TODO: Move motor with Y's result

main()
