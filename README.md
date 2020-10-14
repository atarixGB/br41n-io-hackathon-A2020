# Hand Orthosis controlled by motor Imagery

The main goal of this project was to check if we were able to control a left hand orthosis using right cortex motor imagery.

This code was written for the [BR41N.IO Toronto 2020 hackathon](https://www.br41n.io/Toronto-2020).

## Training dataset

We used the [EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) in order to train our classifier. For this, we only used the task 2 and kept only the T0 and T1 annotations.

## How it works

The [exploration.ipynb](https://github.com/atarixGB/br41n-io-hackathon-A2020/blob/main/exploration.ipynb) was used to perform preprocessing, feature extraction and create classification models. We then saved our models using the joblib library.

Then, using the [app.py](https://github.com/atarixGB/br41n-io-hackathon-A2020/blob/main/realtime/app.py) file, you can run our live prototype. This does the preprocessing, feature extraction and classification to send prediction to an arduino. The arduino then open and close the orthosis hand according to the prediction that was made.

## Test

The OpenBCI GUI was then used to stream Cyton data to our app using pyLSL. This [playback file](https://github.com/atarixGB/br41n-io-hackathon-A2020/blob/main/data/open_bci_raw.txt) was used to stream data.

## Going further

The accuracy of our classification was pretty low. To get a better control over the hand, we should work more on our feature extraction pipeline.
