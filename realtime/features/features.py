import numpy as np
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from scipy.stats import skew, kurtosis

from features.frequency_domain import get_psds_from_epochs, get_mean_psds
from features.time_domain import get_transformer, get_signal_mean_energy
from features.utils import (
    drop_channels,
    create_kernels,
    FREQ_BANDS,
    NYQUIST_FREQ,
    ORDERS,
)

def _get_data_from_epochs(epochs):
    """
    epochs: mne.Epochs
    
    returns np array of shape (nb_epochs, sampling_rate*epoch_length)
    """
    return epochs.get_data().squeeze().reshape(1, -1)

def _get_epochs_per_subband(subband, kernels, time_domain_feature_union):
    freq_range = FREQ_BANDS[subband]

    def filter_freq_band_on_epochs(epochs):
        l_freq = freq_range[0]/NYQUIST_FREQ
        h_freq = freq_range[1]/NYQUIST_FREQ
        b = kernels[subband][0]
        a = kernels[subband][1]
        return epochs.copy().filter(l_freq=l_freq, h_freq=h_freq, method='iir', n_jobs=-1, iir_params = {'order': ORDERS[subband], 'ftype': 'butter','a': a,'b': b}, verbose=False)

    return Pipeline([
        ('filter', FunctionTransformer(filter_freq_band_on_epochs, validate=False)),
        ('epochs_to_data', FunctionTransformer(_get_data_from_epochs, validate=False)),
        ('time_domain_features', time_domain_feature_union)
    ])

def _create_feature_extraction_pipeline():
    frequency_domain_feature_union = FeatureUnion([
        ('absolute_mean_power_band', FunctionTransformer(get_mean_psds, validate=False)),
        ('relative_mean_power_band', FunctionTransformer(lambda psds_with_freq: get_mean_psds(psds_with_freq, are_relative=True), validate=False)),
    ], n_jobs=1)

    frequency_domain_pipeline = Pipeline([
        ('get_psds_from_epochs', FunctionTransformer(get_psds_from_epochs, validate=False)),
        ('frequency_domain_features', frequency_domain_feature_union)
    ])

    time_domain_feature_union = FeatureUnion([
        ('mean', FunctionTransformer(get_transformer(np.mean), validate=True)),
        ('std', FunctionTransformer(get_transformer(np.std), validate=True)),
        ('skew', FunctionTransformer(get_transformer(skew), validate=True)),
        ('kurtosis', FunctionTransformer(get_transformer(kurtosis), validate=True)),
        ('mean_energy', FunctionTransformer(get_transformer(get_signal_mean_energy), validate=True)),
    ], n_jobs=1)

    kernels = create_kernels()
    feature_union = FeatureUnion([
        ('frequency_domain', frequency_domain_pipeline),
        ('subband_feature_union', FeatureUnion([(f'subband_{subband}', _get_epochs_per_subband(subband, kernels, time_domain_feature_union)) for subband in FREQ_BANDS.keys()], n_jobs=1))
    ], n_jobs=1)

    return feature_union

def create_features_matrix(epochs):
    feature_extraction_pipeline = _create_feature_extraction_pipeline()

    X = []
    for chan in epochs.info['ch_names']:
        chan_epoch = drop_channels(epochs, chan)
        X.append(feature_extraction_pipeline.transform(chan_epoch))
    
    return np.hstack(X)
