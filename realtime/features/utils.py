import scipy.signal

SAMPLING_FREQ = 160
NYQUIST_FREQ = SAMPLING_FREQ/2
WOR_N = 2056
FREQ_BANDS = {
    'alpha': [7, 11.5],
    'sigma': [11.5, 15.5],
    'beta': [15.5, 30],
}
ORDERS = {
    'alpha': 7,
    'sigma': 8,
    'beta': 14,
}

def drop_channels(epochs, chan_to_keep):
    return epochs.copy().drop_channels([chan for chan in epochs.info['ch_names'] if chan != chan_to_keep])

def create_kernels():
    kernels = dict.fromkeys(FREQ_BANDS.keys())
    for key, freq_range in FREQ_BANDS.items():
        lower_bound = freq_range[0]/NYQUIST_FREQ
        upper_bound = freq_range[1]/NYQUIST_FREQ
        b, a = scipy.signal.butter(ORDERS[key], [lower_bound, upper_bound], btype='bandpass')
        w, h = scipy.signal.freqz(b, a, worN=WOR_N)
        kernels[key] = (b, a)

    return kernels
