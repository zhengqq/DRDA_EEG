import scipy.signal as scipy_signal
import scipy.io as sio
import numpy as np
from scipy.signal import resample
from scipy.signal import filtfilt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

raw_freq = 1000
freq = 250

dataPath = 'D:/BCI/Data/BCI2019/supervised/mi-t.tar/01/B0101T.mat'

raw = sio.loadmat(dataPath)['EEG']
eeg_data = raw['data'][0, 0]
eeg_event = raw['event'][0, 0]

data = []
label = []
for i in range(len(eeg_event)):
	type = int(eeg_event['type'][i][0])
	if type < 4:
		latency = eeg_event['latency'][i][0][0][0]
		start = int(1.5 * raw_freq)
		end = int(5.5 * raw_freq)
		slide = int(0.5 * raw_freq)
		window = int(2 * raw_freq)
		left = start
		right = start + window

		while right <= end:
			tdata = eeg_data[:, latency + left: latency + right]
			label.append(int(type))
			data.append(tdata)
			left += slide
			right += slide

data = np.stack(data, axis=0)
label = np.stack(label, axis=0)

band = [[4, 8],
        [8, 12],
        [12, 16],
        [16, 20],
        [20, 24],
        [24, 28],
        [28, 32]]

BBAA = []

for band_pass in band:
	BBAA.append(scipy_signal.butter(2, [i * 2 / freq for i in band_pass], btype='bandpass'))


data = data - data.mean(axis=2, keepdims=True)
data = resample(data, int(data.shape[-1]/raw_freq*freq), axis=-1)

_data = []
for BA in BBAA:
	_data.append(filtfilt(BA[0], BA[1], data, axis=-1))

_data = np.stack(_data, axis=-1)
