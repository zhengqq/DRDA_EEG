import scipy.signal as scipy_signal
import scipy.io as sio
import numpy as np
from scipy.signal import resample
from scipy.signal import filtfilt
from scipy.signal import stft
from scipy import interpolate
from sklearn import preprocessing
import warnings
import pandas as pd
import random
import matplotlib.pyplot as plt
warnings.simplefilter(action='ignore', category=FutureWarning)


class loadData:
	def __init__(self, path, s=1.5, e=5.5, w=1, sl=7):
		self.raw_freq = 1000
		self.freq = 1000
		# self.band = [[4,8], [8,12], [12,16], [16,20], [20,24], [24,28], [28,32]]
		self.band = [[4,38]]
		# self.band = []
		# channel_index = list(range(17, 24)) + list(range(26, 33)) + list(range(35, 41))
		self.channel = list(range(17, 24)) + list(range(26, 33)) + list(range(35, 41))
		# self.channel = list(range(60))

		self.BBAA = []
		for band_pass in self.band:
			self.BBAA.append(scipy_signal.butter(3, [i * 2 / self.freq for i in band_pass], btype='bandpass'))

		self.start = int(s * self.raw_freq)
		self.end = int(e * self.raw_freq) #5.5
		self.window = int(w * self.raw_freq)
		self.slide = int(sl * self.raw_freq)

		self.dataPath = path
		self.fold = 40

	def signalCut(self, idx, random_pick=False):
		# random_pick = True

		raw = sio.loadmat(self.dataPath)['EEG']
		eeg_data = raw['data'][0, 0]
		eeg_event = raw['event'][0, 0]

		data = []
		label = []
		idd = 0
		flag = []

		for i in range(len(eeg_event)):
			eeg_type = int(eeg_event['type'][i][0])
			if eeg_type < 4:

				if len(idx):
					if idd not in idx:
						idd += 1
						continue
					else:
						idd += 1

				latency = eeg_event['latency'][i][0][0][0]
				start = self.start
				end = self.end
				slide = self.slide
				window = self.window
				left = start - 1
				right = start + window - 1
				fold_ite = 0
				fold = self.fold

				if random_pick is False:

					while right <= end:
						try:
							tdata = eeg_data[:, latency + left: latency + right]
						except:
							# print(latency)
							# print(int(round(latency)))
							tdata = eeg_data[:, int(round(latency)) + left: int(round(latency)) + right]

						label.append(int(eeg_type))
						data.append(tdata)
						left += slide
						right += slide

				else:
					tdata = eeg_data[:, int(round(latency)) + left: int(round(latency)) + right]
					data.append(tdata)
					label.append(int(eeg_type))
					flag.append(1)
					for fold_ite in range(fold - 1):
						rdn = np.random.random()
						rdn = rdn * 3.0 + 1.5
						left_rdn = int(rdn * self.raw_freq)
						right_rdn = left_rdn + window
						tdata = eeg_data[:, int(round(latency)) + left_rdn: int(round(latency)) + right_rdn]
						data.append(tdata)
						label.append(int(eeg_type))
						flag.append(0)


		data = np.stack(data, axis=0)
		label = np.stack(label, axis=0)
		data = data[:,self.channel,:]
		if random_pick:
			flag = np.stack(flag, axis=0)

		# data = (data - data.mean(axis=2, keepdims=True))

		return data, label

	def signalProcess(self, _data, dsFlag=False):

		##--- mean value subtraction and resample
		# data = (_data - _data.mean(axis=2, keepdims=True)) #/ (_data.std(axis=2, keepdims=True))

		if dsFlag:
			_data = resample(_data, int(_data.shape[2] / self.raw_freq * self.freq), axis=2)

		data = []
		for i in range(_data.shape[0]):
			# try:
			# 	dataFeed = _data[i]

			data.append(self.exponential_running_standardize(_data[i, :, :, 0].T, factor_new=0.01).T)

		data = np.stack(data, axis=0)



		return data

	def exponential_running_standardize(self, data, factor_new=0.001,
										init_block_size=None, eps=1e-4):
		"""
        Perform exponential running standardization.

        Compute the exponental running mean :math:`m_t` at time `t` as
        :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot m_{t-1}`.

        Then, compute exponential running variance :math:`v_t` at time `t` as
        :math:`v_t=\mathrm{factornew} \cdot (m_t - x_t)^2 + (1 - \mathrm{factornew}) \cdot v_{t-1}`.

        Finally, standardize the data point :math:`x_t` at time `t` as:
        :math:`x'_t=(x_t - m_t) / max(\sqrt{v_t}, eps)`.


        Parameters
        ----------
        data: 2darray (time, channels)
        factor_new: float
        init_block_size: int
            Standardize data before to this index with regular standardization.
        eps: float
            Stabilizer for division by zero variance.

        Returns
        -------
        standardized: 2darray (time, channels)
            Standardized data.
        """
		df = pd.DataFrame(data)
		meaned = df.ewm(alpha=factor_new).mean()
		demeaned = df - meaned
		squared = demeaned * demeaned
		square_ewmed = squared.ewm(alpha=factor_new).mean()
		standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
		standardized = np.array(standardized)
		if init_block_size is not None:
			other_axis = tuple(range(1, len(data.shape)))
			init_mean = np.mean(data[0:init_block_size], axis=other_axis,
								keepdims=True)
			init_std = np.std(data[0:init_block_size], axis=other_axis,
							  keepdims=True)
			init_block_standardized = (data[0:init_block_size] - init_mean) / \
									  np.maximum(eps, init_std)
			standardized[0:init_block_size] = init_block_standardized
		return standardized


	def signalFilter(self, _data):
		data = []
		for BBAA in self.BBAA:
			data.append(filtfilt(BBAA[0], BBAA[1], _data, axis=-1))

		if data:
			return np.stack(data, axis=-1)
		else:
			return np.expand_dims(_data, axis=-1)

	def signalNorm(self, _data):
		data = []
		for trail in range(len(_data)):
			data_channel = []
			for channel in range(_data.shape[-1]):
				signal = _data[trail,:,:,channel]
				# data = preprocessing.scale(signal, axis=-1)
				data = preprocessing.minmax_scale(signal, (0, 1), axis=-1)
				# data_channel.append(scale_data)
				data_channel.append(data)

			data.append(np.stack(data_channel, axis=-1))
		dataNorm = np.stack(data, axis =0)

		return dataNorm

	def stft_process(self, _data, mu=(8, 15), beta=(18, 24), wlen=64, nfft=512, fs=250, hop=14, rescale=False):

		f, t, Fstft = stft(_data, fs=fs, window='hamming', nperseg=wlen, noverlap=wlen - hop,
		                   nfft=nfft, return_onesided=True, boundary=None, padded=False)

		mu_left = np.where(f >= mu[0])[0][0]
		mu_right = np.where(f >= mu[1])[0][0]
		beta_left = np.where(f >= beta[0])[0][0]
		beta_right = np.where(f >= beta[1])[0][0]

		stft_image = None
		for i in range(Fstft.shape[0]):
			mu_feature_matrix = np.abs(Fstft[i, mu_left: mu_right, :])
			beta_feature_matrix = np.abs(Fstft[i, beta_left: beta_right, :])

			beta_interp = interpolate.interp2d(t, f[beta_left: beta_right], beta_feature_matrix, kind='cubic')
			interNum = len(mu_feature_matrix)
			f_beta = np.arange(beta[0], beta[1], (beta[1] - beta[0]) / (interNum))
			beta_feature_matrix = beta_interp(t, f_beta)

			if stft_image is None:
				stft_image = np.append(mu_feature_matrix, beta_feature_matrix, axis=0)
				# stft_image = (stft_image - stft_image.min(keepdims=True)) / (
				# 			stft_image.max(keepdims=True) - stft_image.min(keepdims=True))
			else:
				stft_image1 = np.append(mu_feature_matrix, beta_feature_matrix, axis=0)
				# stft_image1 = (stft_image1 - stft_image1.min(keepdims=True)) / (
				# 			stft_image1.max(keepdims=True) - stft_image1.min(keepdims=True))
				stft_image = np.dstack((stft_image, stft_image1))

		stft_image = np.transpose(stft_image, [2, 0, 1])

		return stft_image

	def signalFFT(self,_data):
		from scipy import fftpack
		import matplotlib.pyplot as plt
		N = _data.shape[2]
		F = fftpack.fft(_data, axis=2)
		f = fftpack.fftfreq(N, 1.0/self.freq)
		mask = np.where(f>=0)

		# A = F[0,0,:,0]
		# plt.plot(f[mask], np.abs(A[mask])/N, label='real')
		# plt.xlim(4,38)
		# plt.xlabel("frequency (Hz)", fontsize=14)
		# plt.ylabel("$|F|$", fontsize=14)
		# plt.show()

		return f[mask], np.abs(F[:,:,mask[0].tolist()] / N)

	def theta_alpha_beta_averages(self,f, Y):
		theta_range = (4, 8)
		alpha_range = (8, 12)
		beta_range = (12, 40)
		theta = Y[(f > theta_range[0]) & (f <= theta_range[1])].mean()
		alpha = Y[(f > alpha_range[0]) & (f <= alpha_range[1])].mean()
		beta = Y[(f > beta_range[0]) & (f <= beta_range[1])].mean()
		return theta, alpha, beta

	def psd(self, _data, mu=(8,15), beta=(18,24)):
		freqs, psd = scipy_signal.welch(_data, fs=self.freq)
		mu_band = np.mean(psd[:, (freqs>mu[0])&(freqs<mu[1])], axis=-1, keepdims=False)
		beta_band = np.mean(psd[:, (freqs>beta[0])&(freqs<beta[1])], axis=-1, keepdims=False)

		return mu_band, beta_band


	def augment(self, _data, mu=0, sigma=0.02):

		noise = np.random.normal(mu, sigma, _data.shape)
		data = _data + noise

		return data



if __name__ == '__main__':

	for i in range(6):
		dataPath = './data/dataB/0' + str(i+1) + '/B0' + str(i+1) + '01T.mat'

		dataset = loadData(dataPath)
		data, label = dataset.signalCut()
		data = dataset.signalFilter(data)
		data = dataset.signalProcess(data, dsFlag=True)

		freq, mag = dataset.signalFFT(data)

		theta, alpha, beta = np.zeros([mag.shape[0],mag.shape[1]]), np.zeros([mag.shape[0],mag.shape[1]]), np.zeros([mag.shape[0],mag.shape[1]])
		for ii in range(mag.shape[0]):
			for jj in range(mag.shape[1]):
				mag1 = mag[ii,jj,:,0]
				theta[ii,jj], alpha[ii,jj], beta[ii,jj] = dataset.theta_alpha_beta_averages(freq, mag1)

		data_ = np.stack((theta, alpha, beta), axis=-1)
		data_ = []


		for j in range(len(data)):
			data1 = dataset.stft_process(data[j])
			data_.append(data1)
		data = np.stack(data_, axis=0)

		data = dataset.signalFilter(data)
		data = dataset.signalNorm(data)






