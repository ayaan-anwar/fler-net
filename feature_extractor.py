import constants

import numpy as np
import antropy as ant
import random

from scipy.integrate import simps
from scipy import signal
from scipy.signal import welch
from sklearn.model_selection import train_test_split


class FeatureExtractor:
    def __init__(self, conf: dict) -> None:
        self.WINDOW_SECONDS = conf["WINDOW_SECONDS"]
        self.SAMPLING_FREQUENCY = conf["SAMPLING_FREQUENCY"]
        self.WINDOW_SIZE = self.WINDOW_SECONDS * self.SAMPLING_FREQUENCY
        self.DATA_LEN = conf["DATA_LEN_SECONDS"] * self.SAMPLING_FREQUENCY
        self.START_TIMESTAMP = conf["START_TIMESTAMP_SECONDS"] * self.SAMPLING_FREQUENCY
        self.NUM_CHANNELS = conf["NUM_CHANNELS"]
        self.NUM_SUBJECTS = conf["NUM_SUBJECTS"]
        self.NUM_SITUATIONS = conf["NUM_SITUATIONS"]
    
    def time_features(self, sample, NUM_CHANNELS, SAMPLING_FREQUENCY, WINDOW_SIZE):
        features_for_sample = []
        for i in range(NUM_CHANNELS):
            (h_mobility, h_complexity) = ant.hjorth_params(sample[i])

            higuchi_fd = ant.higuchi_fd(sample[i])
            petrosian = ant.petrosian_fd(sample[i])

            features = [h_mobility, h_complexity, higuchi_fd, petrosian]
            features_for_sample.extend(features)
        return features_for_sample
    
    def bandpower(self, data, sf, band, window_sec=None, relative=False):
        band = np.asarray(band)
        low, high = band

        # Define window length
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf

        # Compute the modified periodogram (Welch)
        freqs, psd = welch(data, sf, nperseg=nperseg)

        # Frequency resolution
        freq_res = freqs[1] - freqs[0]

        # Find closest indices of band in frequency vector
        idx_band = np.logical_and(freqs >= low, freqs <= high)

        # Integral approximation of the spectrum using Simpson's rule.
        bp = simps(psd[idx_band], dx=freq_res)

        if relative:
            bp /= simps(psd, dx=freq_res)
        return bp

    def frequency_features(self, sample, NUM_CHANNELS, SAMPLING_FREQUENCY, WINDOW_SIZE):
        features_for_sample = []
        for i in range(NUM_CHANNELS):
            spectral = ant.spectral_entropy(sample[i], sf=SAMPLING_FREQUENCY, method='welch', normalize=True)
            svd = ant.svd_entropy(sample[i], normalize=True)
            samp = ant.sample_entropy(sample[i])

            theta_bandpower = self.bandpower(sample[i], SAMPLING_FREQUENCY, [4, 8], window_sec=10, relative=True)
            alpha_bandpower = self.bandpower(sample[i], SAMPLING_FREQUENCY, [8, 12], window_sec=10, relative=True)
            beta_bandpower = self.bandpower(sample[i], SAMPLING_FREQUENCY, [12, 35], window_sec=10, relative=True)
            gamma_bandpower = self.bandpower(sample[i], SAMPLING_FREQUENCY, [35, 45], window_sec=10, relative=True)

            features = [spectral, svd, samp, theta_bandpower, alpha_bandpower, beta_bandpower, gamma_bandpower]
            features_for_sample.extend(features)
        return features_for_sample
    
    def extract_features(self, sub_data, sub_labels, out_dir='DEAP_Features/'):
        VALENCE = 0
        AROUSAL = 1
        DOMINANCE = 2

        for sub in range(1, self.NUM_SUBJECTS + 1):
            train_data = []
            train_labels_val = []
            train_labels_aro = []
            train_labels_dom = []

            test_data = []
            test_labels_val = []
            test_labels_aro = []
            test_labels_dom = []

            situations = range(1, self.NUM_SITUATIONS + 1)
            random.Random(sub * constants.SEED2).shuffle(situations)
            train_situations = situations[:32]
            test_situations = situations[32:]

            for i in train_situations:
                sub_sit_data = sub_data[sub][i]
                data = []
                labels_val = []
                labels_aro = []
                labels_dom = []

                for j in range(self.START_TIMESTAMP, self.DATA_LEN - self.WINDOW_SIZE, self.SAMPLING_FREQUENCY):
                    print(f"Sub {sub} Situation {i} / 40: ({j} - {j + self.WINDOW_SIZE}) / {len(sub_sit_data[0])} ; Channels = {len(sub_sit_data)}")
                    sample = sub_sit_data[:, j : j + self.WINDOW_SIZE]

                    features = self.time_features(sample, self.NUM_CHANNELS, self.SAMPLING_FREQUENCY, self.WINDOW_SIZE)
                    features.extend(self.frequency_features(sample, self.NUM_CHANNELS, self.SAMPLING_FREQUENCY, self.WINDOW_SIZE))
                    features = np.array(features)
                    print("Features shape = ", features.shape)
                    data.append(features)
                    labels_val.append(0 if sub_labels[sub][i - 1][VALENCE] < 5 else 1)
                    labels_aro.append(0 if sub_labels[sub][i - 1][AROUSAL] < 5 else 1)
                    labels_dom.append(0 if sub_labels[sub][i - 1][DOMINANCE] < 5 else 1)
                
                train_data.extend(data)
                train_labels_val.extend(labels_val)
                train_labels_aro.extend(labels_aro)
                train_labels_dom.extend(labels_dom)
                
            train_data = np.array(train_data)
            train_labels_val = np.array(train_labels_val)
            train_labels_aro = np.array(train_labels_aro)
            train_labels_dom = np.array(train_labels_dom)

            for i in test_situations:
                sub_sit_data = sub_data[sub][i]
                data = []
                labels_val = []
                labels_aro = []
                labels_dom = []

                for j in range(self.START_TIMESTAMP, self.DATA_LEN - self.WINDOW_SIZE, self.SAMPLING_FREQUENCY):
                    print(f"Sub {sub} Situation {i} / 40: ({j} - {j + self.WINDOW_SIZE}) / {len(sub_sit_data[0])} ; Channels = {len(sub_sit_data)}")
                    sample = sub_sit_data[:, j : j + self.WINDOW_SIZE]

                    features = self.time_features(sample, self.NUM_CHANNELS, self.SAMPLING_FREQUENCY, self.WINDOW_SIZE)
                    features.extend(self.frequency_features(sample, self.NUM_CHANNELS, self.SAMPLING_FREQUENCY, self.WINDOW_SIZE))
                    features = np.array(features)
                    print("Features shape = ", features.shape)
                    data.append(features)
                    labels_val.append(0 if sub_labels[sub][i - 1][VALENCE] < 5 else 1)
                    labels_aro.append(0 if sub_labels[sub][i - 1][AROUSAL] < 5 else 1)
                    labels_dom.append(0 if sub_labels[sub][i - 1][DOMINANCE] < 5 else 1)
                
                test_data.extend(data)
                test_labels_val.extend(labels_val)
                test_labels_aro.extend(labels_aro)
                test_labels_dom.extend(labels_dom)

            test_data = np.array(test_data)
            test_labels_val = np.array(test_labels_val)
            test_labels_aro = np.array(test_labels_aro)
            test_labels_dom = np.array(test_labels_dom)

            with open(out_dir + "/S%02d_X_train.npy" % (sub), "wb") as f:
                np.save(f, train_data)
            del train_data

            with open(out_dir + "/S%02d_Y_train_val.npy" % (sub), "wb") as f:
                np.save(f, train_labels_val)
            del train_labels_val

            with open(out_dir + "/S%02d_Y_train_aro.npy" % (sub), "wb") as f:
                np.save(f, train_labels_aro)
            del train_labels_aro

            with open(out_dir + "/S%02d_Y_train_dom.npy" % (sub), "wb") as f:
                np.save(f, train_labels_dom)
            del train_labels_dom

            with open(out_dir + "/S%02d_X_test.npy" % (sub), "wb") as f:
                np.save(f, test_data)
            del test_data

            with open(out_dir + "/S%02d_Y_test_val.npy" % (sub), "wb") as f:
                np.save(f, test_labels_val)
            del test_labels_val

            with open(out_dir + "/S%02d_Y_test_aro.npy" % (sub), "wb") as f:
                np.save(f, test_labels_aro)
            del test_labels_aro

            with open(out_dir + "/S%02d_Y_test_dom.npy" % (sub), "wb") as f:
                np.save(f, test_labels_dom)
            del test_labels_dom

