import pydub
import librosa
from scipy import signal
from mutagen.mp3 import MP3
import numpy as np
from numpy import sign, trapz
from teager_py import Teager
import statsmodels.api as sm
import wave
from sphfile import SPHFile


class TeoFeatureExtractorAverage:
    def __init__(self, args):
        self.args = args
        self.audio_file = None
        self.audio_data = None
        self.audio_data_all = None
        self.filted_data = []
        self.sampling_rate = 0
        self.max_frequency = 0.
        self.num_filter = 16
        self.delta = self.args.delta_for_teo
        self.teo_feature = None
        self.acf_feature = []
        self.envelope = []
        self.area = []
        self.critical_bands = [
            [100, 200],
            [200, 300],
            [300, 400],
            [400, 510],
            [510, 630],
            [630, 770],
            [770, 920],
            [920, 1080],
            [1080, 1270],
            [1270, 1480],
            [1480, 1720],
            [1720, 2000],
            [2000, 2320],
            [2320, 2700],
            [2700, 3150],
            [3150, 3700]
        ]
        self.all_features = []
        self.result = None


    def clear_data(self):
        self.filted_data = []
        self.acf_feature = []
        self.envelope = []
        self.area = []

    
    def get_teo_feature(self):
        self.teo_feature = Teager(self.filted_data, 'horizontal', 1)
        if self.teo_feature is None:
            raise AssertionError


    def get_audio_data(self, filepath, start, end):
        self.audio_data_all = []
        # check the format: mp3 or sph
        if filepath.endswith("sph"):
            sph = SPHFile(filepath)
            self.audio_data_all.append(sph.content)
            # print(sph.content)
            self.audio_file = sph
        else:
            song = pydub.AudioSegment.from_mp3(filepath)
            self.audio_file = song
            # print(type(song))
            duration = song.duration_seconds * 1000
            # song = np.array(song.get_array_of_samples())

            # print(f"Wav file length: {duration}")
            length = start * 1000
            
            while length + int(self.delta * 1000) <= end * 1000:
                song_piece = song[length:length + int(self.delta * 1000)]
                self.audio_data_all.append(np.array(song_piece.get_array_of_samples()).astype(np.int64))
                length += int(self.delta * 1000)
            # print(len(self.audio_data_all))

    def get_sampling_rate(self, filepath):
        if filepath.endswith("sph"):
            return self.audio_file.format['sample_rate']
        return 22050

    @staticmethod
    def get_max_frequency(filepath, sampling_rate):
        data, _ = librosa.load(filepath, sr=sampling_rate)
        frequencies, times, spectrogram = signal.spectrogram(data, sampling_rate)
        return max(frequencies)


    def cb_filter(self):
        for i in range(self.num_filter):
            wn1, wn2 = 2.9 * self.critical_bands[i][0] / self.max_frequency, 2.9 * self.critical_bands[i][1] / self.max_frequency
            wn1 = min(wn1, 0.999)
            wn2 = min(wn2, 0.999)
            # print(wn1, wn2)
            b, a = signal.butter(1, [wn1, wn2], 'bandpass')
            # print(b, a, self.audio_data.shape)
            self.filted_data.append(signal.filtfilt(b, a, self.audio_data))
        self.filted_data = np.array(self.filted_data)
        if self.filted_data.shape != (self.num_filter, len(self.audio_data)):
            raise AssertionError

    
    def get_acf_feature(self):
        for i in range(self.num_filter):
            self.acf_feature.append(sm.tsa.stattools.acf(self.teo_feature[i]))
        self.acf_feature = np.array(self.acf_feature)

    
    def get_envelope(self):
        for i in range(self.num_filter):
            y_upper, y_lower = self.envelope_extraction(self.acf_feature[i])
            self.envelope.append([y_upper, y_lower])


    def get_area(self):
        for i in range(self.num_filter):
            self.area.append(trapz(self.envelope[i][0]))
        self.area = np.array(self.area)
        # print()


    def __call__(self, filepath, start=None, end=None) -> np.ndarray:
        # self.delta = delta
        self.get_audio_data(filepath, start, end)
        self.sampling_rate = self.get_sampling_rate(filepath)
        self.max_frequency = self.get_max_frequency(filepath, self.sampling_rate)
        # print("Computing TEO Features ...")
        nums = len(self.audio_data_all)
        for i, audio in enumerate(self.audio_data_all):
            self.audio_data = audio
            self.clear_data()
            self.cb_filter()
            self.get_teo_feature()
            self.get_acf_feature()
            self.get_envelope()
            self.get_area()
            self.all_features.append(self.area)

        self.result = np.nan_to_num(np.array(self.all_features))
        return self.get_result()
        
    def get_result(self):
        res = np.mean(self.result, axis=0)
        assert res.shape == (16,), f"{len(self.all_features)}"
        return res


    def envelope_extraction(self, signal):
        s = signal.astype(float)
        q_u = np.zeros(s.shape)
        q_l = np.zeros(s.shape)

        u_x = [0, ] 
        u_y = [s[0], ]  

        l_x = [0, ] 
        l_y = [s[0], ]

        for k in range(1, len(s) - 1):
            if (sign(s[k] - s[k - 1]) == 1) and (sign(s[k] - s[k + 1]) == 1):
                u_x.append(k)
                u_y.append(s[k])

            if (sign(s[k] - s[k - 1]) == -1) and ((sign(s[k] - s[k + 1])) == -1):
                l_x.append(k)
                l_y.append(s[k])

        u_x.append(len(s) - 1)
        u_y.append(s[-1])  

        l_x.append(len(s) - 1)
        l_y.append(s[-1])  

        upper_envelope_y = np.zeros(len(signal))
        lower_envelope_y = np.zeros(len(signal))

        upper_envelope_y[0] = u_y[0]
        upper_envelope_y[-1] = u_y[-1]
        lower_envelope_y[0] = l_y[0]  
        lower_envelope_y[-1] = l_y[-1]

        last_idx, next_idx = 0, 0
        k, b = self.general_equation(u_x[0], u_y[0], u_x[1], u_y[1])
        for e in range(1, len(upper_envelope_y) - 1):
            if e not in u_x:
                v = k * e + b
                upper_envelope_y[e] = v
            else:
                idx = u_x.index(e)
                upper_envelope_y[e] = u_y[idx]
                last_idx = u_x.index(e)
                next_idx = u_x.index(e) + 1
                k, b = self.general_equation(u_x[last_idx], u_y[last_idx], u_x[next_idx], u_y[next_idx])

        last_idx, next_idx = 0, 0
        k, b = self.general_equation(l_x[0], l_y[0], l_x[1], l_y[1])
        for e in range(1, len(lower_envelope_y) - 1):

            if e not in l_x:
                v = k * e + b
                lower_envelope_y[e] = v
            else:
                idx = l_x.index(e)
                lower_envelope_y[e] = l_y[idx]
                last_idx = l_x.index(e)
                next_idx = l_x.index(e) + 1
                k, b = self.general_equation(l_x[last_idx], l_y[last_idx], l_x[next_idx], l_y[next_idx])

        return upper_envelope_y, lower_envelope_y


    def general_equation(self, first_x, first_y, second_x, second_y):
        A = second_y - first_y
        B = first_x - second_x
        C = second_x * first_y - first_x * second_y
        k = -1 * A / B
        b = -1 * C / B
        return k, b

