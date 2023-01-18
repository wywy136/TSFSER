import numpy as np
import pandas as pd
import opensmile
import pydub
import librosa
from scipy import signal
from mutagen.mp3 import MP3
from numpy import sign, trapz
from teager_py import Teager
import statsmodels.api as sm
import wave
from python_speech_features import mfcc

import warnings
warnings.filterwarnings('ignore')


class MFCCFeatureExtractorAverage:
    def __init__(self, args):
        self.args = args
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,    
            #feature_level=opensmile.FeatureLevel.Functionals,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
        self.result = None

    def librosa_to_audiosegment(self, filename):
        y, sr = librosa.load(filename)
        # convert from float to uint16
        y = np.array(y * (1<<15), dtype=np.int16)
        audio_segment = pydub.AudioSegment(
            y.tobytes(), 
            frame_rate=sr,
            sample_width=y.dtype.itemsize, 
            channels=1
        )
        return audio_segment
    
    def get_result(self):
        return np.mean(self.result, axis=1)
        # return self.result

    def __call__(self, filepath, start=None, end=None):
        audio = self.librosa_to_audiosegment(filepath) # convert librosa array to audiosegment
        sr = audio.frame_rate # get sample rate
        length = audio.duration_seconds
        
        audio = np.array(audio.get_array_of_samples(), dtype=np.float32) # convert audiosegment to array

        # print(audio.shape)
        # self.result = mfcc(audio)
        self.result = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40
        )
        # print(self.result.shape)
        # print(self.get_result().shape)
        # audio_size = audio.shape[0]
        
        # if start is not None:
        #     start_point = int(audio_size * start / length)
        #     end_point = int(audio_size * end / length)
        #     audio = audio[start_point:end_point]
        
        # self.result = self.smile.process_signal(audio, sr).to_numpy() # get GeMAPS
        return self.get_result()
