# from sphfile import SPHFile
# import librosa
# from scipy import signal

# from code.feature_extractor.teo import TeoFeatureExtractorAverage

# # filepath = "/project/graziul/data/corpora/susas/speech/actual/roller/f1/freefall/freeze1.sph"
# filepath = "/project/graziul/data/Zone1/2018_12_20/201812202328-822223-27730.mp3"
# # sph = SPHFile("/project/graziul/data/corpora/susas/speech/actual/roller/f1/freefall/freeze1.sph")

# # data, _ = librosa.load("/project/graziul/data/corpora/susas/speech/actual/roller/f1/freefall/freeze1.sph", sr=8000)
# # frequencies, times, spectrogram = signal.spectrogram(data, 8000)
# # print(max(frequencies))
# # print(sph.format['sample_rate'])

# teo = TeoFeatureExtractorAverage()
# teo_feature = teo(filepath, 0.5, 6.5)
# print(teo_feature.shape)

import opensmile

config_str = '''
[componentInstances:cComponentManager]
instance[dataMemory].type=cDataMemory

;;; default source
[componentInstances:cComponentManager]
instance[dataMemory].type=cDataMemory

;;; source

\{\cm[source{?}:include external source]}

;;; main section

[componentInstances:cComponentManager]
instance[framer].type = cFramer
instance[lld].type = cEnergy
instance[func].type=cFunctionals

[framer:cFramer]
reader.dmLevel = wave
writer.dmLevel = frames
copyInputName = 1
frameMode = fixed
frameSize = 0.025000
frameStep = 0.010000
frameCenterSpecial = left
noPostEOIprocessing = 1

[lld:cEnergy]
reader.dmLevel = frames
writer.dmLevel = lld
\{\cm[bufferModeRbConf{?}:path to included config to set the buffer mode for the standard ringbuffer levels]}
nameAppend = energy
copyInputName = 1
rms = 1
log = 1

[func:cFunctionals]
reader.dmLevel=lld
writer.dmLevel=func
copyInputName = 1
\{\cm[bufferModeRbConf]}
\{\cm[frameModeFunctionalsConf{?}:path to included config to set frame mode for all functionals]}
functionalsEnabled=Moments
Moments.variance = 0
Moments.stddev = 1
Moments.skewness = 0
Moments.kurtosis = 0
Moments.amean = 1
Moments.doRatioLimit = 0

;;; sink

\{\cm[sink{?}:include external sink]}

'''

with open('my.conf', 'w') as fp:
    fp.write(config_str)

smile = opensmile.Smile(
    feature_set='my.conf',
    feature_level='lld',
)
print(smile.feature_names)
