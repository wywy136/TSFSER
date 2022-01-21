import torch
from torch.utils.data import Dataset
import numpy as np

import csv
import sys
sys.path.append('..')

from feature_extractor.teo import TeoFeatureExtractorAverage
from feature_extractor.gemaps import GemapsFeatureExtractorAverage


class SusasDataset(Dataset):
    def __init__(self, args):
        Dataset.__init__(self)
        self.args = args
        self.csv_data = open(self.args.susas_path, 'r')
        
        self.ori_data = []
        csv_reader = csv.reader(self.csv_data)
        header = next(csv_reader)
        for row in csv_reader:
            self.ori_data.append(row)
        # self.ori_data[0]: ['0', '/project/graziul/data/corpora/susas/speech/actual/roller/f1/free_oov_all/all1.sph', 'High', 'Negative', 'all']
     
        self.teo_extractor = TeoFeatureExtractorAverage(self.args)
        self.gemaps_extractor = GemapsFeatureExtractorAverage(self.args)
        
    def __len__(self) -> int:
        return len(self.ori_data)
    
    def __getitem__(self, index) -> dict:
        piece = self.ori_data[index]
        path = piece[1]
        arousal = piece[2]
        valence = piece[3]
        text = piece[4]
        
        # (16,)
        teo_feature: np.ndarray = self.teo_extractor(piece[1])
        # (25,)
        gemaps_feature: np.ndarray = self.gemaps_extractor(piece[1])
        label = self.args.label_map[arousal + valence]
        
        return {
            "teo": teo_feature,
            "gemaps": gemaps_feature,
            "label": label
        }

    
class SusasCollator(object):
    def __call__(self, batch: dict) -> dict:
        teo = np.array([each["teo"] for each in batch], dtype=np.float32)
        gemaps = np.array([each["gemaps"] for each in batch], dtype=np.float32)
        label = np.array([each["label"] for each in batch], dtype=np.int32)
        
        teo = torch.from_numpy(teo)
        gemaps = torch.from_numpy(gemaps)
        label = torch.from_numpy(label)
        
        return {
            "teo": teo,
            "gemaps": gemaps,
            "label": label
        }
