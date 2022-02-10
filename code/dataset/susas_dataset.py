import torch
from torch.utils.data import Dataset
import numpy as np

import sys
sys.path.append('..')
import json

from feature_extractor.teo import TeoFeatureExtractorAverage
from feature_extractor.gemaps import GemapsFeatureExtractorAverage


class SusasDataset(Dataset):
    def __init__(self, args, split):
        Dataset.__init__(self)
        self.args = args
        with open(self.args.susas_feature_path, 'r') as f:
            self.ori_data = json.load(f)
        
        # self.ori_data = []
        # csv_reader = csv.reader(self.csv_data)
        # header = next(csv_reader)
        # for row in csv_reader:
        #     self.ori_data.append(row)
        # self.ori_data[0]: ['0', '/project/graziul/data/corpora/susas/speech/actual/roller/f1/free_oov_all/all1.sph', 'High', 'Negative', 'all']
        
        data_size = len(self.ori_data)
        train_size = int(self.args.train_test_split * data_size)
        if split == "train":
            self.ori_data = self.ori_data[0:train_size]
        else:
            self.ori_data = self.ori_data[train_size:]
        
        # self.teo_extractor = TeoFeatureExtractorAverage(self.args)
        # self.gemaps_extractor = GemapsFeatureExtractorAverage(self.args)
        
    def __len__(self) -> int:
        return len(self.ori_data)
    
    def __getitem__(self, index) -> dict:
        piece = self.ori_data[index]
        path = piece["path"]
        arousal = piece["arousal"]
        valence = piece["valence"]
        
        # (16,)
        # teo_feature: np.ndarray = self.teo_extractor(piece[1])
        teo_feature: np.ndarray = np.zeros(1)
        # (25,)
        gemaps_feature: np.ndarray = np.array(piece["gemaps"])
        label = self.args.label_map[arousal + valence]
        
        return {
            "path": path,
            "teo": teo_feature,
            "gemaps": gemaps_feature,
            "label": label
        }

    
class SusasCollator(object):
    def __call__(self, batch: dict) -> dict:
        teo = np.array([each["teo"] for each in batch], dtype=np.float32)
        gemaps = np.array([each["gemaps"] for each in batch], dtype=np.float32)
        label = np.array([each["label"] for each in batch], dtype=np.long)
        path = [each["path"] for each in batch]
        
        teo = torch.from_numpy(teo)
        gemaps = torch.from_numpy(gemaps)
        label = torch.from_numpy(label)

        return {
            "path": path,
            "teo": teo,
            "gemaps": gemaps,
            "label": label
        }
