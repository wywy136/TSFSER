from operator import ge
import torch
from torch.utils.data import Dataset
import numpy as np

import sys
sys.path.append('..')
import json
import csv

from feature_extractor.teo import TeoFeatureExtractorAverage
from feature_extractor.gemaps import GemapsFeatureExtractorAverage


class SusasDataset(Dataset):
    def __init__(self, args, split):
        Dataset.__init__(self)
        self.args = args
        # with open(self.args.susas_feature_path, 'r') as f:
        #     self.ori_data = json.load(f)
        self.csv_data = open(self.args.susas_path, 'r')
        
        self.ori_data = []
        csv_reader = csv.reader(self.csv_data)
        header = next(csv_reader)
        for row in csv_reader:
            self.ori_data.append(row)
        # self.ori_data[0]: ['0', '/project/graziul/data/corpora/susas/speech/actual/roller/f1/free_oov_all/all1.sph', 'High', 'Negative', 'all']
        
        data_size = len(self.ori_data)
        train_size = int(self.args.train_eval_split * data_size)
        if split == "train":
            self.ori_data = self.ori_data
        else:
            self.ori_data = self.ori_data[train_size:]
        
        self.teo_extractor = TeoFeatureExtractorAverage(self.args)
        self.gemaps_extractor = GemapsFeatureExtractorAverage(self.args)
        
    def __len__(self) -> int:
        return len(self.ori_data)
    
    def __getitem__(self, index) -> dict:
        piece = self.ori_data[index]
        # path = piece["path"]
        # arousal = piece["arousal"]
        # valence = piece["valence"]
        path = piece[1]
        arousal = piece[2]
        valence = piece[3]
        
        # (16,)
        # teo_feature: np.ndarray = self.teo_extractor(piece[1])
        teo_feature: np.ndarray = np.zeros(1)
        # (25,)
        gemaps_feature: np.ndarray = self.gemaps_extractor(path)
        # print(gemaps_feature.shape)
        gemaps_feature = gemaps_feature.tolist()
        gemaps_feature = gemaps_feature[:self.args.max_len_susas]
        padding = [0. for i in range(self.args.gemaps_feature_size)]
        # print(len(padding))
        gemaps_feature += [padding] * (self.args.max_len_susas - len(gemaps_feature))
        gemaps_feature = np.array(gemaps_feature)
        # print(gemaps_feature.shape)
        # gemaps_feature: np.ndarray = np.array(piece["gemaps"])
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

        # # padding gemaps
        # max_length = 0
        # padding = [0. for i in range(len(gemaps[0][0]))]
        # for i in range(len(gemaps)):
        #     max_length = max(max_length, len(gemaps[i]))
        # for i in range(len(gemaps)):
        #     gemaps[i] += [padding] * (max_length - len(gemaps[i]))
        # gemaps = np.array(gemaps, dtype=np.float32)
        # # print(gemaps.shape)
        
        teo = torch.from_numpy(teo)
        gemaps = torch.from_numpy(gemaps)
        label = torch.from_numpy(label)

        return {
            "path": path,
            "teo": teo,
            "gemaps": gemaps,
            "label": label
        }
