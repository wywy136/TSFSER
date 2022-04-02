import torch
from torch.utils.data import Dataset
import numpy as np

import csv
import sys
sys.path.append('..')
import json

from feature_extractor.teo import TeoFeatureExtractorAverage
from feature_extractor.gemaps import GemapsFeatureExtractorAverage

class BpcDataset(Dataset):
    def __init__(self, args):
        Dataset.__init__(self)
        self.args = args
        # self.csv_data = open(self.args.bpc_path, 'r')
        with open(self.args.bpc_feature_path, 'r') as f:
            self.ori_data = json.load(f)
        
        # self.ori_data = []
        # csv_reader = csv.reader(self.csv_data)
        # header = next(csv_reader)
        # # [0,/project/graziul/data/Zone1/2018_08_12/201808120932-28710-27730.mp3,00.02.21.252,00.02.31.279,RADIOSHOP TESTING ONE TWO THREE FOUR FIVE FIVE FOUR THREE TWO ONE RADIO SHOP TEST,10.027]
        # for i, row in enumerate(csv_reader):
        #     if i > self.args.bpc_num:
        #         break
        #     self.ori_data.append(row)

        data_size = len(self.ori_data)
        # train_size = int(self.args.train_test_split * data_size)
        # if split == "train":
        #     self.ori_data = self.ori_data[0:train_size]
        # else:
        #     self.ori_data = self.ori_data[train_size:]
        
        # self.teo_extractor = TeoFeatureExtractorAverage(self.args)
        # self.gemaps_extractor = GemapsFeatureExtractorAverage(self.args)
    
    @staticmethod
    def get_second(time: str) -> float:
        [h, m, s, ms] = time.split('.')
        return int(m) * 60 + int(s) + 0.001 * int(ms)
        
    def __len__(self) -> len:
        return len(self.ori_data)
    
    def __getitem__(self, index) -> dict:
        piece = self.ori_data[index]
        path = piece["path"]
        # start = piece[2]
        # end = piece[3]
        # text = piece[4]
        
        # start_second = self.get_second(start)
        # end_second = self.get_second(end)
        # (16,)
        # teo_feature: np.ndarray = self.teo_extractor(piece[1], start_second, end_second)
        teo_feature: np.ndarray = np.zeros(1)
#         if teo_feature.shape != (16,):
#             print(teo_feature)
        gemaps_feature: np.ndarray = np.array(piece["gemaps"]) # np.array(piece["gemaps"])
        
        return {
            "teo": teo_feature,
            "gemaps": gemaps_feature,
            "path": path
        }
    

class BpcCollator(object):
    def __call__(self, batch: dict) -> dict:
        teo = np.array([each["teo"] for each in batch], dtype=np.float32)
        gemaps = np.array([each["gemaps"] for each in batch], dtype=np.float32)
        path = [each["path"] for each in batch]
        # start = [each["start"] for each in batch]
        # end = [each["end"] for each in batch]
        
        teo = torch.from_numpy(teo)
        gemaps = torch.from_numpy(gemaps)
        
        return {
            "teo": teo,
            "gemaps": gemaps,
            "path": path
        }
