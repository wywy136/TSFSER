import csv
import sys
sys.path.append('..')
import json

from feature_extractor.teo import TeoFeatureExtractorAverage
from feature_extractor.gemaps import GemapsFeatureExtractorAverage


class GemapsWriter:
    def __init__(self, args):
        self.args = args
        self.gemaps = GemapsFeatureExtractorAverage(self.args)

        self.susas_csv = open(self.args.susas_path, 'r')
        self.susas_data = []
        csv_reader = csv.reader(self.susas_csv)
        header = next(csv_reader)
        for row in csv_reader:
            self.susas_data.append(row)

        self.bpc_csv = open(self.args.bpc_path, 'r')
        self.bpc_data = []
        csv_reader = csv.reader(self.bpc_csv)
        header = next(csv_reader)
        for row in csv_reader:
            self.bpc_data.append(row)


    def __call__(self):
        json_data = []
        for i, piece in enumerate(self.susas_data):
            if i % 100 == 0:
                print(f"{i}/{len(self.susas_data)}")
            dic = {
                "path": piece[1],
                "arousal": piece[2],
                "valence": piece[3],
                "gemaps": self.gemaps(piece[1]).tolist()
            }
            json_data.append(dic)
        
        json_str = json.dumps(json_data)
        with open(self.args.susas_feature_path, 'w') as f:
            f.write(json_str)
        
        json_data = []
        for i, piece in enumerate(self.bpc_data):
            print(f"{i}/{len(self.bpc_data)}")
            dic = {
                "path": piece[1],
                "gemaps": self.gemaps(piece[1]).tolist()
            }
            json_data.append(dic)
        
        json_str = json.dumps(json_data)
        with open(self.args.bpc_feature_path, 'w') as f:
            f.write(json_str)
