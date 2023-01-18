import csv
import sys
sys.path.append('..')
import json

from feature_extractor.mfcc import MFCCFeatureExtractorAverage
from config import Argument


class MFCCWriter:
    def __init__(self, args):
        self.args = args
        self.mfcc = MFCCFeatureExtractorAverage(self.args)

        self.susas_csv = open("../data/susas_path.csv", 'r')  # self.args.susas_path
        self.susas_data = []
        csv_reader = csv.reader(self.susas_csv)
        header = next(csv_reader)
        for row in csv_reader:
            self.susas_data.append(row)

        # self.bpc_csv = open(self.args.bpc_path, 'r')
        # self.bpc_data = []
        # csv_reader = csv.reader(self.bpc_csv)
        # header = next(csv_reader)
        # for row in csv_reader:
        #     self.bpc_data.append(row)


    def __call__(self):
        # json_data = []
        # for i, piece in enumerate(self.susas_data):
        #     if i % 100 == 0:
        #         print(f"{i}/{len(self.susas_data)}")
        #     dic = {
        #         "path": piece[1],
        #         "arousal": piece[2],
        #         "valence": piece[3],
        #         "gemaps": self.gemaps(piece[1]).tolist()
        #     }
        #     json_data.append(dic)
        
        # json_str = json.dumps(json_data)
        # with open(self.args.susas_feature_path, 'w') as f:
        #     f.write(json_str)
        
        json_data = []
        print("Generating GeMAPS feature for BPC. This will take quite a period of time.")
        for i, piece in enumerate(self.susas_data):
            if i % 10 == 0:
                print(f"Finished: {i}/{len(self.susas_data)}")
            dic = {
                "path": piece[1],
                "gemaps": self.mfcc(piece[1]).tolist()
            }
            json_data.append(dic)
        
        json_str = json.dumps(json_data)
        with open("../data/susas_feature_mfcc.json", 'w') as f:  # self.args.susas_mfcc_feature_path
            f.write(json_str)
        f.close()
        print(f"MFCC generation completed! Features in {self.args.susas_mfcc_feature_path}.")


mfcc_writer = MFCCWriter(Argument)
mfcc_writer()
