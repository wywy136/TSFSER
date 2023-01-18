import sys
sys.path.append('..')
from feature_extractor.teo import TeoFeatureExtractorAverage

import csv
import json


class TeoWriter:
    def __init__(self, args):
        self.args = args
        self.teo = TeoFeatureExtractorAverage(self.args)

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
        print("Generating TEO features. This will take quite a period of time.")
        for i, piece in enumerate(self.susas_data):
            if i % 100 == 0:
                print(f"Finished: {i}/{len(self.susas_data)}")
            dic = {
                "path": piece[1],
                "gemaps": self.teo(piece[1]).tolist()
            }
            json_data.append(dic)

        json_str = json.dumps(json_data)
        with open(self.args.susas_feature_path, 'w') as f:
            f.write(json_str)
        f.close()

        print(f"TEO feature generation completed! Features in {self.args.susas_feature_path}.")
