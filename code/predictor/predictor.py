import csv

import torch
from torch.nn import Softmax

class Predictor:
    def __init__(self, args):
        self.args = args
        self.write_path = self.args.bpc_output_path
        self.softmax = Softmax(dim=1)

    def __call__(self, model, bpc_dataloader):
        print(f"Predicting labels for BPC")
        output = []

        size = len(bpc_dataloader)
        for index_b, batch_b in enumerate(bpc_dataloader):
            if index_b % 100 == 0:
                print(f"{index_b}/{size}")

            logits = model(None, batch_b, "predict")
            
            probs = self.softmax(logits).tolist()
            # print(probs)
            labels = torch.argmax(logits, dim=1).tolist()
            # print(labels)

            for i in range(len(probs)):
                output.append([
                    batch_b["path"][i],
                    batch_b["start"][i],
                    batch_b["end"][i],
                    labels[i],
                    probs[i][0],
                    probs[i][1],
                    probs[i][2]
                ])
        
        print(f"Start writing file: {self.write_path}")
        with open(self.write_path, 'w') as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(["path", "start", "end", "label", "0", "1", "2"])
            writer.writerows(output)
        
        csvfile.close()
