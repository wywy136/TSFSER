import sys
sys.path.append('..')
from dataset.bpc_dataset import BpcDataset


class TeoWriter:
    def __init__(self):
        self.bpc_dataset = BpcDataset(self.args, "train")

    def __call__(self):
        