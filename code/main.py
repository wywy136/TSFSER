from dataset.susas_dataset import SusasDataset
from config import Argument
from trainer.trainer import Trainer
from feature_extractor.gemaps_writer import GemapsWriter


if __name__ == "__main__":
    # g = GemapsWriter(Argument)
    # g()
    t = Trainer(Argument)
    t.train()
