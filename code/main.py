from dataset.susas_dataset import SusasDataset
from config import Argument
from trainer.trainer import Trainer


if __name__ == "__main__":
    t = Trainer(Argument)
    t.train()
    