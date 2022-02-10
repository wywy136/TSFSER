from config import Argument
from trainer.trainer import Trainer


if __name__ == "__main__":
    # g = GemapsWriter(Argument)
    # g()
    t = Trainer(Argument)
    t.train()
