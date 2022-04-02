from feature_extractor.gemaps_writer import GemapsWriter
# from trainer.trainer import Trainer
from config import Argument


# Usage: 
#   - in config.py, set `bpc_path` to your csv path
#   - in config.py, set `prediction` to True
#   - run this script
# It will take hours time to finish the gemaps extraction
class Interface:
    def __init__(self, args):
        self.args = args
        self.gemaps_writer = GemapsWriter(self.args)
        # self.trainer = Trainer(self.args)
    
    def __call__(self):
        # generate Gemaps features for BPC
        self.gemaps_writer()
        # self.trainer.train()


if __name__ == "__main__":
    interface = Interface(Argument)
    interface()
