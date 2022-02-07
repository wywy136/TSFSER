from torch.nn import CrossEntropyLoss, L1Loss
import torch


class ClassificationLoss():
    def __init__(self):
        self.loss_fn = CrossEntropyLoss()
    
    def __call__(self, logits, labels):
        return self.loss_fn(
            input=logits,
            target=labels
        )


class MMDLoss():
    def __init__(self):
        self.loss_fn = L1Loss()

    def __call__(self, susas, bpc):
        s_rps = torch.mean(susas, 0)
        b_rps = torch.mean(bpc, 0)
        return self.loss_fn(
            input=s_rps, 
            target=b_rps
        )