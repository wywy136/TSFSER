from torch.nn import CrossEntropyLoss


class ClassificationLoss():
    def __init__(self):
        self.loss_fn = CrossEntropyLoss()
    
    def __call__(self, logits, labels):
        return self.loss_fn(
            input=logits,
            target=labels
        )